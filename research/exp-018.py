"""exp-018: data audit on saved val logits. Corpus-stratified within-organ abnormality AUROC, corpus-prior baseline,
grouped bootstrap CIs, ICBHI regrouped by patient, per-class AUROC, organ confusion by corpus, and per-class
decomposition of the late macro-F1 swing. CPU only, no training.

Usage (stdlib + torch only, safe with `python -I`):
    .venv/bin/python -I research/exp-018.py --files F1 ... F6 --metadata datasets/BioSARDdBScene/metadata.csv [--epoch 23] [--n-boot 2000] [--tag str]

--files: 3-6 vallogits.pt files (seed order). Each holds {"val_logits": [24, N, 8], "val_labels": [N], "val_groups": list[N] of source_id, ...}.
Row i of the val set is the i-th metadata row with split == val. The ensemble (mean of softmax) is labelled ens<n>.
Prints one `key: value` line per number.
"""
import argparse
import csv
import math

import torch

ORGANS = ["Bowel", "Heart", "Joint", "Lung"]
CLASS_NAMES = [f"{o}-{s}" for o in ORGANS for s in ("n", "a")]
NUM_CLASSES = 8
EPOCH = 23
N_BOOT = 2000
MIN_STRATUM = 5


def binary_auroc(score, labels):
    """Rank-based AUROC (ties averaged); same computation as utils.utils.binary_f1_auroc."""
    score = score.detach().cpu().double()
    pos = labels.detach().cpu() >= 0.5
    n_pos = pos.sum().item()
    n_neg = pos.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = torch.unique(score, return_inverse=True, return_counts=True)
    upper = counts.cumsum(0)
    avg_rank = (upper - (counts - 1) / 2.0)[inverse]
    return (avg_rank[pos].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def prob_auroc(logits, labels):
    """Macro one-vs-rest AUROC on softmax probabilities (present classes only)."""
    p = logits.softmax(dim=1)
    vals = [binary_auroc(p[:, c], (labels == c).float())
            for c in range(NUM_CLASSES) if (labels == c).any() and (labels != c).any()]
    return sum(vals) / len(vals)


def class_f1s(preds, labels):
    """Per-class F1 (all classes present in val)."""
    out = []
    for c in range(NUM_CLASSES):
        is_c, pred_c = labels == c, preds == c
        tp = (is_c & pred_c).sum().item()
        fp = (~is_c & pred_c).sum().item()
        fn = (is_c & ~pred_c).sum().item()
        out.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
    return out


def corpus_key(r):
    return r["primary_source"].split("/")[2].lower().replace("abnormal", "").replace("normal", "")


def patient_id(r):
    """ICBHI patient = first 3 characters of the file name (103_2b2_... -> 103); None for other corpora."""
    if r["organ"] == "Lung" and corpus_key(r) == "icbhi":
        return r["primary_source"].split("/")[-1][:3]
    return None


def fmt(x):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def pooled_auroc(score, y, corp, strata):
    """sum_s P_s N_s AUROC_s / sum_s P_s N_s over the given strata (strata with a missing class are skipped)."""
    num = den = 0.0
    for s in strata:
        m = corp == s
        pos = int((y[m] == 1).sum())
        neg = int((y[m] == 0).sum())
        if pos == 0 or neg == 0:
            continue
        a = binary_auroc(score[m], y[m].float())
        num += pos * neg * a
        den += pos * neg
    return num / den if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description="exp-018: corpus-stratified within-organ AUROC and swing decomposition (CPU only)")
    ap.add_argument("--files", nargs="+", required=True, help="3-6 vallogits.pt files in seed order")
    ap.add_argument("--metadata", required=True, help="BioSAR metadata.csv")
    ap.add_argument("--epoch", type=int, default=EPOCH, help="epoch index to analyse (default 23 = last)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--tag", default="", help="free label printed in the output")
    args = ap.parse_args()
    assert 3 <= len(args.files) <= 6, "need 3-6 files"
    nf = len(args.files)
    ens_name = f"ens{nf}"
    print(f"tag: {args.tag}")
    print(f"files: {nf} epoch={args.epoch} n_boot={args.n_boot}")

    with open(args.metadata, newline="") as f:
        allrows = list(csv.DictReader(f))
    for r in allrows:
        r["corpus"] = corpus_key(r)
        r["abn"] = int(r["label"] == "abnormal")
    meta = [r for r in allrows if r["split"] == "val"]
    n = len(meta)
    organ_idx = torch.tensor([ORGANS.index(r["organ"]) for r in meta])
    meta_labels = 2 * organ_idx + torch.tensor([r["abn"] for r in meta])
    meta_groups = [r["source_id"] for r in meta]
    print(f"n_val: {n}")

    # ---- 1. counts ----
    cnt = {}
    for r in allrows:
        cnt[(r["organ"], r["corpus"], r["split"], r["abn"])] = cnt.get((r["organ"], r["corpus"], r["split"], r["abn"]), 0) + 1
    keys = sorted({(o, c) for (o, c, _, _) in cnt})
    for o, c in keys:
        for s in ("train", "val", "test"):
            print(f"count: organ={o} corpus={c} split={s} normal={cnt.get((o, c, s, 0), 0)} abnormal={cnt.get((o, c, s, 1), 0)}")
    for o in ("Lung",):
        for a, name in ((0, "normal"), (1, "abnormal")):
            tot = sum(v for (oo, _, _, aa), v in cnt.items() if oo == o and aa == a)
            cough = sum(v for (oo, c, _, aa), v in cnt.items() if oo == o and aa == a and c == "coughdataset")
            print(f"lung_{name}_total_all_splits: {tot} coughdataset: {cough} share={cough / tot:.4f}")

    # ---- 2. corpus-prior baseline ----
    prior = {}
    for o, c in keys:
        nn = cnt.get((o, c, "train", 0), 0)
        na = cnt.get((o, c, "train", 1), 0)
        prior[(o, c)] = na / (nn + na) if nn + na > 0 else float("nan")
        print(f"train_prior: organ={o} corpus={c} p_abnormal={fmt(prior[(o, c)])}")
    for o in ("Bowel", "Heart", "Lung"):
        for s in ("val", "test"):
            rows = [r for r in allrows if r["split"] == s and r["organ"] == o]
            sc = torch.tensor([prior[(o, r["corpus"])] for r in rows])
            y = torch.tensor([r["abn"] for r in rows]).float()
            print(f"corpus_prior_auroc: organ={o} split={s} value={fmt(binary_auroc(sc, y))}")

    # ---- load logits ----
    logits = []
    for path in args.files:
        d = torch.load(path, map_location="cpu")
        vl = d["val_logits"]
        z = vl[args.epoch].float()
        assert z.shape == (n, NUM_CLASSES), f"{path}: logits {tuple(z.shape)} vs metadata val rows {n}"
        assert torch.equal(d["val_labels"].long(), meta_labels), f"{path}: val_labels differ from metadata labels"
        assert list(d["val_groups"]) == meta_groups, f"{path}: val_groups differ from metadata source_id order"
        logits.append((z, vl.float()))
    print(f"check_labels_groups_match_metadata: ok epoch={args.epoch} files={nf}")
    labels = meta_labels
    y_abn = (labels % 2)
    corp = [r["corpus"] for r in meta]
    corp_arr = {c: torch.tensor([x == c for x in corp]) for c in {r["corpus"] for r in meta}}
    corp_id = torch.zeros(n, dtype=torch.long)
    corp_names = sorted(corp_arr)
    for i, c in enumerate(corp_names):
        corp_id[corp_arr[c]] = i

    def organ_rows(o):
        return organ_idx == ORGANS.index(o)

    def score_all(p):
        """within-organ abnormality score for every row using its TRUE organ."""
        o = organ_idx
        idx = torch.arange(n)
        pa, pn = p[idx, 2 * o + 1], p[idx, 2 * o]
        return pa / (pn + pa)

    # strata with >= 5 pos and >= 5 neg per organ
    strata = {}
    for o in ("Bowel", "Heart", "Lung"):
        strata[o] = []
        for ci, c in enumerate(corp_names):
            m = organ_rows(o) & (corp_id == ci)
            pos = int((y_abn[m] == 1).sum())
            neg = int((y_abn[m] == 0).sum())
            if pos >= MIN_STRATUM and neg >= MIN_STRATUM:
                strata[o].append(ci)

    def organ_report(tag, p):
        s = score_all(p)
        res = {}
        for o in ("Bowel", "Heart", "Lung"):
            rows = organ_rows(o)
            ov = binary_auroc(s[rows], y_abn[rows].float())
            pooled = pooled_auroc(s[rows], y_abn[rows], corp_id[rows], strata[o])
            res[o] = (ov, pooled)
            print(f"{tag}_{o}: overall_auroc={fmt(ov)} pooled_within_corpus={fmt(pooled)} between_corpus_part={fmt(ov - pooled)}")
            for ci in strata[o]:
                m = rows & (corp_id == ci)
                a = binary_auroc(s[m], y_abn[m].float())
                print(f"{tag}_{o}_corpus={corp_names[ci]}: auroc={fmt(a)} n_pos={int((y_abn[m] == 1).sum())} n_neg={int((y_abn[m] == 0).sum())}")
        return res

    probs = [z.softmax(1) for z, _ in logits]
    ens = torch.stack(probs).mean(0)

    # ---- 3. model scores ----
    results = {}
    for i, p in enumerate(probs):
        results[f"seed{i}"] = organ_report(f"seed{i}", p)
    results[ens_name] = organ_report(ens_name, ens)
    for o in ("Bowel", "Heart", "Lung"):
        ovs = [results[f"seed{i}"][o][0] for i in range(nf)]
        pls = [results[f"seed{i}"][o][1] for i in range(nf)]
        print(f"single_mean_{o}: overall_auroc={fmt(sum(ovs) / nf)} pooled_within_corpus={fmt(sum(pls) / nf)}")
    s_ens = score_all(ens)
    for o in ("Heart", "Lung"):
        for ci, c in enumerate(corp_names):
            for a in (0, 1):
                m = organ_rows(o) & (corp_id == ci) & (y_abn == a)
                if m.any():
                    print(f"mean_score_{ens_name}: organ={o} corpus={c} label={'abnormal' if a else 'normal'} n={int(m.sum())} mean_s={s_ens[m].mean().item():.4f}")

    # ---- 4. ICBHI by patient ----
    pat_val = [patient_id(r) for r in meta]
    train_pats = {patient_id(r) for r in allrows if r["split"] == "train" and patient_id(r) is not None}
    val_pats = {p for p in pat_val if p is not None}
    unseen = {p for p in val_pats if p not in train_pats}
    print(f"icbhi_val_patients: {len(val_pats)} also_in_train={len(val_pats) - len(unseen)} never_in_train={len(unseen)}")
    ic = corp_names.index("icbhi") if "icbhi" in corp_names else None
    if ic is not None:
        m_ic = organ_rows("Lung") & (corp_id == ic)
        print(f"icbhi_val_recordings: n={int(m_ic.sum())} pos={int((y_abn[m_ic] == 1).sum())} neg={int((y_abn[m_ic] == 0).sum())}")
        m_un = torch.tensor([p in unseen if p is not None else False for p in pat_val]) & m_ic
        npos, nneg = int((y_abn[m_un] == 1).sum()), int((y_abn[m_un] == 0).sum())
        print(f"icbhi_unseen_patient_rows: n={int(m_un.sum())} n_pos={npos} n_neg={nneg}")
        if npos >= MIN_STRATUM and nneg >= MIN_STRATUM:
            for tag, p in [(f"seed{i}", probs[i]) for i in range(nf)] + [(ens_name, ens)]:
                print(f"icbhi_unseen_patient_auroc_{tag}: {fmt(binary_auroc(score_all(p)[m_un], y_abn[m_un].float()))}")
        else:
            print("icbhi_unseen_patient_auroc: too_small (need >= 5 pos and >= 5 neg)")

    # ---- 5. grouped bootstrap (ensemble only) ----
    group_keys = []
    for r, pt in zip(meta, pat_val):
        group_keys.append(f"icbhi::{pt}" if pt is not None else r["source_id"])
    gmap = {}
    for i, g in enumerate(group_keys):
        gmap.setdefault(g, []).append(i)
    gidx = [torch.tensor(v) for v in gmap.values()]
    G = len(gidx)
    print(f"bootstrap_groups: {G}")
    ci_cough = corp_names.index("coughdataset")
    stat_names = ["lung_overall", "lung_within_cough", "lung_pooled_within_corpus", "heart_overall",
                  "heart_pooled_within_corpus", "bowel_overall", "lung_overall_minus_pooled"]

    def stats(idx):
        s, y, oi, ci = s_ens[idx], y_abn[idx], organ_idx[idx], corp_id[idx]
        out = {}
        ml = oi == ORGANS.index("Lung")
        mh = oi == ORGANS.index("Heart")
        mb = oi == ORGANS.index("Bowel")
        out["lung_overall"] = binary_auroc(s[ml], y[ml].float())
        mc = ml & (ci == ci_cough)
        out["lung_within_cough"] = binary_auroc(s[mc], y[mc].float())
        out["lung_pooled_within_corpus"] = pooled_auroc(s[ml], y[ml], ci[ml], strata["Lung"])
        out["heart_overall"] = binary_auroc(s[mh], y[mh].float())
        out["heart_pooled_within_corpus"] = pooled_auroc(s[mh], y[mh], ci[mh], strata["Heart"])
        out["bowel_overall"] = binary_auroc(s[mb], y[mb].float())
        out["lung_overall_minus_pooled"] = out["lung_overall"] - out["lung_pooled_within_corpus"]
        return out

    point = stats(torch.arange(n))
    gen = torch.Generator().manual_seed(0)
    draws = {k: [] for k in stat_names}
    for _ in range(args.n_boot):
        pick = torch.randint(G, (G,), generator=gen)
        idx = torch.cat([gidx[j] for j in pick.tolist()])
        st = stats(idx)
        for k in stat_names:
            if not math.isnan(st[k]):
                draws[k].append(st[k])
    ci_res = {}
    for k in stat_names:
        d = torch.tensor(draws[k], dtype=torch.double)
        lo, hi = torch.quantile(d, torch.tensor([0.025, 0.975], dtype=torch.double)).tolist()
        ci_res[k] = (point[k], lo, hi)
        print(f"boot_{ens_name}_{k}: point={fmt(point[k])} ci95=[{fmt(lo)}, {fmt(hi)}] n_valid={len(draws[k])}")

    # ---- 6. per-class prob-AUROC ----
    pc = []
    for i, p in enumerate(probs):
        v = [binary_auroc(p[:, c], (labels == c).float()) for c in range(NUM_CLASSES)]
        pc.append(v)
        print(f"per_class_auroc_seed{i}: " + " ".join(f"{CLASS_NAMES[c]}={v[c]:.4f}" for c in range(NUM_CLASSES)))
    print(f"per_class_auroc_single_mean: " + " ".join(f"{CLASS_NAMES[c]}={sum(v[c] for v in pc) / nf:.4f}" for c in range(NUM_CLASSES)))
    ve = [binary_auroc(ens[:, c], (labels == c).float()) for c in range(NUM_CLASSES)]
    print(f"per_class_auroc_{ens_name}: " + " ".join(f"{CLASS_NAMES[c]}={ve[c]:.4f}" for c in range(NUM_CLASSES)))

    # ---- 7. organ confusion by corpus ----
    for o, other in (("Lung", "Heart"), ("Heart", "Lung")):
        oo = ORGANS.index(other)
        mass = ens[:, 2 * oo] + ens[:, 2 * oo + 1]
        for ci, c in enumerate(corp_names):
            for a in (0, 1):
                m = organ_rows(o) & (corp_id == ci) & (y_abn == a)
                if m.any():
                    print(f"organ_confusion_{ens_name}: true_organ={o} mass_of={other} corpus={c} label={'abnormal' if a else 'normal'} n={int(m.sum())} mean_mass={mass[m].mean().item():.4f}")

    # ---- 8. swing decomposition ----
    e_lo = args.epoch - 4
    abs_sum = torch.zeros(nf, NUM_CLASSES, dtype=torch.double)
    changed = torch.zeros(nf, NUM_CLASSES, dtype=torch.double)
    n_delta = 0
    for si, (_, vl) in enumerate(logits):
        for e in range(e_lo, args.epoch + 1):
            prev, cur = vl[e - 1].argmax(1), vl[e].argmax(1)
            f_prev, f_cur = class_f1s(prev, labels), class_f1s(cur, labels)
            for c in range(NUM_CLASSES):
                abs_sum[si, c] += abs(f_cur[c] - f_prev[c])
                changed[si, c] += int(((prev != cur) & (labels == c)).sum())
        n_delta = args.epoch - e_lo + 1
    mean_abs = abs_sum / n_delta  # [nf, C]
    share_per_seed = mean_abs / mean_abs.sum(1, keepdim=True)
    mean_changed = changed / n_delta
    n_class = [int((labels == c).sum()) for c in range(NUM_CLASSES)]
    print(f"swing_epochs: {e_lo}..{args.epoch} deltas_per_seed={n_delta}")
    for c in range(NUM_CLASSES):
        print(f"swing_class={CLASS_NAMES[c]}: n_val={n_class[c]} mean_abs_dF1={mean_abs[:, c].mean().item():.4f} "
              f"contrib_to_abs_dmacroF1={mean_abs[:, c].mean().item() / NUM_CLASSES:.5f} "
              f"share_seed_mean={share_per_seed[:, c].mean().item():.4f} mean_n_changed={mean_changed[:, c].mean().item():.2f}")
    tiny = [CLASS_NAMES.index(x) for x in ("Joint-n", "Joint-a", "Bowel-a")]
    lung_a = CLASS_NAMES.index("Lung-a")
    share_tiny = share_per_seed[:, tiny].sum(1)
    share_lung_a = share_per_seed[:, lung_a]
    share_tiny_mean = share_tiny.mean().item()
    print(f"swing_share_joint_n_joint_a_bowel_a: seed_mean={share_tiny_mean:.4f} per_seed=" + ",".join(f"{x:.4f}" for x in share_tiny.tolist())
          + f" n_val_recordings={sum(n_class[c] for c in tiny)}")
    print(f"swing_share_lung_a: seed_mean={share_lung_a.mean().item():.4f} per_seed=" + ",".join(f"{x:.4f}" for x in share_lung_a.tolist()))

    # ---- 9. summary and rules ----
    lung = results[ens_name]["Lung"]
    heart = results[ens_name]["Heart"]
    cough_pt, cough_lo, cough_hi = ci_res["lung_within_cough"]
    diff_pt = lung[0] - lung[1]
    h1 = cough_pt <= 0.55 and diff_pt >= 0.01
    l1 = h1
    l2 = cough_pt >= 0.58 and cough_lo > 0.53
    lung_rule = "L1_confirmed" if l1 else ("L2_rejected" if l2 else "inconclusive")
    h_pt, h_lo, h_hi = ci_res["heart_pooled_within_corpus"]
    heart_real = h_pt >= 0.75 and h_lo > 0.70
    h3 = share_tiny_mean >= 0.5
    print("=== summary ===")
    print(f"tag: {args.tag}")
    print(f"sel_{ens_name}_lung: overall={fmt(lung[0])} pooled_within_corpus={fmt(lung[1])} within_cough={fmt(cough_pt)} ci95=[{fmt(cough_lo)}, {fmt(cough_hi)}] overall_minus_pooled={fmt(diff_pt)}")
    print(f"sel_{ens_name}_heart: overall={fmt(heart[0])} pooled_within_corpus={fmt(heart[1])} pooled_boot_ci95=[{fmt(h_lo)}, {fmt(h_hi)}]")
    print(f"H1_lung_confound (within_cough<=0.55 and overall-pooled>=0.01): {h1}")
    print(f"H2_heart_real (pooled within-corpus >= 0.75): {heart[1] >= 0.75}")
    print(f"H3_swing (Joint-n+Joint-a+Bowel-a share >= 0.5): {h3} share={share_tiny_mean:.4f}")
    print(f"L1_lung_confound_confirmed: {l1}")
    print(f"L2_rejected (within_cough>=0.58 and boot_lower>0.53): {l2}")
    print(f"lung_rule_outcome: {lung_rule}")
    print(f"heart_real_beyond_corpus (pooled>=0.75 and boot_lower>0.70): {heart_real}")
    print(f"swing_source_supported (H3): {h3}")


if __name__ == "__main__":
    main()
