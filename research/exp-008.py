"""exp-008: KD headroom and dark knowledge from the saved exp-006 val logits. CPU only, no training.

Usage (stdlib + torch only, safe with `python -I`):
    .venv/bin/python -I research/exp-008.py --files F_a F_b F_c F_d F_e F_f --metadata datasets/BioSARDdBScene/metadata.csv

--files: the six exp-006{a..f}-<jobid>-vallogits.pt in seed order (a=42 ... f=47). The "seeds 45-47" ensemble is the LAST THREE files.
Each file holds {"val_logits": [24, N, 8], "val_labels": [N], "val_groups": list[N] of source_id, "halves", "epoch_val_f1"}.
Row i of the val set is the i-th metadata row with split == val (BioSARMulti(split="val") item order).
Prints one `key: value` line per number.
"""
import argparse
import csv

import torch

ORGANS = ["Bowel", "Heart", "Joint", "Lung"]
NUM_CLASSES = 8
EPOCH = 23
N_PERM = 10_000
PERM_SEED = 0
N_BINS = 5


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
    """Macro one-vs-rest AUROC on softmax probabilities (present classes only). Copied from research/exp-007.py."""
    p = logits.softmax(dim=1)
    vals = [binary_auroc(p[:, c], (labels == c).float())
            for c in range(NUM_CLASSES) if (labels == c).any() and (labels != c).any()]
    return sum(vals) / len(vals)


def macro_f1(preds, labels):
    """Macro-F1 over present classes (same formula as multiclass_metrics)."""
    f1s = []
    for c in range(NUM_CLASSES):
        is_c, pred_c = labels == c, preds == c
        if not is_c.any():
            continue
        tp = (is_c & pred_c).sum().item()
        fp = (~is_c & pred_c).sum().item()
        fn = (is_c & ~pred_c).sum().item()
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s)


def organ_abn_auroc(prob, labels):
    """Within-organ abnormality AUROC for Bowel / Heart / Lung from probabilities [N, 8].
    Among rows whose true organ is o: AUROC of p(o,abn) / p(o,normal + abn) against the abnormal label."""
    out = {}
    for name in ("Bowel", "Heart", "Lung"):
        o = ORGANS.index(name)
        rows = (labels // 2) == o
        score = prob[rows, 2 * o + 1] / (prob[rows, 2 * o] + prob[rows, 2 * o + 1])
        out[name] = binary_auroc(score, (labels[rows] % 2).float())
    return out


def avg_ranks(x):
    _, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
    upper = counts.cumsum(0)
    return (upper - (counts - 1) / 2.0)[inverse]


def spearman_perm(x, y, n_perm=N_PERM, seed=PERM_SEED):
    """Spearman rho (average ranks) and two-sided permutation p-value (plain permutations of y, fixed seed)."""
    x, y = x.double(), y.double()
    rx = avg_ranks(x)
    ry = avg_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    rho = ((rx * ry).sum() / denom).item()
    gen = torch.Generator().manual_seed(seed)
    n = len(x)
    count = 0
    done = 0
    while done < n_perm:
        b = min(500, n_perm - done)
        idx = torch.stack([torch.randperm(n, generator=gen) for _ in range(b)])
        rho_p = (ry[idx] * rx).sum(dim=1) / denom
        count += (rho_p.abs() >= abs(rho) - 1e-12).sum().item()
        done += b
    return rho, (1 + count) / (1 + n_perm)


def load_val_metadata(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] == "val":
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description="exp-008: KD headroom and dark knowledge from saved exp-006 val logits (CPU only)")
    ap.add_argument("--files", nargs=6, required=True, help="six exp-006{a..f}-<jobid>-vallogits.pt in seed order (a..f = seeds 42..47)")
    ap.add_argument("--metadata", required=True, help="BioSAR metadata.csv")
    ap.add_argument("--epoch", type=int, default=EPOCH, help="epoch index to analyse (default 23 = last)")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()

    meta = load_val_metadata(args.metadata)
    n = len(meta)
    organ_idx = torch.tensor([ORGANS.index(r["organ"]) for r in meta])
    meta_labels = 2 * organ_idx + torch.tensor([int(r["label"] == "abnormal") for r in meta])
    snr_sec = torch.tensor([float(r["snr_secondary_db"]) for r in meta])
    snr_amb = torch.tensor([float(r["snr_ambient_db"]) for r in meta])
    meta_groups = [r["source_id"] for r in meta]
    print(f"n_val: {n}")

    logits, labels = [], None
    for path in args.files:
        d = torch.load(path, map_location="cpu")
        z = d["val_logits"][args.epoch].float()
        assert z.shape == (n, NUM_CLASSES), f"{path}: logits {tuple(z.shape)} vs metadata val rows {n}"
        assert torch.equal(d["val_labels"].long(), meta_labels), f"{path}: val_labels differ from metadata labels"
        assert list(d["val_groups"]) == meta_groups, f"{path}: val_groups differ from metadata source_id order"
        logits.append(z)
    labels = meta_labels
    print(f"check_labels_groups_match_metadata: ok epoch={args.epoch} files={len(logits)}")

    # ---- per-seed and ensemble discrimination ----
    singles = []
    for i, z in enumerate(logits):
        pa = prob_auroc(z, labels)
        f1 = macro_f1(z.argmax(1), labels)
        oa = organ_abn_auroc(z.softmax(1), labels)
        singles.append(pa)
        print(f"seed{i}: val_auroc_prob={pa:.6f} val_macro_f1={f1:.6f} "
              f"organ_abn_auroc_bowel={oa['Bowel']:.6f} heart={oa['Heart']:.6f} lung={oa['Lung']:.6f}")
    mean_single = sum(singles) / len(singles)
    print(f"single_mean_val_auroc_prob: {mean_single:.6f}")

    def ensemble(zs):
        return torch.stack([z.softmax(1) for z in zs]).mean(0)

    ens = {"ens6": ensemble(logits), "ens3_last": ensemble(logits[3:])}
    for name, p in ens.items():
        logp = p.log()
        pa = prob_auroc(logp, labels)
        f1 = macro_f1(p.argmax(1), labels)
        oa = organ_abn_auroc(p, labels)
        print(f"{name}: val_auroc_prob={pa:.6f} val_macro_f1={f1:.6f} "
              f"organ_abn_auroc_bowel={oa['Bowel']:.6f} heart={oa['Heart']:.6f} lung={oa['Lung']:.6f} "
              f"organ_abn_auroc_mean={sum(oa.values()) / 3:.6f}")
    s_mean_oa = []
    for z in logits:
        s_mean_oa.append(sum(organ_abn_auroc(z.softmax(1), labels).values()) / 3)
    print(f"single_mean_organ_abn_auroc_mean: {sum(s_mean_oa) / len(s_mean_oa):.6f}")
    head6 = prob_auroc(ens["ens6"].log(), labels) - mean_single
    head3 = prob_auroc(ens["ens3_last"].log(), labels) - mean_single
    print(f"headroom_ens6_minus_mean_single: {head6:.6f}")
    print(f"headroom_ens3_minus_mean_single: {head3:.6f}")

    # ---- dark knowledge: non-target organ mass vs SNR ----
    # primary Lung -> secondary heart, mass m = p(Heart-n) + p(Heart-a); primary Heart -> secondary lung, m = lung mass.
    other = {"Lung": "Heart", "Heart": "Lung"}
    sources = {f"seed{i}": z.softmax(1) for i, z in enumerate(logits)}
    sources["ens6"] = ens["ens6"]
    sources["ens3_last"] = ens["ens3_last"]
    res = {}
    for prim, oth in other.items():
        rows = organ_idx == ORGANS.index(prim)
        o = ORGANS.index(oth)
        print(f"primary_{prim}: n={rows.sum().item()} secondary={oth}")
        for name, p in sources.items():
            m = (p[:, 2 * o] + p[:, 2 * o + 1])[rows]
            r_s, p_s = spearman_perm(m, snr_sec[rows], args.n_perm)
            r_a, p_a = spearman_perm(m, snr_amb[rows], args.n_perm)
            res[(prim, name)] = (r_s, p_s, r_a, p_a)
            print(f"dark_{prim}_{name}: rho_secondary={r_s:.4f} p_secondary={p_s:.5f} rho_ambient={r_a:.4f} p_ambient={p_a:.5f}")
        # binned means for the figure: 5 quantile bins of snr_secondary_db within this primary organ
        s = snr_sec[rows]
        edges = torch.quantile(s.double(), torch.linspace(0, 1, N_BINS + 1, dtype=torch.double))
        bin_id = torch.bucketize(s.double(), edges[1:-1], right=True)
        for name in ("ens6",):
            m = (sources[name][:, 2 * o] + sources[name][:, 2 * o + 1])[rows]
            for b in range(N_BINS):
                sel = bin_id == b
                print(f"bin_{prim}_{name}: bin={b} snr_sec_lo={edges[b].item():.2f} snr_sec_hi={edges[b + 1].item():.2f} "
                      f"n={sel.sum().item()} mean_mass={m[sel].mean().item():.6f} mean_snr_sec={s[sel].mean().item():.2f}")

    # ---- decision summary (informative) ----
    h2 = all(res[(prim, "ens6")][0] < -0.1 and res[(prim, "ens6")][1] < 0.01
             and abs(res[(prim, "ens6")][2]) < abs(res[(prim, "ens6")][0]) for prim in other)
    print("=== summary ===")
    print(f"headroom_ens6: {head6:.6f}  (>=0.005: strong prior for T-ens; <0.002: Stage C only if exp-010 passes its own gate)")
    print(f"H1_headroom_in_0.005_0.02: {0.005 <= head6 <= 0.02}")
    print(f"H2_supported_ens6: {h2}  (rho_secondary < -0.1, p < 0.01, |rho_ambient| < |rho_secondary| for both Lung and Heart primaries)")


if __name__ == "__main__":
    main()
