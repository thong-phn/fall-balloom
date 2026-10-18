"""exp-033 (CPU, torch + stdlib only, runs under `python -I`): operating-point rules and ensembles on saved logits.

Input: `-evallogits.pt` files written by exp-033e (last-epoch TRAIN / VAL / TEST logits of one checkpoint each).
Every rule changes DECISIONS only; the scores are unchanged, so prob-AUROC is identical under every rule (printed as a check).

Rules (n = train counts of that file's own train set)
  A0  argmax (control)
  A1  within-organ, ONE tau fitted on TRAIN macro-F1 over 0.00..1.50 step 0.05 (ties -> smaller |tau|)     PRIMARY
  A2  within-organ, fixed tau 0.5 (val-exploratory: seen on val before pre-registration)
  A3  global prior shift argmax(z - tau log n), tau fitted on TRAIN (same grid)
  A4  per-organ tau (Bowel, Heart, Lung) fitted on TRAIN by coordinate ascent (3 sweeps, start 0.5), Joint fixed 0.5
  A5  8 additive class biases (Lung-normal fixed 0) fitted on TRAIN with fit_class_bias copied from research/exp-004.py
  E1  seed ensemble of all files of the arm (mean softmax), with A0 and A1 (tau fitted on the ensemble's own train predictions)
  E2  ensemble of the first 3 files of the arm (same two rules)
  E3  tail-5 checkpoint ensemble (epochs 19..23 of the run's -vallogits.pt), VAL only, A0
Hard rules: no val / test logit enters any fit (fit_rules takes the TRAIN arrays only); test is printed, never selected on.
Keep rule (shared protocol): keep iff mean(d) > max(t_{0.975,n-1} SD(d)/sqrt(n), 0.015) and the prob-AUROC guard (-0.002) holds.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

NUM_CLASSES = 8
CLASS_NAMES = [f"{o}-{c}" for o in ("Bowel", "Heart", "Joint", "Lung") for c in ("normal", "abnormal")]
ORGAN_NAMES = ["Bowel", "Heart", "Joint", "Lung"]
FITTED_ORGANS = (0, 1, 3)                       # Joint (2) is not fitted in A4
LUNG_NORMAL = CLASS_NAMES.index("Lung-normal")
TAU_GRID = [round(0.05 * i, 2) for i in range(31)]            # 0.00 .. 1.50
BIAS_GRID = [round(-3.0 + 0.1 * i, 1) for i in range(61)]     # -3.0 .. 3.0 (as exp-004.py)
FIXED_TAU = 0.5
FLOOR = 0.015
AUROC_GUARD = -0.002
TAIL = 5
RULES = ["A0", "A1", "A2", "A3", "A4", "A5"]
SECONDARY = ["A2", "A3", "A4", "A5"]
MMACS = {1: 5.88, 2: 19.35, 4: 69.12}           # MMACs per 5 s clip (exp-009 count_macs)
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t975(df):
    return T975[df - 1] if 1 <= df <= len(T975) else 1.96


# ------------------------------------------------------------------ metrics
def confusion(preds, labels):
    return torch.bincount(labels * NUM_CLASSES + preds, minlength=NUM_CLASSES * NUM_CLASSES).view(NUM_CLASSES, NUM_CLASSES)


def fast_macro_f1(preds, labels):
    """Macro-F1 over the classes present in labels; same formula and summation order as multiclass_metrics / exp-004 fast_macro_f1."""
    cm = confusion(preds, labels)
    tp, row, col = cm.diag().tolist(), cm.sum(1).tolist(), cm.sum(0).tolist()
    f1s = [2 * tp[c] / (2 * tp[c] + (col[c] - tp[c]) + (row[c] - tp[c])) for c in range(NUM_CLASSES) if row[c] > 0]
    return sum(f1s) / len(f1s)


def full_metrics(preds, labels):
    """acc, bal_acc, macro_f1, per-class F1 / recall / precision (0 where undefined) and number of predictions per class."""
    cm = confusion(preds, labels)
    tp, row, col = cm.diag().tolist(), cm.sum(1).tolist(), cm.sum(0).tolist()
    f1s, recalls, f1_pc, rec_pc, prec_pc = [], [], [], [], []
    for c in range(NUM_CLASSES):
        fp, fn = col[c] - tp[c], row[c] - tp[c]
        if row[c] > 0:
            f1s.append(2 * tp[c] / (2 * tp[c] + fp + fn))
            recalls.append(tp[c] / (tp[c] + fn))
        f1_pc.append(2 * tp[c] / (2 * tp[c] + fp + fn) if row[c] > 0 else float("nan"))
        rec_pc.append(tp[c] / row[c] if row[c] > 0 else float("nan"))
        prec_pc.append(tp[c] / col[c] if col[c] > 0 else 0.0)
    return {"acc": sum(tp) / max(1, sum(row)), "bal_acc": sum(recalls) / len(recalls), "macro_f1": sum(f1s) / len(f1s),
            "f1_pc": f1_pc, "rec_pc": rec_pc, "prec_pc": prec_pc, "npred": col, "n": row}


def binary_auroc(score, pos):
    """Rank-based AUROC with averaged ties (same as utils.utils.binary_f1_auroc), 1-D tensors."""
    score = score.detach().cpu().double()
    pos = pos.detach().cpu() >= 0.5
    n_pos = pos.sum().item()
    n_neg = pos.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = torch.unique(score, return_inverse=True, return_counts=True)
    upper = counts.cumsum(0)
    avg_rank = (upper - (counts - 1) / 2.0)[inverse]
    return (avg_rank[pos].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def prob_auroc(logits, labels):
    """Macro one-vs-rest AUROC of softmax probabilities (exp-009 prob_auroc; float32 softmax like the training script)."""
    p = logits.softmax(dim=1)
    vals = [binary_auroc(p[:, c], (labels == c).float()) for c in range(NUM_CLASSES)
            if (labels == c).any() and (labels != c).any()]
    return sum(vals) / len(vals)


def organ_abn_auroc(logits, labels):
    """Within-organ abnormality AUROC (Bowel, Heart, Lung) as exp-009 organ_abn_auroc; returns mean."""
    p = logits.softmax(dim=1)
    vals = []
    for o in FITTED_ORGANS:
        rows = (labels // 2) == o
        score = p[rows, 2 * o + 1] / (p[rows, 2 * o] + p[rows, 2 * o + 1])
        vals.append(binary_auroc(score, (labels[rows] % 2).float()))
    return sum(vals) / len(vals)


# ------------------------------------------------------------------ rules
def organ_hat_and_diff(logits):
    """Organ = argmax_o logsumexp(z[2o], z[2o+1]) (= log p marginal up to a constant); diff = z[2o+1] - z[2o] at that organ."""
    z = logits.double()
    pair = z.view(-1, 4, 2)
    organ = torch.logsumexp(pair, dim=2).argmax(dim=1)
    idx = torch.arange(z.shape[0])
    return organ, pair[idx, organ, 1] - pair[idx, organ, 0]


def pick_tau(f_of_tau, grid=TAU_GRID):
    """Grid value maximising f; ties (within 1e-9) -> smaller |tau|. Returns (tau, best_f)."""
    vals = [f_of_tau(t) for t in grid]
    best = max(vals)
    tau = min((t for t, v in zip(grid, vals) if v >= best - 1e-9), key=abs)
    return tau, best


def log_ratio(counts):
    """log(n_abnormal / n_normal) per organ (counts clamped at 1)."""
    c = counts.double().clamp(min=1)
    return (c[1::2] / c[0::2]).log()


def fast_f1_bias_search(logits, labels, ref, sweeps=3):
    """fit_class_bias copied from research/exp-004.py (Lung-normal fixed at 0), macro-F1 via fast_macro_f1."""
    bias = torch.zeros(NUM_CLASSES)
    best = fast_macro_f1(logits.argmax(dim=1), labels)
    for _ in range(sweeps):
        improved = False
        for c in range(NUM_CLASSES):
            if c == ref:
                continue
            for v in BIAS_GRID:
                trial = bias.clone()
                trial[c] = v
                f = fast_macro_f1((logits + trial).argmax(dim=1), labels)
                if f > best + 1e-9:
                    best, bias, improved = f, trial, True
                elif abs(f - best) <= 1e-9 and abs(v) < abs(bias[c].item()):
                    bias = trial  # tie: prefer the smaller bias
        if not improved:
            break
    return bias


def predict(rule, logits):
    """Class predictions of a fitted rule on any logits. rule: {'kind': ..., ...}."""
    kind = rule["kind"]
    if kind == "argmax":
        return logits.argmax(dim=1)
    if kind == "organ":      # abnormal iff diff > thr[organ_hat]
        organ, diff = organ_hat_and_diff(logits)
        return 2 * organ + (diff > rule["thr"][organ]).long()
    if kind == "prior":      # argmax(z - tau log n)
        return (logits.double() - rule["tau"] * rule["logn"]).argmax(dim=1)
    if kind == "bias":
        return (logits + rule["bias"]).argmax(dim=1)
    raise ValueError(kind)


def oracle_thr(rule, counts):
    """Per-organ decision threshold on z_abn - z_norm that the rule applies when the organ is known (for Se / Sp)."""
    kind = rule["kind"]
    if kind == "argmax":
        return torch.zeros(4, dtype=torch.double)
    if kind == "organ":
        return rule["thr"]
    if kind == "prior":
        return rule["tau"] * log_ratio(counts)
    if kind == "bias":
        b = rule["bias"].double()
        return -(b[1::2] - b[0::2])
    raise ValueError(kind)


def fit_rules(train_logits, train_labels, counts):
    """Fit A0..A5 on TRAIN arrays only. Returns {name: rule dict with 'kind', parameters, 'train_f1'}."""
    lr = log_ratio(counts)
    organ, diff = organ_hat_and_diff(train_logits)

    def organ_f1(thr):
        return fast_macro_f1(2 * organ + (diff > thr[organ]).long(), train_labels)

    rules = {"A0": {"kind": "argmax"}}
    tau1, _ = pick_tau(lambda t: organ_f1(t * lr))
    rules["A1"] = {"kind": "organ", "thr": tau1 * lr, "tau": tau1}
    rules["A2"] = {"kind": "organ", "thr": FIXED_TAU * lr, "tau": FIXED_TAU}
    logn = counts.double().clamp(min=1).log()
    tau3, _ = pick_tau(lambda t: fast_macro_f1((train_logits.double() - t * logn).argmax(dim=1), train_labels))
    rules["A3"] = {"kind": "prior", "tau": tau3, "logn": logn}
    taus = [FIXED_TAU] * 4
    for _ in range(3):
        changed = False
        for o in FITTED_ORGANS:
            def f(t, o=o):
                trial = list(taus)
                trial[o] = t
                return organ_f1(torch.tensor(trial, dtype=torch.double) * lr)
            t_new, _ = pick_tau(f)
            if t_new != taus[o]:
                taus[o], changed = t_new, True
        if not changed:
            break
    rules["A4"] = {"kind": "organ", "thr": torch.tensor(taus, dtype=torch.double) * lr, "tau_o": list(taus)}
    rules["A5"] = {"kind": "bias", "bias": fast_f1_bias_search(train_logits, train_labels, LUNG_NORMAL)}
    for r in rules.values():
        r["train_f1"] = fast_macro_f1(predict(r, train_logits), train_labels)
    return rules


def rule_params_str(name, rule):
    if name == "A0":
        return "none"
    if name in ("A1", "A2", "A3"):
        return f"tau={rule['tau']:.2f}"
    if name == "A4":
        return "tau_o=" + ",".join(f"{o}:{t:.2f}" for o, t in zip(ORGAN_NAMES, rule["tau_o"]))
    return "bias=" + ",".join(f"{n}:{b:.1f}" for n, b in zip(CLASS_NAMES, rule["bias"].tolist()))


def organ_se_sp(rule, logits, labels, counts):
    """Per-organ Se / Sp with the organ known: abnormal iff z_abn - z_norm > the rule's per-organ threshold."""
    thr = oracle_thr(rule, counts)
    out = []
    for o in range(4):
        rows = (labels // 2) == o
        z = logits[rows].double()
        dec = (z[:, 2 * o + 1] - z[:, 2 * o]) > thr[o]
        y = labels[rows] % 2
        se = dec[y == 1].float().mean().item() if (y == 1).any() else float("nan")
        sp = (~dec[y == 0]).float().mean().item() if (y == 0).any() else float("nan")
        out.append((se, sp))
    return out


# ------------------------------------------------------------------ ensembles
def ensemble_logits(logit_list):
    """Log of the mean softmax (float64). Each element's K probabilities are sorted before summing, so the result is
    exactly invariant to the order of the files."""
    p = torch.stack([z.double().softmax(dim=1) for z in logit_list]).sort(dim=0).values
    return p.mean(dim=0).log()


# ------------------------------------------------------------------ keep rule
def mean_sd(x):
    m = sum(x) / len(x)
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1)) if len(x) > 1 else float("nan")
    return m, sd


def keep_decision(d, auroc_d=None):
    """Shared keep rule on paired differences d (candidate - control, val macro-F1 per seed)."""
    n = len(d)
    m, sd = mean_sd(d)
    t = t975(n - 1)
    half = t * sd / math.sqrt(n)
    thr = max(half, FLOOR)
    guard = None if auroc_d is None else sum(auroc_d) / len(auroc_d)
    guard_ok = guard is None or guard >= AUROC_GUARD
    ci_excl0 = (m - half > 0) or (m + half < 0)
    if m > thr and guard_ok:
        label = "keep"
    elif m > thr:
        label = "guard_fail"
    elif m > FLOOR and not ci_excl0:
        label = "inconclusive"
    else:
        label = "reject" if m <= 0 else "below_floor"
    return {"n": n, "mean": m, "sd": sd, "t": t, "half": half, "thr": thr, "ci": (m - half, m + half), "guard": guard,
            "keep": label == "keep", "label": label}


def fmt_dec(k):
    g = "n/a" if k["guard"] is None else f"{k['guard']:+.6f}"
    return (f"mean_d={k['mean']:+.4f} sd_d={k['sd']:.4f} t={k['t']:.3f} half_width={k['half']:.4f} threshold={k['thr']:.4f} "
            f"ci=[{k['ci'][0]:+.4f},{k['ci'][1]:+.4f}] auroc_delta={g} keep={'yes' if k['keep'] else 'no'} label={k['label']}")


# ------------------------------------------------------------------ io
def load_eval(path):
    d = torch.load(path, map_location="cpu")
    for s in ("train", "val", "test"):
        d[f"{s}_logits"] = d[f"{s}_logits"].float()
    d["counts"] = torch.bincount(d["train_labels"], minlength=NUM_CLASSES).long()
    return d


def check_metadata(d, meta, name):
    """Labels / splits of the dumped recordings against the metadata csv (warns, never changes anything)."""
    organs = ORGAN_NAMES
    bad = missing = 0
    for s in ("train", "val", "test"):
        for mid, y in zip(d[f"{s}_mixture_ids"], d[f"{s}_labels"].tolist()):
            row = meta.get(mid)
            if row is None:
                missing += 1
            elif row["split"] != s or 2 * organs.index(row["organ"]) + int(row["label"] == "abnormal") != y:
                bad += 1
    print(f"metadata_check: {name} mismatches={bad} not_in_metadata={missing}" + ("" if bad == 0 else "  <-- WARNING"))


def parse_arms(spec):
    arms = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        name, _, files = part.partition("=")
        arms[name.strip()] = [f for f in files.split(",") if f]
    return arms


def find_vallogits(path, vdir):
    name = Path(path).name.replace("-last-evallogits.pt", "-vallogits.pt")
    cands = [Path(path).with_name(name)]
    if vdir:
        cands += [Path(vdir) / name] + sorted(Path(vdir).glob(f"*/{name}"))
    return next((c for c in cands if c.exists()), None)


def fmt_list(x, nd=3):
    return "[" + ",".join("nan" if v != v else f"{v:.{nd}f}" for v in x) + "]"


# ------------------------------------------------------------------ analysis
def analyse(logits_by_split, labels_by_split, counts, tag, quiet=False):
    """Fit all rules on TRAIN, evaluate on val and test. Returns {rule: {'fit': rule, split: metrics, 'se_sp': {split: ..}}}."""
    rules = fit_rules(logits_by_split["train"], labels_by_split["train"], counts)
    res = {}
    for name in RULES:
        r = rules[name]
        res[name] = {"rule": r, "train_f1": r["train_f1"]}
        print(f"fit[{tag},{name}]: {rule_params_str(name, r)} train_macro_f1={r['train_f1']:.6f}")
        for s in ("val", "test"):
            m = full_metrics(predict(r, logits_by_split[s]), labels_by_split[s])
            m["se_sp"] = organ_se_sp(r, logits_by_split[s], labels_by_split[s], counts)
            res[name][s] = m
            print(f"metrics[{tag},{name},{s}]: macro_f1={m['macro_f1']:.6f} bal_acc={m['bal_acc']:.6f} acc={m['acc']:.6f}")
            if not quiet:
                print(f"\tf1_pc[{tag},{name},{s}]: {fmt_list(m['f1_pc'], 4)}")
                print(f"\trecall_pc[{tag},{name},{s}]: {fmt_list(m['rec_pc'], 4)}")
                print(f"\tprecision_pc[{tag},{name},{s}]: {fmt_list(m['prec_pc'], 4)}")
                print(f"\tnpred[{tag},{name},{s}]: {m['npred']}")
                print(f"\tse_sp_oracle_organ[{tag},{name},{s}]: " + " ".join(f"{o}:{se:.3f}/{sp:.3f}" for o, (se, sp) in zip(ORGAN_NAMES, m["se_sp"])))
    return res


def fmt_ms(x):
    m, sd = mean_sd(x)
    return f"{m:.4f} ± {sd:.4f}"


def main():
    ap = argparse.ArgumentParser(description="exp-033: operating-point rules (fitted on TRAIN) and ensembles on exp-033e logit dumps")
    ap.add_argument("--arms", required=True, help='"W1noKD=f1,f2,...;W1organKD=...;..." -evallogits.pt files per arm, in seed order')
    ap.add_argument("--metadata", default=None, help="metadata.csv; only used to check labels / splits of the dumps")
    ap.add_argument("--pairs", default=None, help='cross-arm pairs "cand:ctrl;cand2:ctrl2" paired by seed (default: W1organKD:W1noKD if both exist)')
    ap.add_argument("--vallogits-dir", default=None, help="where the runs' -vallogits.pt live (default: next to the dump; also searched in <dir>/*/)")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--quiet", action="store_true", help="skip the per-class / per-organ lines")
    args = ap.parse_args()

    arms = parse_arms(args.arms)
    meta = None
    if args.metadata:
        with open(args.metadata, newline="") as f:
            meta = {r["mixture_id"]: r for r in csv.DictReader(f)}

    per_arm = {}      # arm -> {"res": [per-file analysis], "data": [dicts]}
    for arm, files in arms.items():
        print(f"\n===== arm {arm}: {len(files)} files =====")
        res, data = [], []
        for i, fpath in enumerate(files):
            d = load_eval(fpath)
            print(f"file: {fpath} arm={arm} idx={i} width={d.get('width')} train_n={d['counts'].tolist()} "
                  f"n_val={len(d['val_labels'])} n_test={len(d['test_labels'])}")
            if meta is not None:
                check_metadata(d, meta, f"{arm}[{i}]")
            lg = {s: d[f"{s}_logits"] for s in ("train", "val", "test")}
            lb = {s: d[f"{s}_labels"] for s in ("train", "val", "test")}
            tag = f"{arm}#{i}"
            r = analyse(lg, lb, d["counts"], tag, args.quiet)
            r["auroc"] = {s: prob_auroc(lg[s], lb[s]) for s in ("val", "test")}
            r["organ_auroc"] = {s: organ_abn_auroc(lg[s], lb[s]) for s in ("val", "test")}
            print(f"auroc[{tag}]: val_prob_auroc={r['auroc']['val']:.6f} test_prob_auroc={r['auroc']['test']:.6f} "
                  f"val_within_organ_auroc={r['organ_auroc']['val']:.6f} test_within_organ_auroc={r['organ_auroc']['test']:.6f} (identical under every rule)")
            res.append(r)
            data.append(d)
        per_arm[arm] = {"res": res, "data": data}

    summary = {}
    disagreements = []

    def report_delta(label, cand, ctrl, cand_auroc, ctrl_auroc, primary_note=""):
        """cand / ctrl: {'val': [per-seed F1], 'test': [...]} -> decision lines."""
        dv = [a - b for a, b in zip(cand["val"], ctrl["val"])]
        dt = [a - b for a, b in zip(cand["test"], ctrl["test"])]
        da = [a - b for a, b in zip(cand_auroc["val"], ctrl_auroc["val"])]
        kv, kt = keep_decision(dv, da), keep_decision(dt, [a - b for a, b in zip(cand_auroc["test"], ctrl_auroc["test"])])
        print(f"decision[{label}] VAL{primary_note}: {fmt_dec(kv)}")
        print(f"report[{label}] TEST (reporting only): {fmt_dec(kt)}")
        print(f"paired_delta[{label}]: val_per_seed={fmt_list(dv, 4)} test_per_seed={fmt_list(dt, 4)}")
        if kv["mean"] * kt["mean"] < 0:
            disagreements.append(label)
            print(f"val_test_sign_disagree: {label} val_mean_d={kv['mean']:+.4f} test_mean_d={kt['mean']:+.4f}")
        summary[label] = {"val": kv, "test": kt}
        return kv, kt

    for arm, pa in per_arm.items():
        res = pa["res"]
        n = len(res)
        print(f"\n===== summary arm {arm} (n={n}) =====")
        au = {s: [r["auroc"][s] for r in res] for s in ("val", "test")}
        print(f"auroc_prob[{arm}]: val={fmt_ms(au['val'])} test={fmt_ms(au['test'])}")
        for rn in RULES:
            v = [r[rn]["val"]["macro_f1"] for r in res]
            t = [r[rn]["test"]["macro_f1"] for r in res]
            tr = [r[rn]["train_f1"] for r in res]
            vb = [r[rn]["val"]["bal_acc"] for r in res]
            tb = [r[rn]["test"]["bal_acc"] for r in res]
            print(f"mean[{arm},{rn}]: val_macro_f1={fmt_ms(v)} test_macro_f1={fmt_ms(t)} val_bal_acc={fmt_ms(vb)} test_bal_acc={fmt_ms(tb)} "
                  f"train_macro_f1={fmt_ms(tr)} train_minus_val={sum(tr) / n - sum(v) / n:+.4f} val_minus_test={sum(v) / n - sum(t) / n:+.4f}")
            pcv = [sum(r[rn]["val"]["f1_pc"][c] for r in res) / n for c in range(NUM_CLASSES)]
            pct = [sum(r[rn]["test"]["f1_pc"][c] for r in res) / n for c in range(NUM_CLASSES)]
            print(f"\tmean_f1_pc[{arm},{rn}]: val={fmt_list(pcv)} test={fmt_list(pct)}")
        ctrl = {s: [r["A0"][s]["macro_f1"] for r in res] for s in ("val", "test")}
        for rn in RULES[1:]:
            cand = {s: [r[rn][s]["macro_f1"] for r in res] for s in ("val", "test")}
            note = " [PRIMARY]" if rn == "A1" else (" [val-exploratory]" if rn == "A2" else " [secondary]")
            kv, kt = report_delta(f"{arm},{rn}-A0", cand, ctrl, au, au, note)
            if rn in SECONDARY:
                supported = kv["keep"] and kt["mean"] >= FLOOR and kt["mean"] * kv["mean"] > 0
                print(f"secondary_supported[{arm},{rn}]: {'yes' if supported else 'no'} (needs val keep AND test mean_d >= {FLOOR} with the same sign)")

        # E1 / E2 / E3 (descriptive, no pairing)
        data = pa["data"]
        width = data[0].get("width")
        for ename, k in (("E1", n), ("E2", 3)):
            if k > n or (ename == "E2" and n <= 3):
                continue
            sel = list(range(k))
            lg = {s: ensemble_logits([data[i][f"{s}_logits"] for i in sel]) for s in ("train", "val", "test")}
            lb = {s: data[0][f"{s}_labels"] for s in ("train", "val", "test")}
            assert all(torch.equal(data[i][f"{s}_labels"], lb[s]) for i in sel for s in lb), "label order differs between files"
            er = analyse(lg, lb, data[0]["counts"], f"{arm}:{ename}(K={k})", args.quiet)
            ea = {s: prob_auroc(lg[s].float(), lb[s]) for s in ("val", "test")}
            cost = f" cost={k * MMACS[width]:.2f}MMACs(single W2={MMACS[2]}, W4={MMACS[4]})" if width in MMACS else ""
            print(f"auroc[{arm}:{ename}]: val_prob_auroc={ea['val']:.6f} test_prob_auroc={ea['test']:.6f} "
                  f"mean_single_val={sum(au['val'][i] for i in sel) / k:.6f} mean_single_test={sum(au['test'][i] for i in sel) / k:.6f}")
            for rn in ("A0", "A1"):
                for s in ("val", "test"):
                    single = [res[i][rn][s]["macro_f1"] for i in sel]
                    e = er[rn][s]["macro_f1"]
                    print(f"ensemble[{arm},{ename},K={k},{rn},{s}]: ens_macro_f1={e:.4f} mean_single={sum(single) / k:.4f} "
                          f"delta_vs_mean_single={e - sum(single) / k:+.4f} ens_bal_acc={er[rn][s]['bal_acc']:.4f}{cost}")
        # E3
        tails = []
        for i, fpath in enumerate(arms[arm]):
            vp = find_vallogits(fpath, args.vallogits_dir)
            if vp is None:
                continue
            ref = torch.load(vp, map_location="cpu")
            vl, vy = ref["val_logits"].float(), ref["val_labels"]
            if vl.shape[0] < TAIL or not torch.equal(vy, data[i]["val_labels"]) or (vl[-1] - data[i]["val_logits"]).abs().max().item() > 1e-4:
                print(f"E3 skipped for {fpath}: vallogits file does not match the dump")
                continue
            ens = ensemble_logits([vl[e] for e in range(vl.shape[0] - TAIL, vl.shape[0])])
            tails.append((fast_macro_f1(ens.argmax(dim=1), vy), res[i]["A0"]["val"]["macro_f1"],
                          prob_auroc(ens.float(), vy), res[i]["auroc"]["val"]))
        if len(tails) == n:
            cand = {"val": [t[0] for t in tails], "test": [float("nan")] * n}
            dv = [t[0] - t[1] for t in tails]
            kv = keep_decision(dv, [t[2] - t[3] for t in tails])
            print(f"E3[{arm}] tail-5 ensemble A0 (VAL only; test unavailable for old runs): mean_val_f1={sum(t[0] for t in tails) / n:.4f} "
                  f"vs last-epoch A0 {sum(t[1] for t in tails) / n:.4f}")
            print(f"decision[{arm},E3-A0] VAL [descriptive]: {fmt_dec(kv)}")
            summary[f"{arm},E3-A0"] = {"val": kv}
        else:
            print(f"E3[{arm}]: -vallogits.pt not found for {n - len(tails)} of {n} files (skipped)")

    pairs = []
    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs.split(";") if p]
    elif "W1organKD" in per_arm and "W1noKD" in per_arm:
        pairs = [("W1organKD", "W1noKD")]
    for cand_arm, ctrl_arm in pairs:
        rc, rk = per_arm[cand_arm]["res"], per_arm[ctrl_arm]["res"]
        if len(rc) != len(rk):
            print(f"pair {cand_arm}:{ctrl_arm} skipped (different number of seeds)")
            continue
        print(f"\n===== cross-arm pairing {cand_arm} - {ctrl_arm} (paired by seed order) =====")
        for rn in ("A0", "A1"):
            cand = {s: [r[rn][s]["macro_f1"] for r in rc] for s in ("val", "test")}
            ctrl = {s: [r[rn][s]["macro_f1"] for r in rk] for s in ("val", "test")}
            ca = {s: [r["auroc"][s] for r in rc] for s in ("val", "test")}
            ka = {s: [r["auroc"][s] for r in rk] for s in ("val", "test")}
            report_delta(f"{cand_arm}-{ctrl_arm},{rn}", cand, ctrl, ca, ka, " [H3 stacking]" if rn == "A1" else "")

    print(f"\nval_test_sign_disagree_count: {len(disagreements)}")
    if args.json_out:
        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            return o
        Path(args.json_out).write_text(json.dumps(clean(summary), indent=1))


if __name__ == "__main__":
    main()
