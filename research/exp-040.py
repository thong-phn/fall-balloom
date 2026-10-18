"""exp-040: murmur-stratified Heart audit of every existing arm. CPU only, no training.

Within the CirCor-source Heart recordings, AUROC of abnormal vs normal (score s = p(Heart-a) / (p(Heart-n) + p(Heart-a)),
true organ) stratified with the CirCor patient metadata: murmur audible at the recorded location (mloc), murmur absent
(mabs), systolic murmur grade, innocent murmurs (normal outcome with a murmur at the location). Patient-grouped bootstrap
CIs, paired-by-seed arm differences (t-CI, df n-1) and Welch differences for unpaired arms.

Usage (stdlib + torch only, safe with `python -I`; run from fall-balloom/):
    .venv/bin/python -I research/exp-040.py \
        --arms "W1noKD=009g,009h,009i,020a,020b,020c;W1organKD=013a,013b,013c,020g,020h,020i;..." \
        --metadata datasets/BioSARDdBScene/metadata.csv --circor-csv datasets/CirCo/training_data.csv

--arms: "name=id1,id2,...;name2=..." in seed order. An id such as 009g (or exp-009g) is resolved to
  <runs-dir>/exp-009g/exp-009g-<jobid>-vallogits.pt (VALIDATION, last epoch) and, when present,
  <runs-dir>/exp-009g/exp-009g-<jobid>-last-evallogits.pt (exp-033e / exp-034 dump, TEST block). A token containing '/' or
  ending in .pt is taken as an explicit -vallogits.pt path (its test dump is looked up next to it).
Decisions use VALIDATION only. The TEST block is labelled "TEST, reporting only; independent replicate".
Prints one `key: value` line per number.
"""
import argparse
import csv
import glob
import math
import os
import sys

import torch

ORGANS = ["Bowel", "Heart", "Joint", "Lung"]
NUM_CLASSES = 8
HEART = ORGANS.index("Heart")
N_BOOT = 2000

STAT_DEFS = [
    ("s1_mloc_abn_vs_cirnorm", "abnormal with murmur at recorded location vs all CirCor normals"),
    ("s2_mabs_abn_vs_cirnorm", "abnormal, murmur Absent vs all CirCor normals"),
    ("s3_cir_abn_vs_norm", "all CirCor abnormal vs all CirCor normals"),
    ("s4_mur_detect_mloc_vs_mabs", "murmur detection: mloc vs mabs, regardless of label"),
    ("s5a_g3_abn_vs_mabsnorm", "abnormal grade III/VI (mloc) vs murmur-free normals"),
    ("s5b_g1_abn_vs_mabsnorm", "abnormal grade I/VI (mloc) vs murmur-free normals"),
    ("s5c_g2_abn_vs_mabsnorm", "abnormal grade II/VI (mloc) vs murmur-free normals (extra)"),
    ("s6_innocent_vs_mabsnorm", "innocent: normal with murmur at location vs murmur-free normals"),
    ("s7a_physionet_abn_vs_norm", "PhysioNet Heart abnormal vs normal (reference)"),
    ("s7b_bmdhs_abn_vs_norm", "BMD-HS Heart abnormal vs normal (reference)"),
]
STAT_NAMES = [n for n, _ in STAT_DEFS]

# (candidate, control, kind): kind paired = same seeds; welch = unpaired
PAIRS = [("W1organKD", "W1noKD", "paired"), ("W1flatKD", "W1noKD", "paired"), ("W2KD", "W2", "paired"),
         ("W4KD", "W4", "paired"), ("W9", "W1noKD", "welch"), ("W2", "W1noKD", "welch"), ("W4", "W1noKD", "welch")]
NOKD_ARMS = ["W1noKD", "W2", "W4", "W9"]


# ------------------------------------------------------------------ small numerics
def binary_auroc(score, labels):
    """Rank-based AUROC (ties averaged); same computation as research/exp-018.py binary_auroc."""
    score = score.detach().cpu().double()
    pos = labels.detach().cpu() >= 0.5
    n_pos = int(pos.sum().item())
    n_neg = pos.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = torch.unique(score, return_inverse=True, return_counts=True)
    upper = counts.cumsum(0)
    avg_rank = (upper - (counts - 1) / 2.0)[inverse]
    return (avg_rank[pos].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return h


def betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    p = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - p if t > 0 else p


def t_crit(df, level=0.975):
    """Two-sided 95% critical value (stdlib bisection on the t CDF); 2.571 for df 5, 4.303 for df 2."""
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mean_sd(x):
    n = len(x)
    m = sum(x) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1)) if n > 1 else float("nan")
    return m, sd


def fmt(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def fmt_list(x, nd=3):
    return "[" + ",".join(fmt(v, nd) for v in x) + "]"


def quant_ci(draws):
    """95% percentile interval of the non-nan draws (tensor [B])."""
    d = draws[~torch.isnan(draws)]
    if d.numel() < 20:
        return float("nan"), float("nan"), int(d.numel())
    lo, hi = torch.quantile(d, torch.tensor([0.025, 0.975], dtype=torch.double)).tolist()
    return lo, hi, int(d.numel())


# ------------------------------------------------------------------ metadata / strata
def corpus_key(r):
    return r["primary_source"].split("/")[2].lower().replace("abnormal", "").replace("normal", "")


def load_circor(path):
    with open(path, newline="") as f:
        return {r["Patient ID"]: r for r in csv.DictReader(f)}


def heart_rows(meta, split, circor):
    """Heart rows of the split, in metadata order, with CirCor stratum info."""
    out = []
    for r in meta:
        if r["split"] != split or r["organ"] != "Heart":
            continue
        d = {"mid": r["mixture_id"], "abn": int(r["label"] == "abnormal"), "corpus": corpus_key(r),
             "stratum": None, "grade": "", "timing": "", "group": r["source_id"]}
        if d["corpus"] == "circo":
            base = os.path.basename(r["primary_source"])[:-4]
            pid, loc = base.split("_", 1)
            p = circor[pid]
            locs = [x for x in (p["Murmur locations"] or "").split("+") if x]
            d["stratum"] = "mloc" if loc in locs else ("mabs" if p["Murmur"] == "Absent" else "mask")
            d["grade"] = p["Systolic murmur grading"]
            d["timing"] = p["Systolic murmur timing"]
            d["group"] = "circor::" + pid
            d["patient"] = pid
            d["loc"] = loc
        out.append(d)
    return out


def build_strata(hrows):
    """stat name -> (pos index list, neg index list) over positions in hrows."""
    idx = lambda f: [i for i, d in enumerate(hrows) if f(d)]
    cir = lambda d: d["corpus"] == "circo"
    cn = idx(lambda d: cir(d) and d["abn"] == 0)
    mabs_norm = idx(lambda d: cir(d) and d["abn"] == 0 and d["stratum"] == "mabs")
    mloc = idx(lambda d: cir(d) and d["stratum"] == "mloc")
    mabs = idx(lambda d: cir(d) and d["stratum"] == "mabs")
    abn_mloc = lambda g: idx(lambda d: cir(d) and d["abn"] == 1 and d["stratum"] == "mloc" and (g is None or d["grade"] == g))
    S = {
        "s1_mloc_abn_vs_cirnorm": (abn_mloc(None), cn),
        "s2_mabs_abn_vs_cirnorm": (idx(lambda d: cir(d) and d["abn"] == 1 and d["stratum"] == "mabs"), cn),
        "s3_cir_abn_vs_norm": (idx(lambda d: cir(d) and d["abn"] == 1), cn),
        "s4_mur_detect_mloc_vs_mabs": (mloc, mabs),
        "s5a_g3_abn_vs_mabsnorm": (abn_mloc("III/VI"), mabs_norm),
        "s5b_g1_abn_vs_mabsnorm": (abn_mloc("I/VI"), mabs_norm),
        "s5c_g2_abn_vs_mabsnorm": (abn_mloc("II/VI"), mabs_norm),
        "s6_innocent_vs_mabsnorm": (idx(lambda d: cir(d) and d["abn"] == 0 and d["stratum"] == "mloc"), mabs_norm),
        "s7a_physionet_abn_vs_norm": (idx(lambda d: d["corpus"] == "physionet" and d["abn"] == 1),
                                      idx(lambda d: d["corpus"] == "physionet" and d["abn"] == 0)),
        "s7b_bmdhs_abn_vs_norm": (idx(lambda d: d["corpus"] == "bmd-hs" and d["abn"] == 1),
                                  idx(lambda d: d["corpus"] == "bmd-hs" and d["abn"] == 0)),
    }
    return S


class Split:
    """Heart rows of one split + strata + bootstrap groups."""

    def __init__(self, name, hrows):
        self.name = name
        self.rows = hrows
        self.strata = build_strata(hrows)
        gl = sorted({d["group"] for d in hrows})
        gid = {g: i for i, g in enumerate(gl)}
        self.group_of_row = torch.tensor([gid[d["group"]] for d in hrows], dtype=torch.long)
        self.n_groups = len(gl)
        self.circor_patients = len({d["group"] for d in hrows if d["corpus"] == "circo"})
        self._boot_w = None

    def boot_weights(self, n_boot):
        """Row weights [B, H] = how many times the row's group (CirCor: patient; others: source_id) was drawn."""
        if self._boot_w is None or self._boot_w.shape[0] != n_boot:
            gen = torch.Generator().manual_seed(0)
            pick = torch.randint(self.n_groups, (n_boot, self.n_groups), generator=gen)
            counts = torch.zeros(n_boot, self.n_groups, dtype=torch.double)
            counts.scatter_add_(1, pick, torch.ones(n_boot, self.n_groups, dtype=torch.double))
            self._boot_w = counts[:, self.group_of_row]
        return self._boot_w

    def describe(self, tag):
        print(f"strata[{tag}]: heart_rows={len(self.rows)} bootstrap_groups={self.n_groups} circor_patients={self.circor_patients}")
        for n, desc in STAT_DEFS:
            p, q = self.strata[n]
            print(f"stratum[{tag}] {n}: n_pos={len(p)} n_neg={len(q)} patients_pos={len({self.rows[i]['group'] for i in p})} "
                  f"patients_neg={len({self.rows[i]['group'] for i in q})}  # {desc}")
        cir = [d for d in self.rows if d["corpus"] == "circo"]
        cnt = {}
        for d in cir:
            k = (d["stratum"], "abn" if d["abn"] else "norm")
            cnt[k] = cnt.get(k, 0) + 1
        print(f"circor_counts[{tag}]: " + " ".join(f"{s}_{l}={cnt.get((s, l), 0)}" for s in ("mloc", "mabs", "mask") for l in ("abn", "norm")))
        tm = {}
        for d in cir:
            if d["abn"] and d["stratum"] == "mloc":
                tm[d["timing"]] = tm.get(d["timing"], 0) + 1
        print(f"circor_abn_mloc_systolic_timing[{tag}]: " + " ".join(f"{k}={v}" for k, v in sorted(tm.items())))


# ------------------------------------------------------------------ scoring
def heart_scores(logits, split, index_of_mid):
    ii = torch.tensor([index_of_mid[d["mid"]] for d in split.rows], dtype=torch.long)
    p = logits[ii].double().softmax(dim=1)
    return p[:, 2 * HEART + 1] / (p[:, 2 * HEART] + p[:, 2 * HEART + 1])


def file_stats(s, split, n_boot):
    """Point AUROCs (rank-based) and bootstrap AUROCs [B] (weighted pair counting, same value as resampling groups)."""
    point, boot = {}, {}
    W = split.boot_weights(n_boot) if n_boot > 0 else None
    for name in STAT_NAMES:
        pos, neg = split.strata[name]
        if not pos or not neg:
            point[name] = float("nan")
            boot[name] = torch.full((max(n_boot, 0),), float("nan"), dtype=torch.double)
            continue
        sp, sn = s[pos], s[neg]
        y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
        point[name] = binary_auroc(torch.cat([sp, sn]), y)
        if n_boot > 0:
            M = (sp[:, None] > sn[None, :]).double() + 0.5 * (sp[:, None] == sn[None, :]).double()
            wp, wn = W[:, pos], W[:, neg]
            num = ((wp @ M) * wn).sum(1)
            den = wp.sum(1) * wn.sum(1)
            boot[name] = torch.where(den > 0, num / den.clamp(min=1e-300), torch.full_like(num, float("nan")))
    return point, boot


# ------------------------------------------------------------------ file resolution
def parse_arms(spec):
    arms = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        name, _, ids = part.partition("=")
        arms[name.strip()] = [x.strip() for x in ids.split(",") if x.strip()]
    return arms


def resolve(token, runs_dir, evalmode):
    """token -> (run label, vallogits path, evallogits path or None)."""
    if "/" in token or token.endswith(".pt"):
        val = token
        stem = os.path.basename(val)[: -len("-vallogits.pt")] if val.endswith("-vallogits.pt") else os.path.basename(val)
        label = stem
        evl = os.path.join(os.path.dirname(val), stem + "-last-evallogits.pt")
    else:
        rid = token if token.startswith("exp-") else "exp-" + token
        label = rid
        c = sorted(glob.glob(os.path.join(runs_dir, rid, f"{rid}-*-vallogits.pt")))
        if len(c) != 1:
            sys.exit(f"cannot resolve {token}: {len(c)} vallogits files in {runs_dir}/{rid}")
        val = c[0]
        evl = val.replace("-vallogits.pt", "-last-evallogits.pt")
    if evalmode == "none" or not os.path.exists(evl):
        evl = None
    return label, val, evl


# ------------------------------------------------------------------ main analysis
def analyse_split(split, tag, arms_files, n_boot, loader):
    """arms_files: arm -> list of (label, logits [N,8], index_of_mid). Returns per-arm results."""
    res = {}
    for arm, items in arms_files.items():
        pts, boots = [], []
        for k, (label, logits, idx_of) in enumerate(items):
            s = heart_scores(logits, split, idx_of)
            pt, bt = file_stats(s, split, n_boot)
            pts.append(pt)
            boots.append(bt)
            print(f"auroc[{tag},{arm}#{k}] run={label}: " + " ".join(f"{n.split('_')[0]}={fmt(pt[n], 4)}" for n in STAT_NAMES))
        arm_boot = {n: torch.stack([b[n] for b in boots]).mean(0) for n in STAT_NAMES}
        res[arm] = {"point": pts, "boot": arm_boot, "n": len(items)}
        for n in STAT_NAMES:
            vals = [p[n] for p in pts]
            m, sd = mean_sd(vals)
            lo, hi, nv = quant_ci(arm_boot[n])
            print(f"arm[{tag},{arm},{n}]: mean={fmt(m)} sd={fmt(sd)} boot_ci95=[{fmt(lo)}, {fmt(hi)}] n_files={len(vals)} "
                  f"boot_valid={nv} per_seed={fmt_list(vals)}")
    return res


def diff_stats(res, tag):
    """Paired / Welch differences for every stat. Returns {(cand, ctrl): {stat: dict}}."""
    out = {}
    for cand, ctrl, kind in PAIRS:
        if cand not in res or ctrl not in res:
            continue
        rc, rr = res[cand], res[ctrl]
        label = f"{cand}-{ctrl}"
        if kind == "paired" and rc["n"] != rr["n"]:
            print(f"diff[{tag},{label}]: skipped, unequal seeds {rc['n']} vs {rr['n']}")
            continue
        out[(cand, ctrl)] = {}
        for n in STAT_NAMES:
            a = [p[n] for p in rc["point"]]
            b = [p[n] for p in rr["point"]]
            bd = rc["boot"][n] - rr["boot"][n]
            blo, bhi, _ = quant_ci(bd)
            if kind == "paired":
                d = [x - y for x, y in zip(a, b)]
                m, sd = mean_sd(d)
                df = len(d) - 1
                tc = t_crit(df)
                half = tc * sd / math.sqrt(len(d))
                extra = f"per_seed={fmt_list(d)}"
            else:
                ma, sa = mean_sd(a)
                mb, sb = mean_sd(b)
                m = ma - mb
                va, vb = sa * sa / len(a), sb * sb / len(b)
                se = math.sqrt(va + vb)
                df = (va + vb) ** 2 / (va * va / (len(a) - 1) + vb * vb / (len(b) - 1)) if se > 0 else float("nan")
                tc = t_crit(df)
                half = tc * se
                sd = float("nan")
                extra = f"means={fmt(ma)}/{fmt(mb)}"
            lo, hi = m - half, m + half
            out[(cand, ctrl)][n] = {"mean": m, "lo": lo, "hi": hi, "kind": kind}
            print(f"diff[{tag},{label},{n}]: kind={kind} mean={fmt(m)} sd={fmt(sd)} df={df if kind == 'paired' else fmt(df, 2)} "
                  f"t_crit={tc:.3f} t_ci95=[{fmt(lo)}, {fmt(hi)}] excludes0={lo > 0 or hi < 0} "
                  f"boot_ci95=[{fmt(blo)}, {fmt(bhi)}] {extra}")
    return out


def arm_mean(res, arm, stat):
    return mean_sd([p[stat] for p in res[arm]["point"]])[0]


def main():
    ap = argparse.ArgumentParser(description="exp-040: murmur-stratified Heart AUROC audit of existing arms (CPU only)")
    ap.add_argument("--arms", required=True, help='"W1noKD=009g,009h,...;W1organKD=...": run ids in seed order (or explicit -vallogits.pt paths)')
    ap.add_argument("--metadata", required=True, help="BioSAR metadata.csv")
    ap.add_argument("--circor-csv", required=True, help="CirCor training_data.csv (patient metadata)")
    ap.add_argument("--runs-dir", default="../autoresearch/runs", help="directory with exp-<id>/ run folders (default ../autoresearch/runs)")
    ap.add_argument("--epoch", type=int, default=-1, help="epoch index of the validation logits (default -1 = last = 23 for 24-epoch runs)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--evallogits", default="auto", choices=["auto", "none"],
                    help="auto: add the TEST block from <id>-<jobid>-last-evallogits.pt dumps when present; none: validation only")
    ap.add_argument("--tag", default="", help="free label printed in the output")
    args = ap.parse_args()

    arms = parse_arms(args.arms)
    print(f"tag: {args.tag}")
    print(f"arms: " + " ".join(f"{a}({len(v)})" for a, v in arms.items()))
    print(f"epoch={args.epoch} n_boot={args.n_boot} bootstrap_seed=0 evallogits={args.evallogits}")
    circor = load_circor(args.circor_csv)
    with open(args.metadata, newline="") as f:
        meta = list(csv.DictReader(f))
    val = Split("val", heart_rows(meta, "val", circor))
    tst = Split("test", heart_rows(meta, "test", circor))
    val_meta = [r for r in meta if r["split"] == "val"]
    val_index = {r["mixture_id"]: i for i, r in enumerate(val_meta)}
    val_labels = torch.tensor([2 * ORGANS.index(r["organ"]) + int(r["label"] == "abnormal") for r in val_meta])
    val_groups = [r["source_id"] for r in val_meta]
    meta_by_mid = {r["mixture_id"]: r for r in meta}
    print(f"n_val_rows: {len(val_meta)}")
    val.describe("val")
    tst.describe("test")

    # ---- load validation logits ----
    val_files, test_files, test_missing = {}, {}, {}
    for arm, ids in arms.items():
        val_files[arm], test_files[arm] = [], []
        test_missing[arm] = []
        for tok in ids:
            label, vpath, epath = resolve(tok, args.runs_dir, args.evallogits)
            d = torch.load(vpath, map_location="cpu", weights_only=False)
            z = d["val_logits"][args.epoch].float()
            assert z.shape == (len(val_meta), NUM_CLASSES), f"{vpath}: logits {tuple(z.shape)} vs {len(val_meta)} val rows"
            assert torch.equal(d["val_labels"].long(), val_labels), f"{vpath}: val_labels differ from metadata"
            assert list(d["val_groups"]) == val_groups, f"{vpath}: val_groups differ from metadata source_id order"
            val_files[arm].append((label, z, val_index))
            print(f"file: arm={arm} idx={len(val_files[arm]) - 1} run={label} val={vpath}")
            if epath is None:
                test_missing[arm].append(label)
                continue
            e = torch.load(epath, map_location="cpu", weights_only=False)
            gate = (e["val_logits"].float() - z).abs().max().item()
            tm = e["test_mixture_ids"]
            tl = e["test_logits"].float()
            bad = sum(1 for mid, y in zip(tm, e["test_labels"].tolist())
                      if meta_by_mid[mid]["split"] != "test"
                      or 2 * ORGANS.index(meta_by_mid[mid]["organ"]) + int(meta_by_mid[mid]["label"] == "abnormal") != y)
            print(f"dump_check: run={label} evallogits={epath} val_logits_vs_vallogits_max_abs_diff={gate:.2e} test_rows={len(tm)} "
                  f"test_label_or_split_mismatch={bad}" + ("" if gate <= 1e-4 and bad == 0 else "  <-- WARNING"))
            test_files[arm].append((label, tl, {mid: i for i, mid in enumerate(tm)}))

    # ---- VALIDATION ----
    print("=== VALIDATION (decisions use this block only) ===")
    vres = analyse_split(val, "val", val_files, args.n_boot, None)
    vdiff = diff_stats(vres, "val")

    # ---- TEST ----
    tres = tdiff = None
    complete = {a: len(test_files[a]) == len(arms[a]) for a in arms}
    avail = {a: len(test_files[a]) for a in arms}
    if any(avail.values()):
        print("=== TEST, reporting only; independent replicate (never used for a decision) ===")
        print("test_dumps_available: " + " ".join(f"{a}={avail[a]}/{len(arms[a])}" for a in arms))
        tarms = {a: test_files[a] for a in arms if complete[a] and avail[a] > 0}
        for a in arms:
            if not complete[a]:
                print(f"test_arm_skipped: {a} (dumps present {avail[a]}/{len(arms[a])}; missing {test_missing[a]})")
        tres = analyse_split(tst, "test", tarms, args.n_boot, None)
        tdiff = diff_stats(tres, "test")
    else:
        print("=== TEST, reporting only; independent replicate: NOT AVAILABLE (no -last-evallogits.pt dumps found; exp-033e pending) ===")

    # ---- rules ----
    print("=== pre-registered reading (val decides; test reported) ===")
    s1, s2 = STAT_NAMES[0], STAT_NAMES[1]
    h1_ok = True
    for arm in [a for a in NOKD_ARMS if a in vres] + [a for a in vres if a not in NOKD_ARMS]:
        a1, a2 = arm_mean(vres, arm, s1), arm_mean(vres, arm, s2)
        lo1 = quant_ci(vres[arm]["boot"][s1])[0]
        hi2 = quant_ci(vres[arm]["boot"][s2])[1]
        ok = a1 >= 0.80 and a2 <= 0.62
        if arm in NOKD_ARMS:
            h1_ok = h1_ok and ok
        print(f"H1_arm[{arm}]{'' if arm in NOKD_ARMS else ' (KD arm, descriptive)'}: audible_s1={fmt(a1)} boot_lower={fmt(lo1)} "
              f"(>=0.80: point {a1 >= 0.80}, ci_lower {lo1 >= 0.80}); nonaudible_s2={fmt(a2)} boot_upper={fmt(hi2)} "
              f"(<=0.62: point {a2 <= 0.62}, ci_upper {hi2 <= 0.62})")
    have_widths = [a for a in NOKD_ARMS if a in vres]
    print(f"H1 (audible >= 0.80 and non-audible <= 0.62 at every no-KD width {have_widths}, point estimate of seed-mean): {h1_ok}")
    if ("W9", "W1noKD") in vdiff:
        g = vdiff[("W9", "W1noKD")]
        g1, g2 = g[s1], g[s2]
        h2 = g1["mean"] >= 0.07 and g2["mean"] <= 0.03
        print(f"H2 (W9-W1noKD gap: audible >= 0.07 and non-audible <= 0.03): {h2} audible_gap={fmt(g1['mean'])} welch_ci95=[{fmt(g1['lo'])}, {fmt(g1['hi'])}] "
              f"nonaudible_gap={fmt(g2['mean'])} welch_ci95=[{fmt(g2['lo'])}, {fmt(g2['hi'])}]")
    else:
        h2 = None
        print("H2: not evaluable (needs arms W9 and W1noKD)")
    h3 = None
    if ("W1organKD", "W1noKD") in vdiff:
        v = vdiff[("W1organKD", "W1noKD")][s1]
        val_pass = v["mean"] > 0.02 and (v["lo"] > 0 or v["hi"] < 0) and v["mean"] > 0
        print(f"H3_val (organ KD - no-KD, audible s1, paired): mean={fmt(v['mean'])} t_ci95=[{fmt(v['lo'])}, {fmt(v['hi'])}] "
              f"mean>0.02: {v['mean'] > 0.02} ci_excludes0: {v['lo'] > 0 or v['hi'] < 0} -> val_condition_met={val_pass}  "
              f"(val number was seen before registration: exploratory)")
        if tdiff is not None and ("W1organKD", "W1noKD") in tdiff:
            t = tdiff[("W1organKD", "W1noKD")][s1]
            same = (t["mean"] > 0) == (v["mean"] > 0)
            test_pass = same and t["mean"] > 0.01
            print(f"H3_test (REPORT, independent replicate): mean={fmt(t['mean'])} t_ci95=[{fmt(t['lo'])}, {fmt(t['hi'])}] "
                  f"same_sign_as_val={same} mean>0.01: {t['mean'] > 0.01} -> test_condition_met={test_pass}")
            h3 = val_pass and test_pass
            print(f"H3 'KD transfers audible-murmur ranking' CLAIMED: {h3}" + ("" if h3 else "  (otherwise: not shown)"))
        else:
            print(f"H3_test: PENDING (organ KD / no-KD test dumps incomplete; avail W1organKD={avail.get('W1organKD')}/{len(arms.get('W1organKD', []))} "
                  f"W1noKD={avail.get('W1noKD')}/{len(arms.get('W1noKD', []))})")
            print("H3 'KD transfers audible-murmur ranking' CLAIMED: NOT DECIDABLE YET (needs test dumps); wording until then: not shown")
    else:
        print("H3: not evaluable (needs arms W1organKD and W1noKD)")
    if ("W1flatKD", "W1noKD") in vdiff:
        v = vdiff[("W1flatKD", "W1noKD")][s1]
        print(f"descriptive (not pre-registered), flat KD - no-KD audible s1: mean={fmt(v['mean'])} t_ci95=[{fmt(v['lo'])}, {fmt(v['hi'])}]")
    print("---")
    print(f"h1_val: {h1_ok}")
    print(f"h2_val: {h2}")
    print(f"h3_claimed: {h3}")


if __name__ == "__main__":
    main()
