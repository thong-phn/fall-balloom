"""exp-033e: eval-only dump of last-epoch TRAIN / VAL / TEST logits of existing -last.pt checkpoints.

No training, no change to any existing result. Every split is decoded ONCE (batch_size 1, one recording per batch,
BioSARMulti(train=False): all non-overlapping 5 s clips, last clip aligned to the end, logits averaged per recording,
exactly the eval branch of run_one_epoch in exp-009.py) and fed to all checkpoints.

Gate (stops before the expensive splits): val logits must equal the last row of the run's -vallogits.pt to 1e-4.
Afterwards: val / test prob-AUROC and test macro-F1 are compared with the run's own log (sel_R2_last / prob_last)
when the log is found (jobs/ar-<id>-<jobid>.out or --logs).

Output per checkpoint: <ckpt stem>-evallogits.pt (next to the checkpoint unless --out-dir is given).
The hpc template passes --metadata --data --num-workers --output; --output is accepted and ignored.
"""
import argparse
import csv
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # as exp-009.py, before importing torch
import re
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.datasets import CLASS_NAMES, BioSARMulti
from utils.models import SimpleCNN, separable_conv
from utils.utils import binary_f1_auroc, set_seed

NUM_CLASSES = len(CLASS_NAMES)
VAL_TOL = 1e-4        # max |val logits - saved vallogits[-1]|
LOG_TOL = 1e-6        # prob-AUROC / test macro-F1 vs the run log (the log prints 6 decimals)


# ---- verbatim from research/exp-009.py (not importable on the cluster: only the candidate and utils/ are synced) ----
class WideSeparableCNN(SimpleCNN):
    """SeparableCNN with a width multiplier. width=1 builds exactly SeparableCNN (same modules, same RNG consumption)."""
    def __init__(self, num_classes=1, width=1):
        super().__init__(num_classes)  # SimpleCNN init first, exactly like SeparableCNN
        c = [16 * width, 32 * width, 64 * width, 128 * width]
        self.features = nn.Sequential(
            nn.Conv2d(1, c[0], 3, padding=1), nn.BatchNorm2d(c[0]), nn.ReLU(), nn.MaxPool2d(2),
            separable_conv(c[0], c[1]), nn.BatchNorm2d(c[1]), nn.ReLU(), nn.MaxPool2d(2),
            separable_conv(c[1], c[2]), nn.BatchNorm2d(c[2]), nn.ReLU(), nn.MaxPool2d(2),
            separable_conv(c[2], c[3]), nn.BatchNorm2d(c[3]), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        if width != 1:
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(c[3], num_classes))

    def forward_features(self, x):
        """Pooled features [B, 128*width] (same input handling as SimpleCNN.forward)."""
        if x.ndim == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return self.features(x).flatten(1)


class GlobalMeanPool(nn.Module):
    """Deterministic drop-in for AdaptiveAvgPool2d(1): same output, but its CUDA backward has no atomic adds."""

    def forward(self, x):
        return x.mean(dim=(-2, -1), keepdim=True)


def make_pooling_deterministic(model):
    """Swap every AdaptiveAvgPool2d with output size 1 for GlobalMeanPool. No parameters change."""
    swapped = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.AdaptiveAvgPool2d) and child.output_size in (1, (1, 1)):
                setattr(parent, name, GlobalMeanPool())
                swapped += 1
    return swapped


def prob_auroc(logits, labels):
    """Macro one-vs-rest AUROC on softmax probabilities (present classes only)."""
    p = logits.softmax(dim=1)
    vals = [binary_f1_auroc(p[:, c], (labels == c).float())[1]
            for c in range(NUM_CLASSES) if (labels == c).any() and (labels != c).any()]
    return sum(vals) / len(vals)


def multiclass_metrics(logits, labels):
    """Accuracy, balanced accuracy, macro-F1 and macro one-vs-rest AUROC (present classes only)."""
    preds = logits.argmax(dim=1)
    recalls, f1s, aurocs = [], [], []
    for c in range(NUM_CLASSES):
        is_c, pred_c = labels == c, preds == c
        if not is_c.any():
            continue
        tp = (is_c & pred_c).sum().item()
        fp = (~is_c & pred_c).sum().item()
        fn = (is_c & ~pred_c).sum().item()
        recalls.append(tp / (tp + fn))
        f1s.append(2 * tp / (2 * tp + fp + fn))
        if (~is_c).any():
            aurocs.append(binary_f1_auroc(logits[:, c], is_c.float())[1])
    return {
        "acc": (preds == labels).float().mean().item(),
        "bal_acc": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "macro_auroc": sum(aurocs) / len(aurocs) if aurocs else float("nan"),
    }
# ---- end verbatim ----


def parse_log(path):
    """sel_R2_last test_f1 / val_f1 and prob_last val/test prob-AUROC from a run log; None entries when absent."""
    out = {"sel_val_f1": None, "sel_test_f1": None, "val_auroc_prob": None, "test_auroc_prob": None}
    try:
        text = Path(path).read_text()
    except OSError:
        return out
    m = re.search(r"^sel_R2_last:.*?val_f1=([0-9.]+).*?test_f1=([0-9.]+)", text, re.M)
    if m:
        out["sel_val_f1"], out["sel_test_f1"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"^prob_last:.*?val_auroc_prob=([0-9.]+).*?test_auroc_prob=([0-9.]+)", text, re.M)
    if m:
        out["val_auroc_prob"], out["test_auroc_prob"] = float(m.group(1)), float(m.group(2))
    return out


def find_log(ckpt, jobs_dir):
    """jobs/ar-<exp-id>-<jobid>.out for a checkpoint named <exp-id>-<jobid>-last.pt, or None."""
    m = re.match(r"^(exp-[0-9a-z]+)-(\d+)-last$", ckpt.stem)
    if not m:
        return None
    p = Path(jobs_dir) / f"ar-{m.group(1)}-{m.group(2)}.out"
    return p if p.exists() else None


@torch.no_grad()
def predict_split(models, loader, device):
    """One pass over the split; every model sees every recording. Returns (list of [N, C] logits, labels, mixture ids)."""
    outs = [[] for _ in models]
    labels_all, ids_all = [], []
    n = len(loader)
    t0 = time.time()
    for i, (features, labels, mixture_id) in enumerate(loader):
        features = features.squeeze(0).to(device)       # [n_clips, 1, 64, T]
        for k, model in enumerate(models):
            outs[k].append(model(features).mean(dim=0, keepdim=True).float().cpu())   # [1, C], as run_one_epoch
        labels_all.append(labels.cpu())
        ids_all.extend(mixture_id)
        if (i + 1) % 2000 == 0 or i + 1 == n:
            print(f"\t{i + 1}/{n} recordings ({time.time() - t0:.0f}s)", flush=True)
    return [torch.cat(o) for o in outs], torch.cat(labels_all), ids_all


def main():
    ap = argparse.ArgumentParser(description="exp-033e: dump last-epoch train / val / test logits of existing -last.pt checkpoints")
    ap.add_argument("--metadata", default="./datasets/BioSARDdBScene/metadata.csv")
    ap.add_argument("--data", default="./datasets/BioSARDdBScene/")
    ap.add_argument("--output", default=None, help="ignored (passed by the hpc template)")
    ap.add_argument("--num-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    ap.add_argument("--ckpts", required=True, help="comma-separated -last.pt paths")
    ap.add_argument("--vallogits", required=True, help="comma-separated -vallogits.pt paths, same order as --ckpts")
    ap.add_argument("--logs", default=None, help="optional comma-separated run logs, same order (default: jobs/ar-<id>-<jobid>.out)")
    ap.add_argument("--jobs-dir", default="jobs")
    ap.add_argument("--out-dir", default=None, help="default: next to each checkpoint")
    ap.add_argument("--splits", default="val,test,train", help="order of the splits (val first so the gate stops early)")
    args = ap.parse_args()

    ckpts = [Path(p) for p in args.ckpts.split(",") if p]
    vallogits = [Path(p) for p in args.vallogits.split(",") if p]
    if len(ckpts) != len(vallogits):
        sys.exit(f"--ckpts ({len(ckpts)}) and --vallogits ({len(vallogits)}) differ in length")
    logs = [Path(p) for p in args.logs.split(",") if p] if args.logs else [find_log(c, args.jobs_dir) for c in ckpts]
    if len(logs) != len(ckpts):
        sys.exit("--logs length differs from --ckpts")

    set_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; checkpoints: {len(ckpts)}; workers: {args.num_workers}")

    models, widths = [], []
    for c in ckpts:
        ck = torch.load(c, map_location="cpu")
        model = WideSeparableCNN(num_classes=NUM_CLASSES, width=ck["width"])
        make_pooling_deterministic(model)
        model.load_state_dict(ck["model_state_dict"])
        models.append(model.to(device).eval())
        widths.append(ck["width"])
        print(f"loaded {c} width={ck['width']} epoch={ck.get('epoch')}")

    with open(args.metadata, newline="") as f:
        source_of = {r["mixture_id"]: r["source_id"] for r in csv.DictReader(f)}

    nw = max(0, args.num_workers)
    opts = {"num_workers": nw, "pin_memory": device.type == "cuda", "persistent_workers": False}
    results = {}   # split -> (logits list, labels, ids)
    failures = []
    for split in args.splits.split(","):
        data = BioSARMulti(data_path=args.data, metadata_path=args.metadata, split=split, train=False)
        loader = DataLoader(data, batch_size=1, shuffle=False, **opts)
        print(f"split {split}: {len(data)} recordings", flush=True)
        results[split] = predict_split(models, loader, device)
        logits_l, labels, ids = results[split]
        if split == "val":
            print("=== val gate (stop here when it fails) ===")
            for k, c in enumerate(ckpts):
                ref = torch.load(vallogits[k], map_location="cpu")
                ref_last = ref["val_logits"][-1].float()
                ok_labels = bool(torch.equal(ref["val_labels"].cpu(), labels))
                diff = (logits_l[k] - ref_last).abs().max().item() if ref_last.shape == logits_l[k].shape else float("inf")
                m = multiclass_metrics(logits_l[k], labels)
                lg = parse_log(logs[k]) if logs[k] else parse_log("/nonexistent")
                pa = prob_auroc(logits_l[k], labels)
                line = (f"gate_val: {c.name} max_abs_diff_vs_vallogits={diff:.3e} labels_equal={ok_labels} "
                        f"val_macro_f1={m['macro_f1']:.6f} val_auroc_prob={pa:.6f}")
                bad = diff > VAL_TOL or not ok_labels
                if lg["val_auroc_prob"] is not None:
                    line += f" log_val_auroc_prob={lg['val_auroc_prob']:.6f}"
                    bad = bad or abs(pa - lg["val_auroc_prob"]) > LOG_TOL
                if lg["sel_val_f1"] is not None:
                    line += f" log_sel_R2_last_val_f1={lg['sel_val_f1']:.6f}"
                    bad = bad or abs(m["macro_f1"] - lg["sel_val_f1"]) > LOG_TOL
                if ref.get("epoch_val_f1"):
                    line += f" saved_epoch_val_f1_last={ref['epoch_val_f1'][-1]:.6f}"
                print(line + (" FAIL" if bad else " ok"), flush=True)
                if bad:
                    failures.append(c.name)
            if failures:
                sys.exit(f"VAL GATE FAILED for {len(failures)} checkpoint(s): {failures}. Wrong file or non-deterministic eval path; nothing saved.")

    if "test" in results:
        print("=== test report (last epoch; compared with sel_R2_last / prob_last of the run log where available) ===")
        for k, c in enumerate(ckpts):
            logits_l, labels, _ = results["test"]
            m = multiclass_metrics(logits_l[k], labels)
            pa = prob_auroc(logits_l[k], labels)
            lg = parse_log(logs[k]) if logs[k] else parse_log("/nonexistent")
            line = f"gate_test_f1: {c.name} test_macro_f1={m['macro_f1']:.6f} test_bal_acc={m['bal_acc']:.6f} test_auroc_prob={pa:.6f}"
            bad = False
            if lg["sel_test_f1"] is not None:
                line += f" log_sel_R2_last_test_f1={lg['sel_test_f1']:.6f} diff={abs(m['macro_f1'] - lg['sel_test_f1']):.2e}"
                bad = bad or abs(m["macro_f1"] - lg["sel_test_f1"]) > LOG_TOL
            if lg["test_auroc_prob"] is not None:
                line += f" log_test_auroc_prob={lg['test_auroc_prob']:.6f}"
                bad = bad or abs(pa - lg["test_auroc_prob"]) > LOG_TOL
            print(line + (" FAIL" if bad else (" ok" if lg["sel_test_f1"] is not None else " (no log)")), flush=True)
            if bad:
                failures.append(c.name)
    if "train" in results:
        for k, c in enumerate(ckpts):
            logits_l, labels, _ = results["train"]
            m = multiclass_metrics(logits_l[k], labels)
            print(f"train_report: {c.name} train_macro_f1={m['macro_f1']:.6f} train_acc={m['acc']:.4f}")

    for k, c in enumerate(ckpts):
        payload = {"width": widths[k], "ckpt": str(c), "vallogits_ref": str(vallogits[k])}
        for split, (logits_l, labels, ids) in results.items():
            payload[f"{split}_logits"] = logits_l[k].float()
            payload[f"{split}_labels"] = labels
            payload[f"{split}_mixture_ids"] = ids
            payload[f"{split}_groups"] = [source_of.get(i, "") for i in ids]
        out_dir = Path(args.out_dir) if args.out_dir else c.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / (c.stem + "-evallogits.pt")
        torch.save(payload, out)
        print(f"saved {out}")
    print("---")
    print(f"checkpoints: {len(ckpts)}")
    print(f"gate_failures: {len(failures)}")
    if failures:
        sys.exit(f"LOG GATE FAILED for {sorted(set(failures))} (files were saved; do not use them until resolved)")


if __name__ == "__main__":
    main()
