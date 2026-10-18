import argparse
import csv
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by use_deterministic_algorithms for cuBLAS
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.datasets import CLASS_NAMES, BioSARMulti
from utils.models import EdgeMamba, SeparableCNN, SimpleCNN, TinyMamba
from utils.utils import binary_f1_auroc, set_seed

NUM_CLASSES = len(CLASS_NAMES)  # 4 organs x (normal, abnormal)


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


def class_counts(labels):
    counts = Counter(labels)
    return torch.tensor([counts.get(c, 0) for c in range(NUM_CLASSES)], dtype=torch.float)


def class_weights(counts, power):
    """Weight ~ (1/n_c)^power, scaled so the average per-sample weight stays 1.

    power=1 fully balances the classes, power=0.5 is a softer compromise, 0 is off.
    """
    w = counts.clamp(min=1) ** -power
    return w * counts.sum() / (w * counts).sum()


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


def per_class_report(logits, labels):
    preds = logits.argmax(dim=1)
    print("\tclass            n   recall")
    for c, name in enumerate(CLASS_NAMES):
        is_c = labels == c
        if is_c.any():
            recall = (preds[is_c] == c).float().mean().item()
            print(f"\t{name:<14}{is_c.sum().item():>5}   {recall:.3f}")


BIAS_GRID = [round(-3.0 + 0.1 * i, 1) for i in range(61)]   # -3.0 .. 3.0
TAU_GRID = [round(-1.0 + 0.05 * i, 2) for i in range(61)]   # -1.0 .. 2.0
TAIL = 5
EMULATED_PATIENCE = 10
SPLIT_SEED = 0
PBN_BATCHES = 200  # precise-BN: fixed number of train batches per epoch state
PBN_SEED = 0  # precise-BN: seed of its own DataLoader generator


def fast_macro_f1(preds, labels):
    """Same macro-F1 formula as multiclass_metrics (present classes only), without AUROC; used inside the searches."""
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


def fit_class_bias(logits, labels, ref, sweeps=3):
    """Coordinate ascent on a per-class additive logit bias (bias[ref] fixed at 0) maximising macro-F1."""
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


def fit_prior_tau(logits, labels, log_prior):
    """1-parameter post-hoc logit adjustment z - tau * log(prior) (Menon et al. 2021), tau chosen for macro-F1."""
    best_tau, best = 0.0, fast_macro_f1(logits.argmax(dim=1), labels)
    for tau in TAU_GRID:
        f = fast_macro_f1((logits - tau * log_prior).argmax(dim=1), labels)
        if f > best + 1e-9 or (abs(f - best) <= 1e-9 and abs(tau) < abs(best_tau)):
            best_tau, best = tau, f
    return best_tau


def grouped_folds(labels, groups, k, seed=0):
    """Assign each source group to one of k folds, stratified by the group's first label. Fixed seed, independent of --seed."""
    gen = torch.Generator().manual_seed(seed)
    first_label = {}
    for y, g in zip(labels.tolist(), groups):
        first_label.setdefault(g, y)
    fold_of = {}
    for c in range(NUM_CLASSES):
        gs = sorted(g for g, y in first_label.items() if y == c)
        for i, j in enumerate(torch.randperm(len(gs), generator=gen).tolist()):
            fold_of[gs[j]] = (i + c) % k
    return torch.tensor([fold_of[g] for g in groups])


def cross_fit(logits, labels, folds, fit, apply):
    """Out-of-fold adjusted logits: for each fold, fit on the other folds and apply to this fold."""
    out, params = logits.clone(), []
    for f in sorted(set(folds.tolist())):
        held = folds == f
        p = fit(logits[~held], labels[~held])
        out[held] = apply(logits[held], p)
        params.append(p)
    return out, params


def select_patience(f, patience):
    """Emulate max-val selection with early stopping on a per-epoch score list. Returns (selected_epoch, stop_epoch)."""
    best, best_e, count = float("-inf"), -1, 0
    for e, x in enumerate(f):
        if x > best:
            best, best_e, count = x, e, 0
        else:
            count += 1
            if count >= patience:
                return best_e, e
    return best_e, len(f) - 1


def select_ma3(f):
    """Centred 3-epoch moving average, first maximum. Returns (centre_epoch, smoothed_value)."""
    ma = [(f[e - 1] + f[e] + f[e + 1]) / 3 for e in range(1, len(f) - 1)]
    i = max(range(len(ma)), key=lambda k: ma[k])
    return i + 1, ma[i]


def tail_ensemble(logit_list):
    """Average softmax probabilities over checkpoints; returns log-probabilities usable as logits."""
    return torch.stack([z.softmax(dim=1) for z in logit_list]).mean(dim=0).log()


def spearman(x, y):
    rx = torch.tensor(x).argsort().argsort().float()
    ry = torch.tensor(y).argsort().argsort().float()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return (rx * ry).sum().item() / max(1e-12, (rx.norm() * ry.norm()).item())


def prob_auroc(logits, labels):
    """Macro one-vs-rest AUROC on softmax probabilities (present classes only). Reporting only; multiclass_metrics is unchanged."""
    p = logits.softmax(dim=1)
    vals = [binary_f1_auroc(p[:, c], (labels == c).float())[1]
            for c in range(NUM_CLASSES) if (labels == c).any() and (labels != c).any()]
    return sum(vals) / len(vals)


def bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def precise_bn(model, loader, device, num_batches, first_batch_sums=None):
    """Recompute BN running stats with frozen weights: plain average over the first num_batches of loader (Wu & Johnson 2021).
    Returns the mean over BN layers of mean(|mu_ema - mu_precise| / sqrt(var_precise + eps)) as a drift diagnostic.
    If first_batch_sums is a list, the float sum of the first batch's features is appended (data-order check)."""
    bns = bn_layers(model)
    ema = [(m.running_mean.clone(), m.running_var.clone()) for m in bns]
    moms = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
    model.train()
    for i, (features, _, _) in enumerate(loader):
        if i >= num_batches:
            break
        if i == 0 and first_batch_sums is not None:
            first_batch_sums.append(features.double().sum().item())
        model(features.to(device))
    model.eval()
    for m, mom in zip(bns, moms):
        m.momentum = mom
    drift = [((mu - m.running_mean).abs() / (m.running_var + m.eps).sqrt()).mean().item() for (mu, _), m in zip(ema, bns)]
    return sum(drift) / len(drift)


def run_one_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    all_logits, all_labels = [], []

    for features, labels, _ in loader:
        labels = labels.to(device)

        if training:
            features = features.to(device)
            optimizer.zero_grad()
            logits = model(features)  # [B, C]
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = labels.numel()
        else:
            # Evaluation returns one or more clips per recording. Average their
            # logits so each recording contributes one prediction.
            features = features.squeeze(0).to(device)
            with torch.no_grad():
                logits = model(features).mean(dim=0, keepdim=True)  # [1, C]
                loss = loss_fn(logits, labels)
            batch_size = 1

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())
        total_loss += loss.item() * batch_size
        count += batch_size

    logits, labels = torch.cat(all_logits), torch.cat(all_labels)
    return total_loss / count, multiclass_metrics(logits, labels), logits, labels


def fmt(metrics):
    return " ".join(f"{k}={v:.3f}" for k, v in metrics.items())


def main():
    parser = argparse.ArgumentParser(description="Train an 8-class (4 organs x normal/abnormal) BioSAR classifier")
    parser.add_argument("--metadata", default="./datasets/BioSARDdBScene/metadata.csv")
    parser.add_argument("--data", default="./datasets/BioSARDdBScene/")
    parser.add_argument("--output", default="models/ckpt/biosar-8class-full.pt")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--early-stop", type=int, default=0, help="early-stop patience in epochs; 0 = off (train all --epochs)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine", help="per-epoch LR schedule; cosine anneals learning-rate -> 1e-5 over --epochs")
    parser.add_argument("--model", choices=["cnn", "simple-cnn", "mamba", "edge-mamba"], default="cnn")
    parser.add_argument(
        "--balance-power",
        type=float,
        default=0.5,
        help="class weight ~ (1/count)^power; 1 = fully balanced, 0 = off",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibrate", choices=["none", "cv"], default="cv", help="post-hoc decision calibration on val (reporting only; checkpoint selection unchanged)")
    parser.add_argument("--calib-folds", type=int, default=5)
    parser.add_argument("--eval-checkpoint", default=None, help="skip training; load this checkpoint and run eval + calibration")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
        help="DataLoader worker processes (defaults to the CPUs allocated by Slurm)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_data = BioSARMulti(data_path=args.data, metadata_path=args.metadata, split="train", train=True)
    val_data = BioSARMulti(data_path=args.data, metadata_path=args.metadata, split="val", train=False)
    test_data = BioSARMulti(data_path=args.data, metadata_path=args.metadata, split="test", train=False)

    num_workers = max(0, args.num_workers)
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    # Class imbalance: class-weighted cross-entropy
    counts = class_counts(train_data.labels())
    weights = class_weights(counts, args.balance_power)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, **loader_options
    )
    # One recording per batch
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, **loader_options)

    sample_features, _, _ = next(iter(train_loader))
    print(f"-"*50)
    print(f"TRAINNING START")
    print(f"Model input shape: {tuple(sample_features.shape)}")
    print(f"Class-weighted loss (power={args.balance_power})")
    for name, n, w in zip(CLASS_NAMES, counts.tolist(), weights.tolist()):
        print(f"\t{name:<14}train={int(n):>6} weight={w:.2f}")

    if args.model == "mamba":
        model = TinyMamba(n_mels=sample_features.shape[-2], num_classes=NUM_CLASSES).to(device)
    elif args.model == "edge-mamba":
        model = EdgeMamba(n_mels=sample_features.shape[-2], num_classes=NUM_CLASSES).to(device)
    elif args.model == "simple-cnn":
        model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    else:
        model = SeparableCNN(num_classes=NUM_CLASSES).to(device)
    n_swapped = make_pooling_deterministic(model)
    print(f"Deterministic mode: cudnn.deterministic=True, use_deterministic_algorithms(warn_only=True), CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}, AdaptiveAvgPool2d->mean swapped={n_swapped}")
    print(f"Model: {args.model}; params: {sum(p.numel() for p in model.parameters()):,}")
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        if args.lr_schedule == "cosine" else None
    )
    eval_only = args.eval_checkpoint is not None
    output_path = Path(args.eval_checkpoint) if eval_only else Path(args.output)
    if not eval_only:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_f1 = float("-inf")
    best_epoch, best_val_metrics = -1, {}
    start_time = time.time()
    early_stop_tolerant = args.early_stop
    early_stop_count = 0

    print(
        f"Device: {device}; "
        f"workers: {num_workers}; "
        f"train: {len(train_data)}; "
        f"val: {len(val_data)}; test: {len(test_data)}"
    )

    epoch, early_stop_count = -1, 0
    val_hist, state_hist, f_logged = [], [], []
    if not eval_only:
        # training loop
        for epoch in range(args.epochs):
            train_loss, train_metrics, _, _ = run_one_epoch(model, train_loader, loss_fn, device, optimizer)
            epoch_lr = optimizer.param_groups[0]["lr"]  # LR used this epoch (read before scheduler.step())
            if scheduler is not None:
                scheduler.step()
            val_loss, val_metrics, val_ep_logits, _ = run_one_epoch(model, val_loader, loss_fn, device)
            val_hist.append(val_ep_logits)
            state_hist.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            f_logged.append(val_metrics["macro_f1"])
            print(
                f"Epoch {epoch}/{args.epochs}: "
                f"lr={epoch_lr:.2e} | "
                f"train_loss={train_loss:.4f} train_acc={train_metrics['acc']:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_metrics['acc']:.3f} "
                f"val_f1={val_metrics['macro_f1']:.4f} val_bal_acc={val_metrics['bal_acc']:.3f}"
            )

            # Checkpoint on val macro-F1: val loss and accuracy are dominated by the
            # majority classes (lung), macro-F1 weights every class equally.
            if val_metrics["macro_f1"] > best_val_f1:
                early_stop_count = 0
                best_val_f1 = val_metrics["macro_f1"]
                best_epoch, best_val_metrics = epoch, val_metrics
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                        "class_names": CLASS_NAMES,
                    },
                    output_path,
                )
            else:
                early_stop_count += 1
                if early_stop_tolerant > 0 and early_stop_count >= early_stop_tolerant:
                    print(f"\tEarly stop at epoch {epoch} ")
                    break

    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if eval_only:
        best_epoch = checkpoint["epoch"]
    _, val_metrics_ckpt, val_logits, val_labels = run_one_epoch(model, val_loader, loss_fn, device)
    print(f"val_f1_recomputed: {val_metrics_ckpt['macro_f1']:.6f}")
    if eval_only:
        best_val_metrics = val_metrics_ckpt
    test_loss, test_metrics, test_logits, test_labels = run_one_epoch(model, test_loader, loss_fn, device)
    print(f"Test: loss={test_loss:.4f} {fmt(test_metrics)}")
    per_class_report(test_logits, test_labels)
    print(f"Best checkpoint saved to {output_path}")

    if args.calibrate == "cv":
        calib_start = time.time()
        with open(args.metadata, newline="") as f:
            src = {r["mixture_id"]: r["source_id"] for r in csv.DictReader(f)}
        val_groups = [src[it["mixture_id"]] for it in val_data.items]
        folds = grouped_folds(val_labels, val_groups, args.calib_folds)
        log_prior = (counts.clamp(min=1) / counts.sum()).log()
        ref = CLASS_NAMES.index("Lung-normal")

        # Bias (primary)
        bias_full = fit_class_bias(val_logits, val_labels, ref)
        oof_bias, fold_biases = cross_fit(
            val_logits, val_labels, folds, lambda z, y: fit_class_bias(z, y, ref), lambda z, b: z + b
        )
        m_bias_cv = multiclass_metrics(oof_bias, val_labels)  # AUROC ignored (mixes fold biases)
        f1_bias_in = fast_macro_f1((val_logits + bias_full).argmax(1), val_labels)
        m_test_bias = multiclass_metrics(test_logits + bias_full, test_labels)

        # Prior shift tau (secondary)
        tau_full = fit_prior_tau(val_logits, val_labels, log_prior)
        oof_tau, fold_taus = cross_fit(
            val_logits, val_labels, folds, lambda z, y: fit_prior_tau(z, y, log_prior), lambda z, t: z - t * log_prior
        )
        f1_tau_cv = fast_macro_f1(oof_tau.argmax(1), val_labels)
        m_test_tau = multiclass_metrics(test_logits - tau_full * log_prior, test_labels)

        pred_arg, pred_cv = val_logits.argmax(1), oof_bias.argmax(1)
        print("Val per-class, argmax vs cross-fitted bias:")
        print("\tclass            n  recall_argmax  prec_argmax  recall_bias_cv  prec_bias_cv  bias_full")
        for c, name in enumerate(CLASS_NAMES):
            is_c = val_labels == c
            if not is_c.any():
                continue
            n = is_c.sum().item()
            r_a = (pred_arg[is_c] == c).float().mean().item()
            r_b = (pred_cv[is_c] == c).float().mean().item()
            p_a = (is_c & (pred_arg == c)).sum().item() / max(1, (pred_arg == c).sum().item())
            p_b = (is_c & (pred_cv == c)).sum().item() / max(1, (pred_cv == c).sum().item())
            print(f"\t{name:<14}{n:>5}  {r_a:>13.3f}  {p_a:>11.3f}  {r_b:>14.3f}  {p_b:>12.3f}  {bias_full[c].item():>9.2f}")
        print("Test recall with val-fitted bias (reporting only):")
        per_class_report(test_logits + bias_full, test_labels)

        print("=== calibration (post-hoc, fit on val only; checkpoint selection unchanged) ===")
        print(f"calib_val_f1_argmax:      {val_metrics_ckpt['macro_f1']:.6f}")
        print(f"calib_val_f1_bias_cv:     {m_bias_cv['macro_f1']:.6f}")
        print(f"calib_val_bal_acc_bias_cv:{m_bias_cv['bal_acc']:.6f}")
        print(f"calib_val_f1_bias_insample:{f1_bias_in:.6f}")
        print("calib_bias_full:          " + ",".join(f"{n}={b:.2f}" for n, b in zip(CLASS_NAMES, bias_full.tolist())))
        print("calib_bias_folds:         " + " ".join("[" + ",".join(f"{b:.1f}" for b in fb.tolist()) + "]" for fb in fold_biases))
        print(f"calib_val_f1_tau_cv:      {f1_tau_cv:.6f}")
        print(f"calib_tau_full:           {tau_full:.6f}")
        print("calib_tau_folds:          [" + ",".join(f"{t:.2f}" for t in fold_taus) + "]")
        print(f"calib_test_f1_bias:       {m_test_bias['macro_f1']:.6f}")
        print(f"calib_test_bal_acc_bias:  {m_test_bias['bal_acc']:.6f}")
        print(f"calib_test_f1_tau:        {m_test_tau['macro_f1']:.6f}")
        print(f"calib_seconds:            {time.time() - calib_start:.1f}")

    if not eval_only and len(val_hist) == args.epochs:
        f = [fast_macro_f1(z.argmax(1), val_labels) for z in val_hist]
        check = max(abs(a - b) for a, b in zip(f, f_logged))
        e0, stop0 = select_patience(f, EMULATED_PATIENCE)
        e1 = max(range(len(f)), key=f.__getitem__)
        e2 = len(f) - 1
        e3, m3 = select_ma3(f)
        tail = list(range(len(f) - TAIL, len(f)))
        r4 = sum(f[e] for e in tail) / len(tail)
        m5 = multiclass_metrics(tail_ensemble([val_hist[e] for e in tail]), val_labels)

        test_cache = {}

        def test_logits_at(e):
            if e not in test_cache:
                model.load_state_dict(state_hist[e])
                test_cache[e] = run_one_epoch(model, test_loader, loss_fn, device)[2]
            return test_cache[e]

        for e in sorted({e0, e1, e3} | set(tail)):
            test_logits_at(e)

        def epoch_report(e):
            mv = multiclass_metrics(val_hist[e], val_labels)
            mt = multiclass_metrics(test_logits_at(e), test_labels)
            return f"val_f1={f[e]:.6f} val_auroc={mv['macro_auroc']:.6f} test_f1={mt['macro_f1']:.6f} test_auroc={mt['macro_auroc']:.6f}"

        test_f1_tail = [multiclass_metrics(test_logits_at(e), test_labels)["macro_f1"] for e in tail]
        m5_test = multiclass_metrics(tail_ensemble([test_logits_at(e) for e in tail]), test_labels)
        test_f1_e3 = multiclass_metrics(test_logits_at(e3), test_labels)["macro_f1"]

        # Split-half validity test (val only)
        with open(args.metadata, newline="") as fh:
            src_sel = {r["mixture_id"]: r["source_id"] for r in csv.DictReader(fh)}
        sel_groups = [src_sel[it["mixture_id"]] for it in val_data.items]
        halves = grouped_folds(val_labels, sel_groups, 2, seed=SPLIT_SEED)
        transfer = {k: [] for k in ("R0", "R1", "R2", "R3", "R4", "R5")}
        rho_late = None
        for h in (0, 1):
            A = halves == h
            B = ~A
            fA = [fast_macro_f1(z[A].argmax(1), val_labels[A]) for z in val_hist]
            fB = [fast_macro_f1(z[B].argmax(1), val_labels[B]) for z in val_hist]
            transfer["R0"].append(fB[select_patience(fA, EMULATED_PATIENCE)[0]])
            transfer["R1"].append(fB[max(range(len(fA)), key=fA.__getitem__)])
            transfer["R2"].append(fB[-1])
            transfer["R3"].append(fB[select_ma3(fA)[0]])
            transfer["R4"].append(sum(fB[e] for e in tail) / len(tail))
            transfer["R5"].append(fast_macro_f1(tail_ensemble([val_hist[e][B] for e in tail]).argmax(1), val_labels[B]))
            if h == 0:
                rho_late = spearman(fA[8:], fB[8:])

        torch.save(
            {"val_logits": torch.stack(val_hist), "val_labels": val_labels, "val_groups": sel_groups, "halves": halves, "epoch_val_f1": f},
            output_path.with_name(output_path.stem + "-vallogits.pt"),
        )

        print("=== selection rules (post hoc, same trajectory; test is reporting only) ===")
        print(f"sel_check_max_abs_diff: {check:.6f}")
        print(f"sel_R0_patience10: epoch={e0} stop={stop0} {epoch_report(e0)}")
        print(f"sel_R1_max: epoch={e1} {epoch_report(e1)}")
        print(f"sel_R2_last: epoch={e2} {epoch_report(e2)}")
        print(f"sel_R3_ma3: epoch={e3} ma3={m3:.6f} val_f1_at_epoch={f[e3]:.6f} test_f1={test_f1_e3:.6f}")
        print(f"sel_R4_tailmean5: val_f1={r4:.6f} test_f1_mean={sum(test_f1_tail) / len(test_f1_tail):.6f}")
        print(f"sel_R5_tailens5: val_f1={m5['macro_f1']:.6f} val_auroc={m5['macro_auroc']:.6f} test_f1={m5_test['macro_f1']:.6f} test_auroc={m5_test['macro_auroc']:.6f}")
        print("sel_split_transfer: " + " ".join(f"{k}={sum(v) / len(v):.6f}" for k, v in transfer.items()))
        print(f"sel_split_spearman_late: {rho_late:.6f}")

        # ---- precise BN (eval-only, post hoc; weights, training and global RNG untouched) ----
        pbn_start = time.time()
        rng_saved = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, np.random.get_state(), random.getstate())
        model_saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bn_gen = torch.Generator()
        bn_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=bn_gen, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=False)
        pval_hist, pstate_hist, pval_loss, drifts, pbn_lines, sums = [], [], [], [], [], []
        for pe in range(args.epochs):
            bn_gen.manual_seed(PBN_SEED)  # same 200 batches for every epoch state
            torch.manual_seed(PBN_SEED)  # crops when num_workers == 0 (global RNG; restored below)
            model.load_state_dict(state_hist[pe])
            drift_e = precise_bn(model, bn_loader, device, PBN_BATCHES, sums if pe in (0, args.epochs - 1) else None)
            drifts.append(drift_e)
            pstate_hist.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            vl, vm, vz, _ = run_one_epoch(model, val_loader, loss_fn, device)
            pval_hist.append(vz)
            pval_loss.append(vl)
            pbn_lines.append(
                f"pbn_epoch: e={pe} val_loss={vl:.6f} val_f1={vm['macro_f1']:.6f} val_bal_acc={vm['bal_acc']:.6f} val_acc={vm['acc']:.6f} "
                f"val_auroc_logit={vm['macro_auroc']:.6f} val_auroc_prob={prob_auroc(vz, val_labels):.6f} "
                f"ema_val_auroc_prob={prob_auroc(val_hist[pe], val_labels):.6f} bn_drift={drift_e:.6f}"
            )

        fp = [fast_macro_f1(z.argmax(1), val_labels) for z in pval_hist]
        p0, pstop = select_patience(fp, EMULATED_PATIENCE)
        p1 = max(range(len(fp)), key=fp.__getitem__)
        p2 = len(fp) - 1
        p4 = sum(fp[e] for e in tail) / len(tail)
        z5p = tail_ensemble([pval_hist[e] for e in tail])

        ptest_cache = {}

        def ptest_logits_at(e):
            if e not in ptest_cache:
                model.load_state_dict(pstate_hist[e])
                ptest_cache[e] = run_one_epoch(model, test_loader, loss_fn, device)[2]
            return ptest_cache[e]

        for e in sorted({p0, p1} | set(tail)):
            ptest_logits_at(e)

        def pm(z, y):
            return multiclass_metrics(z, y)["macro_f1"]

        ptransfer = {k: [] for k in ("P0", "P1", "P2", "P4", "P5")}
        prho_late = None
        for h in (0, 1):
            A = halves == h
            B = ~A
            fpA = [fast_macro_f1(z[A].argmax(1), val_labels[A]) for z in pval_hist]
            fpB = [fast_macro_f1(z[B].argmax(1), val_labels[B]) for z in pval_hist]
            ptransfer["P0"].append(fpB[select_patience(fpA, EMULATED_PATIENCE)[0]])
            ptransfer["P1"].append(fpB[max(range(len(fpA)), key=fpA.__getitem__)])
            ptransfer["P2"].append(fpB[-1])
            ptransfer["P4"].append(sum(fpB[e] for e in tail) / len(tail))
            ptransfer["P5"].append(fast_macro_f1(tail_ensemble([pval_hist[e][B] for e in tail]).argmax(1), val_labels[B]))
            if h == 0:
                prho_late = spearman(fpA[8:], fpB[8:])

        torch.save(
            {"pbn_val_logits": torch.stack(pval_hist), "pbn_epoch_val_f1": fp, "bn_drift": drifts},
            output_path.with_name(output_path.stem + "-pbnlogits.pt"),
        )

        def pline(e):
            return (f"val_f1={fp[e]:.6f} val_auroc_prob={prob_auroc(pval_hist[e], val_labels):.6f} "
                    f"test_f1={pm(ptest_logits_at(e), test_labels):.6f} test_auroc_prob={prob_auroc(ptest_logits_at(e), test_labels):.6f}")

        z5p_test = tail_ensemble([ptest_logits_at(e) for e in tail])
        ptest_f1_tail = [pm(ptest_logits_at(e), test_labels) for e in tail]
        ema_tail_test = tail_ensemble([test_logits_at(e) for e in tail])

        print("=== precise BN (post hoc, same trajectory, 200 fixed train batches; test is reporting only) ===")
        for line in pbn_lines:
            print(line)
        print(f"pbn_data_check: {sums[0]:.6f} {sums[1]:.6f}")
        print(f"pbn_P0_patience10: epoch={p0} stop={pstop} {pline(p0)}")
        print(f"pbn_P1_max: epoch={p1} {pline(p1)}")
        print(
            f"pbn_P2_last: epoch={p2} val_f1={fp[p2]:.6f} val_loss={pval_loss[p2]:.6f} "
            f"val_auroc_prob={prob_auroc(pval_hist[p2], val_labels):.6f} test_f1={pm(ptest_logits_at(p2), test_labels):.6f} "
            f"test_auroc_prob={prob_auroc(ptest_logits_at(p2), test_labels):.6f}"
        )
        print(f"pbn_P4_tailmean5: val_f1={p4:.6f} test_f1_mean={sum(ptest_f1_tail) / len(ptest_f1_tail):.6f}")
        print(
            f"pbn_P5_tailens5: val_f1={pm(z5p, val_labels):.6f} val_auroc_prob={prob_auroc(z5p, val_labels):.6f} "
            f"test_f1={pm(z5p_test, test_labels):.6f} test_auroc_prob={prob_auroc(z5p_test, test_labels):.6f}"
        )
        print("pbn_split_transfer: " + " ".join(f"{k}={sum(v) / len(v):.6f}" for k, v in ptransfer.items()))
        print(f"pbn_split_spearman_late: {prho_late:.6f}")
        print(
            f"ema_auroc_prob: R0={prob_auroc(val_hist[e0], val_labels):.6f} R2={prob_auroc(val_hist[e2], val_labels):.6f} "
            f"R5={prob_auroc(tail_ensemble([val_hist[e] for e in tail]), val_labels):.6f} "
            f"test_R0={prob_auroc(test_logits_at(e0), test_labels):.6f} test_R2={prob_auroc(test_logits_at(e2), test_labels):.6f} "
            f"test_R5={prob_auroc(ema_tail_test, test_labels):.6f}"
        )
        print(f"pbn_seconds: {time.time() - pbn_start:.1f}")

        # restore: model weights/BN stats as at block entry, global RNG streams as at block entry
        model.load_state_dict(model_saved)
        torch.set_rng_state(rng_saved[0])
        if rng_saved[1] is not None:
            torch.cuda.set_rng_state_all(rng_saved[1])
        np.random.set_state(rng_saved[2])
        random.setstate(rng_saved[3])

    # Machine-readable summary read by the autoresearch agents. Selection metric: val_macro_f1.
    print("---")
    print(f"val_macro_f1:    {best_val_metrics['macro_f1']:.6f}")
    print(f"val_bal_acc:     {best_val_metrics['bal_acc']:.6f}")
    print(f"val_macro_auroc: {best_val_metrics['macro_auroc']:.6f}")
    print(f"test_macro_f1:   {test_metrics['macro_f1']:.6f}")
    print(f"test_bal_acc:    {test_metrics['bal_acc']:.6f}")
    print(f"test_macro_auroc:{test_metrics['macro_auroc']:.6f}")
    print(f"best_epoch:      {best_epoch}")
    print(f"epochs_run:      {epoch + 1}")
    print(f"num_params:      {sum(p.numel() for p in model.parameters())}")
    print(f"train_seconds:   {time.time() - start_time:.1f}")

if __name__ == "__main__":
    main()
