import argparse
import contextlib
import copy
import csv
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by use_deterministic_algorithms for cuBLAS
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from utils.datasets import CLASS_NAMES, BioSARMulti
from utils.models import EdgeMamba, SeparableCNN, SimpleCNN, TinyMamba, separable_conv
from utils.utils import binary_f1_auroc, set_seed

NUM_CLASSES = len(CLASS_NAMES)  # 4 organs x (normal, abnormal)
ORGAN_ABN = (("Bowel", 0), ("Heart", 1), ("Lung", 3))  # (name, index in utils.datasets.ORGANS); class = 2 * organ + abnormal


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


def count_macs(model, input_shape=(1, 1, 64, 157)):
    """Multiply-accumulates of one forward pass on a CPU copy in eval mode (Conv2d: out.numel() x in_ch/groups x kh x kw; Linear: in x out). No RNG use."""
    net = copy.deepcopy(model).cpu().eval()
    total = [0]

    def conv_hook(m, inp, out):
        total[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def linear_hook(m, inp, out):
        total[0] += m.in_features * m.out_features * (out.numel() // m.out_features)

    handles = [m.register_forward_hook(conv_hook) for m in net.modules() if isinstance(m, nn.Conv2d)]
    handles += [m.register_forward_hook(linear_hook) for m in net.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        net(torch.zeros(*input_shape))
    for h in handles:
        h.remove()
    return total[0]


def prob_auroc(logits, labels):
    """Macro one-vs-rest AUROC on softmax probabilities (present classes only). Reporting only; multiclass_metrics is unchanged."""
    p = logits.softmax(dim=1)
    vals = [binary_f1_auroc(p[:, c], (labels == c).float())[1]
            for c in range(NUM_CLASSES) if (labels == c).any() and (labels != c).any()]
    return sum(vals) / len(vals)


def organ_abn_auroc(logits, labels):
    """Within-organ abnormality AUROC for Bowel / Heart / Lung: among rows with true organ o, AUROC of
    p(o, abn) / (p(o, normal) + p(o, abn)) against the abnormal label. Returns ([bowel, heart, lung], mean)."""
    p = logits.softmax(dim=1)
    vals = []
    for _, o in ORGAN_ABN:
        rows = (labels // 2) == o
        score = p[rows, 2 * o + 1] / (p[rows, 2 * o] + p[rows, 2 * o + 1])
        vals.append(binary_f1_auroc(score, (labels[rows] % 2).float())[1])
    return vals, sum(vals) / len(vals)


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


@contextlib.contextmanager
def preserve_rng():
    """Restore the python / numpy / torch (CPU and CUDA) global RNG states on exit. Used around teacher construction and the
    teacher val pass: building modules and creating a DataLoader iterator both draw from the global RNG, which would shift the
    student's init-independent data order and crops relative to the no-KD control."""
    py, npy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(npy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def load_teachers(paths, device):
    """Build frozen eval-mode WideSeparableCNN teachers from -last.pt files saved by exp-009 (keys: model_state_dict, epoch, width, class_names).
    Caller wraps this in preserve_rng()."""
    teachers = []
    for path in paths:
        ckpt = torch.load(path, map_location="cpu")
        t = WideSeparableCNN(num_classes=NUM_CLASSES, width=ckpt["width"])
        make_pooling_deterministic(t)
        t.load_state_dict(ckpt["model_state_dict"])
        t.to(device).eval()
        t.requires_grad_(False)
        teachers.append(t)
    return teachers


def teacher_probs(teachers, features, temp):
    """Ensemble teacher probabilities [B, C]: mean over members of softmax(logits / temp). eval mode, no grad, no RNG."""
    with torch.no_grad():
        return torch.stack([F.softmax(t(features) / temp, dim=1) for t in teachers]).mean(0)


def kd_loss(logits, p_t, labels, temp, mode="logit", class_weighted=0, weights=None):
    """T^2 * KL(p_t || softmax(logits / T)). mode 'logit': flat 8-way KL, batch mean (class_weighted=1: per-sample KL weighted by
    the true-class weight, normalised sum(w kl) / sum(w)). mode 'organ': KL(organ marginal) + KL(abnormality | TRUE organ) (exp-013)."""
    if mode == "organ":
        B = logits.shape[0]
        P = p_t.clamp_min(1e-8).view(B, len(CLASS_NAMES) // 2, 2)
        q_t = P.sum(-1)
        c_t = P / q_t.unsqueeze(-1)
        ls = F.log_softmax(logits / temp, dim=1).view(B, -1, 2)
        log_q_s = ls.logsumexp(-1)
        log_c_s = ls - log_q_s.unsqueeze(-1)
        o = labels // 2
        idx = torch.arange(B, device=logits.device)
        ct, lcs = c_t[idx, o], log_c_s[idx, o]
        l_org = (q_t * (q_t.log() - log_q_s)).sum(1).mean()
        l_abn = (ct * (ct.log() - lcs)).sum(1).mean()
        return temp * temp * (l_org + l_abn)
    log_ps = F.log_softmax(logits / temp, dim=1)
    if class_weighted:
        kl = (torch.xlogy(p_t, p_t) - p_t * log_ps).sum(1)
        w = weights[labels]
        return temp * temp * (w * kl).sum() / w.sum()
    return F.kl_div(log_ps, p_t, reduction="batchmean") * temp * temp


def augment_batch(x, aug):
    """Training-time augmentation on log-mel features x [B, ..., n_mels, n_frames]; all randomness from aug["gen"] (a dedicated
    torch.Generator), never the global RNG. Order: (a) per-sample gain (2U - 1) * gain_db * ln(10) / 10 added to the log-mel,
    (b) nf frequency masks then nt time masks, masked bins set to the per-sample mean of the (post-gain) log-mel.
    Returns a new tensor; x is not modified."""
    gen, B = aug["gen"], x.shape[0]
    dev = x.device
    if aug["gain_db"] > 0:
        u = torch.rand(B, generator=gen, device=dev)
        x = x + ((2 * u - 1) * aug["gain_db"] * (np.log(10.0) / 10.0)).view(B, *([1] * (x.ndim - 1))).to(x.dtype)
    if aug["specaug"]:
        mean = x.mean(dim=tuple(range(1, x.ndim)), keepdim=True)
        for dim, width, n_masks in ((-2, aug["F"], aug["nf"]), (-1, aug["T"], aug["nt"])):
            n = x.shape[dim]
            ar = torch.arange(n, device=dev).unsqueeze(0)  # [1, n]
            masked = torch.zeros(B, n, dtype=torch.bool, device=dev)
            for _ in range(n_masks):
                w = torch.floor(torch.rand(B, generator=gen, device=dev) * (width + 1)).long().clamp(max=width)
                u0 = torch.rand(B, generator=gen, device=dev)
                f0 = torch.floor(u0 * (n - w + 1).float()).long().clamp(max=n - w)
                masked |= (ar >= f0.unsqueeze(1)) & (ar < (f0 + w).unsqueeze(1))
            shape = [B] + [1] * (x.ndim - 1)
            shape[dim] = n
            x = torch.where(masked.view(shape), mean.to(x.dtype), x)
    return x


def run_one_epoch(model, loader, loss_fn, device, optimizer=None, kd=None, aug=None):
    """kd (training only): dict(teachers, alpha, temp, mode, class_weighted, weights) or None. With alpha == 0 the teacher forward
    still runs (no RNG, no student state) but the loss is the plain weighted CE, bit-identical to exp-009."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    all_logits, all_labels = [], []

    for features, labels, _ in loader:
        labels = labels.to(device)

        if training:
            features = features.to(device)
            if aug is not None:
                features = augment_batch(features, aug)
            optimizer.zero_grad()
            p_t = teacher_probs(kd["teachers"], features, kd["temp"]) if kd is not None else None
            logits = model(features)  # [B, C]
            loss = loss_fn(logits, labels)
            if kd is not None and kd["alpha"] > 0:
                kd_term = kd_loss(logits, p_t, labels, kd["temp"], kd["mode"], kd["class_weighted"], kd["weights"])
                loss = (1 - kd["alpha"]) * loss + kd["alpha"] * kd_term
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


def per_class_f1(logits, labels):
    """Per-class F1 (nan for classes absent from labels), same formula as multiclass_metrics. Logging only."""
    preds = logits.argmax(dim=1)
    out = []
    for c in range(NUM_CLASSES):
        is_c, pred_c = labels == c, preds == c
        if not is_c.any():
            out.append(float("nan"))
            continue
        tp = (is_c & pred_c).sum().item()
        fp = (~is_c & pred_c).sum().item()
        fn = (is_c & ~pred_c).sum().item()
        out.append(2 * tp / (2 * tp + fp + fn))
    return out


def organ_vs_rest_auroc(logits, labels):
    """Organ-vs-rest AUROC on summed organ probabilities p(o, normal) + p(o, abnormal), for Bowel / Heart / Joint / Lung. Logging only."""
    p = logits.softmax(dim=1)
    out = []
    for o in range(NUM_CLASSES // 2):
        is_o = (labels // 2) == o
        out.append(binary_f1_auroc(p[:, 2 * o] + p[:, 2 * o + 1], is_o.float())[1] if is_o.any() and (~is_o).any() else float("nan"))
    return out


def split_holdout(items, metadata_path, frac, seed):
    """Grouped (by source_id), class-stratified hold-out of training items. For each class c (label of a group's first item) the
    sorted group list is shuffled with torch.Generator().manual_seed(seed) (independent of --seed) and the first
    round(frac * n_groups_c) groups go to the hold-out. Returns (train_items, hold_items, n_hold_groups, src)."""
    with open(metadata_path, newline="") as f:
        src = {r["mixture_id"]: r["source_id"] for r in csv.DictReader(f)}
    first_label = {}
    for it in items:
        first_label.setdefault(src[it["mixture_id"]], it["label"])
    gen = torch.Generator().manual_seed(seed)
    hold_groups = set()
    for c in range(NUM_CLASSES):
        gs = sorted(g for g, y in first_label.items() if y == c)
        k = round(frac * len(gs))
        hold_groups.update(gs[j] for j in torch.randperm(len(gs), generator=gen).tolist()[:k])
    train_items = [it for it in items if src[it["mixture_id"]] not in hold_groups]
    hold_items = [it for it in items if src[it["mixture_id"]] in hold_groups]
    assert not ({src[it["mixture_id"]] for it in train_items} & {src[it["mixture_id"]] for it in hold_items})
    return train_items, hold_items, len(hold_groups), src


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
    parser.add_argument("--width", type=int, default=1, help="channel width multiplier of the separable CNN (--model cnn); 1 = SeparableCNN (13,768 params), 9 = 897,992 params")
    parser.add_argument(
        "--balance-power",
        type=float,
        default=0.5,
        help="class weight ~ (1/count)^power; 1 = fully balanced, 0 = off",
    )
    parser.add_argument("--teacher-ckpts", default="", help="comma-separated exp-009 -last.pt paths (teacher ensemble); empty = no teacher")
    parser.add_argument("--kd-alpha", type=float, default=0.0, help="loss = (1 - alpha) * weighted CE + alpha * T^2 * KL; 0 = plain CE (identical to exp-009)")
    parser.add_argument("--kd-temp", type=float, default=1.0, help="distillation temperature T")
    parser.add_argument("--kd-mode", choices=["logit", "organ"], default="logit", help="logit = flat 8-way KL; organ = organ marginal KL + abnormality KL given the true organ (exp-013)")
    parser.add_argument("--kd-class-weighted", type=int, choices=[0, 1], default=0, help="1 = weight each sample's KL by its true-class weight (exp-012)")
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
    parser.add_argument("--specaug", type=int, choices=[0, 1], default=0, help="SpecAugment (frequency + time masks, per-sample-mean fill), training only")
    parser.add_argument("--specaug-F", type=int, default=8, help="max frequency-mask width (bins)")
    parser.add_argument("--specaug-T", type=int, default=20, help="max time-mask width (frames)")
    parser.add_argument("--specaug-nf", type=int, default=2, help="number of frequency masks")
    parser.add_argument("--specaug-nt", type=int, default=2, help="number of time masks")
    parser.add_argument("--gain-aug-db", type=float, default=0.0, help="random per-sample gain, uniform in +-this many dB, added to the log-mel; 0 = off")
    parser.add_argument("--aug-seed-offset", type=int, default=1000, help="augmentation generator seed = --seed + this (dedicated torch.Generator)")
    parser.add_argument("--holdout-frac", type=float, default=0.0, help="fraction of train source groups (stratified by class) held out of training; 0 = off")
    parser.add_argument("--holdout-seed", type=int, default=0, help="seed of the hold-out split (independent of --seed)")
    parser.add_argument("--save-eval-logits", type=int, choices=[0, 1], default=1, help="1 = save <stem>-evallogits.pt (last-epoch val/test/train/hold logits + tail-5 test logits); no effect on training")
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

    hold_items, n_hold_groups = [], 0
    if args.holdout_frac > 0:
        train_items, hold_items, n_hold_groups, _ = split_holdout(train_data.items, args.metadata, args.holdout_frac, args.holdout_seed)
        train_data.items = train_items  # before class_counts / class_weights / DataLoader

    num_workers = max(0, args.num_workers)
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    # Class imbalance: class-weighted cross-entropy
    counts = class_counts(train_data.labels())
    weights = class_weights(counts, args.balance_power)

    if args.holdout_frac > 0:
        hold_counts = class_counts([it["label"] for it in hold_items])
        print(f"holdout: n_train={len(train_data)} n_hold={len(hold_items)} per_class_train={[int(n) for n in counts.tolist()]} "
              f"per_class_hold={[int(n) for n in hold_counts.tolist()]} n_groups_hold={n_hold_groups} frac={args.holdout_frac} seed={args.holdout_seed}")

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
        model = WideSeparableCNN(num_classes=NUM_CLASSES, width=args.width).to(device)
    n_swapped = make_pooling_deterministic(model)
    print(f"Deterministic mode: cudnn.deterministic=True, use_deterministic_algorithms(warn_only=True), CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}, AdaptiveAvgPool2d->mean swapped={n_swapped}")
    print(f"Model: {args.model}; params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"macs_per_clip: {count_macs(model)}")
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        if args.lr_schedule == "cosine" else None
    )
    eval_only = args.eval_checkpoint is not None
    # Dedicated augmentation RNG (seeded from --seed, never touches the global RNG); None = no augmentation.
    aug = None
    if args.specaug or args.gain_aug_db > 0:
        aug = {"gen": torch.Generator(device=device).manual_seed(args.seed + args.aug_seed_offset), "specaug": args.specaug,
               "F": args.specaug_F, "T": args.specaug_T, "nf": args.specaug_nf, "nt": args.specaug_nt, "gain_db": args.gain_aug_db}
        print(f"Augmentation: specaug={args.specaug} F={args.specaug_F} T={args.specaug_T} nf={args.specaug_nf} nt={args.specaug_nt} "
              f"gain_db={args.gain_aug_db} aug_seed={args.seed + args.aug_seed_offset}")

    # Teacher ensemble (KD). Construction and the val pass run under preserve_rng(): module init and DataLoader iterators draw
    # from the global RNG, which would change the training data order relative to the no-KD control.
    teacher_paths = [p for p in args.teacher_ckpts.split(",") if p.strip()]
    if args.kd_alpha > 0 and not teacher_paths:
        parser.error("--kd-alpha > 0 needs --teacher-ckpts")
    kd = None
    if teacher_paths and not eval_only:
        with preserve_rng():
            teachers = load_teachers(teacher_paths, device)
            # Separate non-persistent val loader: the real val_loader's persistent workers / first iterator must be created at the
            # same point of the RNG stream as in exp-009 (its iterator creation draws a base seed from the global RNG).
            teacher_val_loader = DataLoader(val_data, batch_size=1, shuffle=False, **{**loader_options, "persistent_workers": False})
            member_logits = []
            for path, t in zip(teacher_paths, teachers):
                _, _, z_t, val_labels_all = run_one_epoch(t, teacher_val_loader, loss_fn, device)
                member_logits.append(z_t)
                print(f"teacher_member: {path} width={t.features[0].out_channels // 16} val_auroc_prob={prob_auroc(member_logits[-1], val_labels_all):.6f}")
            print(f"teacher_val_auroc_prob: {prob_auroc(tail_ensemble(member_logits), val_labels_all):.6f} (members={len(teachers)})")
        kd = {"teachers": teachers, "alpha": args.kd_alpha, "temp": args.kd_temp, "mode": args.kd_mode,
              "class_weighted": args.kd_class_weighted, "weights": weights.to(device)}
        print(f"KD: alpha={args.kd_alpha} temp={args.kd_temp} mode={args.kd_mode} class_weighted={args.kd_class_weighted} teachers={len(teachers)}")
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
    val_hist, state_hist, f_logged, val_loss_hist = [], [], [], []
    if not eval_only:
        # training loop
        for epoch in range(args.epochs):
            train_loss, train_metrics, _, _ = run_one_epoch(model, train_loader, loss_fn, device, optimizer, kd, aug)
            epoch_lr = optimizer.param_groups[0]["lr"]  # LR used this epoch (read before scheduler.step())
            if scheduler is not None:
                scheduler.step()
            val_loss, val_metrics, val_ep_logits, val_ep_labels = run_one_epoch(model, val_loader, loss_fn, device)
            val_hist.append(val_ep_logits)
            state_hist.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            f_logged.append(val_metrics["macro_f1"])
            val_loss_hist.append(val_loss)
            print(
                f"Epoch {epoch}/{args.epochs}: "
                f"lr={epoch_lr:.2e} | "
                f"train_loss={train_loss:.4f} train_acc={train_metrics['acc']:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_metrics['acc']:.3f} "
                f"val_f1={val_metrics['macro_f1']:.4f} val_bal_acc={val_metrics['bal_acc']:.3f}"
            )
            print(f"prob_epoch: e={epoch} val_auroc_prob={prob_auroc(val_ep_logits, val_ep_labels):.6f} "
                  f"val_organ_abn_auroc_mean={organ_abn_auroc(val_ep_logits, val_ep_labels)[1]:.6f}")

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

    if not eval_only and state_hist:
        torch.save(
            {"model_state_dict": state_hist[-1], "epoch": len(state_hist) - 1, "width": args.width, "class_names": CLASS_NAMES},
            output_path.with_name(output_path.stem + "-last.pt"),
        )

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

    if args.calibrate == "cv" and args.epochs >= 10:
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

    if not eval_only and len(val_hist) == args.epochs and args.epochs >= 10:
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
        oa_val, oa_val_mean = organ_abn_auroc(val_hist[e2], val_labels)
        oa_test_mean = organ_abn_auroc(test_logits_at(e2), test_labels)[1]
        print(f"prob_last: val_auroc_prob={prob_auroc(val_hist[e2], val_labels):.6f} "
              f"val_organ_abn_auroc={oa_val[0]:.6f},{oa_val[1]:.6f},{oa_val[2]:.6f} mean={oa_val_mean:.6f} "
              f"val_nll={val_loss_hist[e2]:.6f} test_auroc_prob={prob_auroc(test_logits_at(e2), test_labels):.6f} "
              f"test_organ_abn_auroc_mean={oa_test_mean:.6f}")
        print(f"prob_R0: val_auroc_prob={prob_auroc(val_hist[e0], val_labels):.6f}")

        # Logging only (extra lines after the unchanged ones above): organ-vs-rest AUROC, last-epoch per-class F1, test balanced accuracy.
        oav = organ_vs_rest_auroc(val_hist[e2], val_labels)
        oat = organ_vs_rest_auroc(test_logits_at(e2), test_labels)
        mt2 = multiclass_metrics(test_logits_at(e2), test_labels)
        print("aux_organ_auroc: val bowel={:.6f} heart={:.6f} joint={:.6f} lung={:.6f} | test bowel={:.6f} heart={:.6f} joint={:.6f} lung={:.6f}".format(*oav, *oat))
        print("aux_val_f1_last_per_class: " + " ".join(f"{v:.4f}" for v in per_class_f1(val_hist[e2], val_labels)))
        print(f"aux_test_last: test_f1_last={mt2['macro_f1']:.6f} test_bal_acc_last={mt2['bal_acc']:.6f} "
              "test_f1_last_per_class: " + " ".join(f"{v:.4f}" for v in per_class_f1(test_logits_at(e2), test_labels)))

        if args.save_eval_logits:
            # Extra passes of the last-epoch model over the (85% or 100%) train items and the hold-out; RNG-neutral, no printed-line effect.
            with preserve_rng():
                saved_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(state_hist[-1])
                eval_opts = {**loader_options, "persistent_workers": False}

                def eval_items(items):
                    if not items:
                        return torch.zeros(0, NUM_CLASSES), torch.zeros(0, dtype=torch.long), []
                    ds = BioSARMulti(data_path=args.data, metadata_path=args.metadata, split="train", train=False)
                    ds.items = list(items)
                    z, y = run_one_epoch(model, DataLoader(ds, batch_size=1, shuffle=False, **eval_opts), loss_fn, device)[2:]
                    return z, y, [it["mixture_id"] for it in items]

                train_z, train_y, train_ids = eval_items(train_data.items)
                hold_z, hold_y, hold_ids = eval_items(hold_items)
                model.load_state_dict(saved_state)
            torch.save(
                {"train_logits": train_z, "train_labels": train_y, "train_mixture_ids": train_ids,
                 "val_logits": val_hist[-1], "val_labels": val_labels, "val_mixture_ids": [it["mixture_id"] for it in val_data.items],
                 "test_logits": test_logits_at(e2), "test_labels": test_labels, "test_mixture_ids": [it["mixture_id"] for it in test_data.items],
                 "test_logits_tail": torch.stack([test_logits_at(e) for e in tail]), "tail_epochs": tail,
                 "hold_logits": hold_z, "hold_labels": hold_y, "hold_mixture_ids": hold_ids,
                 "holdout_frac": args.holdout_frac, "holdout_seed": args.holdout_seed, "seed": args.seed,
                 "width": args.width, "ckpt": str(output_path.with_name(output_path.stem + "-last.pt"))},
                output_path.with_name(output_path.stem + "-evallogits.pt"),
            )

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
