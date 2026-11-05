import random
from collections import Counter

import numpy as np
import torch
from utils.datasets import CLASS_NAMES, BioSARMulti


NUM_CLASSES = len(CLASS_NAMES)  # 4 organs x (normal, abnormal)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def binary_f1_auroc(logits, labels):
    """F1 (logit >= 0 is positive) and rank-based AUROC (ties averaged) for 1-D tensors."""
    logits = logits.detach().cpu().double()
    labels = labels.detach().cpu() >= 0.5
    pred = logits >= 0
    tp = (pred & labels).sum().item()
    fp = (pred & ~labels).sum().item()
    fn = (~pred & labels).sum().item()
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0

    n_pos = labels.sum().item()
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return f1, float("nan")
    _, inverse, counts = torch.unique(logits, return_inverse=True, return_counts=True)
    upper = counts.cumsum(0)
    avg_rank = (upper - (counts - 1) / 2.0)[inverse]  # 1-based average rank
    auroc = (avg_rank[labels].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return f1, auroc

def multiclass_metrics(logits, labels):
    """Accuracy, macro recall, macro specificity, macro-F1 and macro AUROC."""
    preds = logits.argmax(dim=1)
    recalls, specificities, f1s, aurocs = [], [], [], []
    for c in range(NUM_CLASSES):
        is_c, pred_c = labels == c, preds == c
        tp = (is_c & pred_c).sum().item()
        fp = (~is_c & pred_c).sum().item()
        fn = (is_c & ~pred_c).sum().item()
        tn = (~is_c & ~pred_c).sum().item()
        if tp + fn:
            recalls.append(tp / (tp + fn))
            f1s.append(2 * tp / (2 * tp + fp + fn))
        if tn + fp:
            specificities.append(tn / (tn + fp))
        if (~is_c).any():
            aurocs.append(binary_f1_auroc(logits[:, c], is_c.float())[1])
    return {
        "acc": (preds == labels).float().mean().item(),
        "macro_recall": sum(recalls) / len(recalls) if recalls else float("nan"),
        "bal_acc": sum(recalls) / len(recalls) if recalls else float("nan"),
        "macro_specificity": sum(specificities) / len(specificities) if specificities else float("nan"),
        "macro_f1": sum(f1s) / len(f1s),
        "macro_auroc": sum(aurocs) / len(aurocs) if aurocs else float("nan"),
    }


def class_counts(labels):
    counts = Counter(labels)
    return torch.tensor([counts.get(c, 0) for c in range(NUM_CLASSES)], dtype=torch.float)


def class_weights(counts, power):
    """Weight ~ (1/n_c)^power, scaled so the average per-sample weight stays 1.

    power=1 fully balances the classes, power=0.5 is a softer compromise, 0 is off.
    """
    w = counts.clamp(min=1) ** -power
    return w * counts.sum() / (w * counts).sum()


def per_class_report(logits, labels):
    preds = logits.argmax(dim=1)
    print("\tclass            n   recall  specificity  f1")
    for c, name in enumerate(CLASS_NAMES):
        is_c = labels == c
        pred_c = preds == c
        tp = (is_c & pred_c).sum().item()
        fp = (~is_c & pred_c).sum().item()
        fn = (is_c & ~pred_c).sum().item()
        tn = (~is_c & ~pred_c).sum().item()
        recall = tp / (tp + fn) if tp + fn else float("nan")
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
        print(
            f"\t{name:<14}{is_c.sum().item():>5}   {recall:.3f}"
            f"      {specificity:.3f}    {f1:.3f}"
        )


def binary_detection_metrics(logits, labels):
    """Normal vs abnormal from 8-class logits (class = 2 * organ + is_abnormal)."""
    score = torch.logsumexp(logits[:, 1::2], dim=1) - torch.logsumexp(logits[:, 0::2], dim=1)
    return binary_metrics(score, labels % 2 == 1)


def binary_metrics(score, pos):
    """Sensitivity, specificity, F1, AUROC, AUPRC and MCC. Abnormal = positive.

    score is a 1-D abnormal logit (score >= 0 means P(abnormal) >= 0.5).
    pos is a 1-D bool tensor, True for abnormal.
    """
    pred = score >= 0
    tp = (pred & pos).sum().item()
    fp = (pred & ~pos).sum().item()
    fn = (~pred & pos).sum().item()
    tn = (~pred & ~pos).sum().item()
    nan = float("nan")
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    # AUPRC = average precision: mean precision at each true abnormal, by descending score.
    order = score.argsort(descending=True)
    hits = pos[order].double()
    precision_at_k = hits.cumsum(0) / torch.arange(1, len(hits) + 1, dtype=torch.double)
    return {
        "sensitivity": tp / (tp + fn) if tp + fn else nan,
        "specificity": tn / (tn + fp) if tn + fp else nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else nan,
        "auroc": binary_f1_auroc(score, pos.float())[1],
        "auprc": (precision_at_k * hits).sum().item() / hits.sum().item() if hits.sum() else nan,
        "mcc": (tp * tn - fp * fn) / mcc_den if mcc_den else nan,
    }


def fmt(metrics):
    return " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
