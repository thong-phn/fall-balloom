import random
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
    avg_rank = (upper - (counts - 1) / 2.0)[inverse]  /# 1-based average rank
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