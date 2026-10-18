import argparse
import os
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.datasets import CLASS_NAMES, BioSARMulti
from utils.models import EdgeMamba, SeparableCNN, SimpleCNN, TinyMamba
from utils.utils import binary_f1_auroc, set_seed

NUM_CLASSES = len(CLASS_NAMES)  # 4 organs x (normal, abnormal)


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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--model", choices=["cnn", "simple-cnn", "mamba", "edge-mamba"], default="cnn")
    parser.add_argument(
        "--balance-power",
        type=float,
        default=0.5,
        help="class weight ~ (1/count)^power; 1 = fully balanced, 0 = off",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
        help="DataLoader worker processes (defaults to the CPUs allocated by Slurm)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
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
    print(f"Model: {args.model}; params: {sum(p.numel() for p in model.parameters()):,}")
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output_path = Path(args.output)
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

    # training loop
    for epoch in range(args.epochs):
        train_loss, train_metrics, _, _ = run_one_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_metrics, _, _ = run_one_epoch(model, val_loader, loss_fn, device)
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_metrics['acc']:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['acc']:.3f}"
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
            if early_stop_count >= early_stop_tolerant:
                print(f"\tEarly stop at epoch {epoch} ")
                break

    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics, test_logits, test_labels = run_one_epoch(model, test_loader, loss_fn, device)
    print(f"Test: loss={test_loss:.4f} {fmt(test_metrics)}")
    per_class_report(test_logits, test_labels)
    print(f"Best checkpoint saved to {output_path}")

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
