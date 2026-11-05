import argparse
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.datasets import DATASETS
from utils.models import (
    AuscultationAttentionCNN,
    EdgeMamba,
    SeparableCNN,
    SimpleCNN,
    TinyMamba,
)
from utils.utils import binary_metrics, fmt, set_seed


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
            logits = model(features)  # [B]
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = labels.numel()
        else:
            # Evaluation returns one or more clips per recording. Average their
            # logits so each recording contributes one prediction.
            features = features.squeeze(0).to(device)
            with torch.no_grad():
                logits = model(features).mean().reshape(1)  # [1]
                loss = loss_fn(logits, labels)
            batch_size = 1

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())
        total_loss += loss.item() * batch_size
        count += batch_size

    logits, labels = torch.cat(all_logits), torch.cat(all_labels)
    return total_loss / count, binary_metrics(logits, labels >= 0.5), logits, labels


def main():
    parser = argparse.ArgumentParser(description="Train a binary (normal/abnormal) heart-sound BioSAR classifier")
    parser.add_argument("--dataset", choices=list(DATASETS), default="biosar")
    parser.add_argument("--metadata", help="override the dataset's default metadata.csv")
    parser.add_argument("--data", help="override the dataset's default audio folder")
    parser.add_argument(
        "--test-dataset",
        choices=list(DATASETS),
        help="dataset for the test split (default: same as --dataset); train and val always use --dataset",
    )
    parser.add_argument("--test-data", help="override the test dataset's default audio folder")
    parser.add_argument("--output", help="checkpoint path (default models/ckpt/heart-<dataset>.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--model",
        choices=["cnn", "simple-cnn", "attn-cnn", "mamba", "edge-mamba"],
        default="simple-cnn",
        help="attn-cnn adds low-cost multi-scale temporal and frequency attention",
    )
    parser.add_argument(
        "--balance-power",
        type=float,
        default=0.5,
        help="abnormal pos_weight = (n_normal / n_abnormal)^power; 1 = fully balanced, 0 = off",
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

    dataset_cls, default_metadata, default_data = DATASETS[args.dataset]
    paths = {"metadata_path": args.metadata or default_metadata, "data_path": args.data or default_data}
    train_data = dataset_cls(split="train", train=True, **paths)
    val_data = dataset_cls(split="val", train=False, **paths)
    test_cls, _, test_default_data = DATASETS[args.test_dataset or args.dataset]
    # Same metadata.csv, so train and test use the same rows and split.
    test_paths = {
        "metadata_path": paths["metadata_path"],
        "data_path": args.test_data or (test_default_data if args.test_dataset else paths["data_path"]),
    }
    test_data = test_cls(split="test", train=False, **test_paths)

    num_workers = max(0, args.num_workers)
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    # Class imbalance: positive-class weight in the BCE loss
    n_abnormal = sum(train_data.labels())
    n_normal = len(train_data) - n_abnormal
    pos_weight = torch.tensor((n_normal / max(n_abnormal, 1)) ** args.balance_power)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, **loader_options
    )
    # One recording per batch
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, **loader_options)

    sample_features, _, _ = next(iter(train_loader))
    print("-" * 50)
    print("TRAINING START")
    print(f"Dataset: train/val={args.dataset} test={args.test_dataset or args.dataset}")
    print(f"Model input shape: {tuple(sample_features.shape)}")
    print(
        f"Train: normal={n_normal} abnormal={n_abnormal}; "
        f"pos_weight={pos_weight.item():.2f} (power={args.balance_power})"
    )

    if args.model == "mamba":
        model = TinyMamba(n_mels=sample_features.shape[-2]).to(device)
    elif args.model == "edge-mamba":
        model = EdgeMamba(n_mels=sample_features.shape[-2]).to(device)
    elif args.model == "simple-cnn":
        model = SimpleCNN().to(device)
    elif args.model == "attn-cnn":
        model = AuscultationAttentionCNN().to(device)
    else:
        model = SeparableCNN().to(device)
    print(f"Model: {args.model}; params: {sum(p.numel() for p in model.parameters()):,}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output_path = Path(args.output or f"models/ckpt/heart-{args.dataset}.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_auroc = float("-inf")
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
            f"train_loss={train_loss:.4f} train_auroc={train_metrics['auroc']:.3f} | "
            f"val_loss={val_loss:.4f} val_auroc={val_metrics['auroc']:.3f}"
        )

        # Checkpoint on val AUROC: it does not depend on the 0.5 threshold or the class ratio.
        if val_metrics["auroc"] > best_val_auroc:
            early_stop_count = 0
            best_val_auroc = val_metrics["auroc"]
            best_epoch, best_val_metrics = epoch, val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
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
    test_loss, test_metrics, _, _ = run_one_epoch(model, test_loader, loss_fn, device)
    print(f"Test: loss={test_loss:.4f} {fmt(test_metrics)}")
    print(f"Best checkpoint saved to {output_path}")

    # Machine-readable summary. Selection metric: val_auroc.
    print("---")
    print(f"val_auroc:       {best_val_metrics['auroc']:.6f}")
    for k, v in test_metrics.items():
        print(f"test_{k}: {v:.6f}")
    print(f"best_epoch:      {best_epoch}")
    print(f"epochs_run:      {epoch + 1}")
    print(f"num_params:      {sum(p.numel() for p in model.parameters())}")
    print(f"train_seconds:   {time.time() - start_time:.1f}")


if __name__ == "__main__":
    main()
