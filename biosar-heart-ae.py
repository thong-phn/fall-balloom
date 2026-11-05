"""Anomaly detection of abnormal heart sounds with a CNN autoencoder.

Train: normal recordings of the train split only. The autoencoder learns normal sounds.
Val:   normal + abnormal recordings. Selects the best epoch (AUROC) and the threshold.
Test:  normal + abnormal recordings. Anomaly score = reconstruction error.
"""
import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from utils.datasets import DATASETS
from utils.models import SimpleCNNAutoencoder
from utils.utils import binary_metrics, fmt, set_seed


def train_one_epoch(model, loader, device, optimizer):
    model.train()
    total, count = 0.0, 0
    for features, _, _ in loader:
        features = features.to(device)
        optimizer.zero_grad()
        loss = model.recon_error(features).mean()
        loss.backward()
        optimizer.step()
        total += loss.item() * features.shape[0]
        count += features.shape[0]
    return total / count


@torch.no_grad()
def score_recordings(model, loader, device):
    """One anomaly score per recording: the mean reconstruction error of its clips."""
    model.eval()
    scores, labels = [], []
    for features, label, _ in loader:
        scores.append(model.recon_error(features.squeeze(0).to(device)).mean().reshape(1).cpu())
        labels.append(label.reshape(1))
    return torch.cat(scores), torch.cat(labels) >= 0.5


def youden_threshold(scores, pos):
    """Threshold that maximizes sensitivity + specificity - 1 (score >= threshold is abnormal)."""
    candidates = scores.unique()
    j = [((scores >= t) & pos).sum() / pos.sum() + ((scores < t) & ~pos).sum() / (~pos).sum() for t in candidates]
    return candidates[torch.stack(j).argmax()]


def main():
    parser = argparse.ArgumentParser(description="Heart-sound anomaly detection with a CNN autoencoder")
    parser.add_argument("--dataset", choices=list(DATASETS), default="biosar-bmd")
    parser.add_argument("--metadata", help="override the dataset's default metadata.csv")
    parser.add_argument("--data", help="override the dataset's default audio folder")
    parser.add_argument(
        "--test-dataset",
        choices=list(DATASETS),
        help="dataset for the test split (default: same as --dataset); train and val always use --dataset",
    )
    parser.add_argument("--test-data", help="override the test dataset's default audio folder")
    parser.add_argument("--output", help="checkpoint path (default models/ckpt/heart-ae-<dataset>.pt)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stop", type=int, default=20, help="few normal recordings = few steps per epoch")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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

    dataset_cls, default_metadata, default_data = DATASETS[args.dataset]
    paths = {"metadata_path": args.metadata or default_metadata, "data_path": args.data or default_data}
    train_data = dataset_cls(split="train", train=True, **paths)
    val_data = dataset_cls(split="val", train=False, **paths)
    test_cls, _, test_default_data = DATASETS[args.test_dataset or args.dataset]
    test_paths = {
        "metadata_path": paths["metadata_path"],
        "data_path": args.test_data or (test_default_data if args.test_dataset else paths["data_path"]),
    }
    test_data = test_cls(split="test", train=False, **test_paths)

    # Train on normal recordings only (label 0). Abnormal recordings stay out of training.
    normal_idx = [i for i, y in enumerate(train_data.labels()) if y == 0]
    num_workers = max(0, args.num_workers)
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        Subset(train_data, normal_idx), batch_size=args.batch_size, shuffle=True, **loader_options
    )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, **loader_options)

    print("-" * 50)
    print("TRAINING START")
    print(f"Dataset: train/val={args.dataset} test={args.test_dataset or args.dataset}")
    print(f"Train on {len(normal_idx)} normal of {len(train_data)} recordings")
    print(f"val: {len(val_data)}; test: {len(test_data)}; device: {device}; workers: {num_workers}")

    model = SimpleCNNAutoencoder().to(device)
    print(f"Model: ae-cnn; params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output_path = Path(args.output or f"models/ckpt/heart-ae-{args.dataset}.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_auroc, best_epoch, early_stop_count = float("-inf"), -1, 0
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, device, optimizer)
        val_scores, val_pos = score_recordings(model, val_loader, device)
        val_auroc = binary_metrics(val_scores, val_pos)["auroc"]  # AUROC does not depend on the threshold
        print(
            f"Epoch {epoch}/{args.epochs}: train_recon={train_loss:.4f} | "
            f"val_normal_err={val_scores[~val_pos].mean():.4f} "
            f"val_abnormal_err={val_scores[val_pos].mean():.4f} val_auroc={val_auroc:.3f}"
        )

        if val_auroc > best_val_auroc:
            early_stop_count = 0
            best_val_auroc, best_epoch = val_auroc, epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_auroc": val_auroc}, output_path)
        else:
            early_stop_count += 1
            if early_stop_count >= args.early_stop:
                print(f"\tEarly stop at epoch {epoch} ")
                break

    model.load_state_dict(torch.load(output_path, map_location=device)["model_state_dict"])
    val_scores, val_pos = score_recordings(model, val_loader, device)
    threshold = youden_threshold(val_scores, val_pos)  # chosen on val, applied unchanged to test
    test_scores, test_pos = score_recordings(model, test_loader, device)
    test_metrics = binary_metrics(test_scores - threshold, test_pos)  # score >= 0 means abnormal
    print(f"Threshold (val, Youden): {threshold.item():.4f}")
    print(f"Test: {fmt(test_metrics)}")
    print(f"Best checkpoint saved to {output_path}")

    print("---")
    print(f"val_auroc:       {best_val_auroc:.6f}")
    for k, v in test_metrics.items():
        print(f"test_{k}: {v:.6f}")
    print(f"threshold:       {threshold.item():.6f}")
    print(f"best_epoch:      {best_epoch}")
    print(f"epochs_run:      {epoch + 1}")
    print(f"num_params:      {sum(p.numel() for p in model.parameters())}")
    print(f"train_seconds:   {time.time() - start_time:.1f}")


if __name__ == "__main__":
    main()
