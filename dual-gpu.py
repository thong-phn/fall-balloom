"""
Binary heart classifier, multi-GPU (DDP) version of binary-heart-{biosar,circo}.py.

Launch one process per GPU, e.g. on Kaggle 2xT4:
    torchrun --nproc_per_node=2 dual-gpu.py --dataset circo
Plain `python dual-gpu.py` still works (single process, 1 GPU or CPU).

--batch-size is the GLOBAL batch size (split across GPUs), so the optimisation
settings stay comparable with the single-GPU scripts.
"""
import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from utils.datasets import CirCo, CirCoBioSAR
from utils.models import SimpleCNN
from utils.utils import set_seed

DATASETS = {
    "biosar": {
        "cls": CirCoBioSAR,
        "data": "/kaggle/input/datasets/thongp/biosardbscene-circo/BioSARdBScene-CirCo",
        "output": "models/ckpt/biosarcirco.pt",
    },
    "circo": {
        "cls": CirCo,
        "data": "./datasets/CirCo/training_data/training_data",
        "output": "models/ckpt/circo.pt",
    },
}


def binary_metrics(logits, labels):
    """
    F1 (abnormal = positive class) and AUROC from per-recording logits.
    """
    pred = logits >= 0
    true = labels >= 0.5
    tp = (pred & true).sum().item()
    fp = (pred & ~true).sum().item()
    fn = (~pred & true).sum().item()
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")

    # AUROC = P(score of a random abnormal > score of a random normal), ties count half
    pos, neg = logits[true], logits[~true]
    if len(pos) == 0 or len(neg) == 0:
        auroc = float("nan")
    else:
        diff = pos[:, None] - neg[None, :]
        auroc = ((diff > 0).sum().item() + 0.5 * (diff == 0).sum().item()) / diff.numel()
    return f1, auroc


def run_one_epoch(model, loader, loss_fn, device, optimizer=None):
    """Returns (loss, accuracy), summed over ALL ranks."""
    training = optimizer is not None
    model.train(training)
    # Per-rank sums: [loss * n, correct, n]
    stats = torch.zeros(3, device=device)

    for features, labels, _ in loader:
        labels = labels.to(device)

        if training:
            features = features.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = labels.numel()
        else:
            # Evaluation returns one or more clips per recording. Average their
            # logits so each recording contributes one prediction.
            features = features.squeeze(0).to(device)
            with torch.no_grad():
                logits = model(features).mean().reshape(1)
                loss = loss_fn(logits, labels)
            batch_size = 1

        stats[0] += loss.detach() * batch_size
        stats[1] += ((logits >= 0) == (labels >= 0.5)).sum()
        stats[2] += batch_size

    if dist.is_initialized():
        dist.all_reduce(stats)
    return (stats[0] / stats[2]).item(), (stats[1] / stats[2]).item()


def evaluate_test(model, loader, loss_fn, device):
    """Loss, accuracy, F1, AUROC over the whole test set (single process, no collectives)."""
    model.eval()
    logits, labels = [], []
    for features, label, _ in loader:
        with torch.no_grad():
            # Average the clip logits so each recording contributes one prediction
            logits.append(model(features.squeeze(0).to(device)).mean().reshape(1))
        labels.append(label.to(device))
    logits, labels = torch.cat(logits), torch.cat(labels)

    loss = loss_fn(logits, labels).item()
    accuracy = ((logits >= 0) == (labels >= 0.5)).float().mean().item()
    f1, auroc = binary_metrics(logits.cpu(), labels.cpu())
    return loss, accuracy, f1, auroc


def main():
    parser = argparse.ArgumentParser(description="Train a binary CirCo heart classifier (DDP)")
    parser.add_argument("--dataset", choices=DATASETS, default="biosar")
    parser.add_argument("--metadata", default="./datasets/BioSARdBScene-CirCo/metadata.csv")
    parser.add_argument("--data", help="audio root (defaults per --dataset)")
    parser.add_argument("--output", help="checkpoint path (defaults per --dataset)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32, help="global batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers per GPU process (default: CPUs / number of GPUs)",
    )
    args = parser.parse_args()

    # torchrun sets these; they are absent for a plain `python` launch
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo")
    is_main = rank == 0

    set_seed(args.seed)
    pin_memory = device.type == "cuda"

    config = DATASETS[args.dataset]
    data_path = args.data or config["data"]
    output_path = Path(args.output or config["output"])
    dataset_cls = config["cls"]

    train_data = dataset_cls(args.metadata, data_path, split="train", train=True)
    val_data = dataset_cls(args.metadata, data_path, split="val", train=False)
    test_data = dataset_cls(args.metadata, data_path, split="test", train=False)

    if args.num_workers is None:
        args.num_workers = (os.cpu_count() or 1) // world_size
    num_workers = max(0, args.num_workers)
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    per_gpu_batch = max(1, args.batch_size // world_size)
    train_sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    )
    train_loader = DataLoader(
        train_data, batch_size=per_gpu_batch, sampler=train_sampler, **loader_options
    )
    # One recording per batch. Val is sharded by stride (no padding), so every
    # recording is evaluated exactly once across ranks.
    val_shard = Subset(val_data, range(rank, len(val_data), world_size))
    val_loader = DataLoader(val_shard, batch_size=1, shuffle=False, **loader_options)

    sample_features, _, _ = next(iter(train_loader))
    if is_main:
        print(f"-"*50)
        print(f"TRAINNING START ({args.dataset})")
        print(f"Model input shape (per GPU): {tuple(sample_features.shape)}")

    model = SimpleCNN().to(device)
    ddp_model = DDP(model, device_ids=[local_rank] if use_cuda else None) if world_size > 1 else model
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.learning_rate)
    if is_main:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    early_stop_tolerant = args.early_stop
    early_stop_count = 0

    if is_main:
        print(
            f"Device: {device}; GPUs: {world_size}; "
            f"workers/GPU: {num_workers}; batch/GPU: {per_gpu_batch}; "
            f"train: {len(train_data)}; "
            f"val: {len(val_data)}; test: {len(test_data)}"
        )

    train_start = time.perf_counter()
    # training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_loss, train_accuracy = run_one_epoch(ddp_model, train_loader, loss_fn, device, optimizer)
        # Evaluate the unwrapped model: no DDP collectives with uneven shard sizes
        val_loss, val_accuracy = run_one_epoch(model, val_loader, loss_fn, device)
        if is_main:
            print(
                f"Epoch {epoch}/{args.epochs}: "
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}"
            )

        # val_loss is identical on every rank (all-reduced), so all ranks make
        # the same early-stop decision and leave the loop together.
        if val_loss < best_val_loss:
            early_stop_count = 0
            best_val_loss = val_loss
            if is_main:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                    },
                    output_path,
                )
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop_tolerant:
                if is_main:
                    print(f"\tEarly stop at epoch {epoch} ")
                break

    train_seconds = time.perf_counter() - train_start
    if is_main:
        print(f"Training time: {time.strftime('%H:%M:%S', time.gmtime(train_seconds))} ({train_seconds:.1f}s)")

    if world_size > 1:
        dist.barrier()  # checkpoint written by rank 0 before it is read
    if is_main:
        # Test set is small: rank 0 evaluates all of it so F1/AUROC need no gather
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, **loader_options)
        checkpoint = torch.load(output_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_accuracy, test_f1, test_auroc = evaluate_test(
            model, test_loader, loss_fn, device
        )
        print(
            f"Test loss={test_loss:.4f} acc={test_accuracy:.3f} "
            f"f1={test_f1:.3f} auroc={test_auroc:.3f}"
        )
        print(f"Best checkpoint saved to {output_path}")
    if world_size > 1:
        dist.barrier()  # other ranks wait for rank 0 to finish testing

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
