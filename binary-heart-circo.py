import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.datasets import CirCoBioSAR
from utils.models import SimpleCNN
from utils.utils import set_seed


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0

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

        total_loss += loss.item() * batch_size
        correct += ((logits >= 0) == (labels >= 0.5)).sum().item()
        count += batch_size

    return total_loss / count, correct / count


def main():
    parser = argparse.ArgumentParser(description="Train a binary CirCo heart classifier")
    parser.add_argument("--metadata", default="./datasets/BioSARdBScene-CirCo/metadata.csv")
    parser.add_argument("--data", default="./datasets/BioSARdBScene-CirCo")
    parser.add_argument("--output", default="models/ckpt/biosarcirco.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()


    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_data = CirCoBioSAR(args.metadata, args.data, split="train", train=True)
    val_data = CirCoBioSAR(args.metadata, args.data, split="val", train=False)
    test_data = CirCoBioSAR(args.metadata, args.data, split="test", train=False)


    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory)
    # One recording per batch 
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=pin_memory)

    sample_features, _, _ = next(iter(train_loader))
    print(f"Model input shape: {tuple(sample_features.shape)}")

    model = SimpleCNN().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(
        f"Device: {device};" 
        f"train: {len(train_data)}; "
        f"val: {len(val_data)}; test: {len(test_data)}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_fn, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, loss_fn, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                output_path,
            )

    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = run_epoch(model, test_loader, loss_fn, device)
    print(f"Test loss={test_loss:.4f} acc={test_accuracy:.3f}")
    print(f"Best checkpoint saved to {output_path}")

if __name__ == "__main__":
    main()
