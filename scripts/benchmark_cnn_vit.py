"""
CNN vs ViT Benchmark on CIFAR-10.

Trains both models with the same settings (epochs, batch_size, lr, scheduler),
records loss/accuracy/timing per epoch, and outputs a comparison table + plot.

Usage:
    uv run python scripts/benchmark_cnn_vit.py

Requires: matplotlib (for plot), pytorch (for training)
"""

import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

# Run as a plain script from a checkout that is not pip-installed: the training
# modules are imported as `cv.*` / `ml.*`, so the repository root has to be on
# sys.path before the first project import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv.simplecnn.model import SimpleCNN
from cv.vit.model import ViT


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

NUM_EPOCHS = 30
BATCH_SIZE = 128
LR = 0.001


def _build_transform(augment=False):
    ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else []
    ops.extend([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    return transforms.Compose(ops)


def _transform_batch(batch, fn):
    batch["img"] = [fn(img.convert("RGB")) for img in batch["img"]]
    return batch


def load_data(num_workers=4):
    train_ds = load_dataset("uoft-cs/cifar10", split="train")
    test_ds = load_dataset("uoft-cs/cifar10", split="test")
    train_ds.set_transform(lambda b: _transform_batch(b, _build_transform(augment=True)))
    test_ds.set_transform(lambda b: _transform_batch(b, _build_transform(augment=False)))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, device, name="Model"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {"loss": [], "test_acc": [], "time_per_epoch": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch["img"].to(device), batch["label"].to(device)
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        avg_loss = train_loss / len(train_loader)
        test_acc = correct / total * 100
        epoch_time = time.time() - t0

        history["loss"].append(avg_loss)
        history["test_acc"].append(test_acc)
        history["time_per_epoch"].append(epoch_time)

        print(f"{name:12s} Epoch [{epoch:2d}/{NUM_EPOCHS}]  "
              f"Loss: {avg_loss:.4f}  Test Acc: {test_acc:.2f}%  "
              f"{epoch_time:.1f}s")

    return history


def print_table(cnn_hist, vit_hist, cnn_params, vit_params):
    print("\n" + "=" * 60)
    print("CNN vs ViT Benchmark on CIFAR-10")
    print("=" * 60)

    cnn_acc = cnn_hist["test_acc"][-1]
    vit_acc = vit_hist["test_acc"][-1]
    cnn_time = sum(cnn_hist["time_per_epoch"])
    vit_time = sum(vit_hist["time_per_epoch"])
    cnn_70 = next((i + 1 for i, a in enumerate(cnn_hist["test_acc"]) if a >= 70), NUM_EPOCHS)
    vit_70 = next((i + 1 for i, a in enumerate(vit_hist["test_acc"]) if a >= 70), NUM_EPOCHS)

    print(f"\n{'':<25} {'SimpleCNN':>12} {'ViT':>12}")
    print("-" * 50)
    print(f"{'Parameters':<25} {cnn_params:>10,d} {vit_params:>10,d}")
    print(f"{'Test Accuracy':<25} {cnn_acc:>10.2f}% {vit_acc:>10.2f}%")
    print(f"{'Total Training Time':<25} {cnn_time:>8.1f}s {vit_time:>8.1f}s")
    print(f"{'Epochs to 70% Acc':<25} {cnn_70:>10d} {vit_70:>10d}")
    print("-" * 50)

    winner = "SimpleCNN" if cnn_acc > vit_acc else "ViT" if vit_acc > cnn_acc else "Tie"
    print(f"\nWinner: {winner}")
    return winner


def save_plot(cnn_hist, vit_hist):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(cnn_hist["loss"], label="SimpleCNN", marker="o")
    ax1.plot(vit_hist["loss"], label="ViT", marker="s")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss")
    ax1.legend(); ax1.grid(True)

    ax2.plot(cnn_hist["test_acc"], label="SimpleCNN", marker="o")
    ax2.plot(vit_hist["test_acc"], label="ViT", marker="s")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Test Accuracy (%)"); ax2.set_title("Test Accuracy")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig("benchmark_cnn_vs_vit.png", dpi=150)
    print(f"\nPlot saved to benchmark_cnn_vs_vit.png")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    torch.set_num_threads(4)

    train_loader, test_loader = load_data()
    print(f"Data loaded: {len(train_loader.dataset):,} train, {len(test_loader.dataset):,} test")

    # Train SimpleCNN.
    print("\n── Training SimpleCNN ──")
    cnn_model = SimpleCNN(num_classes=10)
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    cnn_hist = train_model(cnn_model, train_loader, test_loader, device, "SimpleCNN")

    # Train ViT.
    print("\n── Training ViT ──")
    vit_model = ViT(d_model=128, n_heads=4, n_layers=4, d_ff=512,
                    patch_size=4, num_classes=10, dropout=0.1)
    vit_params = sum(p.numel() for p in vit_model.parameters())
    vit_hist = train_model(vit_model, train_loader, test_loader, device, "ViT")

    # Output.
    print_table(cnn_hist, vit_hist, cnn_params, vit_params)
    save_plot(cnn_hist, vit_hist)


if __name__ == "__main__":
    main()
