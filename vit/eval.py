import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vit.data import load_cifar10, CIFAR10_CLASSES
from vit.model import ViT
from utils.config import load_config


def evaluate():
    cfg = load_config("vit/config.yaml")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = cfg["model_path"]
    model = ViT(
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
        patch_size=cfg["patch_size"], num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")
    print(f"Model parameters: {model.num_params():,}")

    _, test_loader = load_cifar10(
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
    )

    correct = 0
    total = 0
    class_correct = [0] * cfg["num_classes"]
    class_total = [0] * cfg["num_classes"]

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            logits = model(images)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1

    print(f"\nTest Accuracy: {correct / total:.1%} ({correct}/{total})")
    print(f"\nPer-class accuracy:")
    for i, name in enumerate(CIFAR10_CLASSES):
        acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"  {name:<12} {acc:.1%}")


if __name__ == "__main__":
    evaluate()
