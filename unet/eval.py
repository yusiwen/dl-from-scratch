import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from unet.data import PetDataset
from unet.model import UNet
from utils.config import load_config
from utils.device import get_device


def compute_iou(preds, targets, num_classes=3, ignore_index=0):
    """Per-class IoU and mean IoU, ignoring ignore_index."""
    ious = []
    for cls in range(1, num_classes + 1):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0, ious


def evaluate():
    cfg = load_config("unet/config.yaml")

    device = get_device()
    print(f"Device: {device}")

    model = UNet(in_channels=3, num_classes=cfg["num_classes"])
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {cfg['model_path']}")
    print(f"Model parameters: {model.num_params():,}")

    test_dataset = PetDataset(split="test", image_size=cfg["image_size"], augment=False)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    correct = 0
    total = 0
    all_ious = []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            mask_valid = masks != 0
            correct += (preds[mask_valid] == masks[mask_valid]).sum().item()
            total += mask_valid.sum().item()

            for i in range(masks.size(0)):
                miou, _ = compute_iou(preds[i], masks[i], cfg["num_classes"])
                all_ious.append(miou)

    avg_loss = total_loss / len(test_loader)
    pixel_acc = correct / total * 100
    mean_iou = np.mean(all_ious)

    print(f"\nTest Loss: {avg_loss:.4f}")
    print(f"Pixel Accuracy: {pixel_acc:.2f}%")
    print(f"Mean IoU: {mean_iou:.4f}")


if __name__ == "__main__":
    evaluate()
