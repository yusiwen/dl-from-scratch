"""Linear probe + kNN on the frozen MAE encoder."""

import json
import os
import sys
import time

import torch

from selfsup.mae.data import load_cifar10_tensors
from selfsup.mae.model import MaskedAutoencoder
from utils.config import load_config
from utils.device import get_device
from utils.probe import knn_accuracy, linear_probe
from utils.seed import set_seed


@torch.no_grad()
def extract_features(model, images, batch_size, device):
    model.eval()
    features = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device)
        features.append(model.encode(batch).cpu())
    return torch.cat(features).numpy()


def main():
    cfg = load_config("selfsup/mae/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")

    if not os.path.exists(cfg["model_path"]):
        sys.exit(
            f"error: checkpoint not found at {cfg['model_path']}; "
            f"run `python -m selfsup.mae.train` first"
        )

    model = MaskedAutoencoder(
        image_size=cfg["image_size"],
        patch_size=cfg["patch_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        decoder_dim=cfg["decoder_dim"],
        decoder_layers=cfg["decoder_layers"],
        mask_ratio=cfg["mask_ratio"],
    ).to(device)
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {cfg['model_path']}")

    train_images, train_labels = load_cifar10_tensors("train")
    test_images, test_labels = load_cifar10_tensors("test")
    train_x = extract_features(model, train_images, cfg["probe_batch_size"], device)
    test_x = extract_features(model, test_images, cfg["probe_batch_size"], device)
    print(f"Features: train {train_x.shape}, test {test_x.shape}")

    started = time.time()
    probe_accuracy, _ = linear_probe(train_x, train_labels.numpy(), test_x, test_labels.numpy())
    print(f"Linear probe: {probe_accuracy:.2%} test top-1 ({time.time() - started:.1f}s)")

    started = time.time()
    knn = knn_accuracy(train_x, train_labels.numpy(), test_x, test_labels.numpy(), k=20)
    print(f"kNN (k=20, cosine): {knn:.2%} test top-1 ({time.time() - started:.1f}s)")

    result = {
        "checkpoint": cfg["model_path"],
        "device": str(device),
        "epochs": int(cfg["num_epochs"]),
        "mask_ratio": cfg["mask_ratio"],
        "feature_dim": int(train_x.shape[1]),
        "linear_probe": round(probe_accuracy, 6),
        "knn_k": 20,
        "knn": round(knn, 6),
    }
    os.makedirs("runs/mae", exist_ok=True)
    with open("runs/mae/eval.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print("Results saved to runs/mae/eval.json")


if __name__ == "__main__":
    main()
