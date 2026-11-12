"""Linear probe + kNN on the frozen BYOL online encoder."""

import json
import os
import sys
import time

import torch

from selfsup.byol.model import BYOL
from selfsup.simclr.data import load_cifar10_eval
from utils.config import load_config
from utils.device import get_device
from utils.probe import knn_accuracy, linear_probe
from utils.seed import set_seed


@torch.no_grad()
def extract_features(encoder, loader, device):
    encoder.eval()
    features, targets = [], []
    for batch in loader:
        features.append(encoder(batch["image"].to(device)).cpu())
        targets.append(batch["label"])
    return torch.cat(features).numpy(), torch.cat(targets).numpy()


def main():
    cfg = load_config("selfsup/byol/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")

    if not os.path.exists(cfg["model_path"]):
        sys.exit(
            f"error: checkpoint not found at {cfg['model_path']}; "
            f"run `python -m selfsup.byol.train` first"
        )

    model = BYOL(
        projector_hidden=cfg["projector_hidden"],
        projector_dim=cfg["projector_dim"],
        predictor_hidden=cfg["predictor_hidden"],
    ).to(device)
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    encoder = model.online_encoder
    encoder.eval()
    print(f"Loaded online encoder from {cfg['model_path']}")

    train_loader, test_loader = load_cifar10_eval(num_workers=cfg["num_workers"])
    train_x, train_y = extract_features(encoder, train_loader, device)
    test_x, test_y = extract_features(encoder, test_loader, device)
    print(f"Features: train {train_x.shape}, test {test_x.shape}")

    started = time.time()
    probe_accuracy, _ = linear_probe(train_x, train_y, test_x, test_y)
    print(f"Linear probe: {probe_accuracy:.2%} test top-1 ({time.time() - started:.1f}s)")

    started = time.time()
    knn = knn_accuracy(train_x, train_y, test_x, test_y, k=20)
    print(f"kNN (k=20, cosine): {knn:.2%} test top-1 ({time.time() - started:.1f}s)")

    result = {
        "checkpoint": cfg["model_path"],
        "device": str(device),
        "epochs": int(cfg["num_epochs"]),
        "ema_decay": cfg["ema_decay"],
        "feature_dim": int(train_x.shape[1]),
        "linear_probe": round(probe_accuracy, 6),
        "knn_k": 20,
        "knn": round(knn, 6),
    }
    os.makedirs("runs/byol", exist_ok=True)
    with open("runs/byol/eval.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print("Results saved to runs/byol/eval.json")


if __name__ == "__main__":
    main()
