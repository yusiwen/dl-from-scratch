"""
Frozen-encoder evaluation for a trained SimCLR model.

The SimCLR objective never sees labels, so it says nothing about whether the
representation is useful. This script measures that downstream:

  linear probe  logistic regression trained on frozen features (the standard
                SimCLR protocol), test top-1 accuracy
  kNN           cosine k-nearest-neighbour vote, k=20, no training at all

Neither probe updates the encoder. The encoder's state_dict is hashed before
and after feature extraction and the hashes must match.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from gen.simclr.data import load_cifar10_eval
from gen.simclr.model import SimCLR
from utils.config import load_config
from utils.device import get_device
from utils.seed import set_seed


def state_dict_hash(model):
    """SHA-256 over every parameter tensor, to prove the encoder is frozen."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def extract_features(encoder, loader, device):
    """Run the encoder in eval mode; returns (features, labels) as NumPy arrays."""
    encoder.eval()
    features, targets = [], []
    for batch in loader:
        images = batch["image"]
        features.append(encoder(images.to(device, non_blocking=True)).cpu())
        targets.append(batch["label"])
    return torch.cat(features).numpy(), torch.cat(targets).numpy()


def knn_accuracy(train_features, train_labels, test_features, test_labels, k=20, chunk=1000):
    """
    Cosine kNN accuracy without a distance-matrix library.

    Features are L2-normalized, so the dot product is the cosine similarity.
    Test rows are processed in chunks to bound the (chunk, n_train) buffer.
    """
    train = train_features / np.maximum(np.linalg.norm(train_features, axis=1, keepdims=True), 1e-12)
    test = test_features / np.maximum(np.linalg.norm(test_features, axis=1, keepdims=True), 1e-12)
    n_classes = int(train_labels.max()) + 1

    correct = 0
    for start in range(0, len(test), chunk):
        similarities = test[start:start + chunk] @ train.T
        neighbours = np.argpartition(-similarities, k, axis=1)[:, :k]
        for row, neighbour_indices in enumerate(neighbours):
            votes = np.bincount(train_labels[neighbour_indices], minlength=n_classes)
            correct += int(votes.argmax() == test_labels[start + row])
    return correct / len(test)


def main():
    cfg = load_config("gen/simclr/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")

    checkpoint = cfg["model_path"]
    if not os.path.exists(checkpoint):
        sys.exit(
            f"error: checkpoint not found at {checkpoint}; "
            f"run `python -m gen.simclr.train` first"
        )

    model = SimCLR(project_dim=cfg["project_dim"], temperature=cfg["temperature"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device)

    train_loader, test_loader = load_cifar10_eval(num_workers=cfg["num_workers"])

    before = state_dict_hash(model)
    started = time.time()
    train_x, train_y = extract_features(model.encoder, train_loader, device)
    test_x, test_y = extract_features(model.encoder, test_loader, device)
    after = state_dict_hash(model)
    assert before == after, "encoder weights changed during evaluation"
    print(f"Features: train {train_x.shape}, test {test_x.shape} "
          f"({time.time() - started:.1f}s, encoder frozen)")

    started = time.time()
    probe = LogisticRegression(max_iter=2000)
    probe.fit(train_x, train_y)
    probe_accuracy = float((probe.predict(test_x) == test_y).mean())
    print(f"Linear probe: {probe_accuracy:.2%} test top-1 "
          f"({time.time() - started:.1f}s)")

    started = time.time()
    knn_accuracy_value = knn_accuracy(train_x, train_y, test_x, test_y, k=20)
    print(f"kNN (k=20, cosine): {knn_accuracy_value:.2%} test top-1 "
          f"({time.time() - started:.1f}s)")

    result = {
        "checkpoint": checkpoint,
        "device": str(device),
        "epochs": int(cfg["num_epochs"]),
        "feature_dim": int(train_x.shape[1]),
        "train_samples": int(len(train_x)),
        "test_samples": int(len(test_x)),
        "linear_probe": round(probe_accuracy, 6),
        "knn_k": 20,
        "knn": round(knn_accuracy_value, 6),
    }
    os.makedirs("runs/simclr", exist_ok=True)
    with open("runs/simclr/eval.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print("Results saved to runs/simclr/eval.json")


if __name__ == "__main__":
    main()
