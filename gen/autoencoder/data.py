"""MNIST tensors for the autoencoder (flattened pixels in [0, 1])."""

import torch

from utils.data import load_mnist_flat


def load_mnist_tensors(max_samples=5000):
    """Return (images, labels) as tensors; images are (n, 784) float32 in [0, 1]."""
    images, labels = load_mnist_flat(max_samples)
    return (
        torch.tensor(images, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
