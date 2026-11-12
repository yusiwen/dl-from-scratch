"""Shared dataset helpers."""

import hashlib

import numpy as np
from datasets import load_dataset


def load_mnist_flat(max_samples=5000):
    """
    Load the first `max_samples` MNIST train images as (n, 784) float64 in [0, 1].

    Taking the first N images (instead of a random sample) keeps every example
    that uses this helper on exactly the same subset without needing an RNG.
    """
    ds = load_dataset("ylecun/mnist", split="train")
    batch = ds[:max_samples]
    images = np.stack(
        [np.array(img, dtype=np.float64).reshape(-1) for img in batch["image"]]
    )
    images /= 255.0
    labels = np.array(batch["label"], dtype=np.int32)
    return images, labels


def subset_checksum(X):
    """SHA-256 of the array contents, for asserting two scripts used one subset."""
    return hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()
