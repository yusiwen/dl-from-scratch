"""
Shared MNIST subset loader for the clustering examples.

GMM and DBSCAN both run on the same 5,000 images -- the first five thousand of
the MNIST train split, matching `ml/basics/pca.py` -- reduced to 50 principal
components. Taking the first N images (instead of a random sample) keeps the
comparison against the PCA demo exact and needs no RNG.
"""

import numpy as np
from datasets import load_dataset

from ml.basics.pca import pca


def load_mnist_flat(max_samples=5000):
    """Load the first `max_samples` MNIST train images as (n, 784) float64."""
    ds = load_dataset("ylecun/mnist", split="train")
    batch = ds[:max_samples]
    images = np.stack(
        [np.array(img, dtype=np.float64).reshape(-1) for img in batch["image"]]
    )
    images /= 255.0
    labels = np.array(batch["label"], dtype=np.int32)
    return images, labels


def load_mnist_pca(max_samples=5000, n_components=50):
    """
    Return an (n, n_components) PCA projection, the digit labels, and the
    variance-explained ratios.

    The projection is already centred; DBSCAN additionally standardizes it so
    that `eps` is comparable across components.
    """
    images, labels = load_mnist_flat(max_samples)
    projection, _, _, var_explained = pca(images, n_components=n_components)
    return projection, labels, var_explained


def standardize(X):
    """Zero mean and unit variance per column."""
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - X.mean(axis=0)) / std
