"""
PCA-reduced MNIST loader for the clustering examples.

GMM and DBSCAN both run on the same 5,000 images -- the first five thousand of
the MNIST train split, matching `ml/basics/pca.py` -- reduced to 50 principal
components. Taking the first N images (instead of a random sample) keeps the
comparison against the PCA demo exact and needs no RNG.
"""

import numpy as np

from ml.basics.pca import pca
from utils.data import load_mnist_flat


def load_mnist_pca(max_samples=5000, n_components=50):
    """
    Return an (n, n_components) PCA projection, the digit labels, and the
    variance-explained ratios.

    The projection is already centred; callers that need unit-variance
    features can scale it themselves.
    """
    images, labels = load_mnist_flat(max_samples)
    projection, _, _, var_explained = pca(images, n_components=n_components)
    return projection, labels, var_explained
