"""
Principal Component Analysis (PCA) on MNIST.

PCA finds a low-dimensional representation of data by projecting onto the
directions of highest variance. These directions are the eigenvectors of the
covariance matrix, corresponding to the largest eigenvalues.

We use SVD (Singular Value Decomposition) which is numerically more stable
than eigendecomposition of the covariance matrix:

    X ≈ U Σ V^T

The principal components are the rows of V (right singular vectors).
Projecting onto the first d components:  X_proj = X · V[:, :d]

On MNIST, the first two PCs capture the coarse structure: digits roughly
separate into clusters even in 2D.
"""

import numpy as np
from datasets import load_dataset


def load_mnist_flat(max_samples=5000):
    """Load MNIST and flatten."""
    ds = load_dataset("ylecun/mnist", split="train")
    batch = ds[:max_samples]
    X = np.stack([np.array(img, dtype=np.float64).reshape(-1) for img in batch["image"]])
    X /= 255.0
    y = np.array(batch["label"], dtype=np.int32)
    return X, y


def pca(X, n_components=2):
    """
    Fit PCA via SVD.

    Steps:
      1. Centre the data (subtract mean along each feature).
      2. Compute SVD: U, S, Vt = svd(X_centered).
      3. The principal components are the rows of Vt[:n_components].
      4. Projection: X_proj = X_centered @ Vt[:n_components].T

    Why SVD instead of eigendecomposition of X^T X?
      - SVD is numerically more stable, especially when n_features is large.
      - It avoids computing the (784×784) covariance matrix explicitly.
    """
    # Centre the data.
    mean = X.mean(axis=0)
    X_centered = X - mean

    # SVD: full_matrices=False avoids creating large unused matrices.
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Project onto top-n_components principal components.
    components = Vt[:n_components]          # (n_components, n_features)
    X_proj = X_centered @ components.T      # (n_samples, n_components)

    # Variance explained ratio.
    var_explained = S[:n_components] ** 2 / (S ** 2).sum()

    return X_proj, components, mean, var_explained


def reconstruct(X_proj, components, mean):
    """
    Reconstruct data from low-dimensional projection.
    Useful to see what information is lost by PCA.
    """
    return X_proj @ components + mean


def print_ascii_scatter(X_2d, y, size=40):
    """
    Print a simple ASCII scatter plot of 2D PCA projection.
    Each digit 0-9 is represented by its class label character.
    """
    # Normalise to [0, 1] grid.
    x_norm = (X_2d[:, 0] - X_2d[:, 0].min()) / (X_2d[:, 0].max() - X_2d[:, 0].min() + 1e-8)
    y_norm = (X_2d[:, 1] - X_2d[:, 1].min()) / (X_2d[:, 1].max() - X_2d[:, 1].min() + 1e-8)

    # Map to integer grid.
    grid = [[' ' for _ in range(size)] for _ in range(size)]
    for i in range(len(x_norm)):
        row = min(int(y_norm[i] * (size - 1)), size - 1)
        col = min(int(x_norm[i] * (size - 1)), size - 1)
        if grid[row][col] == ' ':
            grid[row][col] = str(y[i])
        elif grid[row][col] != '.':
            grid[row][col] = '.'

    print(f"PCA projection (top 2 components, variance explained: "
          f"{var_explained[0]:.1%} + {var_explained[1]:.1%})")
    print()
    for row in grid:
        print(''.join(row))


def demo():
    np.random.seed(42)
    print("Loading MNIST (5000 samples)...")
    X, y = load_mnist_flat(5000)
    print(f"  Data shape: {X.shape}")

    global var_explained
    X_proj, components, mean, var_explained = pca(X, n_components=2)
    print(f"  Variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}")
    print(f"  Total: {var_explained.sum():.1%}")
    print()

    print_ascii_scatter(X_proj, y)

    # Show a reconstruction example.
    print()
    print("Reconstruction (first digit, 784→2→784):")
    x_orig = X[0]
    x_recon = reconstruct(X_proj[0:1], components, mean)[0]
    mse = ((x_orig - x_recon) ** 2).mean()
    print(f"  MSE = {mse:.4f}")
    # Show first 10 pixels for visual comparison.
    print(f"  Original first 10 pixels:    {x_orig[:10].round(3).tolist()}")
    print(f"  Reconstructed first 10 pixels: {x_recon[:10].round(3).tolist()}")


if __name__ == "__main__":
    demo()
