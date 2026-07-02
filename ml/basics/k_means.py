"""
K-Means clustering on MNIST.

K-Means is an unsupervised learning algorithm:
  1. Initialize k cluster centers randomly.
  2. Assign each data point to the nearest center (by Euclidean distance).
  3. Update each center to the mean of its assigned points.
  4. Repeat steps 2-3 until convergence (centers stop changing).

Unlike the other models in this project, K-Means has no labels, no loss
function, no gradient descent — it's purely distance-based clustering.

We cluster 60K MNIST digits into 10 clusters (k=10, one per digit).
The "accuracy" here is cluster purity: for each cluster, we check what
digit label is most common, and measure how many points match it.

Expected purity: ~50-60% (unsupervised, no label information used).
"""

import numpy as np
from datasets import load_dataset
from utils.config import load_config, save_config
from utils.seed import set_seed


def load_mnist_flat():
    """Load full MNIST train set as flattened numpy array (60000, 784)."""
    ds = load_dataset("ylecun/mnist", split="train")
    # Convert all images in one batch.
    batch = ds[:]
    images = np.stack([np.array(img, dtype=np.float32).reshape(-1) for img in batch["image"]])
    images /= 255.0
    labels = np.array(batch["label"], dtype=np.int32)
    return images, labels


def euclidean_distances(a, b):
    """
    Compute pairwise Euclidean distances between two sets of points.

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a @ b.T

    Using this identity avoids explicit loops and is much faster.
    """
    a_norm = np.sum(a ** 2, axis=1, keepdims=True)
    b_norm = np.sum(b ** 2, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(a_norm + b_norm - 2 * a @ b.T, 0))


def kmeans(data, k=10, max_iters=50, tol=1e-4):
    """
    Classic K-Means clustering.

    Parameters:
      data: (n_samples, n_features) numpy array
      k: number of clusters
      max_iters: maximum iterations
      tol: convergence threshold (center movement < tol)

    Returns:
      centers: (k, n_features) final cluster centers
      assignments: (n_samples,) cluster index for each point
    """
    n_samples = data.shape[0]

    # Initialization: pick the first k data points as centers.
    # This is deterministic (no randomness) — more sophisticated init methods
    # like K-Means++ give better results, but this is fine for a baseline.
    centers = data[:k].copy()

    for iteration in range(max_iters):
        # Step 1: Assign each point to the nearest center.
        distances = euclidean_distances(data, centers)
        assignments = np.argmin(distances, axis=1)

        # Step 2: Update centers to the mean of assigned points.
        new_centers = np.zeros_like(centers)
        for i in range(k):
            mask = assignments == i
            if mask.sum() > 0:
                new_centers[i] = data[mask].mean(axis=0)

        # Check for convergence: centers that have no assigned points stay as-is.
        for i in range(k):
            if (assignments == i).sum() == 0:
                new_centers[i] = centers[i]

        # Convergence check: max center movement < tol.
        shift = np.abs(new_centers - centers).max()
        centers = new_centers
        if shift < tol:
            print(f"Converged at iteration {iteration + 1}")
            break

    return centers, assignments


def purity_score(assignments, labels, k=10):
    """
    Compute cluster purity: for each cluster, find the majority class label
    and count how many points match it, then average.

    Purity = (1 / n) * sum_c max_class count_in_cluster_c

    Purity ranges from 1/k (random) to 1 (perfect clusters).
    """
    total = 0
    for i in range(k):
        mask = assignments == i
        if mask.sum() > 0:
            cluster_labels = labels[mask]
            # Find the most common label in this cluster.
            majority = np.bincount(cluster_labels).max()
            total += majority
    return total / len(labels)


def visualize_centers(centers):
    """
    Print a simple ASCII visualization of cluster centers.
    Each center is a 28x28 image; we downsample to 7x7 characters.
    """
    print("\nCluster centers (7x7 downsampled ASCII, '  ' = white, '##' = dark):")
    for c in range(len(centers)):
        img = centers[c].reshape(28, 28)
        print(f"\nCluster {c}:")
        for row in range(0, 28, 4):
            line = ""
            for col in range(0, 28, 4):
                block = img[row:row+4, col:col+4].mean()
                if block > 0.5:
                    line += "##"
                elif block > 0.2:
                    line += ".."
                else:
                    line += "  "
            print(line)


def train():
    cfg = load_config("ml/basics/k_means.yaml")
    set_seed(cfg["seed"])
    print("Loading MNIST (60K images)...")
    images, labels = load_mnist_flat()
    print(f"Data shape: {images.shape}")

    k = cfg["k"]
    centers, assignments = kmeans(images, k=k, max_iters=cfg["max_iters"], tol=cfg["tol"])
    purity = purity_score(assignments, labels, k=k)
    print(f"\nCluster purity: {purity:.2%}")

    # Per-cluster breakdown.
    print(f"\n{'Cluster':<10} {'Majority Class':<15} {'Size':>6}")
    print("-" * 35)
    for i in range(k):
        mask = assignments == i
        size = mask.sum()
        majority_label = np.bincount(labels[mask]).argmax()
        print(f"{i:<10} {majority_label:<15} {size:>6}")

    save_path = "ml/basics/kmeans_centers.npz"
    np.savez(save_path, centers=centers)
    save_config(cfg, save_path.replace(".npz", "_config.yaml"))
    print(f"\nCenters saved to {save_path}")

    visualize_centers(centers)


if __name__ == "__main__":
    train()
