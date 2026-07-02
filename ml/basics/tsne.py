"""
t-Distributed Stochastic Neighbor Embedding (t-SNE) on MNIST.

t-SNE is a nonlinear dimensionality reduction technique that preserves
local structure: points close in high-dimensional space stay close in
the low-dimensional embedding, while distant points can be moved apart.

Key differences from PCA:
  - Linear (PCA) vs nonlinear (t-SNE): t-SNE can capture curved manifolds.
  - Global (PCA) vs local (t-SNE): t-SNE only preserves neighbourhood structure;
    distances between clusters are not meaningful.
  - Deterministic (PCA) vs stochastic (t-SNE): different runs give different results.

We run both PCA and t-SNE on MNIST (5000 digits) and visualise the 2D projections
side by side to highlight these differences.
"""

import numpy as np
from datasets import load_dataset
from utils.seed import set_seed


def load_mnist_flat(max_samples=5000):
    """Load MNIST and flatten."""
    ds = load_dataset("ylecun/mnist", split="train")
    batch = ds[:max_samples]
    X = np.stack([np.array(img, dtype=np.float64).reshape(-1) for img in batch["image"]])
    X /= 255.0
    y = np.array(batch["label"], dtype=np.int32)
    return X, y


# ─────────────────────────────────────────────────────────
#  PCA (for comparison)
# ─────────────────────────────────────────────────────────

def pca_projection(X, n_components=2):
    """Project data to 2D via SVD."""
    mean = X.mean(axis=0)
    X_centered = X - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:n_components].T, S[:n_components] ** 2 / (S ** 2).sum()


# ─────────────────────────────────────────────────────────
#  t-SNE
# ─────────────────────────────────────────────────────────

def _binary_search_perplexity(dist_i, perplexity, tol=1e-5):
    """Binary search for sigma_i such that Perp(P_i) ≈ perplexity."""
    lo, hi = 1e-10, 1e10
    for _ in range(50):
        mid = (lo + hi) / 2.0
        p = np.exp(-dist_i / (2.0 * mid ** 2))
        p = p / (p.sum() + 1e-12)
        ent = -np.sum(p * np.log(p + 1e-12))
        perp = 2.0 ** ent
        if perp > perplexity:
            lo = mid
        else:
            hi = mid
    return mid


def compute_pairwise_affinities(X, perplexity=30):
    """
    Compute symmetric P matrix (joint probabilities in high-dimensional space).

    P_ij = (P_j|i + P_i|j) / 2N
    """
    N = X.shape[0]
    dists = np.sum(X ** 2, axis=1, keepdims=True) + np.sum(X ** 2, axis=1) - 2.0 * X @ X.T
    np.fill_diagonal(dists, 0.0)
    dists = np.maximum(dists, 0.0)

    P = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        dist_i = dists[i]
        dist_i[i] = np.inf  # exclude self
        sigma = _binary_search_perplexity(dist_i, perplexity)
        p = np.exp(-dist_i / (2.0 * sigma ** 2))
        p[i] = 0.0
        P[i] = p / (p.sum() + 1e-12)

    # Symmetrize and normalise.
    P = (P + P.T) / (2.0 * N)
    P = np.maximum(P, 1e-12)  # avoid log(0)
    return P


def tsne(X, n_components=2, perplexity=30, n_iter=1000, lr=200, momentum=0.8, early_exag=4.0):
    """
    t-SNE: reduce X to 2D via gradient descent on KL(P || Q).

    Returns:
        Y: (N, n_components) embedding
        kl_history: KL divergence per iteration
    """
    N = X.shape[0]

    # Compute high-dimensional affinities.
    P = compute_pairwise_affinities(X, perplexity)

    # Initialise low-dimensional embedding (random, scaled small).
    Y = np.random.randn(N, n_components).astype(np.float64) * 0.01
    vel = np.zeros_like(Y)

    kl_history = []
    early_exag_iters = 250

    for it in range(1, n_iter + 1):
        # Compute low-dimensional affinities (Student-t).
        dists = np.sum(Y ** 2, axis=1, keepdims=True) + np.sum(Y ** 2, axis=1) - 2.0 * Y @ Y.T
        np.fill_diagonal(dists, 0.0)
        dists = np.maximum(dists, 0.0)

        Q_num = 1.0 / (1.0 + dists)
        np.fill_diagonal(Q_num, 0.0)
        Q = Q_num / (Q_num.sum() + 1e-12)

        # Calculate gradient.
        PQ_diff = P - Q

        # Early exaggeration: multiply P by early_exag for first iterations.
        if it <= early_exag_iters:
            PQ_diff = early_exag * P - Q

        grad = np.zeros_like(Y)
        for i in range(N):
            diff = Y[i] - Y
            grad[i] = 4.0 * (PQ_diff[i][:, None] * diff).sum(axis=0)

        # Update with momentum.
        vel = momentum * vel + lr * grad
        Y = Y + vel

        # Center embedding (subtract mean).
        Y = Y - Y.mean(axis=0)

        # Compute KL divergence.
        kl = (P * np.log(P / np.maximum(Q, 1e-12))).sum()
        kl_history.append(kl)

        if it % 100 == 0 or it == 1:
            print(f"  t-SNE iter {it:4d}/{n_iter}  KL={kl:.2f}")

    return Y, kl_history


# ─────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────

COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
          "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4"]


def plot_comparison(X_pca, X_tsne, y, var_explained):
    """Matplotlib side-by-side scatter plot of PCA vs t-SNE."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # PCA.
    for c in range(10):
        mask = y == c
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=COLORS[c], label=str(c), s=10, alpha=0.7)
    ax1.set_title(f"PCA (var explained: {var_explained.sum():.1%})")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    ax1.legend(loc="upper right", markerscale=2)

    # t-SNE.
    for c in range(10):
        mask = y == c
        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=COLORS[c], label=str(c), s=10, alpha=0.7)
    ax2.set_title("t-SNE (perplexity=30)")
    ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2")
    ax2.legend(loc="upper right", markerscale=2)

    plt.tight_layout()
    plt.savefig("ml/basics/tsne_vs_pca.png", dpi=150)
    plt.show()
    print("\nSaved to ml/basics/tsne_vs_pca.png")


def demo():
    set_seed(42)
    print("Loading MNIST (5000 samples)...")
    X, y = load_mnist_flat(5000)
    print(f"  Data shape: {X.shape}")

    # ── PCA ──
    print("\nRunning PCA...")
    X_pca, var_explained = pca_projection(X)
    print(f"  Variance explained: {var_explained[0]:.1%} + {var_explained[1]:.1%} = {var_explained.sum():.1%}")

    # ── t-SNE ──
    print("\nRunning t-SNE (this may take a few minutes)...")
    X_tsne, kl_hist = tsne(X, n_components=2, perplexity=30, n_iter=1000)

    # ── Comparison ──
    print(f"\nFinal KL divergence: {kl_hist[-1]:.2f}")
    plot_comparison(X_pca, X_tsne, y, var_explained)


if __name__ == "__main__":
    demo()
