"""
Clustering quality metrics shared by the unsupervised examples.

Three score families are used:

- silhouette: geometric and label-free. Measures how much closer a point is
  to its own cluster than to the nearest other cluster; range [-1, 1].
- NMI (Normalized Mutual Information) and ARI (Adjusted Rand Index): agreement
  with ground-truth labels. They need labels, so they are evaluation-only:
  the clustering never sees them.
- purity: fraction of points whose cluster's majority class matches their
  label. Also label-dependent, reported for continuity with the historical
  K-Means number.

Memory note: silhouette needs the full pairwise distance matrix, O(n^2). MNIST
has 60,000 points, i.e. 3.6e9 float64 entries (~28.8 GB), so the public entry
point subsamples first (default 5,000 -> 200 MB) and never allocates the full
matrix.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def euclidean_distances(a, b):
    """
    Pairwise Euclidean distances between two sets of points.

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a @ b.T

    Using this identity avoids explicit loops and is much faster. The
    arithmetic is done in place so a 5,000 x 5,000 matrix costs one allocation
    (~200 MB) rather than three.
    """
    a_norm = np.sum(a ** 2, axis=1, keepdims=True)
    b_norm = np.sum(b ** 2, axis=1, keepdims=True).T

    dist_sq = a @ b.T
    dist_sq *= -2.0
    dist_sq += a_norm
    dist_sq += b_norm
    np.maximum(dist_sq, 0, out=dist_sq)
    np.sqrt(dist_sq, out=dist_sq)
    return dist_sq


def subsample(X, assignments, sample_size=5000, seed=42):
    """Return a deterministic subsample of at most `sample_size` points."""
    n = X.shape[0]
    if sample_size is None or sample_size >= n:
        return X, assignments
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_size, replace=False)
    idx.sort()
    return X[idx], assignments[idx]


def silhouette_score(X, assignments, sample_size=5000, seed=42):
    """
    Mean silhouette coefficient of a clustering.

    For each point i:

        a(i) = mean distance to the other points of i's own cluster
        b(i) = min over other clusters of the mean distance to that cluster
        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Singleton clusters get s = 0 by convention (a(i) is undefined). Every
    integer label, including -1, is treated as an ordinary cluster; callers
    that use -1 for noise should filter those points out first.

    If `sample_size` is smaller than the number of points, the score is
    computed on a seeded subsample. This bounds memory at sample_size^2
    float64 entries instead of n^2.
    """
    assignments = np.asarray(assignments)
    if assignments.shape[0] != X.shape[0]:
        raise ValueError(
            f"got {X.shape[0]} points but {assignments.shape[0]} assignments"
        )

    X, assignments = subsample(X, assignments, sample_size, seed)

    clusters, inverse = np.unique(assignments, return_inverse=True)
    n_clusters = len(clusters)
    if n_clusters < 2:
        raise ValueError(
            f"silhouette needs at least 2 clusters, got {n_clusters}"
        )

    n = X.shape[0]
    distances = euclidean_distances(X, X)

    # Mean distance from every point to every cluster, as one matrix product
    # against a one-hot indicator (cheaper than a masked copy per cluster).
    indicator = np.zeros((n, n_clusters), dtype=np.float64)
    indicator[np.arange(n), inverse] = 1.0
    counts = indicator.sum(axis=0)
    mean_to = (distances @ indicator) / counts

    # a(i): own-cluster mean, excluding the point itself (distance 0).
    own_mean = mean_to[np.arange(n), inverse]
    a = own_mean * counts[inverse] / np.maximum(counts[inverse] - 1, 1)

    # b(i): closest other cluster.
    mean_to[np.arange(n), inverse] = np.inf
    b = mean_to.min(axis=1)

    denom = np.maximum(a, b)
    s = np.zeros(n)
    valid = (counts[inverse] > 1) & (denom > 0)
    s[valid] = (b[valid] - a[valid]) / denom[valid]
    return float(s.mean())


def nmi_score(labels_true, labels_pred):
    """Normalized Mutual Information (arithmetic normalization). Needs labels."""
    return float(normalized_mutual_info_score(labels_true, labels_pred))


def ari_score(labels_true, labels_pred):
    """Adjusted Rand Index. Needs labels."""
    return float(adjusted_rand_score(labels_true, labels_pred))


def purity_score(assignments, labels):
    """
    Cluster purity: for each cluster, count the points matching its majority
    class; average over all points.

    Purity ranges from 1/k (random) to 1 (perfect clusters). It rewards many
    small clusters, which is why it is reported next to NMI and ARI rather
    than alone. `labels` must be non-negative integers.
    """
    assignments = np.asarray(assignments)
    labels = np.asarray(labels)
    total = 0
    for cluster in np.unique(assignments):
        mask = assignments == cluster
        total += int(np.bincount(labels[mask]).max())
    return total / len(labels)


def _self_check():
    """Assert the score functions on inputs with known answers."""
    rng = np.random.default_rng(0)

    # Two well-separated blobs: both metrics must be ~1.
    blob_a = rng.normal(0.0, 0.3, size=(100, 2))
    blob_b = rng.normal(10.0, 0.3, size=(100, 2))
    X = np.vstack([blob_a, blob_b])
    y = np.array([0] * 100 + [1] * 100)

    score = silhouette_score(X, y)
    assert score > 0.95, f"separated blobs should score ~1, got {score}"
    assert abs(nmi_score(y, y) - 1.0) < 1e-12
    assert abs(ari_score(y, y) - 1.0) < 1e-12
    assert abs(purity_score(y, y) - 1.0) < 1e-12

    # Random labels on one blob: the score must be ~0.
    one_blob = rng.normal(0.0, 1.0, size=(200, 2))
    random_labels = rng.integers(0, 2, size=200)
    score_random = silhouette_score(one_blob, random_labels)
    assert abs(score_random) < 0.1, (
        f"random labels should score ~0, got {score_random}"
    )

    # The subsampled path must agree with the full computation.
    score_sampled = silhouette_score(X, y, sample_size=100, seed=42)
    assert abs(score_sampled - score) < 0.02, (
        f"subsample changed the score: {score} -> {score_sampled}"
    )

    print(
        "metrics self-check passed: "
        f"blobs={score:.4f}, random={score_random:.4f}, "
        f"sampled(100)={score_sampled:.4f}"
    )


if __name__ == "__main__":
    _self_check()
