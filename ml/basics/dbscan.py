"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) in NumPy.

K-Means needs a cluster count and assumes convex, similarly sized clusters;
every point is forced into one. DBSCAN instead grows clusters out of *dense*
regions and leaves sparse points unassigned:

  core point   at least `min_samples` points within distance `eps` (incl. itself)
  border point within `eps` of a core point, but not core itself
  noise        neither -- labelled -1

A cluster is the set of points density-reachable from one core point, so it can
be any shape.

Two demos:

1. Synthetic concentric circles: DBSCAN recovers both rings, K-Means (which
   minimizes variance) splits them arbitrarily.
2. 5,000 MNIST images reduced to 50 principal components (ml/basics/data.py).
   Density is much harder to define in 50 dimensions: the distance to the
   nearest neighbours concentrates, so `eps` trades a few pure clusters against
   a large noise fraction. The measured numbers are printed; the circles are
   where DBSCAN is meant to win.

Labels are used for evaluation only, never for fitting.
"""

import json

import numpy as np

from ml.basics.data import load_mnist_pca
from ml.basics.k_means import kmeans
from ml.basics.metrics import ari_score, euclidean_distances, nmi_score, purity_score, silhouette_score
from utils.config import load_config, save_config
from utils.seed import set_seed

UNVISITED = -2
NOISE = -1


def fit_dbscan(X, eps, min_samples):
    """
    Return per-point labels (-1 = noise), the neighbour graph, and core mask.

    The full (n, n) distance matrix is materialized once; for 5,000 points that
    is 200 MB at float64, which is why the demo keeps the subset at 5,000.
    """
    distances = euclidean_distances(X, X)
    neighbours = distances <= eps
    n = X.shape[0]
    core = neighbours.sum(axis=1) >= min_samples

    labels = np.full(n, UNVISITED, dtype=np.int64)
    cluster = 0
    for i in range(n):
        if labels[i] != UNVISITED:
            continue
        seed_points = np.flatnonzero(neighbours[i])
        if len(seed_points) < min_samples:
            labels[i] = NOISE
            continue

        labels[i] = cluster
        in_queue = np.zeros(n, dtype=bool)
        in_queue[seed_points] = True
        queue = list(seed_points)
        head = 0
        while head < len(queue):
            q = queue[head]
            head += 1
            if labels[q] == NOISE:
                labels[q] = cluster          # border point of this cluster
            if labels[q] != UNVISITED:
                continue
            labels[q] = cluster
            q_neighbours = np.flatnonzero(neighbours[q])
            if len(q_neighbours) < min_samples:
                continue                     # border point: no expansion
            new = q_neighbours[~in_queue[q_neighbours]]
            in_queue[new] = True
            queue.extend(new.tolist())
        cluster += 1

    return labels, neighbours, core


def synthetic_check(seed=42):
    """DBSCAN must recover two concentric circles that K-Means cannot."""
    rng = np.random.default_rng(seed)
    n = 200
    angles_inner = rng.uniform(0.0, 2.0 * np.pi, n)
    angles_outer = rng.uniform(0.0, 2.0 * np.pi, n)
    inner = np.c_[np.cos(angles_inner), np.sin(angles_inner)]
    outer = 2.0 * np.c_[np.cos(angles_outer), np.sin(angles_outer)]
    inner += rng.normal(0.0, 0.03, inner.shape)
    outer += rng.normal(0.0, 0.03, outer.shape)
    X = np.vstack([inner, outer])
    y = np.repeat([0, 1], n)

    labels, _, _ = fit_dbscan(X, eps=0.3, min_samples=3)
    mask = labels >= 0
    dbscan_ari = ari_score(y[mask], labels[mask])

    _, kmeans_assignments = kmeans(X, k=2, max_iters=50, tol=1e-4)
    kmeans_ari = ari_score(y, kmeans_assignments)

    assert dbscan_ari >= 0.99, f"DBSCAN missed the rings: ARI={dbscan_ari:.3f}"
    assert kmeans_ari < 0.2, f"K-Means unexpectedly recovered the rings: ARI={kmeans_ari:.3f}"
    print(
        f"Synthetic circles: DBSCAN ARI = {dbscan_ari:.4f}, "
        f"K-Means ARI = {kmeans_ari:.4f}\n"
    )


def main():
    cfg = load_config("ml/basics/dbscan.yaml")
    set_seed(cfg["seed"])

    synthetic_check(cfg["seed"])

    print(f"Loading MNIST train[:{cfg['max_samples']}] and reducing to "
          f"{cfg['pca_components']} PCs...")
    X, labels, var_explained = load_mnist_pca(cfg["max_samples"], cfg["pca_components"])
    print(f"  Data shape: {X.shape}, variance explained: {var_explained.sum():.1%}")
    print(f"  eps={cfg['eps']}, min_samples={cfg['min_samples']}")

    assignments, _, core = fit_dbscan(X, cfg["eps"], cfg["min_samples"])
    n_clusters = int(assignments.max() + 1) if (assignments >= 0).any() else 0
    noise_ratio = float((assignments == NOISE).mean())

    print(f"\nClusters: {n_clusters}, core points: {int(core.sum()):,}, "
          f"noise: {noise_ratio:.2%}")

    metrics = {
        "eps": float(cfg["eps"]),
        "min_samples": int(cfg["min_samples"]),
        "seed": int(cfg["seed"]),
        "max_samples": int(cfg["max_samples"]),
        "pca_components": int(cfg["pca_components"]),
        "clusters": n_clusters,
        "core_points": int(core.sum()),
        "noise_ratio": round(noise_ratio, 6),
        "silhouette_sample_size": int(min(cfg["metric_sample_size"], len(X))),
    }

    mask = assignments >= 0
    if n_clusters >= 2:
        # Labels are defined on clustered points only; noise has no cluster to
        # compare, so every score below is restricted to `mask`.
        metrics["purity"] = round(purity_score(assignments[mask], labels[mask]), 6)
        metrics["silhouette"] = round(silhouette_score(
            X[mask], assignments[mask], sample_size=cfg["metric_sample_size"], seed=cfg["seed"]
        ), 6)
        metrics["nmi"] = round(nmi_score(labels[mask], assignments[mask]), 6)
        metrics["ari"] = round(ari_score(labels[mask], assignments[mask]), 6)

        print(f"\nClustering quality on the {(mask).sum():,} clustered points "
              f"(labels used for evaluation only):")
        print(f"  purity     {metrics['purity']:7.2%}")
        print(f"  silhouette {metrics['silhouette']:7.4f}")
        print(f"  NMI        {metrics['nmi']:7.4f}")
        print(f"  ARI        {metrics['ari']:7.4f}")
    else:
        print("\nFewer than 2 clusters; quality metrics skipped.")

    metrics_path = "ml/basics/dbscan_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")

    np.savez("ml/basics/dbscan_labels.npz", assignments=assignments, core=core)
    save_config(cfg, "ml/basics/dbscan_config.yaml")
    print(f"\nLabels saved to ml/basics/dbscan_labels.npz")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
