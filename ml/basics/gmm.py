"""
Gaussian Mixture Model (GMM) with Expectation-Maximization, in pure NumPy.

K-Means makes a hard assignment and assumes spherical clusters of similar size.
A GMM instead models the data as a mixture of Gaussians:

    p(x) = sum_k pi_k * N(x | mu_k, Sigma_k)

with three parameter groups -- weights pi, means mu, full covariances Sigma --
fitted by EM:

  E-step: responsibilities  r_ik = p(z_i = k | x_i)  (Bayes, using log-sum-exp)
  M-step: pi, mu, Sigma = weighted maximum likelihood estimates

The log-likelihood is guaranteed not to decrease across EM iterations (up to
floating-point error), which this script asserts. `n_clusters` still has to be
chosen, but BIC/AIC score candidate values.

Demo: 5,000 MNIST images reduced to 50 principal components (ml/basics/data.py).
Labels are used for evaluation only, never for fitting.
"""

import json

import numpy as np

from ml.basics.data import load_mnist_pca
from ml.basics.metrics import ari_score, nmi_score, purity_score, silhouette_score
from utils.config import load_config, save_config
from utils.seed import set_seed


def logsumexp(a, axis=1, keepdims=True):
    """Numerically stable log(sum(exp(a))) along `axis`."""
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def kmeans_plus_plus_init(X, n_clusters, rng):
    """Pick starting means with k-means++ so EM rarely collapses a component."""
    n = X.shape[0]
    centers = np.empty((n_clusters, X.shape[1]), dtype=X.dtype)
    centers[0] = X[rng.integers(n)]
    closest = np.sum((X - centers[0]) ** 2, axis=1)
    for c in range(1, n_clusters):
        total = closest.sum()
        if total <= 0:
            centers[c] = X[rng.integers(n)]
        else:
            centers[c] = X[rng.choice(n, p=closest / total)]
        closest = np.minimum(closest, np.sum((X - centers[c]) ** 2, axis=1))
    return centers


class GaussianMixture:
    """Full-covariance Gaussian mixture fitted with EM."""

    def __init__(self, n_clusters=10, max_iters=100, tol=1e-4, reg_covar=1e-6, seed=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.reg_covar = reg_covar
        self.seed = seed

    def _e_step(self, X):
        """Return responsibilities (n, k) and the total log-likelihood."""
        n, d = X.shape
        log_prob = np.empty((n, self.n_clusters), dtype=np.float64)
        for k in range(self.n_clusters):
            covariance = self.covariances_[k]
            try:
                chol = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError:
                # A collapsed component can lose positive-definiteness; nudge it.
                covariance = covariance + 1e-6 * np.eye(d)
                self.covariances_[k] = covariance
                chol = np.linalg.cholesky(covariance)

            diff = X - self.means_[k]                       # (n, d)
            solved = np.linalg.solve(chol, diff.T)           # (d, n)
            mahalanobis = np.sum(solved ** 2, axis=0)
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            log_prob[:, k] = -0.5 * (d * np.log(2 * np.pi) + log_det + mahalanobis)

        weighted = log_prob + np.log(self.weights_)
        log_norm = logsumexp(weighted, keepdims=False)       # (n,)
        responsibilities = np.exp(weighted - log_norm[:, None])
        return responsibilities, float(log_norm.sum())

    def _m_step(self, X, responsibilities):
        n, d = X.shape
        nk = responsibilities.sum(axis=0) + 1e-10
        self.weights_ = nk / n
        self.means_ = (responsibilities.T @ X) / nk[:, None]
        for k in range(self.n_clusters):
            diff = X - self.means_[k]
            covariance = (responsibilities[:, k, None] * diff).T @ diff / nk[k]
            covariance.flat[:: d + 1] += self.reg_covar
            self.covariances_[k] = covariance

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)

        self.means_ = kmeans_plus_plus_init(X, self.n_clusters, rng)
        shared = np.cov(X, rowvar=False) + self.reg_covar * np.eye(d)
        self.covariances_ = np.tile(shared, (self.n_clusters, 1, 1))
        self.weights_ = np.full(self.n_clusters, 1.0 / self.n_clusters)

        self.log_likelihood_history_ = []
        self.converged_ = False
        previous = -np.inf
        for _ in range(self.max_iters):
            responsibilities, log_likelihood = self._e_step(X)
            self.log_likelihood_history_.append(log_likelihood)
            self._m_step(X, responsibilities)
            if abs(log_likelihood - previous) < self.tol * n:
                self.converged_ = True
                break
            previous = log_likelihood

        # Recompute with the final parameters so responsibilities and the
        # reported likelihood belong to the same model.
        self.responsibilities_, self.log_likelihood_ = self._e_step(X)
        self.log_likelihood_history_.append(self.log_likelihood_)
        return self

    def predict(self, X):
        responsibilities, _ = self._e_step(np.asarray(X, dtype=np.float64))
        return responsibilities.argmax(axis=1)

    @property
    def n_parameters(self):
        """Free parameters of a full-covariance mixture."""
        k, d = self.n_clusters, self.means_.shape[1]
        return (k - 1) + k * d + k * d * (d + 1) // 2

    def bic(self, n_samples):
        return -2.0 * self.log_likelihood_ + self.n_parameters * np.log(n_samples)

    def aic(self, n_samples):
        return -2.0 * self.log_likelihood_ + 2.0 * self.n_parameters


def synthetic_check(seed=42):
    """Three anisotropic, well-separated blobs must be recovered (ARI >= 0.99)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [8.0, 2.0], [3.0, 9.0]])
    covariances = [
        [[1.0, 0.8], [0.8, 1.0]],
        [[1.5, -0.6], [-0.6, 1.0]],
        [[0.8, 0.0], [0.0, 2.0]],
    ]
    X = np.vstack([
        rng.multivariate_normal(center, covariance, size=300)
        for center, covariance in zip(centers, covariances)
    ])
    y = np.repeat([0, 1, 2], 300)

    model = GaussianMixture(n_clusters=3, seed=seed).fit(X)
    ari = ari_score(y, model.predict(X))
    assert ari >= 0.99, f"GMM failed to recover the blobs: ARI={ari:.3f}"
    print(f"Synthetic blobs: GMM ARI = {ari:.4f} (3 components, 300 points each)\n")


def main():
    cfg = load_config("ml/basics/gmm.yaml")
    set_seed(cfg["seed"])

    synthetic_check(cfg["seed"])

    print(f"Loading MNIST train[:{cfg['max_samples']}] and reducing to "
          f"{cfg['pca_components']} PCs...")
    X, labels, var_explained = load_mnist_pca(cfg["max_samples"], cfg["pca_components"])
    print(f"  Data shape: {X.shape}, variance explained: {var_explained.sum():.1%}")

    model = GaussianMixture(
        n_clusters=cfg["n_clusters"],
        max_iters=cfg["max_iters"],
        tol=cfg["tol"],
        reg_covar=cfg["reg_covar"],
        seed=cfg["seed"],
    ).fit(X)

    for iteration, log_likelihood in enumerate(model.log_likelihood_history_, start=1):
        print(f"  EM iter {iteration:3d}  log-likelihood {log_likelihood:14.2f}")

    deltas = np.diff(model.log_likelihood_history_)
    tolerance = 1e-6 * max(1.0, abs(model.log_likelihood_history_[-1]))
    assert deltas.min() >= -tolerance, (
        f"log-likelihood decreased by {deltas.min():.3e} between iterations"
    )

    assignments = model.predict(X)
    metric_sample_size = cfg["metric_sample_size"]
    metrics = {
        "k": int(cfg["n_clusters"]),
        "seed": int(cfg["seed"]),
        "max_samples": int(cfg["max_samples"]),
        "pca_components": int(cfg["pca_components"]),
        "iterations": len(model.log_likelihood_history_),
        "converged": bool(model.converged_),
        "log_likelihood": round(model.log_likelihood_, 3),
        "bic": round(model.bic(len(X)), 3),
        "aic": round(model.aic(len(X)), 3),
        "purity": round(purity_score(assignments, labels), 6),
        "silhouette": round(silhouette_score(
            X, assignments, sample_size=metric_sample_size, seed=cfg["seed"]
        ), 6),
        "nmi": round(nmi_score(labels, assignments), 6),
        "ari": round(ari_score(labels, assignments), 6),
        "silhouette_sample_size": int(min(metric_sample_size, len(X))),
    }

    status = "converged" if model.converged_ else "stopped at max_iters"
    print(f"\nEM {status} after {len(model.log_likelihood_history_)} evaluations")
    print(f"  log-likelihood {model.log_likelihood_:.2f}")
    print(f"  BIC {metrics['bic']:.2f}  AIC {metrics['aic']:.2f}")
    print("\nClustering quality (labels used for evaluation only):")
    print(f"  purity     {metrics['purity']:7.2%}")
    print(f"  silhouette {metrics['silhouette']:7.4f}")
    print(f"  NMI        {metrics['nmi']:7.4f}")
    print(f"  ARI        {metrics['ari']:7.4f}")

    metrics_path = "ml/basics/gmm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")

    np.savez(
        cfg["model_path"],
        means=model.means_,
        covariances=model.covariances_,
        weights=model.weights_,
        assignments=assignments,
    )
    save_config(cfg, cfg["model_path"].replace(".npz", "_config.yaml"))
    print(f"\nModel saved to {cfg['model_path']}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
