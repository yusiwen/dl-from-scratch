"""
k-Nearest Neighbors (k-NN) on MNIST.

k-NN is the simplest machine learning algorithm: memorise all training data
and predict by majority vote of the k closest training samples.

There is no training step — the decision function is just a distance
computation + argmax. This makes k-NN a non-parametric, instance-based
learner.

Key concepts:
  - No training: the "model" is the training set itself.
  - Distance metric: Euclidean (L2) is the most common.
  - Curse of dimensionality: k-NN degrades in high dimensions because
    distances become less meaningful (all points look equally far).
  - Bias-variance: small k = low bias, high variance; large k = higher bias, lower variance.

MNIST 784D is a high-dimensional space, so k-NN's performance is limited
by the curse of dimensionality. Even so, with enough data it reaches ~97%.
"""

import numpy as np
from datasets import load_dataset
from utils.seed import set_seed


def load_mnist_flat():
    """Load MNIST train/test as flattened numpy arrays."""
    ds_train = load_dataset("ylecun/mnist", split="train")
    ds_test = load_dataset("ylecun/mnist", split="test")

    def to_numpy(ds):
        batch = ds[:]
        imgs = np.stack([np.array(img, dtype=np.float64).reshape(-1) for img in batch["image"]])
        imgs /= 255.0
        labels = np.array(batch["label"], dtype=np.int32)
        return imgs, labels

    return to_numpy(ds_train), to_numpy(ds_test)


def euclidean_distances(a, b):
    """
    Pairwise Euclidean distances ||a - b|| via the quadratic identity.

      ||a - b||^2 = ||a||^2 + ||b||^2 - 2·a·b^T

    Vectorised over all pairs, so no Python loops.
    """
    a_norm = np.sum(a ** 2, axis=1, keepdims=True)
    b_norm = np.sum(b ** 2, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(a_norm + b_norm - 2 * a @ b.T, 0))


class KNN:
    """
    k-Nearest Neighbors classifier.

    Parameters:
      k: number of neighbors to consider.
    """

    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        """Store the training data—the only 'training' is memorisation."""
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        """
        Predict by majority vote of k nearest neighbors.

        For each test sample:
          1. Compute distances to all training samples.
          2. Find the k closest indices.
          3. Return the most common label among them.
        """
        dists = euclidean_distances(X, self.X)
        n_test = X.shape[0]
        predictions = np.zeros(n_test, dtype=np.int32)

        for i in range(n_test):
            nearest = np.argsort(dists[i])[:self.k]
            labels = self.y[nearest]
            predictions[i] = np.bincount(labels).argmax()

        return predictions

    def score(self, X, y):
        return (self.predict(X) == y).mean()


def demo():
    set_seed(42)
    print("Loading MNIST...")
    (X_tr, y_tr), (X_te, y_te) = load_mnist_flat()
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")

    # Subset training data to keep prediction time reasonable.
    # k-NN on 60K training samples × 10K test samples × 784 dims
    # would take a while with pure NumPy.
    n_train = 2000
    n_test = 500
    X_tr_sub = X_tr[:n_train]
    y_tr_sub = y_tr[:n_train]
    X_te_sub = X_te[:n_test]
    y_te_sub = y_te[:n_test]

    print(f"\nk-NN on MNIST ({n_train} train, {n_test} test)")
    print("=" * 40)

    for k in [1, 3, 5, 10]:
        model = KNN(k=k)
        model.fit(X_tr_sub, y_tr_sub)
        acc = model.score(X_te_sub, y_te_sub)
        print(f"  k={k:<2}  Test acc: {acc:.1%}")

    print()
    print("Observations:")
    print("  - k=1 gives the most flexible decision boundary (low bias, high variance).")
    print("  - Larger k smooths the boundary (higher bias, lower variance).")
    print("  - With unlimited data, k-NN approaches the Bayes optimal error rate.")
    print("  - In 784D, the 'curse of dimensionality' makes distances less meaningful,")
    print("    so performance plateaus below simpler models like Logistic Regression.")


if __name__ == "__main__":
    demo()
