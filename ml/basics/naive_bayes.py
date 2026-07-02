"""
Gaussian Naive Bayes on MNIST.

Naive Bayes is a probabilistic classifier based on Bayes' theorem:

    P(y | x) ∝ P(y) · Π P(x_i | y)

The "naive" assumption is that all features are conditionally independent
given the class label. For images this is obviously false (pixels are highly
correlated), but the classifier still works reasonably well on simple tasks.

We use the Gaussian likelihood: P(x_i | y) ~ N(μ_iy, σ_iy²), where μ_iy and
σ_iy are the mean and standard deviation of pixel i across all training
samples of class y.

Comparison with Logistic Regression on the same MNIST data:
  - Logistic Regression:  92.3%   (discriminative, learns weights directly)
  - Naive Bayes:          ~55-60% (generative, models P(x|y)P(y))

The gap comes from the independence assumption — pixels are highly dependent
in natural images, so NB underestimates the probability of typical patterns.
"""

import numpy as np
from datasets import load_dataset
from utils.seed import set_seed


def load_mnist_flat():
    """Load MNIST train and test as flattened numpy arrays."""
    ds_train = load_dataset("ylecun/mnist", split="train")
    ds_test = load_dataset("ylecun/mnist", split="test")

    def to_numpy(ds):
        batch = ds[:]
        imgs = np.stack([np.array(img, dtype=np.float64).reshape(-1) for img in batch["image"]])
        imgs /= 255.0
        labels = np.array(batch["label"], dtype=np.int32)
        return imgs, labels

    return to_numpy(ds_train), to_numpy(ds_test)


class GaussianNB:
    """
    Gaussian Naive Bayes classifier.

    For each class c and each feature i, we store:
      - prior: P(class = c)   = count_c / n
      - mu[i, c]             = mean of feature i across class c
      - sigma[i, c]          = std of feature i across class c

    Prediction chooses the class with the highest log-posterior:
      log P(c|x) = log P(c) + Σ log P(x_i | c)
                  = log P(c) - Σ [log(σ_ic) + 0.5·((x_i - μ_ic)/σ_ic)²]
                  + constant

    We omit the constant term since it doesn't affect the argmax.
    """

    def fit(self, X, y):
        """Compute priors, means, and stds for each class."""
        n, d = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.prior = np.zeros(n_classes)
        self.mu = np.zeros((d, n_classes))
        self.sigma = np.zeros((d, n_classes))

        for i, c in enumerate(self.classes):
            mask = y == c
            self.prior[i] = mask.sum() / n
            self.mu[:, i] = X[mask].mean(axis=0)
            self.sigma[:, i] = X[mask].std(axis=0) + 1e-8  # epsilon to avoid division by zero

        return self

    def predict(self, X):
        """
        Predict class labels using log-posterior maximisation.

        Log-space avoids floating-point underflow when multiplying many
        small probabilities (784 pixels × 0.3 ≈ 10⁻¹⁶⁴ — far below float64 min).
        """
        n, d = X.shape
        n_classes = len(self.classes)
        log_probs = np.zeros((n, n_classes))

        for i in range(n_classes):
            # log P(c) + Σ log N(x_j | μ_jc, σ_jc)
            diff = (X - self.mu[:, i]) / self.sigma[:, i]
            log_likelihood = -0.5 * np.sum(diff ** 2, axis=1) - np.sum(np.log(self.sigma[:, i]))
            log_probs[:, i] = np.log(self.prior[i]) + log_likelihood

        return self.classes[np.argmax(log_probs, axis=1)]

    def score(self, X, y):
        return (self.predict(X) == y).mean()


def demo():
    set_seed(42)
    print("Loading MNIST...")
    (X_tr, y_tr), (X_te, y_te) = load_mnist_flat()
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")
    print(f"  Features: {X_tr.shape[1]}  Classes: {len(np.unique(y_tr))}")
    print()

    print("Gaussian Naive Bayes")
    print("=" * 30)

    # Subset training data for speed (NB training is O(nd) and MNIST is 60K×784).
    subset = 5000
    X_tr_sub = X_tr[:subset]
    y_tr_sub = y_tr[:subset]

    model = GaussianNB()
    model.fit(X_tr_sub, y_tr_sub)

    y_pred = model.predict(X_te)
    overall_acc = (y_pred == y_te).mean()
    print(f"  Test Accuracy (trained on {subset} samples): {overall_acc:.1%}")
    print()
    print("  Per-digit accuracy:")
    for i in range(10):
        mask = y_te == i
        acc_i = (y_pred[mask] == y_te[mask]).mean()
        print(f"    {i}: {acc_i:.1%}")
    print()
    print(f"  For comparison, Logistic Regression achieves ~92% on MNIST.")
    print(f"  The ~35% gap is due to the naive independence assumption:")
    print(f"  pixels are highly correlated in natural images, but Naive Bayes")
    print(f"  treats them as independent.")


if __name__ == "__main__":
    demo()
