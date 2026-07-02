"""
Perceptron — the first neural network (Rosenblatt, 1958).

The Perceptron is a single neuron:
    y = sign(w·x + b)

Unlike modern neural networks, the Perceptron has:
  - Step activation function (not smooth like sigmoid/ReLU).
  - Online learning (updates one sample at a time).
  - Convergence guarantee only for linearly separable data.

The Perceptron Convergence Theorem states that if the data is linearly
separable, the Perceptron will find a separating hyperplane in a finite
number of steps.

We generate 2D synthetic data for visualisation, and also test on a
linearly separable subset of MNIST (0 vs 1).
"""

import numpy as np
from utils.config import load_config
from utils.seed import set_seed


class Perceptron:
    """
    Single-layer Perceptron.

    y_hat = sign(w·x + b)

    Update rule (when sample is misclassified):
      w ← w + y·x
      b ← b + y

    This pulls the decision boundary toward the misclassified sample.
    """

    def __init__(self, lr=1.0):
        self.lr = lr
        self.w = None
        self.b = 0.0

    def fit(self, X, y, epochs=100):
        """
        Online training: one sample at a time.

        y must be {+1, -1}.
        """
        n, d = X.shape
        self.w = np.zeros(d, dtype=np.float64)
        self.b = 0.0
        self.converged_at = None

        for epoch in range(epochs):
            errors = 0
            for i in range(n):
                # If misclassified, update.
                if y[i] * (X[i] @ self.w + self.b) <= 0:
                    self.w += self.lr * y[i] * X[i]
                    self.b += self.lr * y[i]
                    errors += 1
            if errors == 0:
                self.converged_at = epoch + 1
                break

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

    def score(self, X, y):
        return (self.predict(X) == y).mean()


def make_linear_2d(n=100):
    """Generate linearly separable 2D data."""
    np.random.seed(42)
    X = np.random.randn(n, 2) * 2
    y = np.sign(X[:, 0] - X[:, 1] + 0.5 * np.random.randn(n))  # slight noise
    # Ensure not all same sign.
    return X, y.astype(np.float64)


def load_mnist_01(max_samples=500):
    """Load MNIST digits 0 and 1 only (linearly separable)."""
    from datasets import load_dataset
    ds = load_dataset("ylecun/mnist", split="train")
    X_list, y_list = [], []
    for item in ds:
        lbl = item["label"]
        if lbl == 0:
            X_list.append(np.array(item["image"], dtype=np.float64).reshape(-1) / 255.0)
            y_list.append(1.0)
        elif lbl == 1:
            X_list.append(np.array(item["image"], dtype=np.float64).reshape(-1) / 255.0)
            y_list.append(-1.0)
        if len(X_list) >= max_samples:
            break
    X = np.array(X_list)
    y = np.array(y_list)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def demo_2d(cfg):
    """Perceptron on 2D linearly separable data."""
    print("Perceptron — 2D synthetic data")
    print("=" * 35)
    X, y = make_linear_2d(200)
    split = 140
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    model = Perceptron(lr=cfg["lr"])
    model.fit(X_tr, y_tr, epochs=cfg["epochs_2d"])
    status = f"converged at epoch {model.converged_at}" if model.converged_at else "did not converge"
    print(f"  {status}")
    print(f"  Test acc: {model.score(X_te, y_te):.1%}")
    print(f"  Learned weights: w=[{model.w[0]:.3f}, {model.w[1]:.3f}], b={model.b:.3f}")
    print()


def demo_mnist(cfg):
    """Perceptron on MNIST 0 vs 1."""
    print("Perceptron — MNIST (0 vs 1)")
    print("=" * 35)
    X, y = load_mnist_01(500)
    split = 350
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    model = Perceptron(lr=cfg["lr"])
    model.fit(X_tr, y_tr, epochs=cfg["epochs_mnist"])
    status = f"converged at epoch {model.converged_at}" if model.converged_at else "did not converge"
    print(f"  {status}")
    print(f"  Test acc: {model.score(X_te, y_te):.1%}")
    print(f"  Weights shape: {model.w.shape} (784 dims, no hidden layer)")
    print()


def demo():
    cfg = load_config("ml/basics/perceptron.yaml")
    set_seed(cfg["seed"])
    demo_2d(cfg)
    demo_mnist(cfg)


if __name__ == "__main__":
    demo()
