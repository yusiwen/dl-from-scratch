"""
Support Vector Machine — two implementations:

  1. SVM_GD:   Linear SVM via gradient descent on the primal hinge-loss objective.
  2. SVM_SMO:  Full SVM via Sequential Minimal Optimization (Platt 1998),
              supports Linear and RBF kernels.

Demos:
  - 2D synthetic blobs (visualize linear vs RBF separation)
  - MNIST binary (digit 3 vs 5)

SVMs find the hyperplane that maximises the margin between two classes.
The primal objective (linear case):

    minimize  (1/n) Σ max(0, 1 - y·(w·x + b))  +  λ||w||²
             └─────── hinge loss ────────┘    └─ L2 regulariser ─┘

The dual formulation (kernel-friendly):

    maximise  Σα_i - ½ ΣΣ α_i α_j y_i y_j K(x_i, x_j)
    subject to  0 ≤ α_i ≤ C,  Σ α_i y_i = 0

Points with α_i > 0 are *support vectors* — they alone define the boundary.
"""

import numpy as np
from datasets import load_dataset


# ─────────────────────────────────────────────────────────
# 1.  Linear SVM via Primal Gradient Descent
# ─────────────────────────────────────────────────────────

class SVM_GD:
    """
    Linear SVM trained by gradient descent on the primal hinge-loss objective.

    Loss = (1/n) Σ max(0, 1 - y·f(x)) + λ||w||²

    where f(x) = w·x + b.  The gradient is:

      dw = 2λw - (1/n) Σ y·x   for samples where y·f(x) < 1
      db = -(1/n) Σ y          for samples where y·f(x) < 1

    This is a simple, fast linear classifier.  No kernel trick.
    """

    def __init__(self, lam=0.01, lr=0.01, epochs=500):
        self.lam = lam
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        y must be {+1, -1}.
        X shape: (n, d)
        """
        n, d = X.shape
        self.w = np.zeros(d, dtype=np.float64)
        self.b = 0.0

        for epoch in range(self.epochs):
            # Forward: margin = y · (w·x + b)
            margins = y * (X @ self.w + self.b)
            # Hinge: which samples violate the margin?
            violations = margins < 1.0

            if violations.sum() == 0:
                # All correctly classified with margin ≥ 1 → optimal.
                if epoch % 200 == 0 and epoch > 0:
                    print(f"  GD epoch {epoch}: no violations, early stop")
                break

            X_v = X[violations]
            y_v = y[violations]

            # Gradient of the regulariser + hinge loss.
            dw = 2.0 * self.lam * self.w - (y_v * X_v.T).sum(axis=1) / n
            db = -y_v.sum() / n

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 200 == 0:
                loss = self._loss(X, y)
                print(f"  GD epoch {epoch}: loss={loss:.4f}, margin_violations={violations.sum()}")

    def _loss(self, X, y):
        margins = y * (X @ self.w + self.b)
        hinge = np.maximum(0, 1 - margins).mean()
        return hinge + self.lam * (self.w @ self.w)

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

    def score(self, X, y):
        return (self.predict(X) == y).mean()


# ─────────────────────────────────────────────────────────
# 2.  SVM via Sequential Minimal Optimization (SMO)
# ─────────────────────────────────────────────────────────

def _linear_kernel(X1, X2):
    return X1 @ X2.T

def _rbf_kernel(X1, X2, gamma=0.1):
    """RBF (Gaussian) kernel: K(x,z) = exp(-γ||x - z||²)."""
    dists = np.sum(X1 ** 2, axis=1, keepdims=True) \
          + np.sum(X2 ** 2, axis=1) \
          - 2.0 * X1 @ X2.T
    return np.exp(-gamma * np.maximum(dists, 0))


class SVM_SMO:
    """
    SVM trained by Sequential Minimal Optimisation (Platt 1998).

    Solves the dual problem, supports Linear and RBF kernels.
    Produces α coefficients; samples with α > 0 are support vectors.

    Parameters
    ----------
    C : float
        Regularisation strength (soft-margin parameter).
    kernel : str {'linear', 'rbf'}
    gamma : float
        RBF kernel width (ignored for linear kernel).
    tol : float
        KKT violation tolerance.
    max_passes : int
        Max passes without α change before convergence.
    """

    def __init__(self, C=1.0, kernel='linear', gamma=0.1, tol=1e-3, max_passes=10):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.alpha = None
        self.b = 0.0
        self.sv_idx = None
        self.X = None
        self.y = None
        self._K = None

    def _compute_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return _linear_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return _rbf_kernel(X1, X2, self.gamma)
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):
        n = X.shape[0]
        self.X = X
        self.y = y
        self.alpha = np.zeros(n, dtype=np.float64)
        self.b = 0.0
        self._K = self._compute_kernel(X, X)

        # Error cache: E_i = f(x_i) - y_i.  Initially f(x) = 0, so E = -y.
        E = np.full(n, -y, dtype=np.float64)

        passes = 0
        epoch = 0

        while passes < self.max_passes:
            num_changed = 0
            epoch += 1

            for i in range(n):
                ri = self.y[i] * (E[i] + self.y[i]) - 1  # = y_i·f(x_i) - 1

                if ((self.alpha[i] < self.C and ri < -self.tol) or
                    (self.alpha[i] > 0 and ri > self.tol)):

                    j = self._second_choice(i, E)
                    if j == -1:
                        continue
                    if self._take_step(i, j, E):
                        num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

            if epoch % 20 == 0:
                viol = self._count_violations(E)
                print(f"  SMO ({self.kernel}) epoch {epoch}: "
                      f"α_changed={num_changed}, KKT_violations={viol}")

        self.sv_idx = np.where(self.alpha > 1e-6)[0]
        print(f"  SMO ({self.kernel}) converged: "
              f"{len(self.sv_idx)} / {n} support vectors")

    def _second_choice(self, i, E):
        """Pick j ≠ i that maximises |E_i - E_j| (standard SMO heuristic)."""
        # Prefer non-bound examples.
        non_bound = np.where((0 < self.alpha) & (self.alpha < self.C))[0]
        non_bound = non_bound[non_bound != i]
        if len(non_bound) > 0:
            return non_bound[np.argmax(np.abs(E[non_bound] - E[i]))]
        # Fallback to all others.
        rest = np.arange(len(self.alpha))
        rest = rest[rest != i]
        if len(rest) > 0:
            return rest[np.argmax(np.abs(E[rest] - E[i]))]
        return -1

    def _take_step(self, i, j, E):
        """Optimise α_i, α_j analytically and maintain error cache for all samples."""
        if i == j:
            return False

        alpha_i_old = self.alpha[i]
        alpha_j_old = self.alpha[j]
        yi, yj = self.y[i], self.y[j]

        Kii = self._K[i, i]
        Kjj = self._K[j, j]
        Kij = self._K[i, j]

        # Bounds L, H
        if yi != yj:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            L = max(0, alpha_i_old + alpha_j_old - self.C)
            H = min(self.C, alpha_i_old + alpha_j_old)
        if L >= H - 1e-10:
            return False

        Ei, Ej = E[i], E[j]
        eta = Kii + Kjj - 2.0 * Kij

        if eta <= 1e-12:
            return False

        alpha_j_new = alpha_j_old + yj * (Ei - Ej) / eta
        alpha_j_new = np.clip(alpha_j_new, L, H)

        if abs(alpha_j_new - alpha_j_old) < 1e-8:
            return False

        alpha_i_new = alpha_i_old + yi * yj * (alpha_j_old - alpha_j_new)
        alpha_i_new = np.clip(alpha_i_new, 0, self.C)

        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new

        # Bias update.
        bi = self.b - Ei - yi * (alpha_i_new - alpha_i_old) * Kii \
                        - yj * (alpha_j_new - alpha_j_old) * Kij
        bj = self.b - Ej - yi * (alpha_i_new - alpha_i_old) * Kij \
                        - yj * (alpha_j_new - alpha_j_old) * Kjj
        if 0 < alpha_i_new < self.C:
            b_new = bi
        elif 0 < alpha_j_new < self.C:
            b_new = bj
        else:
            b_new = (bi + bj) / 2.0

        delta_b = b_new - self.b
        self.b = b_new

        # Update error cache for ALL samples using the incremental formula:
        #   E_k += y_i * Δα_i * K_ik + y_j * Δα_j * K_jk + Δb
        di = yi * (alpha_i_new - alpha_i_old)
        dj = yj * (alpha_j_new - alpha_j_old)
        for k in range(len(E)):
            E[k] += di * self._K[i, k] + dj * self._K[j, k] + delta_b

        return True

    def _count_violations(self, E):
        """Count KKT violations for convergence monitoring."""
        n = len(self.alpha)
        count = 0
        for i in range(n):
            ri = self.y[i] * (E[i] + self.y[i]) - 1
            if ((self.alpha[i] < self.C and ri < -self.tol) or
                (self.alpha[i] > 0 and ri > self.tol)):
                count += 1
        return count

    def decision_function(self, X):
        """Compute f(x) = Σ α_i y_i K(x_i, x) + b."""
        Kx = self._compute_kernel(self.X, X)
        return (self.alpha * self.y) @ Kx + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        return (self.predict(X) == y).mean()


# ─────────────────────────────────────────────────────────
# 3.  Demos
# ─────────────────────────────────────────────────────────

def _make_moons(n=200, noise=0.15):
    """Generate two interleaving half-moons (2-class, 2D)."""
    np.random.seed(42)
    n_per = n // 2
    t = np.linspace(0, np.pi, n_per)
    X1 = np.column_stack([np.cos(t), np.sin(t)]) + np.random.randn(n_per, 2) * noise
    X2 = np.column_stack([1 - np.cos(t), -np.sin(t) + 0.5]) + np.random.randn(n_per, 2) * noise
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_per), -np.ones(n_per)])
    return X, y


def _load_mnist_binary(digit_a=3, digit_b=5, max_samples=500):
    """Load MNIST and filter to two digits, relabel y ∈ {+1, -1}."""
    ds = load_dataset("ylecun/mnist", split="train")
    X_list, y_list = [], []
    for item in ds:
        lbl = item["label"]
        if lbl == digit_a:
            X_list.append(np.array(item["image"], dtype=np.float64).reshape(-1) / 255.0)
            y_list.append(1.0)
        elif lbl == digit_b:
            X_list.append(np.array(item["image"], dtype=np.float64).reshape(-1) / 255.0)
            y_list.append(-1.0)
        if len(X_list) >= max_samples:
            break
    X = np.array(X_list)
    y = np.array(y_list)
    # Shuffle.
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def demo_2d():
    """Compare GD, SMO-linear, and SMO-RBF on 2D moon-shaped data."""
    print("\n" + "=" * 60)
    print("SVM Demo 1: 2D synthetic moons (n=200)")
    print("=" * 60)

    X, y = _make_moons(200)
    split = 140
    mean = X[:split].mean(axis=0)
    std = X[:split].std(axis=0) + 1e-8
    X = (X - mean) / std
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    n_feat = X.shape[1]
    gamma_rbf = 1.0 / n_feat
    print(f"  Train: {len(y_tr)}  Test: {len(y_te)}  Features: {n_feat}")
    print(f"  RBF γ=1/feat = {gamma_rbf:.2f}\n")

    print("── Primal GD (linear) ──")
    gd = SVM_GD(lam=0.01, lr=0.1, epochs=1000)
    gd.fit(X_tr, y_tr)
    print(f"  Test acc: {gd.score(X_te, y_te):.1%}\n")

    print("── SMO (linear, C=10) ──")
    smo_lin = SVM_SMO(C=10.0, kernel='linear', max_passes=10)
    smo_lin.fit(X_tr, y_tr)
    print(f"  Test acc: {smo_lin.score(X_te, y_te):.1%}\n")

    print(f"── SMO (RBF, C=10, γ={gamma_rbf:.2f}) ──")
    smo_rbf = SVM_SMO(C=10.0, kernel='rbf', gamma=gamma_rbf, max_passes=10)
    smo_rbf.fit(X_tr, y_tr)
    print(f"  Test acc: {smo_rbf.score(X_te, y_te):.1%}")


def demo_mnist():
    """Compare GD, SMO-linear, and SMO-RBF on MNIST digits 3 vs 5."""
    print("\n" + "=" * 60)
    print("SVM Demo 2: MNIST (3 vs 5, max 500 samples)")
    print("=" * 60)

    X, y = _load_mnist_binary(3, 5, 500)
    split = 350

    mean = X[:split].mean(axis=0)
    std = X[:split].std(axis=0) + 1e-8
    X = (X - mean) / std
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    n_feat = X.shape[1]
    gamma_rbf = 1.0 / n_feat
    print(f"  Train: {len(y_tr)}  Test: {len(y_te)}  Features: {n_feat}")
    print(f"  RBF γ=1/feat = {gamma_rbf:.4f}\n")

    print("── Primal GD (linear) ──")
    gd = SVM_GD(lam=0.01, lr=0.1, epochs=500)
    gd.fit(X_tr, y_tr)
    print(f"  Test acc: {gd.score(X_te, y_te):.1%}\n")

    print("── SMO (linear, C=10) ──")
    smo_lin = SVM_SMO(C=10.0, kernel='linear', max_passes=10)
    smo_lin.fit(X_tr, y_tr)
    print(f"  Test acc: {smo_lin.score(X_te, y_te):.1%}\n")

    print(f"── SMO (RBF, C=10, γ={gamma_rbf:.4f}) ──")
    smo_rbf = SVM_SMO(C=10.0, kernel='rbf', gamma=gamma_rbf, max_passes=10)
    smo_rbf.fit(X_tr, y_tr)
    print(f"  Test acc: {smo_rbf.score(X_te, y_te):.1%}")


def main():
    demo_2d()
    demo_mnist()


if __name__ == "__main__":
    main()
