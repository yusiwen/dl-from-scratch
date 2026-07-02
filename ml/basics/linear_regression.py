"""
Linear Regression on California Housing dataset.

Two implementations:
  1. Normal Equation — closed-form solution, one step (no training loop).
  2. Gradient Descent — iterative, same pattern as Logistic Regression elsewhere
     in this project.

Both reduce to the same core math: find θ = (θ0, θ1, ..., θn) that minimizes
MSE = (1/m) * Σ(y - Xθ)^2.

Dataset: gvlassis/california_housing
  - 8 features (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,
    Latitude, Longitude)
  - Target: MedHouseVal (median house value in units of $100K)
  - 16,640 samples, split 80/20 train/test
"""

import numpy as np
from datasets import load_dataset
from utils.config import load_config, save_config
from utils.seed import set_seed


def load_data(test_ratio=0.2):
    """Load California Housing from HF, split into train/test."""
    ds = load_dataset("gvlassis/california_housing", split="train")

    # Convert to numpy arrays.
    features = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude",
    ]
    n = len(ds)
    X = np.zeros((n, len(features)), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        row = ds[i]
        for j, f in enumerate(features):
            X[i, j] = row[f]
        y[i] = row["MedHouseVal"]

    # Shuffle.
    indices = np.random.permutation(n)
    X, y = X[indices], y[indices]

    # Split.
    split = int(n * (1 - test_ratio))
    return X[:split], y[:split], X[split:], y[split:], features


def standardize(X_train, X_test):
    """Z-score normalization (mean=0, std=1) for each feature."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled, mean, std


def add_bias_term(X):
    """Prepend a column of ones for the intercept term θ₀."""
    return np.column_stack([np.ones(X.shape[0]), X])


def normal_equation(X, y):
    """
    Closed-form solution: θ = (X^T X)^(-1) X^T y

    Setting the gradient of MSE to zero and solving for θ gives:
      ∇MSE(θ) = (2/m) * X^T (Xθ - y) = 0
      → X^T Xθ = X^T y
      → θ = (X^T X)^(-1) X^T y

    This is exact (up to floating-point precision) and requires no learning
    rate or iterations. However, inverting X^T X is O(n^3) in the number of
    features, so it becomes expensive when n_features > ~10,000.
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y


def predict(X, theta):
    """Linear prediction: ŷ = Xθ"""
    return X @ theta


def mse(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    """
    Coefficient of Determination R² = 1 - SS_res / SS_tot.

    R² measures how much of the variance in y is explained by the model.
      - 1.0 = perfect fit
      - 0.0 = model predicts the mean of y for every sample
      - < 0  = worse than predicting the mean
    """
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def gradient_descent(X, y, lr=0.1, epochs=2000, batch_size=None):
    """
    Gradient Descent solution: iteratively step toward the minimum of MSE.

    Gradient of MSE w.r.t θ:  ∇MSE = (2/m) * X^T (Xθ - y)

    Update rule:  θ ← θ - lr * ∇MSE

    If batch_size is None, use full-batch (all samples per step).
    If batch_size is set, use mini-batch gradient descent with that batch size.
    """
    m, n = X.shape
    theta = np.zeros(n, dtype=np.float64)
    loss_history = []

    if batch_size is None or batch_size >= m:
        # Full-batch gradient descent.
        for epoch in range(epochs):
            grad = 2.0 * X.T @ (X @ theta - y) / m
            theta -= lr * grad
            if epoch % 200 == 0 or epoch == epochs - 1:
                loss_history.append(mse(y, predict(X, theta)))
    else:
        # Mini-batch gradient descent.
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled, y_shuffled = X[indices], y[indices]
            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                grad = 2.0 * X_batch.T @ (X_batch @ theta - y_batch) / len(y_batch)
                theta -= lr * grad
            if epoch % 200 == 0 or epoch == epochs - 1:
                loss_history.append(mse(y, predict(X, theta)))

    return theta, loss_history


def train():
    cfg = load_config("ml/basics/linear_regression.yaml")
    set_seed(cfg["seed"])

    X_train, y_train, X_test, y_test, features = load_data(test_ratio=cfg["test_ratio"])
    X_train_s, X_test_s, mean, std = standardize(X_train, X_test)
    X_train_b = add_bias_term(X_train_s)
    X_test_b = add_bias_term(X_test_s)

    print("Linear Regression — California Housing")
    print("=" * 45)
    print(f"Features ({len(features)}): {', '.join(features)}")
    print(f"Target: MedHouseVal (median house value in $100K)")
    print(f"Samples: train={len(y_train)}, test={len(y_test)}")
    print()

    # --- Normal Equation ---
    theta_ne = normal_equation(X_train_b, y_train)
    y_pred_ne = predict(X_test_b, theta_ne)
    mse_ne = mse(y_test, y_pred_ne)
    r2_ne = r2_score(y_test, y_pred_ne)
    print(f"Normal Equation:   MSE = {mse_ne:.3f}, R² = {r2_ne:.3f}")

    # --- Gradient Descent ---
    theta_gd, loss_hist = gradient_descent(
        X_train_b, y_train, lr=cfg["lr"], epochs=cfg["num_epochs"],
        batch_size=cfg.get("batch_size"),
    )
    y_pred_gd = predict(X_test_b, theta_gd)
    mse_gd = mse(y_test, y_pred_gd)
    r2_gd = r2_score(y_test, y_pred_gd)
    print(f"Gradient Descent:  MSE = {mse_gd:.3f}, R² = {r2_gd:.3f}  "
          f"(epochs={cfg['num_epochs']}, lr={cfg['lr']})")

    # --- Compare parameters ---
    diff = np.abs(theta_ne - theta_gd).max()
    print(f"\nMax parameter difference: {diff:.2e}  "
          f"{'✓ identical' if diff < 1e-4 else '— minor numerical diff'}")
    print()

    # --- Feature importance ---
    # The bias term (θ₀) is at index 0; feature weights start at index 1.
    print(f"{'Feature':<15} {'Weight':>8}  {'Std Impact':>12}")
    print("-" * 40)
    # Sort by absolute weight, descending.
    feature_weights = list(zip(features, theta_ne[1:]))
    feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, w in feature_weights:
        print(f"{feat:<15} {w:>8.3f}  {'':>12}")
    print(f"{'Bias (intercept)':<15} {theta_ne[0]:>8.3f}  {'':>12}")
    print()

    # --- Save weights ---
    save_path = "ml/basics/linear_regression.npz"
    np.savez(save_path, theta=theta_ne, feature_names=features)
    save_config(cfg, save_path.replace(".npz", "_config.yaml"))
    print("Weights saved to ml/basics/linear_regression.npz")


if __name__ == "__main__":
    train()
