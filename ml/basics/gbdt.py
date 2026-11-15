"""
Gradient Boosted Decision Tree (GBDT) — regression (MSE) and binary
classification (LogLoss).

Core idea:
  Fit an ensemble of shallow regression trees sequentially, where each new
  tree predicts the *negative gradient* of the loss w.r.t. the current
  prediction.  This is functional gradient descent.

Key concepts:
  - Stage-wise additive model:  F_m(x) = F_{m-1}(x) + η · h_m(x)
  - Regression trees fit pseudo-residuals (negative gradients)
  - Shrinkage (learning rate η) prevents overfitting
"""

import json
import os

import numpy as np

from utils.config import load_config
from utils.seed import set_seed
from ml.basics.decision_tree import build_tree, predict


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))


class GBDT:
    """Gradient Boosted Decision Tree — regression ('mse') or binary ('logloss')."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=2, loss='mse', seed=42, max_thresholds=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss
        self.seed = seed
        self.max_thresholds = max_thresholds
        self.trees = []
        self.init_pred_ = None
        self.feature_names_ = None
        self.class_names_ = None
        self._is_fitted = False

    def fit(self, X, y):
        set_seed(self.seed)
        n = X.shape[0]
        self.n_features_ = X.shape[1]

        if self.loss == 'mse':
            self.init_pred_ = float(np.mean(y))
            raw_pred = np.full(n, self.init_pred_)
        elif self.loss == 'logloss':
            p = max(np.mean(y), 1e-8)
            self.init_pred_ = float(np.log(p / (1 - p)))
            raw_pred = np.full(n, self.init_pred_)
            self.n_classes_ = 2
        else:
            raise ValueError(f"Unsupported loss: {self.loss}")

        self.trees = []

        for i in range(self.n_estimators):
            if self.loss == 'mse':
                pseudo_residual = y - raw_pred
            else:
                prob = sigmoid(raw_pred)
                pseudo_residual = y - prob

            tree = build_tree(X, pseudo_residual, max_depth=self.max_depth,
                              min_samples=self.min_samples_leaf, criterion='mse',
                              max_thresholds=self.max_thresholds)
            self.trees.append(tree)

            tree_pred = predict(tree, X)
            raw_pred += self.learning_rate * tree_pred

        self._is_fitted = True
        return self

    def predict_raw(self, X):
        pred = np.full(X.shape[0], self.init_pred_)
        for tree in self.trees:
            pred += self.learning_rate * predict(tree, X)
        return pred

    def predict(self, X):
        raw = self.predict_raw(X)
        if self.loss == 'logloss':
            return (sigmoid(raw) >= 0.5).astype(np.int32)
        return raw

    def save(self, path):
        data = {
            "format_version": 1,
            "model_type": "GBDT",
            "config": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "loss": self.loss,
                "seed": self.seed,
            },
            "n_features": self.n_features_,
            "init_pred": self.init_pred_,
            "feature_names": self.feature_names_,
            "trees": self.trees,
        }
        if hasattr(self, 'n_classes_'):
            data["n_classes"] = self.n_classes_
        if self.loss == 'logloss':
            data["class_names"] = self.class_names_
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        gbdt = cls(**data["config"])
        gbdt.n_features_ = data["n_features"]
        gbdt.init_pred_ = data["init_pred"]
        gbdt.feature_names_ = data.get("feature_names")
        gbdt.trees = data["trees"]
        gbdt._is_fitted = True
        if "n_classes" in data:
            gbdt.n_classes_ = data["n_classes"]
        if data.get("class_names"):
            gbdt.class_names_ = data["class_names"]
        return gbdt


def demo_regression():
    """GBDT regression on sin(x) with noise — visualise the stage-wise fit."""
    cfg = load_config("ml/basics/gbdt.yaml")
    set_seed(cfg["seed"])

    np.random.seed(cfg["seed"])
    X = np.linspace(0, 2 * np.pi, 300).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.15, 300)

    split = 240
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    gbdt = GBDT(
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        loss="mse",
        seed=cfg["seed"],
    )
    gbdt.fit(X_tr, y_tr)

    y_pred = gbdt.predict(X_te)
    mse_val = np.mean((y_pred - y_te) ** 2)

    print("GBDT — Regression (sin(x) + noise)")
    print("=" * 40)
    print(f"Trees: {cfg['n_estimators']}  LR: {cfg['learning_rate']}  Max depth: {cfg['max_depth']}")
    print(f"Train: {split}  Test: {len(y_te)}")
    print(f"Test MSE: {mse_val:.4f}")

    save_path = "runs/gbdt/regression_model.json"
    gbdt.save(save_path)
    print(f"Model saved to {save_path}")


def _logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def demo_classification():
    """GBDT binary classification on MNIST digits 3 vs 8, with a held-out split."""
    from ml.basics.data import load_mnist_pca

    cfg = load_config("ml/basics/gbdt.yaml")
    set_seed(cfg["seed"])

    negative, positive = cfg["digits"]
    X, labels, _ = load_mnist_pca(max_samples=cfg["max_samples"],
                                  n_components=cfg["n_components"])
    keep = (labels == negative) | (labels == positive)
    X, y = X[keep], (labels[keep] == positive).astype(np.int32)

    order = np.random.permutation(len(X))
    split = int(len(X) * cfg["train_fraction"])
    train_idx, test_idx = order[:split], order[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    gbdt = GBDT(
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        loss="logloss",
        seed=cfg["seed"],
        max_thresholds=cfg["max_thresholds"],
    )
    gbdt.fit(X_train, y_train)

    train_acc = float((gbdt.predict(X_train) == y_train).mean())
    proba = sigmoid(gbdt.predict_raw(X_test))
    test_acc = float(((proba > 0.5) == y_test).mean())
    baseline = float(max(y_test.mean(), 1.0 - y_test.mean()))

    print(f"GBDT — Binary Classification (MNIST {negative} vs {positive})")
    print("=" * 56)
    print(f"Trees: {cfg['n_estimators']}  LR: {cfg['learning_rate']}  Max depth: {cfg['max_depth']}")
    print(f"Features: {X.shape[1]} PCA components of {cfg['max_samples']} MNIST images")
    print(f"Train: {len(y_train)}  Test: {len(y_test)}  ")
    print(f"Train Accuracy : {train_acc:.1%}")
    print(f"Test Accuracy  : {test_acc:.1%}   (majority-class baseline {baseline:.1%})")
    print(f"Test LogLoss   : {_logloss(y_test, proba):.4f}")

    save_path = "runs/gbdt/classification_model.json"
    gbdt.save(save_path)
    print(f"Model saved to {save_path}")


def demo():
    demo_regression()
    print()
    demo_classification()


if __name__ == "__main__":
    demo()
