"""
Random Forest — ensemble of decision trees trained on bootstrapped data
with random feature subsets, then aggregated by majority vote (classification)
or mean (regression).

Key concepts:
  - Bagging (Bootstrap Aggregating): each tree sees a different bootstrap sample
  - Feature randomness: at each node only a random subset of features is considered
  - Diversity → reduced variance without increasing bias
"""

import json
import os

import numpy as np

from utils.config import load_config
from utils.seed import set_seed

from ml.basics.decision_tree import build_tree, predict


class RandomForest:
    """Random Forest classifier/regressor."""

    def __init__(self, n_estimators=100, max_depth=5, min_samples=2,
                 max_features='sqrt', bootstrap=True, seed=42, max_thresholds=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.seed = seed
        self.max_thresholds = max_thresholds
        self.trees = []
        self.feature_names_ = None
        self.class_names_ = None

    def fit(self, X, y):
        set_seed(self.seed)
        n, m = X.shape
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = m

        self.feature_subsets_ = []

        for i in range(self.n_estimators):
            if self.bootstrap:
                idx = np.random.choice(n, n, replace=True)
                X_boot, y_boot = X[idx], y[idx]
            else:
                X_boot, y_boot = X, y

            subset = self._pick_features(m)
            self.feature_subsets_.append(subset)

            tree = build_tree(X_boot, y_boot, max_depth=self.max_depth,
                              min_samples=self.min_samples, criterion='entropy',
                              feature_subset=subset, max_thresholds=self.max_thresholds)
            self.trees.append(tree)

        return self

    def _pick_features(self, n_total):
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                k = max(1, int(np.sqrt(n_total)))
            elif self.max_features == 'log2':
                k = max(1, int(np.log2(n_total)))
            else:
                k = n_total
        else:
            k = max(1, min(int(self.max_features), n_total))
        if k >= n_total:
            return None
        return sorted(np.random.choice(n_total, k, replace=False).tolist())

    def predict(self, X):
        preds = np.array([predict(t, X) for t in self.trees])
        if self.n_classes_ > 1:
            return np.array([np.bincount(preds[:, i]).argmax() for i in range(X.shape[0])])
        return preds.mean(axis=0)

    def save(self, path):
        data = {
            "format_version": 1,
            "model_type": "RandomForest",
            "config": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples": self.min_samples,
                "max_features": self.max_features,
                "bootstrap": self.bootstrap,
                "seed": self.seed,
            },
            "n_features": self.n_features_,
            "n_classes": self.n_classes_,
            "feature_names": self.feature_names_,
            "class_names": self.class_names_,
            "trees": self.trees,
            "feature_subsets": self.feature_subsets_,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        rf = cls(**data["config"])
        rf.n_features_ = data["n_features"]
        rf.n_classes_ = data["n_classes"]
        rf.feature_names_ = data.get("feature_names")
        rf.class_names_ = data.get("class_names")
        rf.trees = data["trees"]
        rf.feature_subsets_ = data.get("feature_subsets", [None] * len(rf.trees))
        return rf


def demo():
    cfg = load_config("ml/basics/random_forest.yaml")
    # Seed before the train/test split: fit() seeds again for the forest, but the
    # permutation below runs first, so without this every run used a different
    # split (100.0%, 96.7% and 93.3% on the same code).
    set_seed(cfg["seed"])
    from ml.basics.decision_tree import IRIS_DATA, IRIS_LABELS, IRIS_FEATURE_NAMES, IRIS_CLASS_NAMES

    X, y = IRIS_DATA, IRIS_LABELS
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = 120
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    rf = RandomForest(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples=cfg["min_samples"],
        max_features=cfg["max_features"],
        bootstrap=cfg["bootstrap"],
        seed=cfg["seed"],
    )
    rf.feature_names_ = IRIS_FEATURE_NAMES
    rf.class_names_ = IRIS_CLASS_NAMES
    rf.fit(X_tr, y_tr)

    y_pred = rf.predict(X_te)
    acc = (y_pred == y_te).mean()

    print("Random Forest — Iris Dataset")
    print("=" * 35)
    print(f"Trees: {cfg['n_estimators']}  Max features: {cfg['max_features']}  Bootstrap: {cfg['bootstrap']}")
    print(f"Train: {split}  Test: {len(y_te)}")
    print(f"Test Accuracy: {acc:.1%}")

    save_path = "runs/random_forest/model.json"
    rf.save(save_path)
    print(f"Model saved to {save_path}")

    from ml.basics.decision_tree import save_tree, load_tree
    single_tree = build_tree(X_tr, y_tr, max_depth=cfg["max_depth"], min_samples=cfg["min_samples"])
    single_pred = predict(single_tree, X_te)
    single_acc = (single_pred == y_te).mean()
    print(f"Single tree baseline: {single_acc:.1%}")
    delta = acc - single_acc
    print(f"RF vs single tree: {delta:+.1%}")


if __name__ == "__main__":
    demo()
