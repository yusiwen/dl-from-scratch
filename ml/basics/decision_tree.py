"""
Decision Tree (ID3 / CART-style) on the Iris dataset.

A decision tree recursively splits the data on feature thresholds to maximise
information gain at each node. The result is a set of interpretable if-else
rules that can be printed as a text tree.

Key concepts:
  - Entropy:  H(S) = -Σ p_i log₂(p_i)    (measure of impurity)
  - Information Gain:  IG = H(S) - Σ |S_v|/|S| · H(S_v)
  - Recursive partitioning: split until purity or max_depth
  - Pruning: limiting depth prevents overfitting

Unlike the gradient-based models in this project, decision trees have no
weights, no learning rate, and no backprop — they just partition the data.
"""

import json
import os

import numpy as np

from utils.config import load_config
from utils.seed import set_seed


# ─────────────────────────────────────────────────────────
# Iris dataset  (150 samples, 4 features, 3 classes)
# Hard-coded so no external dependency is needed.
# Features: sepal length, sepal width, petal length, petal width (all in cm).
# Classes: 0 = setosa, 1 = versicolor, 2 = virginica.
# ─────────────────────────────────────────────────────────

IRIS_DATA = np.array([
    [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4],
    [4.6, 3.4, 1.4, 0.3], [5.0, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2],
    [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
    [4.8, 3.0, 1.4, 0.1], [4.3, 3.0, 1.1, 0.1], [5.8, 4.0, 1.2, 0.2],
    [5.7, 4.4, 1.5, 0.4], [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3],
    [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3], [5.4, 3.4, 1.7, 0.2],
    [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1.0, 0.2], [5.1, 3.3, 1.7, 0.5],
    [4.8, 3.4, 1.9, 0.2], [5.0, 3.0, 1.6, 0.2], [5.0, 3.4, 1.6, 0.4],
    [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2],
    [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4], [5.2, 4.1, 1.5, 0.1],
    [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.2], [5.0, 3.2, 1.2, 0.2],
    [5.5, 3.5, 1.3, 0.2], [4.9, 3.6, 1.4, 0.1], [4.4, 3.0, 1.3, 0.2],
    [5.1, 3.4, 1.5, 0.2], [5.0, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3],
    [4.4, 3.2, 1.3, 0.2], [5.0, 3.5, 1.6, 0.6], [5.1, 3.8, 1.9, 0.4],
    [4.8, 3.0, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2],
    [5.3, 3.7, 1.5, 0.2], [5.0, 3.3, 1.4, 0.2],
    [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5],
    [5.5, 2.3, 4.0, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
    [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1.0], [6.6, 2.9, 4.6, 1.3],
    [5.2, 2.7, 3.9, 1.4], [5.0, 2.0, 3.5, 1.0], [5.9, 3.0, 4.2, 1.5],
    [6.0, 2.2, 4.0, 1.0], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3],
    [6.7, 3.1, 4.4, 1.4], [5.6, 3.0, 4.5, 1.5], [5.8, 2.7, 4.1, 1.0],
    [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1], [5.9, 3.2, 4.8, 1.8],
    [6.1, 2.8, 4.0, 1.3], [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2],
    [6.4, 2.9, 4.3, 1.3], [6.6, 3.0, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4],
    [6.7, 3.0, 5.0, 1.7], [6.0, 2.9, 4.5, 1.5], [5.7, 2.6, 3.5, 1.0],
    [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1.0], [5.8, 2.7, 3.9, 1.2],
    [6.0, 2.7, 5.1, 1.6], [5.4, 3.0, 4.5, 1.5], [6.0, 3.4, 4.5, 1.6],
    [6.7, 3.1, 4.7, 1.5], [6.3, 2.3, 4.4, 1.3], [5.6, 3.0, 4.1, 1.3],
    [5.5, 2.5, 4.0, 1.3], [5.5, 2.6, 4.4, 1.2], [6.1, 3.0, 4.6, 1.4],
    [5.8, 2.6, 4.0, 1.2], [5.0, 2.3, 3.3, 1.0], [5.6, 2.7, 4.2, 1.3],
    [5.7, 3.0, 4.2, 1.2], [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3],
    [5.1, 2.5, 3.0, 1.1], [5.7, 2.8, 4.1, 1.3],
    [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1],
    [6.3, 2.9, 5.6, 1.8], [6.5, 3.0, 5.8, 2.2], [7.6, 3.0, 6.6, 2.1],
    [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8], [6.7, 2.5, 5.8, 1.8],
    [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2.0], [6.4, 2.7, 5.3, 1.9],
    [6.8, 3.0, 5.5, 2.1], [5.7, 2.5, 5.0, 2.0], [5.8, 2.8, 5.1, 2.4],
    [6.4, 3.2, 5.3, 2.3], [6.5, 3.0, 5.5, 1.8], [7.7, 3.8, 6.7, 2.2],
    [7.7, 2.6, 6.9, 2.3], [6.0, 2.2, 5.0, 1.5], [6.9, 3.2, 5.7, 2.3],
    [5.6, 2.8, 4.9, 2.0], [7.7, 2.8, 6.7, 2.0], [6.3, 2.7, 4.9, 1.8],
    [6.7, 3.3, 5.7, 2.1], [7.2, 3.2, 6.0, 1.8], [6.2, 2.8, 4.8, 1.8],
    [6.1, 3.0, 4.9, 1.8], [6.4, 2.8, 5.6, 2.1], [7.2, 3.0, 5.8, 1.6],
    [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2.0], [6.4, 2.8, 5.6, 2.2],
    [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4], [7.7, 3.0, 6.1, 2.3],
    [6.3, 3.4, 5.6, 2.4], [6.4, 3.1, 5.5, 1.8], [6.0, 3.0, 4.8, 1.8],
    [6.9, 3.1, 5.4, 2.1], [6.7, 3.1, 5.6, 2.4], [6.9, 3.1, 5.1, 2.3],
    [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3], [6.7, 3.3, 5.7, 2.5],
    [6.7, 3.0, 5.2, 2.3], [6.3, 2.5, 5.0, 1.9], [6.5, 3.0, 5.2, 2.0],
    [6.2, 3.4, 5.4, 2.3], [5.9, 3.0, 5.1, 1.8],
], dtype=np.float64)

IRIS_LABELS = np.array(
    [0]*50 + [1]*50 + [2]*50, dtype=np.int32
)

IRIS_FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
IRIS_CLASS_NAMES = ["setosa", "versicolor", "virginica"]


# ─────────────────────────────────────────────────────────
#  Decision Tree
# ─────────────────────────────────────────────────────────

def entropy(y):
    """
    Shannon entropy: H = -Σ p_i log₂(p_i)

    Measures the impurity of a set of labels.
    H=0 when all labels are the same (pure).
    H=max when labels are uniformly distributed.
    """
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-12))


def information_gain(y, y_left, y_right):
    """
    Information Gain = H(parent) - weighted_average(H(left), H(right))

    Measures how much entropy is reduced by a binary split.
    We choose the split that maximises this value.
    """
    n = len(y)
    return entropy(y) - (len(y_left) / n * entropy(y_left) + len(y_right) / n * entropy(y_right))


def mse(y):
    """Mean squared error — used as impurity measure for regression trees."""
    return float(np.mean((y - np.mean(y)) ** 2))


def variance_reduction(y, y_left, y_right):
    """Variance reduction = MSE(parent) - weighted_avg(MSE(left), MSE(right))."""
    n = len(y)
    return mse(y) - (len(y_left) / n * mse(y_left) + len(y_right) / n * mse(y_right))


def best_split(X, y, criterion='entropy', feature_subset=None, max_thresholds=None):
    """
    Find the feature and threshold that give the highest gain.

    For each feature, try every pair of adjacent sorted values as a candidate
    threshold. Return the (feature_index, threshold) with the highest gain.

    criterion: 'entropy' → information_gain, 'mse' → variance_reduction.
    feature_subset: list of feature indices to consider (None = all features).
    max_thresholds: when set, evaluate that many quantile cuts per feature
        instead of every adjacent pair. The exact scan is O(m·n) per node, which
        is fine for Iris and hopeless for anything MNIST-sized; boosting on real
        data needs the quantile version. None (the default) keeps the exact scan.
    """
    n, m = X.shape
    gain_fn = information_gain if criterion == 'entropy' else variance_reduction
    best_gain = 0
    best_feat = None
    best_thresh = None

    features = feature_subset if feature_subset is not None else range(m)

    for feat in features:
        values = X[:, feat]

        if max_thresholds is not None:
            # Quantile cuts: the candidate set real GBDT implementations use.
            candidates = np.unique(
                np.quantile(values, np.linspace(0.0, 1.0, max_thresholds + 2)[1:-1])
            )
        else:
            ordered = np.sort(values)
            midpoints = (ordered[1:] + ordered[:-1]) / 2.0
            candidates = midpoints[ordered[1:] != ordered[:-1]]

        for thresh in candidates:
            left_mask = values <= thresh
            y_left = y[left_mask]
            y_right = y[~left_mask]
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            gain = gain_fn(y, y_left, y_right)
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh

    return best_feat, best_thresh, best_gain


def build_tree(X, y, depth=0, max_depth=5, min_samples=2, criterion='entropy',
               feature_subset=None, max_thresholds=None):
    """
    Recursively build a binary decision tree.

    Parameters
    ----------
    criterion : 'entropy' (classification) or 'mse' (regression)
        Leaf values are int class labels for entropy or float means for mse.
    feature_subset : list of feature indices or None
        If set, only consider these features at each split (used by Random Forest).
    max_thresholds : int or None
        Quantile cuts per feature instead of the exact adjacent-pair scan; see
        `best_split`.

    Returns a node dict:
      Leaf:   {'is_leaf': True, 'value': int|float, 'size': int, 'depth': int}
      Internal: {'is_leaf': False, 'feature': int, 'threshold': float,
                 'gain': float, 'left': node, 'right': node, 'size': int, 'depth': int}
    """
    n = len(y)

    def leaf_value(y):
        if criterion == 'entropy':
            return int(np.bincount(y).argmax())
        return float(np.mean(y))

    if criterion == 'entropy' and len(np.unique(y)) == 1:
        return {'is_leaf': True, 'value': leaf_value(y), 'size': n, 'depth': depth}

    if depth >= max_depth or n < min_samples:
        return {'is_leaf': True, 'value': leaf_value(y), 'size': n, 'depth': depth}

    feat, thresh, gain = best_split(X, y, criterion, feature_subset,
                                     max_thresholds=max_thresholds)

    if feat is None or gain < 1e-10:
        return {'is_leaf': True, 'value': leaf_value(y), 'size': n, 'depth': depth}

    left_mask = X[:, feat] <= thresh
    node = {
        'is_leaf': False,
        'feature': int(feat),
        'threshold': float(thresh),
        'gain': float(gain),
        'size': n,
        'depth': depth,
        'left': build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples,
                                 criterion, feature_subset, max_thresholds),
        'right': build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth, min_samples,
                                  criterion, feature_subset, max_thresholds),
    }
    return node


def predict_sample(tree, x):
    """Traverse the tree to predict the value for a single sample."""
    while not tree['is_leaf']:
        if x[tree['feature']] <= tree['threshold']:
            tree = tree['left']
        else:
            tree = tree['right']
    return tree['value']


def predict(tree, X):
    """Predict labels for all samples in X."""
    return np.array([predict_sample(tree, x) for x in X])


def print_tree(tree, feature_names=None, class_names=None, indent=""):
    """
    Print the decision tree as readable ASCII text.

    Example:
      petal_width <= 0.80 [ig=0.918]
      ├── yes → setosa (50)
      └── no
          └── petal_width <= 1.75 [ig=0.580]
              ├── yes → versicolor (50)
              └── no  → virginica (46)
    """
    fnames = feature_names or [f"feat_{i}" for i in range(4)]

    if tree['is_leaf']:
        v = tree['value']
        if class_names and isinstance(v, int) and 0 <= v < len(class_names):
            label = class_names[v]
        elif isinstance(v, float):
            label = f"{v:.4f}"
        else:
            label = str(v)
        print(f"→ {label} ({tree['size']})")
    else:
        fname = fnames[tree['feature']]
        gain_label = "ig" if tree.get('gain', 0) > 0 else "gain"
        print(f"{fname} <= {tree['threshold']:.2f}  [{gain_label}={tree.get('gain', 0):.3f}]")
        print(f"{indent}├── yes ", end="")
        new_indent = indent + "│   "
        print_tree(tree['left'], fnames, class_names, new_indent)
        print(f"{indent}└── no  ", end="")
        new_indent = indent + "    "
        print_tree(tree['right'], fnames, class_names, new_indent)


def save_tree(tree, path, metadata=None):
    """Save a tree dict to a JSON file (numpy-safe, human-readable)."""
    data = {
        "format_version": 1,
        "model_type": "DecisionTree",
        "tree": tree,
    }
    if metadata:
        data.update(metadata)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_tree(path):
    """Load a tree dict from a JSON file saved by save_tree()."""
    with open(path) as f:
        data = json.load(f)
    return data["tree"]


# ─────────────────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────────────────

def demo():
    cfg = load_config("ml/basics/decision_tree.yaml")
    set_seed(cfg["seed"])
    X, y = IRIS_DATA, IRIS_LABELS

    # Shuffle and split 80/20.
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = 120
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    print("Decision Tree — Iris Dataset")
    print("=" * 35)
    print(f"Features ({len(IRIS_FEATURE_NAMES)}): {', '.join(IRIS_FEATURE_NAMES)}")
    print(f"Classes: {IRIS_CLASS_NAMES}")
    print(f"Train: {split}  Test: {len(y_te)}")
    print()

    tree = build_tree(X_tr, y_tr, max_depth=cfg["max_depth"], min_samples=cfg["min_samples"], criterion=cfg.get("criterion", "entropy"))

    print("Trained tree:")
    print_tree(tree, IRIS_FEATURE_NAMES, IRIS_CLASS_NAMES)
    print()

    y_pred = predict(tree, X_te)
    acc = (y_pred == y_te).mean()
    print(f"Test Accuracy: {acc:.1%}")

    save_path = "runs/decision_tree/model.json"
    save_tree(tree, save_path, metadata={
        "config": dict(cfg),
        "n_features": X.shape[1],
        "n_classes": 3,
        "feature_names": IRIS_FEATURE_NAMES,
        "class_names": IRIS_CLASS_NAMES,
    })
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    demo()
