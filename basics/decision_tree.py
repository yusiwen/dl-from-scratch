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

import numpy as np


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


def best_split(X, y):
    """
    Find the feature and threshold that give the highest information gain.

    For each feature, try every pair of adjacent sorted values as a candidate
    threshold. Return the (feature_index, threshold) with the highest IG.
    """
    n, m = X.shape
    best_ig = 0
    best_feat = None
    best_thresh = None

    for feat in range(m):
        # Sort values to find candidate thresholds (midpoints between unique values).
        sorted_idx = np.argsort(X[:, feat])
        X_sorted = X[sorted_idx, feat]
        y_sorted = y[sorted_idx]

        for i in range(1, n):
            if X_sorted[i] == X_sorted[i - 1]:
                continue
            thresh = (X_sorted[i] + X_sorted[i - 1]) / 2.0
            left_mask = X[:, feat] <= thresh
            y_left = y[left_mask]
            y_right = y[~left_mask]
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            ig = information_gain(y, y_left, y_right)
            if ig > best_ig:
                best_ig = ig
                best_feat = feat
                best_thresh = thresh

    return best_feat, best_thresh, best_ig


def build_tree(X, y, depth=0, max_depth=5, min_samples=2):
    """
    Recursively build a binary decision tree.

    Stopping criteria (to prevent overfitting):
      - max_depth: limits tree size
      - min_samples: stops splitting when a node has too few samples
      - pure node: all samples have the same label

    Returns a node dict:
      {'is_leaf': True, 'label': int, 'size': int, 'depth': int}
      or
      {'is_leaf': False, 'feature': int, 'threshold': float,
       'left': node, 'right': node, 'gain': float, 'size': int, 'depth': int}
    """
    n = len(y)
    unique = np.unique(y)

    # Stopping conditions.
    if len(unique) == 1 or depth >= max_depth or n < min_samples:
        return {'is_leaf': True, 'label': np.bincount(y).argmax(), 'size': n, 'depth': depth}

    feat, thresh, ig = best_split(X, y)

    if feat is None or ig < 1e-10:
        return {'is_leaf': True, 'label': np.bincount(y).argmax(), 'size': n, 'depth': depth}

    left_mask = X[:, feat] <= thresh
    node = {
        'is_leaf': False,
        'feature': feat,
        'threshold': thresh,
        'gain': ig,
        'size': n,
        'depth': depth,
        'left': build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples),
        'right': build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth, min_samples),
    }
    return node


def predict_sample(tree, x):
    """Traverse the tree to predict the label for a single sample."""
    while not tree['is_leaf']:
        if x[tree['feature']] <= tree['threshold']:
            tree = tree['left']
        else:
            tree = tree['right']
    return tree['label']


def predict(tree, X):
    """Predict labels for all samples in X."""
    return np.array([predict_sample(tree, x) for x in X])


def print_tree(tree, feature_names=None, indent=""):
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
    cnames = IRIS_CLASS_NAMES

    if tree['is_leaf']:
        label_name = cnames[tree['label']] if cnames else str(tree['label'])
        print(f"→ {label_name} ({tree['size']})")
    else:
        fname = fnames[tree['feature']]
        print(f"{fname} <= {tree['threshold']:.2f}  [ig={tree['gain']:.3f}]")
        print(f"{indent}├── yes ", end="")
        new_indent = indent + "│   "
        print_tree(tree['left'], fnames, new_indent)
        print(f"{indent}└── no  ", end="")
        new_indent = indent + "    "
        print_tree(tree['right'], fnames, new_indent)


# ─────────────────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────────────────

def demo():
    np.random.seed(42)
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

    tree = build_tree(X_tr, y_tr, max_depth=4, min_samples=3)

    print("Trained tree:")
    print_tree(tree, IRIS_FEATURE_NAMES)
    print()

    y_pred = predict(tree, X_te)
    acc = (y_pred == y_te).mean()
    print(f"Test Accuracy: {acc:.1%}")


if __name__ == "__main__":
    demo()
