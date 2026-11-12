"""Frozen-feature probes shared by the self-supervised methods.

Both probes never update the encoder: features are extracted once, then
  linear probe  logistic regression (the standard SimCLR/MAE protocol)
  kNN           cosine nearest-neighbour vote, no training
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def linear_probe(train_features, train_labels, test_features, test_labels, max_iter=2000):
    """Fit logistic regression on frozen features; return (accuracy, classifier)."""
    classifier = LogisticRegression(max_iter=max_iter)
    classifier.fit(train_features, train_labels)
    accuracy = float((classifier.predict(test_features) == test_labels).mean())
    return accuracy, classifier


def knn_accuracy(train_features, train_labels, test_features, test_labels, k=20, chunk=1000):
    """Cosine kNN accuracy, chunked to bound the (chunk, n_train) buffer."""
    train = train_features / np.maximum(
        np.linalg.norm(train_features, axis=1, keepdims=True), 1e-12
    )
    test = test_features / np.maximum(
        np.linalg.norm(test_features, axis=1, keepdims=True), 1e-12
    )
    n_classes = int(train_labels.max()) + 1

    correct = 0
    for start in range(0, len(test), chunk):
        similarities = test[start:start + chunk] @ train.T
        neighbours = np.argpartition(-similarities, k, axis=1)[:, :k]
        for row, neighbour_indices in enumerate(neighbours):
            votes = np.bincount(train_labels[neighbour_indices], minlength=n_classes)
            correct += int(votes.argmax() == test_labels[start + row])
    return correct / len(test)
