"""Cora citation network dataset loader."""

import os
import urllib.request
import numpy as np
import torch


CORA_URL = "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/"


def _download(raw_dir="gcn/raw"):
    os.makedirs(raw_dir, exist_ok=True)
    for fname in ["cora.content", "cora.cites"]:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(CORA_URL + fname, path)


def _encode_onehot(labels):
    classes = sorted(set(labels))
    mapping = {c: i for i, c in enumerate(classes)}
    return torch.eye(len(classes))[torch.tensor([mapping[l] for l in labels])], mapping


def load_cora(raw_dir="gcn/raw"):
    _download(raw_dir)

    # Parse content: node_id features label
    content = np.genfromtxt(os.path.join(raw_dir, "cora.content"), dtype=np.dtype(str))
    node_ids = content[:, 0].astype(int)
    features = torch.tensor(content[:, 1:-1].astype(np.float32))
    labels_onehot, class_mapping = _encode_onehot(content[:, -1])
    labels = labels_onehot.argmax(dim=1)

    n = len(node_ids)
    id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}

    # Parse cites: source target
    cites = np.genfromtxt(os.path.join(raw_dir, "cora.cites"), dtype=np.int32)
    adj = torch.zeros((n, n), dtype=torch.float32)
    for src, tgt in cites:
        if src in id_to_idx and tgt in id_to_idx:
            adj[id_to_idx[src], id_to_idx[tgt]] = 1.0
            adj[id_to_idx[tgt], id_to_idx[src]] = 1.0  # undirected

    # Normalize: D^{-1/2} @ A @ D^{-1/2}
    rowsum = adj.sum(dim=1).clamp(min=1e-8)
    d_inv_sqrt = torch.diag(rowsum ** -0.5)
    adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt

    # Standard splits (20 nodes/class for train, 500 val, 1000 test).
    n_classes = len(class_mapping)
    idx_per_class = [torch.where(labels == c)[0] for c in range(n_classes)]

    train_idx = torch.cat([idx[:20] for idx in idx_per_class])
    rest = torch.cat([idx[20:] for idx in idx_per_class])
    val_idx = rest[:500]
    test_idx = rest[500:1500]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"Cora: {n} nodes, {features.size(1)} features, {n_classes} classes")
    print(f"  Train: {train_idx.size(0)}  Val: {val_idx.size(0)}  Test: {test_idx.size(0)}")

    return features, adj_norm, labels, train_mask, val_mask, test_mask, class_mapping
