import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Graph Convolution: H' = σ(Â @ H @ W)"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, out_features) * 0.01)

    def forward(self, x, adj_norm):
        # x: (N, in_features), adj_norm: (N, N)
        return adj_norm @ x @ self.W


class GCN(nn.Module):
    """2-layer Graph Convolutional Network (Kipf & Welling, 2017)."""

    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GraphConv(in_features, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        x = self.conv1(x, adj_norm)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, adj_norm)
        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
