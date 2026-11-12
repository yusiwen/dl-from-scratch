"""
Deep Embedded Clustering (DEC, Xie et al. 2016) on top of the MLP autoencoder.

DEC keeps the autoencoder's reconstruction objective and adds a KL term that
pulls latent codes towards trainable cluster centres:

  q_ij = (1 + ||z_i - mu_j||^2)^-1 / sum_k (...)      Student-t soft assignment
  p_ij = q_ij^2 / f_j / sum_k (q_ik^2 / f_k)          sharpened target
  loss = BCE(x, x_hat) + gamma * KL(P || Q)

The centres are initialized with K-Means on the latent space, so the
comparison is "same k, same data, better objective".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_assign(latent, centers):
    """Student-t responsibilities, shape (n, k)."""
    distances = torch.cdist(latent, centers) ** 2
    q = 1.0 / (1.0 + distances)
    return q / q.sum(dim=1, keepdim=True)


def target_distribution(q):
    weight = q ** 2 / q.sum(dim=0)
    return (weight.t() / weight.sum(dim=1)).t()


class DeepEmbeddedClustering(nn.Module):
    def __init__(self, autoencoder, n_clusters, latent_dim):
        super().__init__()
        self.autoencoder = autoencoder
        self.centers = nn.Parameter(torch.zeros(n_clusters, latent_dim))

    def initialize_centers(self, centers):
        with torch.no_grad():
            self.centers.copy_(centers)

    def forward(self, images):
        latent = self.autoencoder.encode(images)
        reconstructed = self.autoencoder.decode(latent)
        q = soft_assign(latent, self.centers)
        return latent, reconstructed, q

    @staticmethod
    def kl_loss(q, p):
        return torch.sum(p * torch.log((p + 1e-12) / (q + 1e-12)), dim=1).mean()

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
