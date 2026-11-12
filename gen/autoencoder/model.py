"""Symmetric MLP autoencoder for flattened 28x28 MNIST images."""

import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Encoder 784 -> hidden... -> latent_dim, decoder mirrored, Sigmoid output.

    No probabilistic latent and no KL term: this is the reconstruction-only
    model that sits between PCA and the VAE.
    """

    def __init__(self, latent_dim=32, hidden_dims=(256, 128), input_dim=784):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        encoder = []
        last = input_dim
        for hidden in hidden_dims:
            encoder += [nn.Linear(last, hidden), nn.ReLU()]
            last = hidden
        encoder.append(nn.Linear(last, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        last = latent_dim
        for hidden in reversed(hidden_dims):
            decoder += [nn.Linear(last, hidden), nn.ReLU()]
            last = hidden
        decoder.append(nn.Linear(last, input_dim))
        self.decoder = nn.Sequential(*decoder, nn.Sigmoid())

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
