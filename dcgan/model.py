import torch
import torch.nn as nn


def _weights_init(m):
    """DCGAN paper: Normal(0, 0.02) for all weights."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, mean=1.0, std=0.02)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    """DCGAN Generator: latent z → 64×64×3 image."""

    def __init__(self, latent_dim=100, feature_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_dim, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(_weights_init)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """DCGAN Discriminator: 64×64×3 image → probability (real/fake)."""

    def __init__(self, feature_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, feature_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.apply(_weights_init)

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)
