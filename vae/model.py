import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Image → μ, logσ² in latent space."""

    def __init__(self, latent_dim=100, feature_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, feature_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
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
        )
        self.fc_mu = nn.Linear(feature_dim * 8 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim * 8 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    """Latent z → image."""

    def __init__(self, latent_dim=100, feature_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, feature_dim * 8 * 4 * 4)
        self.net = nn.Sequential(
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
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        return self.net(x)


class VAE(nn.Module):
    """Variational Autoencoder with reparameterization trick."""

    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def reparameterize(self, mu, logvar):
        """z = μ + ε * σ, where ε ~ N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def generate(self, num_samples, device):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            return self.decoder(z)

    def interpolate(self, z1, z2, steps=8):
        """Linear interpolation between two latent codes."""
        alphas = torch.linspace(0, 1, steps, device=z1.device)
        interp = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])
        return self.decoder(interp)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
