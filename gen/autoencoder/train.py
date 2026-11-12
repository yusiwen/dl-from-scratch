"""
Train a reconstruction-only autoencoder on MNIST.

The loss is plain BCE between the input and its reconstruction -- no KL term,
no sampling. That is exactly the gap between the PCA demo (linear projection)
and the VAE (probabilistic latent): this model shows what an encoder/decoder
pair learns without any probabilistic machinery, and why the VAE adds one.

Reconstruction is compared with PCA at the same bottleneck width on the same
5,000 images (utils/data.py), so the two numbers are directly comparable.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from gen.autoencoder.data import load_mnist_tensors
from gen.autoencoder.model import Autoencoder
from ml.basics.pca import pca, reconstruct
from utils.config import load_config, save_config
from utils.data import load_mnist_flat, subset_checksum
from utils.device import get_device
from utils.seed import set_seed


def reconstruction_mse(model, images, device):
    model.eval()
    with torch.no_grad():
        recon = model(images.to(device)).cpu()
    return torch.mean((recon - images) ** 2).item()


def pca_reconstruction_mse(images, n_components):
    projection, components, mean, _ = pca(images.numpy(), n_components=n_components)
    recon = reconstruct(projection, components, mean)
    return float(((recon - images.numpy()) ** 2).mean())


def train():
    cfg = load_config("gen/autoencoder/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")

    images, labels = load_mnist_tensors(cfg["max_samples"])
    assert images.shape == (cfg["max_samples"], 784)
    fresh, _ = load_mnist_flat(cfg["max_samples"])
    # float64 -> float32 is a lossy cast, so compare at the storage precision.
    assert np.array_equal(images.numpy(), fresh.astype(np.float32)), (
        "MNIST subset changed between loads"
    )
    checksum = subset_checksum(fresh)
    print(f"Data: {tuple(images.shape)}, subset sha256: {checksum[:16]}")

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    model = Autoencoder(latent_dim=cfg["latent_dim"]).to(device)
    print(f"Parameters: {model.num_params():,} (latent_dim={cfg['latent_dim']})")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    writer = SummaryWriter(log_dir="runs/autoencoder")

    num_epochs = cfg["num_epochs"]
    sample_interval = cfg.get("sample_interval", 5)
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_images, _ in loader:
            batch_images = batch_images.to(device)
            recon = model(batch_images)
            loss = criterion(recon, batch_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        writer.add_scalar("train/bce", avg_loss, epoch)
        print(f"Epoch [{epoch:2d}/{num_epochs}]  BCE: {avg_loss:.4f}")

        if epoch % sample_interval == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                samples = model.decode(torch.randn(64, cfg["latent_dim"], device=device))
            writer.add_images("reconstructions", samples.cpu().view(-1, 1, 28, 28), epoch)

    writer.close()

    ae_mse = reconstruction_mse(model, images, device)
    pca_mse = pca_reconstruction_mse(images, cfg["latent_dim"])
    print(f"\nReconstruction MSE on the same {len(images):,} images:")
    print(f"  AE  (latent {cfg['latent_dim']:2d}) {ae_mse:.5f}")
    print(f"  PCA ({cfg['latent_dim']:2d} components) {pca_mse:.5f}")

    save_path = cfg["model_path"]
    torch.save(model.state_dict(), save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()
