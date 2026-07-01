import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vae.model import VAE
from vae.data import load_celeba
from utils.config import load_config, save_config
from utils.seed import set_seed
from utils.device import get_device


def vae_loss(recon, x, mu, logvar):
    """VAE loss: reconstruction BCE + KL divergence."""
    recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
    # KL divergence: KL(N(μ,σ²) ∥ N(0,1)) = ½ Σ(μ² + σ² - log(σ²) - 1)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl_loss


def train():
    cfg = load_config("vae/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")

    dataset = load_celeba(cfg["num_samples"], cfg["image_size"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    print(f"Dataset: {len(dataset):,} images")

    model = VAE(latent_dim=cfg["latent_dim"]).to(device)
    print(f"Parameters: {model.num_params():,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    num_epochs = cfg["num_epochs"]
    writer = SummaryWriter(log_dir="runs/vae")
    sample_interval = cfg.get("sample_interval", 5)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_recon = 0.0
        total_kl = 0.0

        for x in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)
            loss = recon_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        avg_recon = total_recon / len(dataset)
        avg_kl = total_kl / len(dataset)
        avg_loss = avg_recon + avg_kl

        writer.add_scalar("loss/total", avg_loss, epoch)
        writer.add_scalar("loss/recon", avg_recon, epoch)
        writer.add_scalar("loss/kl", avg_kl, epoch)

        print(f"Epoch [{epoch:2d}/{num_epochs}]  "
              f"Loss: {avg_loss:.0f}  Recon: {avg_recon:.0f}  "
              f"KL: {avg_kl:.2f}")

        if epoch % sample_interval == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                samples = model.generate(64, device).cpu()
            writer.add_images("generated", samples, epoch)

    writer.close()
    save_path = cfg["model_path"]
    torch.save(model.state_dict(), save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()
