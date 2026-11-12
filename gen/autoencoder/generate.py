"""Sample the autoencoder and, when the latent is 2-D, plot it against digits."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from PIL import Image

from gen.autoencoder.model import Autoencoder
from utils.config import load_config
from utils.device import get_device

COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
          "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4"]


def generate():
    cfg = load_config("gen/autoencoder/config.yaml")
    device = get_device()

    model = Autoencoder(latent_dim=cfg["latent_dim"])
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Loaded model from {cfg['model_path']}")

    with torch.no_grad():
        samples = model.decode(torch.randn(64, cfg["latent_dim"], device=device)).cpu()
    grid = vutils.make_grid(samples.view(-1, 1, 28, 28), nrow=8, padding=2)
    grid_img = grid.mul(255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    Image.fromarray(grid_img).save("gen/autoencoder/samples.png")
    print("Saved samples to gen/autoencoder/samples.png")

    if cfg["latent_dim"] != 2:
        print(f"latent_dim={cfg['latent_dim']}: set latent_dim: 2 and retrain for the 2-D latent plot")
        return

    from gen.autoencoder.data import load_mnist_tensors

    images, labels = load_mnist_tensors(cfg["max_samples"])
    with torch.no_grad():
        codes = model.encode(images.to(device)).cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    for digit in range(10):
        mask = labels.numpy() == digit
        ax.scatter(codes[mask, 0], codes[mask, 1], s=6, alpha=0.6,
                   c=COLORS[digit], label=str(digit))
    ax.set_title("Autoencoder latent space (2-D)")
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.legend(loc="best", markerscale=2)
    fig.tight_layout()
    fig.savefig("gen/autoencoder/latent_2d.png", dpi=150)
    print("Saved latent plot to gen/autoencoder/latent_2d.png")


if __name__ == "__main__":
    generate()
