"""Generate a grid of fake images from a trained DCGAN."""

import torch
import torchvision.utils as vutils
from PIL import Image

from gen.dcgan.model import Generator
from utils.config import load_config
from utils.device import get_device


def generate():
    cfg = load_config("gen/dcgan/config.yaml")

    device = get_device()
    latent_dim = cfg["latent_dim"]

    netG = Generator(latent_dim=latent_dim).to(device)
    checkpoint = torch.load(cfg["model_path"], map_location=device, weights_only=True)
    netG.load_state_dict(checkpoint["G"])
    netG.eval()
    print(f"Loaded Generator from {cfg['model_path']}")

    with torch.no_grad():
        noise = torch.randn(64, latent_dim, 1, 1, device=device)
        fake = netG(noise).cpu()
    fake = (fake + 1) / 2  # denormalize [-1, 1] → [0, 1]

    grid = vutils.make_grid(fake, nrow=8, padding=2, normalize=False)
    grid_img = grid.mul(255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    Image.fromarray(grid_img).save("gen/dcgan/samples.png")
    print("Saved grid to gen/dcgan/samples.png")


if __name__ == "__main__":
    generate()
