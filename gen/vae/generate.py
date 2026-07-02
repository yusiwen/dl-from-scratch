"""VAE inference: generate samples and interpolate in latent space."""

import torch
import torchvision.utils as vutils
from PIL import Image

from gen.vae.model import VAE
from utils.config import load_config


def generate():
    cfg = load_config("gen/vae/config.yaml")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = VAE(latent_dim=cfg["latent_dim"])
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {cfg['model_path']}")

    # Generate samples.
    with torch.no_grad():
        samples = model.generate(64, device).cpu()
    grid = vutils.make_grid(samples, nrow=8, padding=2)
    grid_img = grid.mul(255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    Image.fromarray(grid_img).save("gen/vae/samples.png")
    print("Saved samples to gen/vae/samples.png")

    # Interpolate between two random latent codes.
    z1 = torch.randn(1, cfg["latent_dim"], device=device)
    z2 = torch.randn(1, cfg["latent_dim"], device=device)
    with torch.no_grad():
        interp = model.interpolate(z1, z2, steps=8).cpu()
    grid = vutils.make_grid(interp, nrow=8, padding=2)
    grid_img = grid.mul(255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    Image.fromarray(grid_img).save("gen/vae/interpolation.png")
    print("Saved interpolation to gen/vae/interpolation.png")


if __name__ == "__main__":
    generate()
