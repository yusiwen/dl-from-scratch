"""FID-style evaluation for a trained VAE (CelebA 64x64)."""

import os
import sys

import torch

from gen.vae.data import load_celeba
from gen.vae.model import VAE
from utils.config import load_config
from utils.device import get_device
from utils.fid import compute_and_save
from utils.seed import set_seed


def main():
    cfg = load_config("gen/vae/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")

    if not os.path.exists(cfg["model_path"]):
        sys.exit(
            f"error: checkpoint not found at {cfg['model_path']}; "
            f"run `python -m gen.vae.train` first"
        )

    num_samples = cfg["fid_num_samples"]

    model = VAE(latent_dim=cfg["latent_dim"]).to(device)
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {cfg['model_path']}")

    with torch.no_grad():
        fake = model.generate(num_samples, device).cpu()
    real = load_celeba(num_samples, cfg["image_size"])
    print(f"Real {tuple(real.shape)}, fake {tuple(fake.shape)}")

    compute_and_save(real, fake, "vae", device, {
        "image_size": cfg["image_size"],
        "checkpoint": cfg["model_path"],
    })


if __name__ == "__main__":
    main()
