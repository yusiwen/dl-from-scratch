"""FID-style evaluation for a trained DCGAN (CelebA 64x64)."""

import os
import sys

import torch
from torchvision import transforms

from gen.dcgan.data import CelebADataset
from gen.dcgan.model import Generator
from utils.config import load_config
from utils.device import get_device
from utils.fid import compute_and_save
from utils.seed import set_seed


def main():
    cfg = load_config("gen/dcgan/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")

    if not os.path.exists(cfg["model_path"]):
        sys.exit(
            f"error: checkpoint not found at {cfg['model_path']}; "
            f"run `python -m gen.dcgan.train` first"
        )

    num_samples = cfg["fid_num_samples"]

    netG = Generator(latent_dim=cfg["latent_dim"]).to(device)
    state = torch.load(cfg["model_path"], map_location=device, weights_only=True)
    netG.load_state_dict(state["G"])
    netG.eval()
    print(f"Loaded generator from {cfg['model_path']}")

    with torch.no_grad():
        fake = netG(torch.randn(num_samples, cfg["latent_dim"], 1, 1, device=device)).cpu()
    fake = (fake + 1) / 2

    # Match the training preprocessing (resize 72, then a 64x64 crop) without
    # the random crop, so real and fake images have the same geometry.
    transform = transforms.Compose([
        transforms.Resize(cfg["image_size"] + 8),
        transforms.CenterCrop(cfg["image_size"]),
        transforms.ToTensor(),
    ])
    dataset = CelebADataset(num_samples=num_samples, transform=transform)
    real = torch.stack([dataset[i] for i in range(len(dataset))])
    print(f"Real {tuple(real.shape)}, fake {tuple(fake.shape)}")

    compute_and_save(real, fake, "dcgan", device, {
        "image_size": cfg["image_size"],
        "checkpoint": cfg["model_path"],
    })


if __name__ == "__main__":
    main()
