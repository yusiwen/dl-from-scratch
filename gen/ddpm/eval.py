"""FID-style evaluation for a trained DDPM (CIFAR-10 32x32).

Sampling is T sequential UNet calls per image, so this is by far the slowest
eval script: lower fid_num_samples in the config for a quick look.
"""

import os
import sys

import torch

from gen.ddpm.data import load_cifar10
from gen.ddpm.model import DDPM
from utils.config import load_config
from utils.device import get_device
from utils.fid import compute_and_save
from utils.seed import set_seed


def main():
    cfg = load_config("gen/ddpm/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")

    if not os.path.exists(cfg["model_path"]):
        sys.exit(
            f"error: checkpoint not found at {cfg['model_path']}; "
            f"run `python -m gen.ddpm.train` first"
        )

    num_samples = cfg["fid_num_samples"]

    model = DDPM(
        in_channels=3,
        model_channels=cfg["model_channels"],
        channel_mult=cfg["channel_mult"],
        num_res_blocks=cfg["num_res_blocks"],
        T=cfg["T"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
    ).to(device)
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {cfg['model_path']}")

    with torch.no_grad():
        fake = model.sample(
            batch_size=num_samples, image_size=cfg["image_size"], device=device
        ).cpu()
    fake = (fake + 1) / 2

    dataset = load_cifar10()
    real = torch.stack([dataset[i]["img"] for i in range(num_samples)])
    print(f"Real {tuple(real.shape)}, fake {tuple(fake.shape)}")

    compute_and_save(real, fake, "ddpm", device, {
        "image_size": cfg["image_size"],
        "checkpoint": cfg["model_path"],
    })


if __name__ == "__main__":
    main()
