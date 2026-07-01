"""DDPM inference: generate CIFAR-10 samples."""

import torch
import torchvision.utils as vutils
from PIL import Image

from ddpm.model import DDPM
from utils.config import load_config


def generate():
    cfg = load_config("ddpm/config.yaml")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = DDPM(
        in_channels=3,
        model_channels=cfg["model_channels"],
        channel_mult=cfg["channel_mult"],
        num_res_blocks=cfg["num_res_blocks"],
        T=cfg["T"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
    )
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {cfg['model_path']}")

    with torch.no_grad():
        samples = model.sample(batch_size=64, image_size=cfg["image_size"], device=device).cpu()
    samples = (samples + 1) / 2  # [-1,1] → [0,1]

    grid = vutils.make_grid(samples.clamp(0, 1), nrow=8, padding=2)
    grid_img = grid.mul(255).permute(1, 2, 0).to(torch.uint8).numpy()
    Image.fromarray(grid_img).save("ddpm/samples.png")
    print("Saved samples to ddpm/samples.png")


if __name__ == "__main__":
    generate()
