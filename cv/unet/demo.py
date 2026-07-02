"""UNet inference demo: segment a single image."""

import argparse
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

from cv.unet.model import UNet
from utils.config import load_config


CLASS_CMAP = np.array([[0, 0, 0], [0, 180, 60], [180, 60, 0], [60, 0, 180]], dtype=np.uint8)


def demo():
    parser = argparse.ArgumentParser(description="UNet: segment a single image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--save", default="unet_output.png", help="Save path for output")
    args = parser.parse_args()

    cfg = load_config("cv/unet/config.yaml")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model.
    model = UNet(in_channels=3, num_classes=cfg["num_classes"])
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Preprocess image.
    image = Image.open(args.image).convert("RGB")
    size = cfg["image_size"]
    image_resized = TF.resize(image, size, TF.InterpolationMode.BILINEAR)
    input_tensor = TF.to_tensor(image_resized)
    input_tensor = TF.normalize(input_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Inference.
    with torch.no_grad():
        logits = model(input_tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    mask_colored = CLASS_CMAP[pred]
    overlay = (np.array(image_resized) * 0.5 + mask_colored * 0.5).astype(np.uint8)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_resized)
    axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(mask_colored)
    axes[1].set_title("Segmentation Mask"); axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(args.save, dpi=150)
    print(f"\nSaved to {args.save}")

    import subprocess, sys
    subprocess.run(["open", args.save] if sys.platform == "darwin" else ["xdg-open", args.save])


if __name__ == "__main__":
    demo()
