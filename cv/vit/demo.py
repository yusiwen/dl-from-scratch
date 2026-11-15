"""ViT inference demo: classify a single image."""

import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image

from cv.vit.model import ViT
from cv.vit.data import CIFAR10_CLASSES
from utils.config import load_config
from utils.device import get_device


def demo():
    parser = argparse.ArgumentParser(description="ViT: classify a single image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--topk", type=int, default=3, help="Number of top predictions")
    parser.add_argument("--save", default="vit_output.png", help="Save path for output")
    args = parser.parse_args()

    cfg = load_config("cv/vit/config.yaml")

    device = get_device()

    # Load model.
    model = ViT(
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
        patch_size=cfg["patch_size"], num_classes=cfg["num_classes"],
        dropout=0,
    )
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Preprocess image.
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    image = Image.open(args.image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference.
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, args.topk)

    print(f"\nInput: {args.image}")
    print(f"Top-{args.topk} predictions:")
    for i in range(args.topk):
        idx = top_indices[i].item()
        prob = top_probs[i].item()
        print(f"  {i+1}. {CIFAR10_CLASSES[idx]:<15} {prob:.1%}")

    # Save image with label overlay.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.axis("off")
    label = "\n".join(f"{CIFAR10_CLASSES[top_indices[i].item()]}: {top_probs[i].item():.1%}"
                      for i in range(args.topk))
    plt.title(f"Top-{args.topk}\n{label}", fontsize=10, loc="left")
    plt.tight_layout()
    plt.savefig(args.save, dpi=150)
    print(f"\nSaved to {args.save}")
    # Open the result when there is a desktop to open it on; headless runs
    # (containers, CI) must not fail after a successful save.
    import shutil, subprocess, sys
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    if shutil.which(opener):
        try:
            subprocess.run([opener, args.save], check=False)
        except OSError:
            pass


if __name__ == "__main__":
    demo()
