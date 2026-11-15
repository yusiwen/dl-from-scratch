"""YOLO inference demo: detect objects in a single image."""

import argparse
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from cv.yolo.model import YOLO
from cv.yolo.loss import nms
from cv.yolo.data import VOC_CLASSES
from utils.config import load_config
from utils.device import get_device


def demo():
    parser = argparse.ArgumentParser(description="YOLO: detect objects in a single image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--save", default="yolo_output.png", help="Save path for output")
    args = parser.parse_args()

    cfg = load_config("cv/yolo/config.yaml")

    device = get_device()

    # Load model.
    model = YOLO(S=cfg["S"], B=cfg["B"], C=cfg["C"])
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Preprocess image.
    image = Image.open(args.image).convert("RGB")
    orig_w, orig_h = image.size
    transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference.
    with torch.no_grad():
        output = model(input_tensor).squeeze(0).cpu()  # (S, S, B*5+C)

    # NMS.
    boxes, scores, labels = nms(output, args.conf, args.iou)

    # Draw results.
    draw = ImageDraw.Draw(image)
    scale_x = orig_w / cfg["image_size"]
    scale_y = orig_h / cfg["image_size"]

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        cls_name = VOC_CLASSES[label]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 15), f"{cls_name} {score:.2f}", fill="red")

    image.save(args.save)
    print(f"\nInput: {args.image}")
    print(f"Detected {len(boxes)} objects:")
    for box, score, label in zip(boxes, scores, labels):
        cls_name = VOC_CLASSES[label]
        print(f"  {cls_name:<15} {score:.2f}  [{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")
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
