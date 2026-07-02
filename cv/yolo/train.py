import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from cv.yolo.model import YOLO
from cv.yolo.loss import yolo_loss
from cv.yolo.data import load_voc, VOC_CLASSES
from utils.config import load_config, save_config
from utils.seed import set_seed
from utils.device import get_device


def train():
    cfg = load_config("cv/yolo/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")
    torch.set_num_threads(4)

    train_loader, test_loader = load_voc(
        batch_size=cfg["batch_size"], image_size=cfg["image_size"],
        S=cfg["S"], B=cfg["B"], C=cfg["C"], num_workers=cfg["num_workers"],
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = YOLO(S=cfg["S"], B=cfg["B"], C=cfg["C"]).to(device)
    print(f"Parameters: {model.num_params():,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    num_epochs = cfg["num_epochs"]
    writer = SummaryWriter(log_dir="runs/yolo")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            pred = model(images)
            loss = yolo_loss(pred, targets, cfg["S"], cfg["B"], cfg["C"],
                             cfg["coord_scale"], cfg["noobj_scale"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        writer.add_scalar("train/loss", avg_loss, epoch)
        print(f"Epoch [{epoch:2d}/{num_epochs}]  Loss: {avg_loss:.4f}")

    writer.close()
    save_path = cfg["model_path"]
    torch.save(model.state_dict(), save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()
