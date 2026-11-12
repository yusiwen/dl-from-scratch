"""Train BYOL on CIFAR-10 (no labels used)."""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from selfsup.byol.model import BYOL
from selfsup.simclr.data import load_cifar10_simclr
from utils.config import load_config, save_config
from utils.device import get_device
from utils.seed import set_seed


def train():
    cfg = load_config("selfsup/byol/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")

    loader = load_cifar10_simclr(
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"]
    )
    print(f"Dataset: 50,000 CIFAR-10 images (two augmented views)")

    model = BYOL(
        projector_hidden=cfg["projector_hidden"],
        projector_dim=cfg["projector_dim"],
        predictor_hidden=cfg["predictor_hidden"],
        center_momentum=cfg["center_momentum"],
    ).to(device)
    print(f"Parameters: {model.num_params():,} (online + target)")

    optimizer = optim.AdamW(
        [
            {"params": model.online_encoder.parameters()},
            {"params": model.online_projector.parameters()},
            {"params": model.predictor.parameters()},
        ],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    writer = SummaryWriter(log_dir="runs/byol")
    num_epochs = cfg["num_epochs"]
    warmup_epochs = cfg.get("warmup_epochs", 0)

    for epoch in range(1, num_epochs + 1):
        if warmup_epochs > 0:
            scale = min(1.0, epoch / warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = cfg["lr"] * scale
        model.train()
        total_loss = 0.0
        for batch in loader:
            view1 = batch["view1"].to(device)
            view2 = batch["view2"].to(device)

            loss = model(view1, view2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target(cfg["ema_decay"])

            total_loss += loss.item()

        average_loss = total_loss / len(loader)
        writer.add_scalar("train/loss", average_loss, epoch)
        print(f"Epoch [{epoch:3d}/{num_epochs}]  Loss: {average_loss:.4f}")

    writer.close()
    save_path = cfg["model_path"]
    torch.save(model.state_dict(), save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()
