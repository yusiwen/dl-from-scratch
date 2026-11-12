"""Train the miniature MAE on CIFAR-10."""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from selfsup.mae.data import load_cifar10_tensors
from selfsup.mae.model import MaskedAutoencoder
from utils.config import load_config, save_config
from utils.device import get_device
from utils.seed import set_seed


def train():
    cfg = load_config("selfsup/mae/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")

    images, labels = load_cifar10_tensors("train")
    print(f"Dataset: {len(images):,} CIFAR-10 images")
    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    model = MaskedAutoencoder(
        image_size=cfg["image_size"],
        patch_size=cfg["patch_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        decoder_dim=cfg["decoder_dim"],
        decoder_layers=cfg["decoder_layers"],
        mask_ratio=cfg["mask_ratio"],
    ).to(device)
    print(f"Parameters: {model.num_params():,}, mask_ratio={cfg['mask_ratio']}")

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    writer = SummaryWriter(log_dir="runs/mae")
    num_epochs = cfg["num_epochs"]

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_images, _ in loader:
            batch_images = batch_images.to(device)
            loss = model(batch_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(loader)
        writer.add_scalar("train/masked_mse", average_loss, epoch)
        print(f"Epoch [{epoch:3d}/{num_epochs}]  masked MSE: {average_loss:.4f}")

    writer.close()
    save_path = cfg["model_path"]
    torch.save(model.state_dict(), save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()
