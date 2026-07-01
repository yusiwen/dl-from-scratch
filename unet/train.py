import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from unet.data import PetDataset
from unet.model import UNet
from utils.config import load_config, save_config
from utils.seed import set_seed
from utils.device import get_device


def train():
    cfg = load_config("unet/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")
    torch.set_num_threads(4)

    train_dataset = PetDataset(split="train", image_size=cfg["image_size"], augment=True)
    val_dataset = PetDataset(split="test", image_size=cfg["image_size"], augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )

    print(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}")

    model = UNet(in_channels=3, num_classes=cfg["num_classes"]).to(device)
    print(f"Model parameters: {model.num_params():,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    num_epochs = cfg["num_epochs"]
    writer = SummaryWriter(log_dir="runs/unet")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                mask_valid = masks != 0
                correct += (preds[mask_valid] == masks[mask_valid]).sum().item()
                total += mask_valid.sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        pixel_acc = correct / total * 100 if total > 0 else 0

        writer.add_scalar("train/loss", avg_train_loss, epoch)
        writer.add_scalar("val/loss", avg_val_loss, epoch)
        writer.add_scalar("val/pixel_acc", pixel_acc, epoch)

        print(f"Epoch [{epoch:2d}/{num_epochs}]  "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  "
              f"Pixel Acc: {pixel_acc:.2f}%")

        # Log sample predictions.
        if epoch == 1 or epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_images, sample_masks = next(iter(val_loader))
                sample_images = sample_images[:4].to(device)
                preds = torch.argmax(model(sample_images), dim=1, keepdim=True).float()
                preds = preds / (cfg["num_classes"] - 1)
                writer.add_images("val/input", sample_images.cpu() * 0.25 + 0.5, epoch)
                writer.add_images("val/prediction", preds.cpu(), epoch)

        if device.type == "mps":
            torch.mps.empty_cache()

    writer.close()
    save_path = cfg["model_path"]
    torch.save(model.state_dict(), save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()
