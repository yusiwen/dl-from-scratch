import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from resnet34.data import CelebADataset, CELEBA_ATTR_ORDER, train_transform
from resnet34.model import resnet34
from utils.config import load_config, save_config
from utils.seed import set_seed
from utils.device import get_device


def compute_pos_weight(loader, device):
    """Compute pos_weight per attribute for imbalanced multi-label classification."""
    total = 0
    pos_counts = torch.zeros(40, device=device)
    for _, labels in loader:
        labels = labels.to(device)
        pos_counts += labels.sum(dim=0)
        total += labels.size(0)
    neg_counts = total - pos_counts
    return (neg_counts / pos_counts).clamp(min=1.0)


def train():
    cfg = load_config("resnet34/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")
    torch.set_num_threads(4)

    transform = train_transform()
    train_dataset = CelebADataset(split="train", transform=transform)
    val_dataset = CelebADataset(split="val", transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
    )

    # Compute pos_weight from a single pass over the training set.
    pos_weight = compute_pos_weight(train_loader, device)
    print(f"pos_weight computed: [{pos_weight[0]:.2f} ... {pos_weight[-1]:.2f}] (min={pos_weight.min():.2f}, max={pos_weight.max():.2f})")

    model = resnet34(num_classes=40).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Training on {len(train_dataset):,} samples, validating on {len(val_dataset):,}")
    print(f"Attributes: all 40")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["T_max"])

    scaler = None
    if cfg.get("amp", False):
        try:
            from torch.amp import autocast, GradScaler
            if device.type == "mps":
                scaler = GradScaler("mps")
            else:
                scaler = GradScaler()
        except (ImportError, RuntimeError):
            pass

    num_epochs = cfg["num_epochs"]
    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    patience = cfg.get("early_stopping_patience", 0)

    writer = SummaryWriter(log_dir="runs/resnet34")

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = cfg["model_path"]

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            if scaler:
                with autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels) / grad_accum
                scaler.scale(loss).backward()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels) / grad_accum
                loss.backward()

            if (i + 1) % grad_accum == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                if scaler:
                    with autocast(device_type=device.type):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        lr = scheduler.get_last_lr()[0]

        writer.add_scalar("train/loss", avg_train_loss, epoch)
        writer.add_scalar("val/loss", avg_val_loss, epoch)
        writer.add_scalar("lr", lr, epoch)

        print(f"Epoch [{epoch:2d}/{num_epochs}]  "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  "
              f"LR: {lr:.2e}")

        # Early stopping.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model, best_model_path)
            save_config(cfg, best_model_path.replace(".pt", "_config.yaml"))
            print(f"  -> saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

        if device.type == "mps":
            torch.mps.empty_cache()

    writer.close()
    print(f"\nTraining complete. Best model saved to {best_model_path} (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    train()
