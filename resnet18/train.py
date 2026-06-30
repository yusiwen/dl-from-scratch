import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from resnet18.data import CelebADataset, ATTRIBUTES
from resnet18.model import resnet18
from utils.config import load_config, save_config
from utils.seed import set_seed


def train():
    cfg = load_config("resnet18/config.yaml")
    set_seed(cfg["seed"])

    # --- Device setup ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "mps":
        print(f"  MPS is available on M4 (48GB, capped at ~24GB usage)")

    torch.set_num_threads(4)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = CelebADataset(ATTRIBUTES, num_samples=cfg["num_samples"], transform=transform)

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [cfg["train_split"], cfg["val_split"]],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
    )

    model = resnet18(num_classes=len(ATTRIBUTES)).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    scaler = None
    use_amp = cfg.get("amp", False)
    if use_amp:
        try:
            from torch.amp import autocast, GradScaler
            if device.type == "mps":
                scaler = GradScaler("mps")
            else:
                scaler = GradScaler()
        except (ImportError, RuntimeError):
            use_amp = False

    num_epochs = cfg["num_epochs"]
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)}")
    print(f"Attributes: {ATTRIBUTES}")
    print(f"Using AMP: {use_amp}")
    print()

    writer = SummaryWriter(log_dir="runs/resnet")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                if use_amp:
                    with autocast(device_type=device.type):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total * 100

        writer.add_scalar("train/loss", avg_train_loss, epoch)
        writer.add_scalar("val/loss", avg_val_loss, epoch)
        writer.add_scalar("val/acc", accuracy, epoch)

        print(f"Epoch [{epoch:2d}/{num_epochs}]  "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  "
              f"Val Acc: {accuracy:.2f}%")

        if device.type == "mps":
            torch.mps.empty_cache()

    writer.close()
    print("\nTraining complete!")
    torch.save(model, "resnet18/resnet18_celeba.pt")
    save_config(cfg, "resnet18/resnet18_celeba_config.yaml")
    print("Model saved to resnet18/resnet18_celeba.pt")


if __name__ == "__main__":
    train()
