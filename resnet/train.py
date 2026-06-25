import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from resnet.data import CelebADataset, ATTRIBUTES
from resnet.model import resnet18


def train():
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

    full_dataset = CelebADataset(ATTRIBUTES, num_samples=1000, transform=transform)

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [800, 200], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    model = resnet18(num_classes=len(ATTRIBUTES)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scaler = None
    try:
        from torch.amp import autocast, GradScaler
        if device.type == "mps":
            scaler = GradScaler("mps")
        else:
            scaler = GradScaler()
        use_amp = True
    except (ImportError, RuntimeError):
        use_amp = False

    num_epochs = 50
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)}")
    print(f"Attributes: {ATTRIBUTES}")
    print(f"Using AMP: {use_amp}")
    print()

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

        print(f"Epoch [{epoch:2d}/{num_epochs}]  "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  "
              f"Val Acc: {accuracy:.2f}%")

        if device.type == "mps":
            torch.mps.empty_cache()

    print("\nTraining complete!")
    torch.save(model, "resnet/resnet18_celeba.pt")
    print("Model saved to resnet/resnet18_celeba.pt")


if __name__ == "__main__":
    train()
