import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from cv.mobilenet.model import MobileNet
from cv.mobilenet.data import load_cifar10, CIFAR10_CLASSES
from utils.config import load_config, save_config
from utils.seed import set_seed
from utils.device import get_device


def train():
    cfg = load_config("cv/mobilenet/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")
    torch.set_num_threads(4)

    train_loader, test_loader = load_cifar10(
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
    )

    model = MobileNet(num_classes=cfg["num_classes"], width_multiplier=cfg["width_multiplier"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Width multiplier: {cfg['width_multiplier']}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["T_max"])

    num_epochs = cfg["num_epochs"]
    writer = SummaryWriter(log_dir="runs/mobilenet")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = train_total = 0

        for batch in train_loader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch["img"].to(device), batch["label"].to(device)
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                test_correct += (pred == labels).sum().item()
                test_total += labels.size(0)

        avg_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total * 100
        test_acc = test_correct / test_total * 100

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("test/acc", test_acc, epoch)

        print(f"Epoch [{epoch:2d}/{num_epochs}]  "
              f"Loss: {avg_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Test Acc: {test_acc:.2f}%")

    writer.close()
    save_path = cfg["model_path"]
    torch.save(model.state_dict(), save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()
