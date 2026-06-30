import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from cnn.data import load_cifar10, CIFAR10_CLASSES
from cnn.model import SimpleCNN
from utils.config import load_config, save_config
from utils.seed import set_seed


def train():
    cfg = load_config("cnn/config.yaml")
    set_seed(cfg["seed"])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    torch.set_num_threads(4)

    train_loader, test_loader = load_cifar10(
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
    )

    model = SimpleCNN(num_classes=len(CIFAR10_CLASSES)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["T_max"])

    num_epochs = cfg["num_epochs"]
    writer = SummaryWriter(log_dir="runs/cnn")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            images, labels = batch["img"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch["img"].to(device), batch["label"].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_acc = train_correct / train_total * 100
        test_acc = test_correct / test_total * 100

        writer.add_scalar("train/loss", avg_train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("test/loss", avg_test_loss, epoch)
        writer.add_scalar("test/acc", test_acc, epoch)
        writer.add_scalar("lr", lr, epoch)

        print(
            f"Epoch [{epoch:2d}/{num_epochs}]  "
            f"Train Loss: {avg_train_loss:.4f}  Acc: {train_acc:.2f}%  |  "
            f"Test Loss: {avg_test_loss:.4f}  Acc: {test_acc:.2f}%  |  "
            f"LR: {lr:.2e}"
        )

    writer.close()
    print("\nTraining complete!")
    torch.save(model, "cnn/simple_cnn_cifar10.pt")
    save_config(cfg, "cnn/simple_cnn_cifar10_config.yaml")
    print("Model saved to cnn/simple_cnn_cifar10.pt")


if __name__ == "__main__":
    train()
