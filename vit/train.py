import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from vit.data import load_cifar10, CIFAR10_CLASSES
from vit.model import ViT
from utils.config import load_config, save_config
from utils.seed import set_seed
from utils.device import get_device


def train():
    cfg = load_config("vit/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")
    torch.set_num_threads(4)

    train_loader, test_loader = load_cifar10(
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
    )

    model = ViT(
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
        patch_size=cfg["patch_size"], num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
    ).to(device)
    print(f"Model parameters: {model.num_params():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    num_epochs = cfg["num_epochs"]
    writer = SummaryWriter(log_dir="runs/vit")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            images, labels = batch["img"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch["img"].to(device), batch["label"].to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                test_loss += loss.item()
                _, predicted = torch.max(logits, 1)
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

        print(f"Epoch [{epoch:2d}/{num_epochs}]  "
              f"Train Loss: {avg_train_loss:.4f}  Acc: {train_acc:.2f}%  |  "
              f"Test Loss: {avg_test_loss:.4f}  Acc: {test_acc:.2f}%")

    writer.close()
    save_path = cfg["model_path"]
    torch.save(model.state_dict(), save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()
