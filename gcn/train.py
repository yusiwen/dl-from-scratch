import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from gcn.model import GCN
from gcn.data import load_cora
from utils.config import load_config, save_config
from utils.seed import set_seed
from utils.device import get_device


def train():
    cfg = load_config("gcn/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")

    features, adj_norm, labels, train_mask, val_mask, test_mask, _ = load_cora()
    features, adj_norm, labels = features.to(device), adj_norm.to(device), labels.to(device)
    train_mask, val_mask = train_mask.to(device), val_mask.to(device)

    model = GCN(
        in_features=features.size(1),
        hidden_dim=cfg["hidden_dim"],
        num_classes=labels.max().item() + 1,
        dropout=cfg["dropout"],
    ).to(device)
    print(f"Parameters: {model.num_params():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    num_epochs = cfg["num_epochs"]
    writer = SummaryWriter(log_dir="runs/gcn")

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_norm)
        loss = criterion(output[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            output = model(features, adj_norm)
            val_loss = criterion(output[val_mask], labels[val_mask])
            val_acc = (output[val_mask].argmax(dim=1) == labels[val_mask]).float().mean().item()

        writer.add_scalar("train/loss", loss.item(), epoch)
        writer.add_scalar("val/loss", val_loss.item(), epoch)
        writer.add_scalar("val/acc", val_acc, epoch)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{num_epochs}]  Loss: {loss.item():.4f}  Val Acc: {val_acc:.2%}")

    writer.close()
    save_path = cfg["model_path"]
    torch.save(model.state_dict(), save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()
