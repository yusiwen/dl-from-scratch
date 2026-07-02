import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from gen.simclr.model import SimCLR
from gen.simclr.data import load_cifar10_simclr
from utils.config import load_config, save_config
from utils.seed import set_seed
from utils.device import get_device


def train():
    cfg = load_config("gen/simclr/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")
    torch.set_num_threads(4)

    loader = load_cifar10_simclr(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    print(f"Dataset: 50,000 CIFAR-10 images (with augmentations)")

    model = SimCLR(project_dim=cfg["project_dim"], temperature=cfg["temperature"]).to(device)
    print(f"Parameters: {model.num_params():,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    num_epochs = cfg["num_epochs"]
    writer = SummaryWriter(log_dir="runs/simclr")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            x1 = batch["view1"].to(device)
            x2 = batch["view2"].to(device)

            z1 = model(x1)
            z2 = model(x2)
            loss = model.nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        writer.add_scalar("train/loss", avg_loss, epoch)
        print(f"Epoch [{epoch:2d}/{num_epochs}]  Loss: {avg_loss:.4f}")

    writer.close()
    save_path = cfg["model_path"]
    torch.save(model.state_dict(), save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()
