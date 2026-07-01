import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ddpm.model import DDPM
from ddpm.data import load_cifar10
from utils.config import load_config, save_config
from utils.seed import set_seed
from utils.device import get_device


def train():
    cfg = load_config("ddpm/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")
    torch.set_num_threads(4)

    dataset = load_cifar10()
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    print(f"Dataset: {len(dataset):,} images")

    model = DDPM(
        in_channels=3,
        model_channels=cfg["model_channels"],
        channel_mult=cfg["channel_mult"],
        num_res_blocks=cfg["num_res_blocks"],
        T=cfg["T"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
    ).to(device)
    print(f"Parameters: {model.num_params():,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    num_epochs = cfg["num_epochs"]
    writer = SummaryWriter(log_dir="runs/ddpm")
    sample_interval = cfg.get("sample_interval", 10)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            images = batch["img"].to(device)
            loss = model(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        writer.add_scalar("train/loss", avg_loss, epoch)
        print(f"Epoch [{epoch:2d}/{num_epochs}]  Loss: {avg_loss:.6f}")

        if epoch % sample_interval == 0 or epoch == 1:
            model.eval()
            samples = model.sample(batch_size=64, image_size=cfg["image_size"], device=device).cpu()
            samples = (samples + 1) / 2  # [-1,1] → [0,1]
            writer.add_images("generated", samples.clamp(0, 1), epoch)

    writer.close()
    save_path = cfg["model_path"]
    torch.save(model.state_dict(), save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()
