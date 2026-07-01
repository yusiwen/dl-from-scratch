import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dcgan.data import CelebADataset
from dcgan.model import Generator, Discriminator
from utils.config import load_config, save_config
from utils.seed import set_seed
from utils.device import get_device


def train():
    cfg = load_config("dcgan/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")
    torch.set_num_threads(4)

    dataset = CelebADataset(num_samples=cfg["num_samples"])
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    print(f"Dataset: {len(dataset):,} images, {len(loader)} batches")

    latent_dim = cfg["latent_dim"]
    netG = Generator(latent_dim=latent_dim).to(device)
    netD = Discriminator().to(device)

    g_params = sum(p.numel() for p in netG.parameters())
    d_params = sum(p.numel() for p in netD.parameters())
    print(f"Generator: {g_params:,} params")
    print(f"Discriminator: {d_params:,} params")

    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], 0.999))

    real_label = 0.9 if cfg.get("label_smoothing") else 1.0
    fake_label = 0.0

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    writer = SummaryWriter(log_dir="runs/dcgan")

    num_epochs = cfg["num_epochs"]
    sample_interval = cfg["sample_interval"]
    save_interval = cfg.get("save_interval", 0)
    n_critic = cfg.get("n_critic", 1)

    for epoch in range(1, num_epochs + 1):
        for i, real_images in enumerate(loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # ── Train Discriminator ──
            netD.zero_grad()
            output = netD(real_images)
            label = torch.full((batch_size,), real_label, device=device)
            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = netG(noise)
            output = netD(fake_images.detach())
            label.fill_(fake_label)
            lossD_fake = criterion(output, label)
            lossD_fake.backward()
            D_G_z1 = output.mean().item()
            optimizerD.step()

            # ── Train Generator (every n_critic steps) ──
            if i % n_critic == 0:
                netG.zero_grad()
                output = netD(fake_images)
                label.fill_(real_label)
                lossG = criterion(output, label)
                lossG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

        # ── Logging ──
        writer.add_scalar("D/loss_real", lossD_real.item(), epoch)
        writer.add_scalar("D/loss_fake", lossD_fake.item(), epoch)
        writer.add_scalar("D/x", D_x, epoch)
        writer.add_scalar("D/G_z1", D_G_z1, epoch)
        writer.add_scalar("G/loss", lossG.item(), epoch)
        writer.add_scalar("G/D_z2", D_G_z2, epoch)

        print(f"Epoch [{epoch:2d}/{num_epochs}]  "
              f"D_loss: {lossD_real.item()+lossD_fake.item():.4f}  "
              f"G_loss: {lossG.item():.4f}  "
              f"D(x): {D_x:.3f}  D(G(z)): {D_G_z1:.3f}/{D_G_z2:.3f}")

        # ── Sample images ──
        if epoch % sample_interval == 0 or epoch == 1:
            with torch.no_grad():
                fake = netG(fixed_noise).cpu()
            # Denormalize from [-1, 1] to [0, 1] for TensorBoard display.
            fake = (fake + 1) / 2
            writer.add_images("generated", fake, epoch)

        # ── Save checkpoint ──
        if save_interval > 0 and epoch % save_interval == 0:
            ckpt_path = cfg["model_path"].replace(".pt", f"_epoch{epoch}.pt")
            torch.save({"G": netG.state_dict(), "D": netD.state_dict(), "cfg": cfg, "epoch": epoch}, ckpt_path)
            print(f"  -> saved checkpoint to {ckpt_path}")

        if device.type == "mps":
            torch.mps.empty_cache()

    writer.close()
    save_path = cfg["model_path"]
    torch.save({"G": netG.state_dict(), "D": netD.state_dict(), "cfg": cfg, "epoch": num_epochs}, save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()
