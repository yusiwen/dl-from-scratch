"""
Deep clustering on 5,000 MNIST images: raw-pixel K-Means, K-Means on the
autoencoder latent, then DEC refinement. Prints ARI for all three on the same
subset, plus a linear probe on the latent codes (labels are evaluation-only).

The subset matches `ml/basics/pca.py` and the autoencoder demo.
"""

import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from gen.autoencoder.data import load_mnist_tensors
from gen.autoencoder.model import Autoencoder
from ml.basics.k_means import kmeans
from ml.basics.metrics import ari_score, nmi_score, purity_score
from selfsup.deep_cluster.model import (
    DeepEmbeddedClustering,
    soft_assign,
    target_distribution,
)
from utils.config import load_config, save_config
from utils.device import get_device
from utils.probe import linear_probe
from utils.seed import set_seed


def train_autoencoder(ae, loader, device, epochs, lr):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        ae.train()
        total = 0.0
        for images, _ in loader:
            images = images.to(device)
            recon = ae(images)
            loss = criterion(recon, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"  AE epoch [{epoch:2d}/{epochs}]  BCE: {total / len(loader):.4f}")


def latent_codes(ae, images, device):
    ae.eval()
    with torch.no_grad():
        return ae.encode(images.to(device)).cpu().numpy()


def cluster_metrics(assignments, labels):
    return {
        "ari": round(ari_score(labels, assignments), 6),
        "nmi": round(nmi_score(labels, assignments), 6),
        "purity": round(purity_score(assignments, labels), 6),
    }


def main():
    cfg = load_config("selfsup/deep_cluster/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")

    images, labels = load_mnist_tensors(cfg["max_samples"])
    print(f"Data: {tuple(images.shape)}")

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    ae = Autoencoder(latent_dim=cfg["latent_dim"]).to(device)
    print(f"AE parameters: {ae.num_params():,} (latent {cfg['latent_dim']})")
    print("\n1) Pretraining the autoencoder (reconstruction only)")
    train_autoencoder(ae, loader, device, cfg["ae_epochs"], cfg["lr"])

    latent_np = latent_codes(ae, images, device)
    _, pixel_assignments = kmeans(
        images.numpy(), k=cfg["n_clusters"], max_iters=50, tol=1e-4
    )

    print("\n2) K-Means on the frozen latent space")
    latent_centers, latent_assignments = kmeans(
        latent_np, k=cfg["n_clusters"], max_iters=50, tol=1e-4
    )

    model = DeepEmbeddedClustering(ae, cfg["n_clusters"], cfg["latent_dim"]).to(device)
    model.initialize_centers(torch.tensor(latent_centers, dtype=torch.float32, device=device))
    optimizer = optim.Adam(model.parameters(), lr=cfg["dec_lr"])
    writer = SummaryWriter(log_dir="runs/deep_cluster")
    criterion = nn.BCELoss()

    print("\n3) DEC refinement")
    for epoch in range(1, cfg["dec_epochs"] + 1):
        model.train()
        total_recon = 0.0
        total_kl = 0.0
        for batch_images, _ in loader:
            batch_images = batch_images.to(device)
            latent, reconstructed, q = model(batch_images)
            p = target_distribution(q).detach()
            recon_loss = criterion(reconstructed, batch_images)
            kl_loss = model.kl_loss(q, p)
            loss = recon_loss + cfg["gamma"] * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        writer.add_scalar("train/recon", total_recon / len(loader), epoch)
        writer.add_scalar("train/kl", total_kl / len(loader), epoch)
        print(f"  DEC epoch [{epoch:2d}/{cfg['dec_epochs']}]  "
              f"recon: {total_recon / len(loader):.4f}  kl: {total_kl / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        latent_tensor = torch.tensor(latent_np, dtype=torch.float32, device=device)
        q = soft_assign(latent_tensor, model.centers)
        dec_assignments = q.argmax(dim=1).cpu().numpy()

    raw_scores = cluster_metrics(pixel_assignments, labels.numpy())
    latent_scores = cluster_metrics(latent_assignments, labels.numpy())
    dec_scores = cluster_metrics(dec_assignments, labels.numpy())

    # Linear probe on the latent codes with a held-out split (labels used here
    # only for evaluation, exactly as in the SimCLR/MAE probes).
    rng = np.random.default_rng(cfg["seed"])
    order = rng.permutation(len(images))
    split = int(len(images) * cfg["probe_train_fraction"])
    train_idx, test_idx = order[:split], order[split:]
    probe, _ = linear_probe(
        latent_np[train_idx], labels.numpy()[train_idx],
        latent_np[test_idx], labels.numpy()[test_idx],
    )

    print("\nClustering quality (same 5,000 images, k=10):")
    print(f"  K-Means on pixels   ARI {raw_scores['ari']:.4f}  NMI {raw_scores['nmi']:.4f}  "
          f"purity {raw_scores['purity']:.2%}")
    print(f"  K-Means on latent   ARI {latent_scores['ari']:.4f}  NMI {latent_scores['nmi']:.4f}  "
          f"purity {latent_scores['purity']:.2%}")
    print(f"  DEC                 ARI {dec_scores['ari']:.4f}  NMI {dec_scores['nmi']:.4f}  "
          f"purity {dec_scores['purity']:.2%}")
    print(f"  Latent linear probe {probe:.2%} (80/20 split)")

    metrics = {
        "seed": int(cfg["seed"]),
        "k": int(cfg["n_clusters"]),
        "latent_dim": int(cfg["latent_dim"]),
        "kmeans_pixels": raw_scores,
        "kmeans_latent": latent_scores,
        "dec": dec_scores,
        "latent_linear_probe": round(float(probe), 6),
    }
    with open("selfsup/deep_cluster/dec_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")

    torch.save(model.state_dict(), cfg["model_path"])
    save_config(cfg, cfg["model_path"].replace(".pt", "_config.yaml"))
    print(f"\nMetrics saved to selfsup/deep_cluster/dec_metrics.json")


if __name__ == "__main__":
    main()
