#!/usr/bin/env python3
"""Generate the DEC (deep clustering) notebook."""

import nbformat as nbf
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from nb_bootstrap import bootstrap_cell, config_cell, stabilize_ids  # noqa: E402

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"},
}

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""\
# DEC: Deep Embedded Clustering

先用自编码器学隐空间，再在隐空间上做 K-Means，最后用 KL 散度把聚类中心和编码器**一起**微调。
""")

md("""\
## 背景

聚类和无监督表示学习是同一个问题的两面：

1. 直接在 784 维像素上 K-Means → 受维度灾难影响，簇质量一般；
2. 先用自编码器压到低维隐空间，再 K-Means → 明显更好；
3. **DEC**（Deep Embedded Clustering）不满足于此：它定义一个从隐空间到簇的软分配，
   再用一个目标分布把分配推向"更硬、更均衡"，同时更新编码器。

第 3 步的关键是：聚类中心不是固定的，编码器也在跟着动 —— 表示和聚类互相塑造。
""")

md(r"""\
## 数学原理

**软分配（Student-t 核）：**

$$q_{ij} = \frac{\left(1 + \|z_i - \mu_j\|^2\right)^{-1}}{\sum_{k} \left(1 + \|z_i - \mu_k\|^2\right)^{-1}}$$

**目标分布（锐化并均衡）：**

$$p_{ij} = \frac{q_{ij}^2 / f_j}{\sum_{k} q_{ik}^2 / f_k},
\qquad f_j = \sum_i q_{ij}$$

平方让分配更"硬"（置信度高的更高），除以 $f_j$ 防止所有样本塌到同一个簇。

**KL 散度损失：**

$$\mathcal{L}_{\text{KL}} = \mathrm{KL}(P \,\|\, Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

总损失是重建项 + $\gamma \cdot \mathcal{L}_{\text{KL}}$，$\gamma$ 在 `config.yaml` 里。
""")

code("""\
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from gen.autoencoder.data import load_mnist_tensors
from gen.autoencoder.model import Autoencoder
from ml.basics.k_means import kmeans
from selfsup.deep_cluster.model import DeepEmbeddedClustering, soft_assign, target_distribution
from selfsup.deep_cluster.train import train_autoencoder, latent_codes, cluster_metrics
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

md("""## 数据（5,000 张 MNIST）""")

code("""\
images, labels = load_mnist_tensors(cfg["max_samples"])
loader = DataLoader(
    TensorDataset(images, labels),
    batch_size=cfg["batch_size"],
    shuffle=True,
    drop_last=True,
)
print(f"Data: {tuple(images.shape)}  k={cfg['n_clusters']}  latent_dim={cfg['latent_dim']}")
""")

md("""\
## 第 1 步：自编码器预训练

> ⏱ 全部超参来自 `config.yaml`（`ae_epochs` / `lr` / `latent_dim`）。
> `python -m selfsup.deep_cluster.train` 跑的是同一套配置。
""")

code("""\
ae = Autoencoder(latent_dim=cfg["latent_dim"]).to(device)
print(f"AE parameters: {ae.num_params():,}")
train_autoencoder(ae, loader, device, cfg["ae_epochs"], cfg["lr"])
""")

md("""## 第 2 步：在隐空间上 K-Means""")

code("""\
latent_np = latent_codes(ae, images, device)

# Baseline 1: K-Means straight on the 784-d pixels.
_, pixel_assignments = kmeans(images.numpy(), k=cfg["n_clusters"], max_iters=50, tol=1e-4)
pixel_scores = cluster_metrics(pixel_assignments, labels.numpy())

# Baseline 2: K-Means on the frozen latent codes.
centers, latent_assignments = kmeans(latent_np, k=cfg["n_clusters"], max_iters=50, tol=1e-4)
latent_scores = cluster_metrics(latent_assignments, labels.numpy())

print(f"K-Means on pixels  ARI {pixel_scores['ari']:.4f}  NMI {pixel_scores['nmi']:.4f}")
print(f"K-Means on latent  ARI {latent_scores['ari']:.4f}  NMI {latent_scores['nmi']:.4f}")
""")

md("""## 第 3 步：DEC 微调""")

code("""\
model = DeepEmbeddedClustering(ae, cfg["n_clusters"], cfg["latent_dim"]).to(device)
model.initialize_centers(torch.tensor(centers, dtype=torch.float32, device=device))

optimizer = optim.Adam(model.parameters(), lr=cfg["dec_lr"])
criterion = nn.BCELoss()

for epoch in range(1, cfg["dec_epochs"] + 1):
    model.train()
    total_recon = total_kl = 0.0
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

    if epoch % 10 == 0 or epoch == 1:
        print(f"DEC epoch [{epoch:2d}/{cfg['dec_epochs']}]  "
              f"recon: {total_recon / len(loader):.4f}  kl: {total_kl / len(loader):.4f}")
""")

md("""## 三种方案对比""")

code("""\
model.eval()
with torch.no_grad():
    latent_tensor = torch.tensor(latent_np, dtype=torch.float32, device=device)
    q = soft_assign(latent_tensor, model.centers)
    dec_assignments = q.argmax(dim=1).cpu().numpy()
dec_scores = cluster_metrics(dec_assignments, labels.numpy())

print(f"{'method':<20}{'ARI':>8}{'NMI':>8}{'purity':>10}")
print("-" * 46)
for name, s in [("K-Means on pixels", pixel_scores),
                ("K-Means on latent", latent_scores),
                ("DEC", dec_scores)]:
    print(f"{name:<20}{s['ari']:>8.4f}{s['nmi']:>8.4f}{s['purity']:>9.2%}")
""")

md("""## 隐空间可视化（按 DEC 簇着色）""")

code("""\
plt.figure(figsize=(7, 6))
plt.scatter(latent_np[:, 0], latent_np[:, 1], c=dec_assignments, cmap="tab10", s=4)
plt.title(f"DEC clusters in the {cfg['latent_dim']}-d latent space")
plt.xlabel("z[0]"); plt.ylabel("z[1]"); plt.grid(True)
plt.show()
""")

md("""\
## 思考题

1. 为什么"像素上 K-Means"和"隐空间上 K-Means"差距这么大？
2. 目标分布里的平方项和除以 $f_j$ 分别解决什么问题？去掉任一个会怎样？
3. DEC 的 KL 损失会不会让重建变差？`gamma` 调大调小分别有什么风险？
4. 聚类质量用 ARI/NMI 评估，但训练时并没有用标签。这种"评估用标签、训练不用"的做法
   和无监督表示学习的线性探针（`utils/probe.py`）有什么共同点？
""")

nb.cells = [bootstrap_cell(), config_cell("selfsup/deep_cluster/config.yaml"), *cells]
stabilize_ids(nb)
with open("selfsup/deep_cluster/dec.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated selfsup/deep_cluster/dec.ipynb")
