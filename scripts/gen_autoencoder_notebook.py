#!/usr/bin/env python3
"""Generate the autoencoder notebook."""

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
# Autoencoder: PCA → VAE 的中间站

只做重建的瓶颈自编码器 — 没有 KL 项、没有采样，用来看清"学一个隐空间"
在最朴素的设定下能到什么程度。
""")

md("""\
## 背景

PCA 是**线性**降维：它找一组正交方向，把所有样本投影上去。
自编码器把这件事变成**可学习**的：编码器和解码器都是神经网络，瓶颈宽度相同。

它正好卡在 PCA 和 VAE（Variational Autoencoder，变分自编码器）之间：

| 模型 | 隐空间 | 目标 |
|---|---|---|
| PCA | 线性子空间 | 最大方差 |
| **Autoencoder** | 学到的非线性映射 | 重建误差最小 |
| VAE | 概率分布 | 重建 + KL 正则 |

训练日志里会同时给出同宽度下 PCA 的重建 MSE 作为对照（同一批 5000 张图）。
""")

md(r"""\
## 数学原理

**编码 / 解码：**

$$z = f_\theta(x), \qquad \hat{x} = g_\phi(z)$$

**重建损失（MNIST 像素已归一化到 [0,1]，用二元交叉熵）：**

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{d=1}^{D}
\left[ x_{id} \log \hat{x}_{id} + (1 - x_{id}) \log (1 - \hat{x}_{id}) \right]$$

**与 PCA 的对照（同为 2 维瓶颈）：**

$$\text{MSE}_{\text{PCA}} = \frac{1}{ND}\sum_{i,d} \left( x_{id} - \hat{x}^{\text{PCA}}_{id} \right)^2$$
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
from ml.basics.pca import pca, reconstruct
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

md("""## 数据""")

code("""\
images, labels = load_mnist_tensors(cfg["max_samples"])
loader = DataLoader(
    TensorDataset(images, labels),
    batch_size=cfg["batch_size"],
    shuffle=True,
    drop_last=True,
)
print(f"Data: {tuple(images.shape)}  batches: {len(loader)}")
""")

md("""## 模型""")

code("""\
model = Autoencoder(latent_dim=cfg["latent_dim"]).to(device)
print(f"Parameters: {model.num_params():,}  (latent_dim={cfg['latent_dim']})")
""")

md("""\
## 训练

> ⏱ epoch / lr / latent_dim 来自 `config.yaml`。数据固定用前 `max_samples` 张，
> 与 `python -m gen.autoencoder.train` 完全一致（同一个子集、同一份配置）。
""")

code("""\
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

num_epochs = cfg["num_epochs"]

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    for batch_images, _ in loader:
        batch_images = batch_images.to(device)
        optimizer.zero_grad()
        recon = model(batch_images)
        loss = criterion(recon, batch_images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg = total_loss / len(loader)
    if epoch % cfg.get("sample_interval", 5) == 0 or epoch == 1 or epoch == num_epochs:
        print(f"Epoch [{epoch:2d}/{num_epochs}]  BCE: {avg:.4f}")
""")

md("""## 重建效果 vs PCA""")

code("""\
model.eval()
sample = images[:8].to(device)
with torch.no_grad():
    recon = model(sample).cpu()

# Same bottleneck width, same images: PCA as the linear baseline.
projection, components, mean, _ = pca(images.numpy(), n_components=cfg["latent_dim"])
pca_recon = reconstruct(projection, components, mean)

print(f"Autoencoder MSE: {torch.mean((recon - sample.cpu()) ** 2).item():.4f}")
print(f"PCA MSE        : {float(np.mean((pca_recon[:8] - images[:8].numpy()) ** 2)):.4f}")

fig, axes = plt.subplots(3, 8, figsize=(12, 5))
for i in range(8):
    axes[0, i].imshow(sample[i].cpu().view(28, 28), cmap="gray"); axes[0, i].axis("off")
    axes[1, i].imshow(recon[i].view(28, 28), cmap="gray"); axes[1, i].axis("off")
    axes[2, i].imshow(pca_recon[i].reshape(28, 28), cmap="gray"); axes[2, i].axis("off")
axes[0, 0].set_ylabel("input", fontsize=10)
axes[1, 0].set_ylabel("AE", fontsize=10)
axes[2, 0].set_ylabel("PCA", fontsize=10)
plt.suptitle("Original / Autoencoder / PCA")
plt.tight_layout(); plt.show()
""")

md("""## 隐空间（2 维）""")

code("""\
model.eval()
with torch.no_grad():
    latents = model.encoder(images.to(device)).cpu().numpy()

plt.figure(figsize=(7, 6))
plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap="tab10", s=4)
plt.colorbar(label="digit"); plt.title("Autoencoder latent space (2D)")
plt.xlabel("z[0]"); plt.ylabel("z[1]"); plt.grid(True)
plt.show()
""")

md("""\
## 思考题

1. 为什么这里用 BCE 而不是 MSE？两者对 [0,1] 像素的梯度有什么不同？
2. 自编码器把 784 维压到 2 维，MSE 大约是多少？PCA 呢？差距来自哪里？
3. 为什么自编码器的隐空间不连续，而 VAE 的连续？KL 项起了什么作用？
4. 把 `latent_dim` 改成 100，重建误差会怎么变？此时还能直接画散点图吗？
""")

nb.cells = [bootstrap_cell(), config_cell("gen/autoencoder/config.yaml"), *cells]
stabilize_ids(nb)
with open("gen/autoencoder/autoencoder.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated gen/autoencoder/autoencoder.ipynb")
