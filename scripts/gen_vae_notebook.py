#!/usr/bin/env python3
"""Generate the VAE notebook."""

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
# VAE: Variational Autoencoder

给自编码器加上概率结构：编码器输出一个**分布**而不是一个点，解码器从这个分布里采样。
""")

md(r"""## 背景

普通自编码器把图像压成一个点 $z$。问题是：隐空间里"点与点之间"没有定义，任意采样一个 $z$
解码出来往往是无意义的噪声。

VAE（Variational Autoencoder，变分自编码器）改成：

- 编码器输出 $\mu$ 和 $\log \sigma^2$，定义一个高斯分布 $q(z\mid x)$；
- 训练时从该分布采样 $z$（重参数化，让梯度能穿过采样）；
- 损失里加一个 KL 项，把 $q(z\mid x)$ 拉向标准正态 $\mathcal{N}(0, I)$。

这样隐空间变成连续、可采样的，前后两个思路（自编码器 ↔ 生成模型）就连起来了。

数据集：CelebA 64×64 子集（`config.yaml` 的 `num_samples`）。
""")

md(r"""\
## 数学原理

**ELBO（证据下界）：**

$$\log p(x) \geq \mathbb{E}_{q(z\mid x)}\left[\log p(x \mid z)\right]
- \mathrm{KL}\left(q(z\mid x) \,\|\, p(z)\right)$$

**重参数化技巧（让采样可导）：**

$$z = \mu + \sigma \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

**两项损失：**

$$\mathcal{L}_{\text{recon}} = -\sum_{d} \left[ x_d \log \hat{x}_d + (1 - x_d)\log(1 - \hat{x}_d) \right]$$

$$\mathcal{L}_{\text{KL}} = -\frac{1}{2}\sum_{j}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

第一项要求重建得准，第二项要求隐空间规整；两者互相拉扯，这正是 VAE 的难点。
""")

code("""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from gen.vae.data import load_celeba
from gen.vae.model import VAE
from gen.vae.train import vae_loss
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

md("""## 数据""")

code("""\
dataset = load_celeba(cfg["num_samples"], cfg["image_size"])
loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
print(f"Dataset: {len(dataset):,} images at {cfg['image_size']}×{cfg['image_size']}")
""")

md("""## 模型""")

code("""\
model = VAE(latent_dim=cfg["latent_dim"]).to(device)
print(f"Parameters: {model.num_params():,}  (latent_dim={cfg['latent_dim']})")
""")

md("""\
## 训练

> ⏱ epoch / lr / latent_dim / 数据量全部来自 `config.yaml`。
> 完整训练用 `python -m gen.vae.train`（同一份配置），会同时保存权重。
""")

code("""\
optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
recon_hist, kl_hist = [], []

for epoch in range(1, cfg["num_epochs"] + 1):
    model.train()
    total_recon = total_kl = 0.0
    for x in loader:
        x = x.to(device)
        recon, mu, logvar = model(x)
        recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)
        loss = recon_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    n = len(dataset)
    recon_hist.append(total_recon / n)
    kl_hist.append(total_kl / n)
    if epoch % cfg.get("sample_interval", 5) == 0 or epoch == 1:
        print(f"Epoch [{epoch:2d}/{cfg['num_epochs']}]  "
              f"Recon: {recon_hist[-1]:.0f}  KL: {kl_hist[-1]:.2f}")
""")

md("""## 重建与采样""")

code("""\
model.eval()

# Reconstruction of real images.
with torch.no_grad():
    batch = next(iter(loader))[:8].to(device)
    recon, mu, logvar = model(batch)

# Fresh samples from the prior.
samples = model.generate(8, device).cpu()

fig, axes = plt.subplots(3, 8, figsize=(12, 5))
for i in range(8):
    axes[0, i].imshow(batch[i].cpu().permute(1, 2, 0).clamp(0, 1)); axes[0, i].axis("off")
    axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0).clamp(0, 1)); axes[1, i].axis("off")
    axes[2, i].imshow(samples[i].permute(1, 2, 0).clamp(0, 1)); axes[2, i].axis("off")
axes[0, 0].set_ylabel("input"); axes[1, 0].set_ylabel("recon"); axes[2, 0].set_ylabel("sample")
plt.suptitle("VAE: input / reconstruction / prior sample")
plt.tight_layout(); plt.show()
""")

md("""## 隐空间插值""")

code("""\
model.eval()
with torch.no_grad():
    z1 = torch.randn(1, cfg["latent_dim"], device=device)
    z2 = torch.randn(1, cfg["latent_dim"], device=device)
    interp = model.interpolate(z1, z2, steps=8).cpu()

fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i in range(8):
    axes[i].imshow(interp[i].permute(1, 2, 0).clamp(0, 1)); axes[i].axis("off")
plt.suptitle("Latent interpolation z1 → z2")
plt.tight_layout(); plt.show()
""")

md("""\
## 思考题

1. 为什么普通自编码器的隐空间不能直接采样，而 VAE 可以？
2. 重参数化技巧解决了什么问题？如果直接 `z ~ N(mu, sigma)` 会怎样？
3. KL 项太强会发生什么（recon 变差、隐空间塌缩）？太弱呢？
4. 对比 `gen/autoencoder` 的重建 MSE：VAE 通常更差，为什么我们仍然用 VAE？
""")

nb.cells = [bootstrap_cell(), config_cell("gen/vae/config.yaml"), *cells]
stabilize_ids(nb)
with open("gen/vae/vae.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated gen/vae/vae.ipynb")
