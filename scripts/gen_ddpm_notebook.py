#!/usr/bin/env python3
"""Generate the DDPM notebook."""

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
# DDPM: Denoising Diffusion Probabilistic Model

先按固定日程给图像加噪，再学习"从噪声里一步步去噪"，最终从纯高斯噪声生成图像。
""")

md("""\
## 背景

GAN（如 DCGAN）一次前向就生成整张图，训练不稳定；VAE 生成模糊。
DDPM（Denoising Diffusion Probabilistic Model）走另一条路：

1. **前向过程**：对真实图像逐步加高斯噪声，$T = 1000$ 步后变成纯噪声（无参数、固定日程）；
2. **反向过程**：训练一个 UNet 预测每一步加进去的噪声，然后从噪声出发逐步去噪。

训练目标极其简单：给图加噪，让网络猜"加了什么噪声"。生成则要做 1000 次前向 ——
所以 DDPM 慢，但训练稳定、样本质量高。
""")

md(r"""\
## 数学原理

**前向过程（闭式解，可一步到位）：**

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\right)$$

其中 $\alpha_t = 1 - \beta_t$，$\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$。

**训练目标（预测噪声 $\varepsilon$）：**

$$\mathcal{L} = \mathbb{E}_{t, x_0, \varepsilon}\left[\left\|\varepsilon - \varepsilon_\theta\left(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\ t\right)\right\|^2\right]$$

**反向采样（从 $t=T$ 到 $1$）：**

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t, t)\right) + \sigma_t z$$

$\beta_t$ 从 `beta_start` 线性增长到 `beta_end`，都是 `config.yaml` 里的超参。
""")

code("""\
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from gen.ddpm.data import load_cifar10
from gen.ddpm.model import DDPM
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

md("""\
## 数据

> ⏱ 这是**演示规模**：CIFAR-10 只取前 `SUBSET_SIZE` 张。
> 完整训练（全量 50,000，100 epoch）用 `python -m gen.ddpm.train`，读同一份
> `config.yaml`；把 `SUBSET_SIZE` 改大即可在 notebook 里复现。
""")

code("""\
SUBSET_SIZE = 10000    # 演示子集；全量是 50,000

dataset = load_cifar10()
dataset = dataset.select(range(SUBSET_SIZE))
loader = torch.utils.data.DataLoader(
    dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
)
print(f"Data: {len(dataset):,} images at {cfg['image_size']}×{cfg['image_size']}")
""")

md("""## 模型""")

code("""\
model = DDPM(
    in_channels=3,
    model_channels=cfg["model_channels"],
    channel_mult=tuple(cfg["channel_mult"]),
    num_res_blocks=cfg["num_res_blocks"],
    T=cfg["T"],
    beta_start=cfg["beta_start"],
    beta_end=cfg["beta_end"],
).to(device)
print(f"Parameters: {model.num_params():,}  T={cfg['T']}")
""")

md("""\
## 训练

> ⏱ epoch / T / β 日程 / 通道数全部来自 `config.yaml`；`forward` 直接返回损失。
> 采样间隔用 `sample_interval`，采样 1000 步比较慢，所以不要太频繁。
""")

code("""\
optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
loss_hist = []
sample_interval = max(cfg.get("sample_interval", 10), 10)

for epoch in range(1, cfg["num_epochs"] + 1):
    model.train()
    total_loss, num_batches = 0.0, 0
    for batch in loader:
        images = batch["img"].to(device)
        loss = model(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    loss_hist.append(total_loss / num_batches)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:3d}/{cfg['num_epochs']}]  Loss: {loss_hist[-1]:.6f}")
""")

md("""## 损失曲线""")

code("""\
plt.plot(loss_hist)
plt.xlabel("Epoch"); plt.ylabel("Noise-prediction MSE")
plt.title("DDPM training loss"); plt.grid(True); plt.show()
""")

md("""\
## 从噪声生成

采样走完整的 $t = T \\to 1$ 反向链，这一步比训练本身慢得多。
""")

code("""\
model.eval()
# 采样 batch 与训练 batch 是两回事：生成 16 张足够看清效果
SAMPLE_BATCH = 16
samples = model.sample(
    batch_size=SAMPLE_BATCH, image_size=cfg["image_size"], device=device,
).cpu()
samples = ((samples + 1) / 2).clamp(0, 1)

fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i].permute(1, 2, 0)); ax.axis("off")
plt.suptitle(f"DDPM samples (t={cfg['T']} → 0)")
plt.tight_layout(); plt.show()
""")

md("""\
## 思考题

1. 为什么前向过程可以写成闭式解（一步加噪到 $t$），而反向必须一步步走？
2. 训练目标是预测噪声而不是直接预测 $x_0$，这两者等价吗？为什么前者更好训？
3. $T$ 越大采样越慢，生成质量一定越好吗？`beta_start`/`beta_end` 影响什么？
4. DDPM 采样要 1000 次前向，而 GAN 只要 1 次。后来有哪些加速采样的方法（DDIM 等）？
""")

nb.cells = [bootstrap_cell(), config_cell("gen/ddpm/config.yaml"), *cells]
stabilize_ids(nb)
with open("gen/ddpm/ddpm.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated gen/ddpm/ddpm.ipynb")
