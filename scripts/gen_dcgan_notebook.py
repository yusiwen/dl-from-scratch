#!/usr/bin/env python3
"""Generate DCGAN notebook."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"},
}

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""\
# DCGAN: Deep Convolutional GAN

Image generation with adversarial training — Generator vs Discriminator.
""")

md("""\
## 背景

DCGAN 将 CNN 与 GAN 结合，用转置卷积（transposed convolution）作为生成器。
核心思想：两个网络对抗训练——

- **Generator (G)**：从随机噪声生成逼真图像，目标是骗过 D
- **Discriminator (D)**：区分真实图像和生成的假图像，目标是识破 G

训练收敛时，G 能生成以假乱真的图像，D 无法区分真假（输出 ≈ 0.5）。
""")

md("""\
## 数学原理

### 对抗训练

$$\\min_G \\max_D V(D, G) = \\mathbb{E}_{x \\sim p_{\\text{data}}} [\\log D(x)] + \\mathbb{E}_{z \\sim p_z} [\\log(1 - D(G(z)))]$$

- D 的目标：最大化 $\\log D(x) + \\log(1 - D(G(z)))$
- G 的目标：最小化 $\\log(1 - D(G(z)))$，等价于最大化 $\\log D(G(z))$

### 架构

```
Generator:  z(100) → ConvTranspose × 4 → image(64×64×3), Tanh
Discriminator: image(64×64×3) → Conv × 4 → 1, Sigmoid
```

关键技巧：除第一层外使用 BatchNorm，G 用 ReLU，D 用 LeakyReLU(0.2)。
""")

code("""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from dcgan.model import Generator, Discriminator
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

code("""\
# 加载 CelebA 子集
transform = transforms.Compose([
    transforms.Resize(72),
    transforms.RandomCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

ds = load_dataset("eurecom-ds/celeba", split="train[:10000]")
images = [transform(item["image"]).unsqueeze(0) for item in ds]
dataset = torch.cat(images)
loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
print(f"Dataset: {len(dataset):,} images, {len(loader)} batches")
""")

code("""\
latent_dim = 100
netG = Generator(latent_dim=latent_dim).to(device)
netD = Discriminator().to(device)
print(f"G params: {sum(p.numel() for p in netG.parameters()):,}")
print(f"D params: {sum(p.numel() for p in netD.parameters()):,}")
""")

md("""\
## 训练

> ⏱ 预估耗时：**30 epoch × ~30s/epoch ≈ 15 分钟**（M4 Max, batch_size=128）
> GAN 训练较慢，如果想快速看效果可以把 `NUM_EPOCHS` 改到 10。
""")

code("""\
NUM_EPOCHS = 30
LR = 0.0002
BETA1 = 0.5
LABEL_SMOOTHING = True

criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))

real_label = 0.9 if LABEL_SMOOTHING else 1.0
fake_label = 0.0

fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

g_losses, d_losses, d_x_vals = [], [], []

for epoch in range(1, NUM_EPOCHS + 1):
    for i, real_images in enumerate(loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        # Train D
        netD.zero_grad()
        output = netD(real_images)
        label = torch.full((batch_size,), real_label, device=device)
        lossD_real = criterion(output, label)
        lossD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake = netG(noise)
        output = netD(fake.detach())
        label.fill_(fake_label)
        lossD_fake = criterion(output, label)
        lossD_fake.backward()
        optimizerD.step()

        # Train G
        netG.zero_grad()
        output = netD(fake)
        label.fill_(real_label)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

    d_loss = lossD_real.item() + lossD_fake.item()
    g_losses.append(lossG.item())
    d_losses.append(d_loss)
    d_x_vals.append(D_x)
    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  D: {d_loss:.3f}  G: {lossG.item():.3f}  D(x): {D_x:.3f}")
""")

md("""## Loss 曲线""")

code("""\
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(g_losses, label='G loss', alpha=0.8)
plt.plot(d_losses, label='D loss', alpha=0.8)
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
plt.title("GAN Training Loss"); plt.show()
""")

md("""## 生成样本

训练完成后，用固定噪声生成一批图像看看效果。""")

code("""\
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from utils.device import get_device

with torch.no_grad():
    fake = netG(fixed_noise).cpu()
fake = (fake + 1) / 2  # [-1,1] → [0,1]
grid = vutils.make_grid(fake, nrow=8, padding=2, normalize=False)

plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(grid.permute(1, 2, 0).clamp(0, 1))
plt.title("Generated CelebA Samples"); plt.show()
""")

md("""\
## 思考题

1. D(x) 接近 0 或 1 分别意味着什么？理想值应该是多少？
2. 为什么 G 的第一层不用 BatchNorm，D 的第一层也不用？
3. 标签平滑（label smoothing）为什么能帮助 GAN 训练？
4. 如果 G 和 D 的 Loss 都趋于稳定但生成效果很差，可能是什么问题？
5. 试试把 `NUM_EPOCHS` 加到 100，观察生成质量的变化。
""")

nb.cells = cells
out = "dcgan/dcgan.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Generated {out}")
