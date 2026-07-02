#!/usr/bin/env python3
"""Generate SimCLR notebook."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},"language_info": {"name": "python", "version": "3.12.0"}}

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("# SimCLR: Contrastive Learning\n\nSelf-supervised representation learning with NT-Xent loss on CIFAR-10.")

md("""## 背景

SimCLR（Chen et al. 2020）通过**对比学习**在没有标签的情况下学习图像表示。
核心思路：同一张图的不同增强视图应该得到相似的表示，不同图的视图应该不同。

关键组件：
- **数据增强**：随机裁剪、颜色抖动、高斯模糊、灰度化
- **Encoder**：ResNet18（去掉最后一层）
- **Projector**：MLP 将表示投影到对比空间
- **NT-Xent Loss**：归一化温度标度的交叉熵损失

训练完成后，encoder 可以迁移到下游分类任务，只需加一个线性分类器。
""")

md("""## 数学原理

### NT-Xent Loss

对每个批次 $N$ 张图，生成两个增强视图，共 $2N$ 个样本：

$$\ell(i, j) = -\\log \\frac{\\exp(\\text{sim}(z_i, z_j) / \\tau)}{\\sum_{k=1}^{2N} \\mathbb{1}_{[k \\neq i]} \\exp(\\text{sim}(z_i, z_k) / \\tau)}$$

其中 $\\text{sim}(u, v) = \\frac{u^\\top v}{\\|u\\|\\|v\\|}$ 是余弦相似度，$(i, j)$ 是一对正样本（同一图的两种增强）。
""")

code("""\
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from simclr.model import SimCLR
from utils.config import load_config
from utils.seed import set_seed
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

code("""\
from simclr.data import SimCLRTransform, load_cifar10_simclr

loader = load_cifar10_simclr(batch_size=256, num_workers=4)
print(f"Batches per epoch: {len(loader)}")
""")

code("""\
model = SimCLR(project_dim=128, temperature=0.5).to(device)
print(f"Parameters: {model.num_params():,}")

# Count encoder vs projector params
enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
proj = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
print(f"  Encoder (ResNet18): {enc:,}")
print(f"  Projector (MLP):    {proj:,}")
""")

md("""## 训练

> ⏱ 预估耗时：**100 epoch × ~40s/epoch ≈ 1 小时**（M4 Max, batch_size=256）
> 如果太久，把下面 `NUM_EPOCHS` 改到 10 先看 loss 趋势。
""")

code("""\
NUM_EPOCHS = 100
LR = 0.0003

optimizer = optim.Adam(model.parameters(), lr=LR)
loss_hist = []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        x1, x2 = batch["view1"].to(device), batch["view2"].to(device)
        z1, z2 = model(x1), model(x2)
        loss = model.nt_xent_loss(z1, z2)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item(); num_batches += 1

    avg = total_loss / num_batches
    loss_hist.append(avg)
    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  Loss: {avg:.4f}")
""")

md("""## Loss 曲线""")

code("""\
import matplotlib.pyplot as plt
plt.plot(loss_hist)
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("SimCLR Contrastive Loss"); plt.grid(True)
plt.show()
""")

md("""\
## 思考题

1. SimCLR 为什么需要 Projector？直接用 encoder 的输出做对比学习效果会差吗？
2. 数据增强的质量对对比学习有多重要？如果只用翻转，loss 会怎样？
3. NT-Xent 中的 temperature $\\tau$ 起什么作用？增大/减小各有什么影响？
4. SimCLR 为什么需要大 batch size？（提示：负样本数量）
""")

nb.cells = cells
with open("simclr/simclr.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated simclr/simclr.ipynb")
