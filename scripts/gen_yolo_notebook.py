#!/usr/bin/env python3
"""Generate YOLO notebook."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},"language_info": {"name": "python", "version": "3.12.0"}}

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("# YOLO: You Only Look Once\n\nSimplified object detection with grid-based bounding box regression on Pascal VOC.")

md("""## 背景

YOLO（Redmon et al. 2016）是首个单阶段目标检测器，将检测视为回归问题。
一张图通过 CNN 一次前向传播，直接输出边界框和类别概率。

核心思想：将图像分成 $S \\times S$ 网格，每个网格预测 $B$ 个边界框和 $C$ 个类别的概率。

与两阶段检测器（Faster R-CNN）的区别：
- YOLO：一次前向 → 端到端，速度快但精度略低
- Faster R-CNN：候选区域 → 分类，精度高但速度慢

数据集：**Pascal VOC** — 20 类物体，含边界框标注。
""")

md("""## 数学原理

### 输出表示

每个网格单元预测 $B$ 个边界框，每个框 5 个值：

$$(x, y, w, h, \\text{confidence})$$

- $x, y$: 框中心相对于网格单元的偏移（0~1）
- $w, h$: 框宽高相对于图像尺寸的比例
- $\\text{confidence}$: $P(\\text{object}) \\times \\text{IoU}_{\\text{pred}}^{\\text{truth}}$

再加上 $C$ 个类别概率 $P(\\text{class}_i \\mid \\text{object})$

输出张量：$S \\times S \\times (B \\times 5 + C)$

### 损失函数

$$\\mathcal{L} = \\lambda_{\\text{coord}} \\sum \\mathbb{1}_{ij}^{\\text{obj}} [(x - \\hat{x})^2 + (y - \\hat{y})^2 + (\\sqrt{w} - \\sqrt{\\hat{w}})^2 + (\\sqrt{h} - \\sqrt{\\hat{h}})^2] + \\sum \\mathbb{1}_{ij}^{\\text{obj}} (C - \\hat{C})^2 + \\lambda_{\\text{noobj}} \\sum \\mathbb{1}_{ij}^{\\text{noobj}} (C - \\hat{C})^2 + \\sum \\mathbb{1}_{i}^{\\text{obj}} \\sum_{c=1}^C (p_i(c) - \\hat{p}_i(c))^2$$

### 非极大值抑制（NMS）

对同一类别的重叠框，保留得分最高的，移除与其 IoU 超过阈值的框。
""")

code("""\
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from yolo.model import YOLO
from yolo.loss import yolo_loss
from utils.config import load_config
from utils.seed import set_seed
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

code("""\
from yolo.data import load_voc, VOC_CLASSES

train_loader, test_loader = load_voc(
    batch_size=32, image_size=224, S=7, B=2, C=20, num_workers=4,
)
print(f"Classes ({len(VOC_CLASSES)}): {VOC_CLASSES}")
print(f"Train batches: {len(train_loader)}")
""")

code("""\
model = YOLO(S=7, B=2, C=20).to(device)
print(f"Parameters: {model.num_params():,}")
""")

md("""## 训练

> ⏱ 预估耗时：**50 epoch × ~120s/epoch ≈ 1.5 小时**（M4 Max, batch_size=32）
> 如果太久，把下面 `NUM_EPOCHS` 改到 5 先看 loss 趋势。
""")

code("""\
NUM_EPOCHS = 50
LR = 0.0001

optimizer = optim.Adam(model.parameters(), lr=LR)
loss_hist = []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        pred = model(images)
        loss = yolo_loss(pred, targets, S=7, B=2, C=20, coord_scale=5, noobj_scale=0.5)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()

    avg = total_loss / len(train_loader)
    loss_hist.append(avg)
    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  Loss: {avg:.4f}")
""")

md("""## Loss 曲线""")

code("""\
import matplotlib.pyplot as plt
plt.plot(loss_hist)
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("YOLO Training Loss"); plt.grid(True)
plt.show()
""")

md("""\
## 思考题

1. YOLO 的 $S \\times S$ 网格中，一个网格只能预测一个物体（每个类）。这对检测小物体有什么影响？
2. 为什么边界框的 $w, h$ 用平方根而不是直接用？这有什么物理意义？
3. NMS 中的 IoU 阈值高低各有什么影响？
4. YOLO 和两阶段检测器（Faster R-CNN）的核心区别是什么？
""")

nb.cells = cells
with open("yolo/yolo.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated yolo/yolo.ipynb")
