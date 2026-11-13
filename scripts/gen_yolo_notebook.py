#!/usr/bin/env python3
"""Generate YOLO notebook."""

import nbformat as nbf
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from nb_bootstrap import bootstrap_cell  # noqa: E402

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

from cv.yolo.model import YOLO
from cv.yolo.loss import yolo_loss
from utils.config import load_config
from utils.seed import set_seed
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

code("""\
from cv.yolo.data import load_voc, VOC_CLASSES

# 用 VOC2007 子集控制 notebook 训练时长（trainval 共 5011 张）。
# 训练取前 3000 张，评测取之后的 1000 张：两段不相交，避免训练集与评测集重叠。
train_loader, test_loader = load_voc(
    batch_size=32, image_size=224, S=7, B=2, C=20, num_workers=4,
    max_train=3000, max_test=1000,
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

md(r"""
## 评测：IoU 与 mAP

### 为什么训练 loss 不够？

训练 loss 衡量的是「网格里的数字和标签差多少」。这个数据集里绝大多数网格是空的，
模型只要把空网格的 confidence 压到 0，loss 就会很低 —— 即使它一个物体都没检测出来。
而且 loss 并不直接关心预测框的**位置**有没有和真实框重合。

所以要分开评测两件事：

- **框得准不准** → IoU
- **按置信度排序后，各类别的检测质量如何** → AP / mAP
""")

md(r"""
### IoU（Intersection over Union，交并比）

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}
= \frac{\text{交叠面积}}{\text{并集面积}}$$

- 两个框完全重合 → IoU = 1；完全不相交 → IoU = 0
- 它同时惩罚「框太小」（漏掉目标）和「框太大」（框进了背景）
- 评测时用 IoU ≥ 阈值 来判定「这个预测框是否匹配上了某个真实框」

本 notebook 用 **IoU ≥ 0.5** 作为匹配阈值，并且匹配是**一对一**的：
一个真实框只能被一个预测框认领，同类别多余的预测框算作误检（false positive）。
""")

md(r"""
### mAP（mean Average Precision，平均精度均值）

单个 IoU 只说明「这一对框像不像」，我们需要一个能概括整个数据集、所有类别的数字：

1. 按置信度从高到低排列所有预测框；
2. 逐个累加，算出每个位置上的 precision（查准率）和 recall（查全率）；
3. 画 P-R 曲线，曲线下的面积就是 **AP**（Average Precision，单类别平均精度）；
4. 对 20 个类别取平均，得到 **mAP**。

| 指标 | 含义 |
|---|---|
| `mAP@0.5` | 只要求 IoU ≥ 0.5 即算匹配，最常用的宽松标准 |
| `mAP@0.5:0.95` | 在 0.5、0.55、…、0.95 十个阈值上取平均，更严格，反映定位精度 |

**为什么需要它：** VOC 各类别样本数差异很大（person 远多于 horse），只看总体准确率会被大类主导；
mAP 对每个类别各自算 AP 再平均，且只衡量排序质量，因此可以跨模型、跨类别直接比较。

> 评测在**留出集**上进行：训练用 trainval 的前 3000 张，评测用其后的 1000 张，两段完全不相交。
> （注意：VOC 官方的 `val` 列表本身就是 `trainval` 的子集，直接拿它评测会把训练图算进评测集。）
""")

code("""\
from cv.yolo.eval import evaluate_map

# test_loader.dataset 就是 VOCDataset；评测会用它还原原始边界框
metrics = evaluate_map(model, test_loader.dataset, device, image_size=224, batch_size=32)

print(f"评测图片: {metrics['num_images']}   真实框: {metrics['num_ground_truths']}")
print(f"mAP@0.5      : {metrics['mAP@0.5']:.3f}")
print(f"mAP@0.5:0.95 : {metrics['mAP@0.5:0.95']:.3f}")
print("\\n各类别 AP@0.5:")
for name, ap in metrics["AP@0.5"].items():
    print(f"  {name:<14} {ap:.3f}")
""")

md("""\
## 思考题

1. YOLO 的 $S \\times S$ 网格中，一个网格只能预测一个物体（每个类）。这对检测小物体有什么影响？
2. 为什么边界框的 $w, h$ 用平方根而不是直接用？这有什么物理意义？
3. NMS 中的 IoU 阈值高低各有什么影响？
4. YOLO 和两阶段检测器（Faster R-CNN）的核心区别是什么？
5. 把评测里的匹配阈值从 0.5 改到 0.75，`mAP` 会怎么变？为什么它比 `mAP@0.5` 更能反映定位质量？
""")

nb.cells = [bootstrap_cell(), *cells]
with open("cv/yolo/yolo.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated cv/yolo/yolo.ipynb")
