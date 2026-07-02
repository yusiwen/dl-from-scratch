#!/usr/bin/env python3
"""Generate MobileNet notebook."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},"language_info": {"name": "python", "version": "3.12.0"}}

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("# MobileNet: Depthwise Separable Convolutions\n\nEfficient CNN with depthwise separable convolutions for CIFAR-10.")

md("""## 背景

MobileNet（Howard et al. 2017）专为移动和嵌入式设备设计，核心创新是
**深度可分离卷积（Depthwise Separable Convolution）**——将标准卷积分解为两步：

| | 标准卷积 | 深度可分离卷积 |
|---|---|---|
| 计算量 | $K_h \\cdot K_w \\cdot C_{in} \\cdot C_{out}$ | $K_h \\cdot K_w \\cdot C_{in} + C_{in} \\cdot C_{out}$ |
| 3×3, 64→128 | $3\\times3\\times64\\times128 = 73,728$ | $3\\times3\\times64 + 64\\times128 = 8,768$ |
| **节省** | — | **~8.4×** |

额外参数：**Width Multiplier** $\\alpha \\in (0, 1]$——以 $\\alpha$ 比例缩小所有通道数，
可在精度和效率之间做权衡。
""")

md("""## 数学原理

### 标准卷积

$$\\text{Conv}(x)_{i,j,k} = \\sum_{c=1}^{C_{in}} \\sum_{p,q=1}^K K_{p,q,c,k} \\cdot x_{i+p, j+q, c}$$

### 深度可分离卷积

深度卷积（每个通道独立做 2D 卷积）：

$$\\text{Depthwise}(x)_{i,j,c} = \\sum_{p,q=1}^K K_{p,q,c} \\cdot x_{i+p, j+q, c}$$

逐点卷积（1×1 卷积组合跨通道信息）：

$$\\text{Pointwise}(x)_{i,j,k} = \\sum_{c=1}^{C_{in}} K_{c,k} \\cdot \\text{Depthwise}(x)_{i,j,c}$$

### 参数量对比

$$\\text{Ratio} = \\frac{\\text{DepthwiseSeparable}}{\\text{Standard}} = \\frac{1}{C_{out}} + \\frac{1}{K^2} \\approx \\frac{1}{K^2} \\quad (C_{out} \\gg K)$$
""")

code("""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from cv.mobilenet.model import MobileNet
from utils.config import load_config
from utils.seed import set_seed
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

code("""\
from cv.mobilenet.data import load_cifar10, CIFAR10_CLASSES

train_loader, test_loader = load_cifar10(batch_size=128, num_workers=4)
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
""")

code("""\
# 比较不同 width_multiplier 的参数量
for wm in [1.0, 0.5, 0.25]:
    m = MobileNet(num_classes=10, width_multiplier=wm)
    print(f"  width={wm:.2f}: {sum(p.numel() for p in m.parameters()):,} params")

# 与 SimpleCNN 对比
from cv.simplecnn.model import SimpleCNN
cnn = SimpleCNN(num_classes=10)
cnn_params = sum(p.numel() for p in cv.simplecnn.parameters())
print(f"  SimpleCNN:  {cnn_params:,} params")
print(f"  MobileNet × {cnn_params / 135562:.1f} smaller")
""")

code("""\
model = MobileNet(num_classes=10, width_multiplier=1.0).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
""")

md("""## 训练

> ⏱ 预估耗时：**50 epoch × ~25s/epoch ≈ 20 分钟**（M4 Max, batch_size=128）
""")

code("""\
NUM_EPOCHS = 50
LR = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

train_loss_hist, test_acc_hist = [], []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        images, labels = batch["img"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        train_loss += loss.item()
    scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    avg_loss = train_loss / len(train_loader)
    acc = correct / total * 100
    train_loss_hist.append(avg_loss)
    test_acc_hist.append(acc)
    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  Loss: {avg_loss:.4f}  Test Acc: {acc:.2f}%")
""")

md("""## Loss 曲线""")

code("""\
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_loss_hist); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.grid(True)
ax2.plot(test_acc_hist, color='green'); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Test Acc (%)"); ax2.grid(True)
plt.tight_layout(); plt.show()
""")

md("""\
## 思考题

1. 深度可分离卷积为什么能大幅减少参数量？计算量减少来自哪一步？
2. Width Multiplier $\\alpha=0.5$ 时参数量减半，你认为精度会下降多少？
3. 把 `width_multiplier` 改到 0.5 重新训练，对比精度差异。
4. MobileNet 的设计哲学适用于 Transformer 吗？（提示：MQA / GQA）
""")

nb.cells = cells
with open("cv/mobilenet/mobilenet.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated cv/mobilenet/mobilenet.ipynb")
