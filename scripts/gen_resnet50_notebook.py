#!/usr/bin/env python3
"""Generate the ResNet50 notebook."""

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
# ResNet50: Bottleneck 残差网络

更深的 ResNet 靠 **Bottleneck** 结构控制参数量：1×1 降维 → 3×3 → 1×1 升维。
""")

md("""\
## 背景

ResNet18/34 用的是 **BasicBlock**（两个 3×3 卷积）。堆到 50 层时，如果还用 3×3 卷积，
参数量和计算量会爆炸。ResNet50 改用 **Bottleneck**：

```
1×1 conv (降维 4 倍) → 3×3 conv → 1×1 conv (升维 4 倍) → + shortcut
```

- 3×3 卷积在一个很窄的通道数上做（64 → 64），计算量大幅下降；
- 首尾两个 1×1 负责通道数的"压缩和扩张"；
- 总层数 50（3 + 4×(3+4+6+3) = 50）。

这体现了一个通用思路：**深度靠结构设计买，不是靠堆参数堆出来的**。
""")

md(r"""\
## 数学原理

**Bottleneck block（输入 $x$，输出维度 $4\times$）：**

$$y = \mathcal{F}(x) + W_s x, \qquad
\mathcal{F}(x) = W_3 \,\sigma\!\left(W_2 \,\sigma(W_1 x)\right)$$

- $W_1$：$1\times1$，把通道压到 64
- $W_2$：$3\times3$，在 64 通道上卷积
- $W_3$：$1\times1$，再扩到 $4\times64 = 256$
- $W_s$：$1\times1$ shortcut，把输入通道对齐到 256

**参数量对比（单块，输入/输出均 256 通道）：**

| 结构 | 参数 |
|---|---|
| BasicBlock（两个 3×3） | $2 \times 3^2 \times 256^2 \approx 1.18\text{M}$ |
| Bottleneck | $256{\cdot}64 + 3^2{\cdot}64^2 + 64{\cdot}256 \approx 0.07\text{M}$ |

**损失（多标签分类，40 个 CelebA 属性）：**

$$\mathcal{L} = -\frac{1}{N}\sum_{i}\sum_{c}
\left[ w_c\, y_{ic}\log\sigma(z_{ic}) + (1 - y_{ic})\log(1 - \sigma(z_{ic})) \right]$$

$w_c$ 是每个属性的正样本权重（`pos_weight`），用来缓解属性不平衡。
""")

code("""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from cv.resnet50.data import CelebADataset, CELEBA_ATTR_ORDER, train_transform
from cv.resnet50.model import resnet50
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

md("""\
## 数据

> ⏱ 这是**演示规模**：只取 CelebA 训练集前 `SUBSET_SIZE` 张。
> 完整训练（全量 162K，100 epoch）用 `python -m cv.resnet50.train`，读同一份
> `config.yaml`；把下面 `SUBSET_SIZE` 改大即可在 notebook 里复现全量。
""")

code("""\
SUBSET_SIZE = 10000    # 演示子集；全量是 162,770

train_dataset = CelebADataset(split="train", transform=train_transform(), num_samples=SUBSET_SIZE)
val_dataset = CelebADataset(split="val", transform=train_transform(), num_samples=2000)
print(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}")
print(f"Attributes: {len(CELEBA_ATTR_ORDER)}")

train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=cfg["num_workers"], pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=True)
""")

md("""## 模型""")

code("""\
model = resnet50(num_classes=len(CELEBA_ATTR_ORDER)).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
""")

md("""## pos_weight""")

code("""\
pos_counts = torch.zeros(len(CELEBA_ATTR_ORDER))
for _, labels in train_loader:
    pos_counts += labels.sum(dim=0)
neg_counts = len(train_dataset) - pos_counts
pos_weight = (neg_counts / pos_counts).clamp(min=1.0)
print(f"pos_weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
""")

md("""\
## 训练

> ⏱ epoch / lr / momentum / weight_decay / T_max 全部来自 `config.yaml`；
> `amp` 为 true 时自动启用混合精度（CPU 上自动关闭）。
""")

code("""\
NUM_EPOCHS = cfg["num_epochs"]
LR = cfg["lr"]

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=cfg["momentum"],
                      weight_decay=cfg["weight_decay"])
scheduler = CosineAnnealingLR(optimizer, T_max=cfg["T_max"])

use_amp = bool(cfg.get("amp")) and device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
print(f"AMP: {use_amp}")

train_loss_hist, val_loss_hist, val_acc_hist = [], [], []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    scheduler.step()

    model.eval()
    val_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()

    train_loss_hist.append(train_loss / len(train_loader))
    val_loss_hist.append(val_loss / len(val_loader))
    val_acc_hist.append(correct / total * 100)
    print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}]  Train: {train_loss_hist[-1]:.4f}  "
          f"Val: {val_loss_hist[-1]:.4f}  Acc: {val_acc_hist[-1]:.2f}%")
""")

md("""## 训练曲线""")

code("""\
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_loss_hist, label="train"); ax1.plot(val_loss_hist, label="val")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True)
ax2.plot(val_acc_hist, color="green"); ax2.set_xlabel("Epoch")
ax2.set_ylabel("Val Acc (%)"); ax2.grid(True)
plt.tight_layout(); plt.show()
""")

md("""## 每属性准确率""")

code("""\
model.eval()
preds_all, labels_all = [], []
with torch.no_grad():
    for images, labels in val_loader:
        outputs = torch.sigmoid(model(images.to(device))).cpu()
        preds_all.append((outputs > 0.5).float())
        labels_all.append(labels)
preds_all = torch.cat(preds_all); labels_all = torch.cat(labels_all)

accs = sorted(
    ((CELEBA_ATTR_ORDER[i], (preds_all[:, i] == labels_all[:, i]).float().mean().item() * 100)
     for i in range(len(CELEBA_ATTR_ORDER))),
    key=lambda x: -x[1],
)
print(f"{'Attribute':<20}{'Accuracy':>10}")
print("-" * 32)
for name, acc in accs:
    print(f"{name:<20}{acc:>9.2f}%")
""")

md("""\
## 思考题

1. Bottleneck 的 1×1 卷积为什么能大幅减少计算量？给出一个具体的通道数例子。
2. 为什么更深不一定更好？ResNet 的残差连接解决了什么"退化"问题？
3. `pos_weight` 在 40 个属性上差异很大，不加权会怎样？
4. ResNet50 和 ResNet34 在同样数据/epoch 下，参数量和准确率各差多少？
5. 为什么 `T_max` 要和 `num_epochs` 一致？
""")

nb.cells = [bootstrap_cell(), config_cell("cv/resnet50/config.yaml"), *cells]
stabilize_ids(nb)
with open("cv/resnet50/resnet50.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated cv/resnet50/resnet50.ipynb")
