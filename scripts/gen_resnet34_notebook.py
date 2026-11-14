#!/usr/bin/env python3
"""Generate ResNet34 notebook."""

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
# ResNet34: Advanced ResNet

Extended ResNet with [3,4,6,3] blocks, full CelebA (40 attributes), data augmentation.
""")

md("""\
## 背景

ResNet34 是 ResNet18 的升级版：从 [2,2,2,2] 扩展到 [3,4,6,3] 个残差块。
除了更深的网络，ResNet34 在此项目中还展示了更完整的训练流程：

- **SGD + Momentum** 替代 Adam（更通用的优化器）
- **CosineAnnealingLR** 学习率调度
- **数据增强**：随机翻转、颜色抖动、旋转
- **Loss 加权**：`pos_weight` 处理属性不平衡
- **Early stopping**：按验证 loss 保存最优模型
""")

md("""\
## 架构对比

```
ResNet18: [BasicBlock×2] → [BasicBlock×2] → [BasicBlock×2] → [BasicBlock×2]
ResNet34: [BasicBlock×3] → [BasicBlock×4] → [BasicBlock×6] → [BasicBlock×3]
```

ResNet34 的每个 `BasicBlock` 结构与 ResNet18 完全相同（2× Conv3×3 + BN + ReLU）。
差异只在于 block 数量。

> 本项目中的 ResNet34 直接从 `cv.resnet18.model` 复用 `ResNet` 类和 `BasicBlock`，
> 仅通过 `num_blocks` 参数实现架构升级。
""")

code("""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from cv.resnet34.data import CelebADataset, CELEBA_ATTR_ORDER, train_transform
from cv.resnet34.model import resnet34
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")
cells.append(config_cell("cv/resnet34/config.yaml"))

code("""\
# 使用 20K 子集控制 notebook 训练时间（全量 162K）
SUBSET_SIZE = 20000

# num_samples 在 HF split 上切片，避免先把 162K 张图全部载入内存
train_dataset = CelebADataset(split="train", transform=train_transform(), num_samples=SUBSET_SIZE)
val_dataset = CelebADataset(split="val", transform=train_transform(), num_samples=2000)

train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)

print(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}")
print(f"Attributes: {len(CELEBA_ATTR_ORDER)}")
""")

code("""\
# 计算 pos_weight（从训练数据中统计正负样本比例）
print("Computing pos_weight...")
pos_counts = torch.zeros(40)
for _, labels in train_loader:
    pos_counts += labels.sum(dim=0)
neg_counts = len(train_dataset) - pos_counts
pos_weight = (neg_counts / pos_counts).clamp(min=1.0)
print(f"pos_weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
""")

code("""\
model = resnet34(num_classes=40).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
""")

md("""\
## 训练

> ⏱ epoch / lr / batch size 全部来自 config.yaml，上面的配置格会打印实际值。
""")

code("""\
NUM_EPOCHS = cfg["num_epochs"]
LR = cfg["lr"]
MOMENTUM = cfg["momentum"]
WEIGHT_DECAY = cfg["weight_decay"]

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=cfg["T_max"])

# amp（Automatic Mixed Precision，自动混合精度）来自 config.yaml；CPU 上
# enabled=False，与全精度训练完全等价。
use_amp = bool(cfg.get("amp")) and device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
print(f"AMP: {use_amp}")

train_loss_hist, val_loss_hist = [], []

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
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    acc = correct / total * 100
    train_loss_hist.append(avg_train)
    val_loss_hist.append(avg_val)
    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  Train: {avg_train:.4f}  Val: {avg_val:.4f}  Acc: {acc:.2f}%")
""")

md("""## Loss 曲线""")

code("""\
import matplotlib.pyplot as plt
from utils.device import get_device

plt.figure(figsize=(8, 4))
plt.plot(train_loss_hist, label='train', marker='o')
plt.plot(val_loss_hist, label='val', marker='o')
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
plt.title("ResNet34 Training on CelebA (20K subset)"); plt.show()
""")

md("""\
## 思考题

1. SGD + Momentum 和 Adam 各自的优缺点是什么？什么场景下 SGD 更好？
2. `pos_weight` 的作用是什么？不使用时对哪些属性影响最大？
3. 数据增强（翻转、颜色抖动、旋转）为什么会提升泛化能力？
4. 把 `SUBSET_SIZE` 改到 162770（全量），训练 30 epoch 观察效果。
5. 对比 ResNet18 和 ResNet34 在此任务上的表现差异。
""")

nb.cells = [bootstrap_cell(), *cells]
out = "cv/resnet34/resnet34.ipynb"
stabilize_ids(nb)
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Generated {out}")
