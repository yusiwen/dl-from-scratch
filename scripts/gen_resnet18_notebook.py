#!/usr/bin/env python3
"""Generate ResNet18 notebook."""

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
# ResNet18: Residual Networks

Deep residual learning with skip connections for image classification.
""")

md("""\
## 背景

ResNet（2015）解决了深度网络的退化问题：层数增加后训练误差反而上升。
核心创新是**残差连接（skip connection / shortcut）**：

$$y = \\mathcal{F}(x) + x$$

让梯度可以直接通过 shortcut 传播，使得训练上百层的网络成为可能。
ResNet18 是其中最轻量的版本，用 4 个 stage 共 18 个卷积层。
""")

md("""\
## 数学原理

### 残差块

$$\\mathcal{F}(x) = \\text{Conv3×3} \\to \\text{BN} \\to \\text{ReLU} \\to \\text{Conv3×3} \\to \\text{BN}$$

$$\\text{Output} = \\text{ReLU}(\\mathcal{F}(x) + x)$$

当维度不匹配时（跨 stage 下采样），shortcut 需额外 Conv1×1 来对齐：

$$\\text{Output} = \\text{ReLU}(\\mathcal{F}(x) + W_s \\cdot x)$$

### 架构

```
7×7 Conv(stride=2) → BN → ReLU → 3×3 MaxPool(stride=2)
→ [BasicBlock×2, 64]  stage1
→ [BasicBlock×2, 128] stage2
→ [BasicBlock×2, 256] stage3
→ [BasicBlock×2, 512] stage4
→ AvgPool → FC(512 → 15)
```
""")

code("""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from resnet18.data import CelebADataset, ATTRIBUTES
from resnet18.model import resnet18

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
""")

code("""\
# 数据加载
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = CelebADataset(ATTRIBUTES, num_samples=1000, transform=transform)
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [800, 200], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")
print(f"Attributes ({len(ATTRIBUTES)}): {', '.join(ATTRIBUTES)}")
""")

code("""\
model = resnet18(num_classes=len(ATTRIBUTES)).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
""")

md("""\
## 训练

> ⏱ 预估耗时：**30 epoch × ~20s/epoch ≈ 10 分钟**（M4 Max, batch_size=128）
""")

code("""\
NUM_EPOCHS = 30
LR = 1e-3

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_loss_hist, val_loss_hist, val_acc_hist = [], [], []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

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
    val_acc_hist.append(acc)
    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  Train: {avg_train:.4f}  Val: {avg_val:.4f}  Acc: {acc:.2f}%")
""")

md("""## Loss 曲线 & 验证准确率""")

code("""\
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_loss_hist, label='train', marker='o')
ax1.plot(val_loss_hist, label='val', marker='o')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True)

ax2.plot(val_acc_hist, marker='o', color='green')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Acc (%)"); ax2.grid(True)

plt.tight_layout(); plt.show()
""")

md("""## 每属性准确率""")

code("""\
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = torch.sigmoid(model(images)).cpu()
        all_preds.append((outputs > 0.5).float())
        all_labels.append(labels)

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

print(f"{'Attribute':<20} {'Accuracy':>8}")
print("-" * 30)
for i, attr in enumerate(ATTRIBUTES):
    acc = (all_preds[:, i] == all_labels[:, i]).sum().item() / all_labels.size(0) * 100
    print(f"{attr:<20} {acc:>7.2f}%")
""")

md("""\
## 思考题

1. 残差连接为什么能缓解梯度消失？画出有/无 shortcut 的梯度路径。
2. 1×1 卷积在 ResNet 中有什么作用？（提示：改变通道数）
3. ResNet18 换成 ResNet34（[3,4,6,3] blocks），参数量增加多少？
4. 尝试去掉 shortcut 训练（把 `+ x` 注释掉），观察 loss 有何不同。
""")

nb.cells = cells
out = "resnet18/resnet18.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Generated {out}")
