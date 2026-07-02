#!/usr/bin/env python3
"""Generate CNN notebook."""

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
# CNN: Convolutional Neural Network

Simple CNN for CIFAR-10 image classification — Conv×3 + Pool×3 + FC×2.
""")

md("""\
## 背景

卷积神经网络（CNN）是计算机视觉的基础模型。核心操作是**卷积**：
一个可学习的滤波器（kernel）在输入上滑动，提取局部特征。

相比全连接网络，CNN 有三个关键特性：
- **局部连接**：每个神经元只关注局部区域
- **权重共享**：同一个滤波器在整个输入上复用
- **平移不变性**：特征检测器对位置不敏感

这些归纳偏置让 CNN 在处理图像时远优于 MLP。
""")

md("""\
## 数学原理

### 卷积

$$(I * K)_{i,j} = \\sum_{m} \\sum_{n} I_{i+m, j+n} \\cdot K_{m,n}$$

其中 $I$ 是输入图像，$K$ 是卷积核。每个通道独立做二维卷积，再跨通道求和。

### 最大池化

取 $k \\times k$ 区域内的最大值，降低空间分辨率：

$$\\text{MaxPool}(I)_{i,j} = \\max_{p,q \\in [0,k)} I_{i+p, j+q}$$

### 架构

```
Input(3×32×32) → Conv(3→64) → ReLU → MaxPool(2) →  (16×16)
              → Conv(64→128) → ReLU → MaxPool(2) → (8×8)
              → Conv(128→256) → ReLU → MaxPool(2) → (4×4)
              → Flatten → FC(256×4×4 → 256) → ReLU
              → FC(256 → 10)
```
""")

code("""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

code("""\
# CIFAR-10 data loading
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def build_transform(augment=False):
    ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else []
    ops += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(ops)

def transform_batch(batch, fn):
    batch["img"] = [fn(img.convert("RGB")) for img in batch["img"]]
    return batch

ds_train = load_dataset("uoft-cs/cifar10", split="train")
ds_test = load_dataset("uoft-cs/cifar10", split="test")
ds_train.set_transform(lambda b: transform_batch(b, build_transform(augment=True)))
ds_test.set_transform(lambda b: transform_batch(b, build_transform(augment=False)))

train_loader = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(ds_test, batch_size=128, shuffle=False, num_workers=4)

CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
print(f"Train: {len(ds_train):,}  Test: {len(ds_test):,}")
""")

code("""\
from cv.simplecnn.model import SimpleCNN

model = SimpleCNN(num_classes=10).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
""")

md("""\
## 训练

> ⏱ 预估耗时：**30 epoch × ~30s/epoch ≈ 15 分钟**（M4 Max, batch_size=128）
""")

code("""\
NUM_EPOCHS = 30
LR = 1e-3

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
        loss.backward()
        optimizer.step()
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
    test_acc = correct / total * 100
    train_loss_hist.append(avg_loss)
    test_acc_hist.append(test_acc)
    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  Loss: {avg_loss:.4f}  Test Acc: {test_acc:.2f}%")
""")

md("""## Loss 曲线 & 测试准确率""")

code("""\
import matplotlib.pyplot as plt
from utils.device import get_device

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_loss_hist, marker='o')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss"); ax1.grid(True)

ax2.plot(test_acc_hist, marker='o', color='green')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Test Acc (%)"); ax2.set_title("Test Accuracy"); ax2.grid(True)

plt.tight_layout(); plt.show()
""")

md("""## 类别准确率分析""")

code("""\
model.eval()
class_correct = [0] * 10
class_total = [0] * 10
with torch.no_grad():
    for batch in test_loader:
        images, labels = batch["img"].to(device), batch["label"].to(device)
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if pred[i].item() == label:
                class_correct[label] += 1

print(f"{'Class':<12} {'Accuracy':>8}")
print("-" * 22)
for i, name in enumerate(CIFAR10_CLASSES):
    acc = class_correct[i] / class_total[i] * 100
    print(f"{name:<12} {acc:>7.2f}%")
""")

md("""\
## 思考题

1. 为什么 CNN 比全连接网络更适合图像分类？（提示：三个归纳偏置）
2. 池化层的作用是什么？去掉池化层会怎样？
3. 卷积核大小（3×3 vs 5×5）对感受野和参数量有什么影响？
4. 试着把 Conv 层数从 3 加到 4，参数量和准确率会怎么变？
""")

nb.cells = cells
out = "cv/simplecnn/cnn.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Generated {out}")
