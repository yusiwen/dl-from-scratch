#!/usr/bin/env python3
"""Generate UNet notebook."""

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
# UNet: Semantic Segmentation

Encoder-decoder with skip connections for pixel-wise classification.
""")

md("""\
## 背景

UNet 最初为医学图像分割设计，核心创新是 **跳跃连接（skip connections）**：
编码器逐步下采样提取高层语义，解码器逐步上采样恢复空间分辨率，
跳跃连接将编码器每层的特征直接拼接到对应解码器层，保留细节信息。

数据集：Oxford-IIIT Pet — 图像 + 逐像素标注（3 类：前景、背景、轮廓）。
""")

md("""\
## 数学原理

### 逐像素分类

每个像素被分类为 $C$ 类之一：

$$\\hat{y}_{i,j} = \\arg\\max_c \\, \\text{logits}_{i,j,c}$$

### 损失函数

$$\\mathcal{L} = -\\frac{1}{N} \\sum_{i,j} \\sum_c y_{i,j,c} \\log(\\hat{y}_{i,j,c})$$

忽略未标注像素（`ignore_index=0`）。

### 架构

```
Input → Conv+ReLU →           ← skip ─── UpConv → Conv+ReLU → Conv1×1 → C classes
         ↓ MaxPool                                ↑
       Conv+ReLU →           ← skip ─── UpConv → Conv+ReLU
         ↓ MaxPool                                ↑
       Conv+ReLU →           ← skip ─── UpConv → Conv+ReLU
         ↓ MaxPool                                ↑
       Conv+ReLU →           ← skip ─── UpConv → Conv+ReLU
         ↓ MaxPool                                ↑
      Conv+ReLU × 2 (bottleneck)
```
""")

code("""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from datasets import load_dataset
import random

from cv.unet.model import UNet
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

code("""\
class PetDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", image_size=128, augment=False):
        self.image_size = image_size
        self.augment = augment and split == "train"
        ds = load_dataset("tchevrou/oxford-iiit-pet", split=split)
        self.images = [item["image"].convert("RGB") for item in ds]
        self.masks = [item["label"] for item in ds]
        del ds

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = TF.resize(image, self.image_size, TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, TF.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image); mask = TF.hflip(mask)
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask = torch.tensor(list(mask.getdata()), dtype=torch.long).view(self.image_size, self.image_size)
        return image, mask

train_dataset = PetDataset(split="train", image_size=128, augment=True)
test_dataset = PetDataset(split="test", image_size=128, augment=False)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
print(f"Train: {len(train_dataset):,}  Test: {len(test_dataset):,}")
""")

code("""\
model = UNet(in_channels=3, num_classes=3).to(device)
print(f"Parameters: {model.num_params():,}")
""")

md("""\
## 训练

> ⏱ 预估耗时：**30 epoch × ~40s/epoch ≈ 20 分钟**（M4 Max, batch_size=16）
> 如果太久，把下面 `NUM_EPOCHS` 改小到 10 先看趋势。
""")

code("""\
NUM_EPOCHS = 30
LR = 1e-3

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=LR)

train_loss_hist, val_loss_hist, pixel_acc_hist = [], [], []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            valid = masks != 0
            correct += (preds[valid] == masks[valid]).sum().item()
            total += valid.sum().item()

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(test_loader)
    acc = correct / total * 100
    train_loss_hist.append(avg_train)
    val_loss_hist.append(avg_val)
    pixel_acc_hist.append(acc)
    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  Train: {avg_train:.4f}  Val: {avg_val:.4f}  Pixel Acc: {acc:.2f}%")
""")

md("""## Loss 曲线 & 像素准确率""")

code("""\
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_loss_hist, label='train', marker='o')
ax1.plot(val_loss_hist, label='val', marker='o')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True)

ax2.plot(pixel_acc_hist, marker='o', color='green')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Pixel Acc (%)"); ax2.grid(True)

plt.tight_layout(); plt.show()
""")

md("""## 分割效果可视化""")

code("""\
import matplotlib.pyplot as plt
import numpy as np
from utils.device import get_device

model.eval()
images, masks = next(iter(test_loader))
with torch.no_grad():
    preds = torch.argmax(model(images.to(device)), dim=1).cpu()

CLASS_CMAP = np.array([[0,0,0], [0,180,60], [180,60,0], [60,0,180]], dtype=np.uint8)

fig, axes = plt.subplots(4, 3, figsize=(12, 16))
for i in range(4):
    img = images[i].permute(1,2,0).numpy()
    img = img * [0.229,0.224,0.225] + [0.485,0.456,0.406]
    img = np.clip(img, 0, 1)
    axes[i,0].imshow(img); axes[i,0].set_title("Input"); axes[i,0].axis("off")

    mask_true = CLASS_CMAP[masks[i].numpy()]
    axes[i,1].imshow(mask_true); axes[i,1].set_title("Ground Truth"); axes[i,1].axis("off")

    mask_pred = CLASS_CMAP[preds[i].numpy()]
    axes[i,2].imshow(mask_pred); axes[i,2].set_title("Prediction"); axes[i,2].axis("off")

plt.tight_layout(); plt.show()
""")

md("""\
## 思考题

1. 跳跃连接（skip connection）在 UNet 中起到什么作用？没有它会怎样？
2. 上采样为什么用转置卷积而不用双线性插值？各自的优缺点是什么？
3. 如果使用 Dice Loss 代替 CrossEntropy，分割效果会有什么变化？
4. 试试把 `NUM_EPOCHS` 加到 100，观察 IoU 和 pixel accuracy 的变化。
""")

nb.cells = cells
out = "cv/unet/unet.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Generated {out}")
