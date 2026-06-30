#!/usr/bin/env python3
"""Generate ViT notebook."""

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
# ViT: Vision Transformer

Apply Transformer Encoder (from BERT) to image patches for classification.
""")

md("""\
## 背景

ViT（Vision Transformer）将 Transformer 从 NLP 引入 CV。核心思路：
将图像切成固定大小的 patch，线性投影为 token 序列，加上 [CLS] 标记后用标准
Transformer Encoder 处理，最后用 [CLS] 的输出来分类。

与 CNN 不同，ViT 完全没有卷积操作，完全依赖自注意力捕捉全局依赖。
""")

md("""\
## 数学原理

### Patch Embedding

输入图像 $x \\in \\mathbb{R}^{H \\times W \\times C}$，切分为 $P \\times P$ 的 patch：

$$N = \\frac{H \\cdot W}{P^2}, \\quad x_p \\in \\mathbb{R}^{N \\times (P^2 \\cdot C)}$$

线性投影到 $d$ 维：

$$\\text{Embed}(x_p) = x_p \\cdot W_e, \\quad W_e \\in \\mathbb{R}^{(P^2 \\cdot C) \\times d}$$

### Transformer Encoder

同一套 Multi-Head Self-Attention + FFN + LayerNorm + Residual，定义在 `nlp.bert.model` 中。
""")

md("""\
## 架构

```
Image (32×32×3) → PatchEmbed(4×4) → 64 patches × 48-dim → Linear → 128-dim
→ +[CLS] token → +PosEmbed → EncoderBlock×4 → LayerNorm → CLS head → 10 classes
```
""")

code("""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from vit.model import ViT

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
""")

code("""\
# CIFAR-10 数据加载
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

def transform_batch(batch, fn):
    batch["img"] = [fn(img.convert("RGB")) for img in batch["img"]]
    return batch

ds_train = load_dataset("uoft-cs/cifar10", split="train")
ds_test = load_dataset("uoft-cs/cifar10", split="test")
ds_train.set_transform(lambda b: transform_batch(b, train_transform))
ds_test.set_transform(lambda b: transform_batch(b, test_transform))

train_loader = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(ds_test, batch_size=128, shuffle=False, num_workers=4)

CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
print(f"Train: {len(ds_train):,}  Test: {len(ds_test):,}")
""")

code("""\
model = ViT(
    d_model=128, n_heads=4, n_layers=4, d_ff=512,
    patch_size=4, num_classes=10, dropout=0.1,
).to(device)
print(f"Parameters: {model.num_params():,}")
""")

md("""\
## 训练

> ⏱ 预估耗时：**30 epoch × ~30s/epoch ≈ 15 分钟**（M4 Max, batch_size=128）
> 如果太久，把下面 `NUM_EPOCHS` 改小到 10 先看趋势。
""")

code("""\
NUM_EPOCHS = 30
LR = 1e-3

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_loss_hist, test_acc_hist = [], []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        images, labels = batch["img"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            logits = model(images)
            _, pred = torch.max(logits, 1)
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_loss_hist, marker='o')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss"); ax1.grid(True)

ax2.plot(test_acc_hist, marker='o', color='green')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Test Acc (%)"); ax2.set_title("Test Accuracy"); ax2.grid(True)

plt.tight_layout(); plt.show()
""")

md("""\
## 思考题

1. ViT 没有卷积的归纳偏置（平移不变性、局部性），它靠什么学到空间结构？
2. Patch size 越大越好还是越小越好？为什么？（提示：序列长度 vs 局部信息）
3. 把 epoch 数加到 100，ViT 能超过 CNN（82.4%）吗？
4. ViT 的 [CLS] token 和 BERT 的 [CLS] 有什么区别和联系？
""")

nb.cells = cells
out = "vit/vit.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Generated {out}")
