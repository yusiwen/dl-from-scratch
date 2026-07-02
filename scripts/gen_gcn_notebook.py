#!/usr/bin/env python3
"""Generate GCN notebook."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},"language_info": {"name": "python", "version": "3.12.0"}}

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("# GCN: Graph Convolutional Network\n\nNode classification on citation graphs using spectral graph convolution.\n")

md("""## 背景

GCN（Kipf & Welling, 2017）将卷积操作推广到图结构数据。核心思想：
每个节点的特征由其邻居节点加权聚合而来。

与图像 CNN 的区别：
- CNN：固定网格结构，卷积核在空间上滑动
- GCN：任意图结构，卷积由邻接矩阵定义的消息传递实现

数据集：**Cora** — 2708 篇论文，每篇用 1433 维词袋向量表示，分为 7 类。边表示引用关系。
""")

md("""## 数学原理

### 图卷积层

$$H^{(l+1)} = \\sigma\\left(\\hat{A} H^{(l)} W^{(l)}\\right)$$

其中 $\\hat{A} = D^{-1/2} A D^{-1/2}$ 是归一化邻接矩阵。

- $A$: 邻接矩阵（加自环后）
- $D$: 度矩阵 $D_{ii} = \\sum_j A_{ij}$
- $H^{(l)}$: 第 $l$ 层的节点表示
- $W^{(l)}$: 可学习的权重矩阵

### 2 层 GCN

$$Z = \\text{softmax}\\left(\\hat{A}\\ \\text{ReLU}\\left(\\hat{A} X W^{(0)}\\right) W^{(1)}\\right)$$

半监督学习：只用少量标注节点（每类 20 个）训练，模型通过图结构传播标签信息到未标注节点。
""")

code("""\
import torch
import torch.nn as nn
import torch.optim as optim

from graph.gcn.model import GCN
from graph.gcn.data import load_cora
from utils.config import load_config
from utils.seed import set_seed
from utils.device import get_device

device = get_device()
print(f"Device: {device}")

features, adj_norm, labels, train_mask, val_mask, test_mask, classes = load_cora()
features = features.to(device)
adj_norm = adj_norm.to(device)
labels = labels.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
""")

code("""\
model = GCN(
    in_features=features.size(1),
    hidden_dim=16,
    num_classes=labels.max().item() + 1,
    dropout=0.5,
).to(device)
print(f"Parameters: {model.num_params():,}")
""")

md("""## 训练

> ⏱ 预估耗时：**200 epoch × ~0.1s/epoch ≈ 20 秒**（CPU 即可完成）
""")

code("""\
NUM_EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 5e-4

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

loss_hist, acc_hist = [], []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj_norm)
    loss = criterion(output[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(features, adj_norm)
        val_acc = (output[val_mask].argmax(dim=1) == labels[val_mask]).float().mean().item()
        loss_hist.append(loss.item())
        acc_hist.append(val_acc)

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}]  Loss: {loss.item():.4f}  Val Acc: {val_acc:.2%}")
""")

md("""## Loss 曲线""")

code("""\
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(loss_hist); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss"); ax1.grid(True)
ax2.plot(acc_hist, color='green'); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Acc"); ax2.set_title("Validation Accuracy"); ax2.grid(True)
plt.tight_layout(); plt.show()
""")

md("""## 测试准确率""")

code("""\
model.eval()
with torch.no_grad():
    output = model(features, adj_norm)
    pred = output[test_mask].argmax(dim=1)
    test_acc = (pred == labels[test_mask]).float().mean().item()
print(f"Test Accuracy: {test_acc:.2%}")
""")

md("""\
## 思考题

1. 为什么 GCN 的归一化用 $D^{-1/2} A D^{-1/2}$ 而不是 $D^{-1} A$？
2. GCN 能处理归纳式（inductive）任务吗？还是只能直推式（transductive）？
3. 如果不用邻接矩阵只用节点特征，准确率会降到多少？
4. GCN 层数加深为什么会导致性能下降？（提示：过平滑问题）
""")

nb.cells = cells
with open("graph/gcn/gcn.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated graph/gcn/gcn.ipynb")
