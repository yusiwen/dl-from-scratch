#!/usr/bin/env python3
"""Generate the MLP notebook."""

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
# MLP: Multi-Layer Perceptron (pure NumPy)

A fully connected network on MNIST with hand-written forward and backward passes —
no autograd, no PyTorch tensors.
""")

md("""\
## 背景

多层感知机是最基本的神经网络：若干全连接层 + 非线性激活。

这个实现刻意**不用**任何自动求导框架：前向、反向传播、SGD 全部手写，
为的是看清 `loss.backward()` 背后到底发生了什么。模型代码见 `ml/mlp/model.py`。

结构：`784 → 256 → 128 → 10`（见 `config.yaml` 的 `layer_dims`）。
""")

md(r"""\
## 数学原理

**前向（ReLU 激活）：**

$$h_1 = \mathrm{ReLU}(x W_1 + b_1), \quad
h_2 = \mathrm{ReLU}(h_1 W_2 + b_2), \quad
z = h_2 W_3 + b_3$$

**Softmax + 交叉熵：**

$$\hat{y}_c = \frac{e^{z_c}}{\sum_j e^{z_j}}, \qquad
\mathcal{L} = -\sum_{c} y_c \log \hat{y}_c$$

**反向（链式法则）：**

$$\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y, \qquad
\frac{\partial \mathcal{L}}{\partial W_3} = h_2^\top \frac{\partial \mathcal{L}}{\partial z}$$

$$\frac{\partial \mathcal{L}}{\partial h_2} = \frac{\partial \mathcal{L}}{\partial z} W_3^\top,
\qquad \frac{\partial h_2}{\partial a_2} = \mathbb{1}[a_2 > 0]$$

其中 $a_2 = h_1 W_2 + b_2$ 是 ReLU 的输入；ReLU 的导数在 $a_2 > 0$ 时是 1，否则是 0。

**SGD 更新：**

$$W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}$$
""")

code("""\
import numpy as np
import matplotlib.pyplot as plt

from ml.mlp.data import load_mnist, MNIST_CLASSES
from ml.mlp.model import MLP, SGD

train_loader, test_loader = load_mnist(batch_size=cfg["batch_size"])
print(f"batch_size={cfg['batch_size']}  epochs={cfg['num_epochs']}  lr={cfg['lr']}")
""")

md("""## 模型""")

code("""\
model = MLP(cfg["layer_dims"])
optimizer = SGD(model, lr=cfg["lr"])
print(f"Layer dims: {cfg['layer_dims']}")
print(f"Parameters: {sum(p.size for p, _ in model.params()):,}")
""")

md("""\
## 训练

> ⏱ epoch / lr / 学习率衰减点全部来自 `config.yaml`。
> 这是完整训练（不是子集演示）；`python -m ml.mlp.train` 跑的是同一套配置。
""")

code("""\
num_epochs = cfg["num_epochs"]
lr_decay_epochs = set(cfg["lr_decay_epochs"])
lr_decay_factor = cfg["lr_decay_factor"]

loss_hist, acc_hist = [], []

for epoch in range(1, num_epochs + 1):
    epoch_loss, num_batches = 0.0, 0
    for images, labels, labels_one_hot in train_loader():
        logits = model.forward(images)
        epoch_loss += model.compute_loss(logits, labels_one_hot)
        model.backward(labels_one_hot)
        optimizer.step()
        num_batches += 1

    correct = total = 0
    for images, labels, _ in test_loader():
        logits = model.forward(images)
        predictions = np.argmax(logits, axis=1)
        correct += (predictions == labels).sum()
        total += len(labels)

    loss_hist.append(epoch_loss / num_batches)
    acc_hist.append(correct / total * 100)
    print(f"Epoch [{epoch:2d}/{num_epochs}]  Loss: {loss_hist[-1]:.4f}  "
          f"Test Acc: {acc_hist[-1]:.2f}%  LR: {optimizer.lr:.2e}")

    if epoch in lr_decay_epochs:
        optimizer.set_lr(optimizer.lr / lr_decay_factor)
""")

md("""## 训练曲线""")

code("""\
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(loss_hist, marker="o"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Training Loss"); ax1.grid(True)
ax2.plot(acc_hist, marker="o", color="green"); ax2.set_xlabel("Epoch")
ax2.set_ylabel("Test Acc (%)"); ax2.set_title("Test Accuracy"); ax2.grid(True)
plt.tight_layout(); plt.show()
""")

md("""## 每个数字的准确率""")

code("""\
per_class_correct = np.zeros(10)
per_class_total = np.zeros(10)
for images, labels, _ in test_loader():
    predictions = np.argmax(model.forward(images), axis=1)
    for cls in range(10):
        mask = labels == cls
        per_class_correct[cls] += (predictions[mask] == cls).sum()
        per_class_total[cls] += mask.sum()

print(f"{'Digit':<7}{'Accuracy':>10}")
print("-" * 18)
for cls in range(10):
    acc = per_class_correct[cls] / max(per_class_total[cls], 1) * 100
    print(f"{cls:<7}{acc:>9.2f}%")
""")

md("""\
## 思考题

1. 为什么 ReLU 比 sigmoid 更适合深层网络？（提示：梯度消失）
2. 交叉熵对 logits 的梯度是 $\\hat{y} - y$，这个形式是怎么来的？
3. 学习率在 epoch 5/10/15 各降 10 倍，如果不降会怎样？
4. 把 `layer_dims` 改成 `[784, 64, 10]`，参数量减少多少？准确率掉多少？
5. 纯 NumPy 版本和 PyTorch 版本相比，最大的性能瓶颈在哪里？
""")

nb.cells = [bootstrap_cell(), config_cell("ml/mlp/config.yaml"), *cells]
stabilize_ids(nb)
with open("ml/mlp/mlp.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated ml/mlp/mlp.ipynb")
