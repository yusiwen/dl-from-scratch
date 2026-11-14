#!/usr/bin/env python3
"""Generate the BYOL notebook."""

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
# BYOL: Bootstrap Your Own Latent

不用负样本的自监督学习：一个在线网络去预测另一个**动量目标网络**的输出，
学到的表示用线性探针评估。
""")

md("""\
## 背景

SimCLR 这类对比学习依赖**负样本**：同一张图的两个增强要靠近，不同图要远离。
负样本需要大 batch，而且会引入"哪些算负例"的采样问题。

BYOL（Bootstrap Your Own Latent）的做法完全不同：

- **在线网络**（online）：编码器 + projector + predictor，只有它接收梯度；
- **目标网络**（target）：编码器 + projector，参数是在线网络参数的**指数滑动平均**（EMA），
  并且**不接收梯度**；
- 损失是两路输出之间的余弦相似度。

没有负样本，模型本来会"塌缩"（所有输入映射到同一个点，损失为 0）。BYOL 靠两个机制避免它：
**predictor 的不对称性**和**目标网络的 EMA + 中心化**。
""")

md(r"""\
## 数学原理

**对称化的余弦损失（两个视角各算一次）：**

$$\mathcal{L} = 2 - 2 \cdot \frac{\langle q_\theta(z),\, z'_{\xi}\rangle}{\|q_\theta(z)\| \, \|z'_{\xi}\|}
+ 2 - 2 \cdot \frac{\langle q_\theta(z'),\, z_{\xi}\rangle}{\|q_\theta(z')\| \, \|z_{\xi}\|}$$

其中 $q_\theta$ 是带 predictor 的在线分支，$z'_\xi$ 是目标分支的输出（stop-gradient）。

**目标网络 EMA 更新：**

$$\xi \leftarrow m\,\xi + (1 - m)\,\theta, \qquad m = 0.999$$

**中心化（防止某个维度主导）：**

$$c \leftarrow m_c\, c + (1 - m_c)\,\bar{z}, \qquad
\mathcal{L} \mathrel{+}= \text{center penalty}$$

$\theta$ 更新、$\xi$ 慢慢跟随 —— 这就是"自己教自己"。
""")

code("""\
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from selfsup.byol.model import BYOL
from selfsup.simclr.data import load_cifar10_simclr, load_cifar10_eval
from selfsup.byol.eval import extract_features, knn_accuracy
from utils.probe import linear_probe
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

md("""## 数据（两个增强视角）""")

code("""\
loader = load_cifar10_simclr(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
print("50,000 CIFAR-10 images, two augmented views each")
print(f"Batch size: {cfg['batch_size']}")
""")

md("""## 模型""")

code("""\
model = BYOL(
    projector_hidden=cfg["projector_hidden"],
    projector_dim=cfg["projector_dim"],
    predictor_hidden=cfg["predictor_hidden"],
    center_momentum=cfg["center_momentum"],
).to(device)
print(f"Parameters: {model.num_params():,} (online + target)")
""")

md("""\
## 训练

> ⏱ epoch / lr / EMA 衰减 / warmup 全部来自 `config.yaml`。
> `python -m selfsup.byol.train` 跑的是同一套配置；本 notebook 额外做一次线性探针。
""")

code("""\
optimizer = optim.AdamW(
    [
        {"params": model.online_encoder.parameters()},
        {"params": model.online_projector.parameters()},
        {"params": model.predictor.parameters()},
    ],
    lr=cfg["lr"], weight_decay=cfg["weight_decay"],
)

warmup_epochs = cfg.get("warmup_epochs", 0)
loss_hist = []

for epoch in range(1, cfg["num_epochs"] + 1):
    if warmup_epochs > 0:
        scale = min(1.0, epoch / warmup_epochs)
        for group in optimizer.param_groups:
            group["lr"] = cfg["lr"] * scale

    model.train()
    total_loss, num_batches = 0.0, 0
    for batch in loader:
        view1 = batch["view1"].to(device)
        view2 = batch["view2"].to(device)

        loss = model(view1, view2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_target(cfg["ema_decay"])

        total_loss += loss.item()
        num_batches += 1

    loss_hist.append(total_loss / num_batches)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:3d}/{cfg['num_epochs']}]  Loss: {loss_hist[-1]:.4f}")
""")

md("""## 损失曲线""")

code("""\
plt.plot(loss_hist)
plt.xlabel("Epoch"); plt.ylabel("BYOL loss"); plt.title("BYOL training loss")
plt.grid(True); plt.show()
""")

md("""\
## 评估：线性探针 + kNN

自监督训练不碰标签；评估时冻结编码器，只在特征上训练一个线性分类器。
""")

code("""\
torch.save(model.state_dict(), cfg["model_path"])

encoder = model.online_encoder
encoder.eval()
train_loader, test_loader = load_cifar10_eval(num_workers=cfg["num_workers"])
train_x, train_y = extract_features(encoder, train_loader, device)
test_x, test_y = extract_features(encoder, test_loader, device)

probe_accuracy, _ = linear_probe(train_x, train_y, test_x, test_y)
knn = knn_accuracy(train_x, train_y, test_x, test_y, k=20)

print(f"Features: {train_x.shape}  per-dim std {train_x.std(axis=0).mean():.3f}")
print(f"Linear probe: {probe_accuracy:.2%}")
print(f"kNN (k=20):   {knn:.2%}")
""")

md("""\
## 思考题

1. 没有负样本，BYOL 为什么不会塌缩？predictor 和 EMA 各起什么作用？
2. 如果去掉目标网络的 stop-gradient 会怎样？
3. 为什么更新的是目标网络用 EMA、而梯度只在在线分支上？
4. 线性探针准确率和训练损失之间是什么关系？损失降了就一定表示更好吗？
""")

nb.cells = [bootstrap_cell(), config_cell("selfsup/byol/config.yaml"), *cells]
stabilize_ids(nb)
with open("selfsup/byol/byol.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated selfsup/byol/byol.ipynb")
