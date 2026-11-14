#!/usr/bin/env python3
"""Generate the MAE notebook."""

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
# MAE: Masked Autoencoder

遮住 75% 的图像块，让模型从剩下的 25% 重建被遮住的部分 —— 一种自监督预训练目标。
""")

md("""\
## 背景

MAE（Masked Autoencoder，掩码自编码器）把 NLP 里 BERT 的"完形填空"搬到图像上：

1. 把图像切成小块（patch），随机遮住大部分（这里 75%）；
2. **编码器只看可见块**（这是关键：省算力，也让任务更难）；
3. 解码器把编码结果和 mask token 拼回去，重建整张图；
4. 损失只在**被遮住的块**上计算。

学到的编码器可以直接拿去做下游任务（这里是线性探针）。
""")

md(r"""\
## 数学原理

**切块与遮罩：**

$$N = \frac{HW}{P^2}, \qquad n_{\text{keep}} = \lfloor N(1 - r) \rfloor$$

其中 $P$ 是 patch 边长，$r$ 是 mask ratio（`config.yaml` 里的 `mask_ratio`）。

**重建损失（只算被遮住的块）：**

$$\mathcal{L} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}}
\left\| \hat{p}_i - p_i \right\|^2$$

**非对称结构（MAE 的核心）：**

| 部分 | 输入 | 参数量 |
|---|---|---|
| 编码器 | 只有可见块（25%） | 大 |
| 解码器 | 可见块 + mask token（100%） | 小 |

因为编码器只处理 25% 的 token，MAE 的训练成本远低于"全图重建"。
""")

code("""\
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from selfsup.mae.data import load_cifar10_tensors
from selfsup.mae.model import MaskedAutoencoder
from selfsup.mae.eval import extract_features
from utils.probe import linear_probe, knn_accuracy
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

md("""## 数据""")

code("""\
images, labels = load_cifar10_tensors("train")
loader = DataLoader(
    TensorDataset(images, labels),
    batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
)
print(f"Data: {tuple(images.shape)}  batches: {len(loader)}")
""")

md("""## 模型""")

code("""\
model = MaskedAutoencoder(
    image_size=cfg["image_size"], patch_size=cfg["patch_size"],
    d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
    decoder_dim=cfg["decoder_dim"], decoder_layers=cfg["decoder_layers"],
    mask_ratio=cfg["mask_ratio"],
).to(device)
print(f"Parameters: {model.num_params():,}  mask_ratio={cfg['mask_ratio']}")
""")

md("""\
## 训练

> ⏱ epoch / lr / mask_ratio / 模型维度全部来自 `config.yaml`。
> `python -m selfsup.mae.train` 跑的是同一套配置；本 notebook 额外做一次线性探针。
""")

code("""\
optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

loss_hist = []
for epoch in range(1, cfg["num_epochs"] + 1):
    model.train()
    total_loss, num_batches = 0.0, 0
    for batch_images, _ in loader:
        batch_images = batch_images.to(device)
        loss = model(batch_images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    loss_hist.append(total_loss / num_batches)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:3d}/{cfg['num_epochs']}]  MSE: {loss_hist[-1]:.4f}")
""")

md("""## 损失曲线""")

code("""\
plt.plot(loss_hist)
plt.xlabel("Epoch"); plt.ylabel("Masked-patch MSE"); plt.title("MAE training loss")
plt.grid(True); plt.show()
""")

md("""\
## 评估：线性探针 + kNN

编码器冻结，只用 `encode()` 出来的 CLS 特征训练线性分类器 —— 标签从不参与预训练。
""")

code("""\
torch.save(model.state_dict(), cfg["model_path"])

train_images, train_labels = load_cifar10_tensors("train")
test_images, test_labels = load_cifar10_tensors("test")
train_x = extract_features(model, train_images, cfg["probe_batch_size"], device)
test_x = extract_features(model, test_images, cfg["probe_batch_size"], device)

probe_accuracy, _ = linear_probe(train_x, train_labels.numpy(), test_x, test_labels.numpy())
knn = knn_accuracy(train_x, train_labels.numpy(), test_x, test_labels.numpy(), k=20)

print(f"Features: {train_x.shape}")
print(f"Linear probe: {probe_accuracy:.2%}")
print(f"kNN (k=20):   {knn:.2%}")
""")

md("""\
## 思考题

1. mask ratio 75% 是不是太高了？如果改成 25% 或 95%，任务难度和学到的表示会怎么变？
2. 为什么编码器只处理可见块？这对训练成本有多大影响？
3. 损失为什么只在被遮住的块上算，而不是全图？
4. 线性探针准确率和重建 MSE 是什么关系？重建得好就一定有好的表示吗？
""")

nb.cells = [bootstrap_cell(), config_cell("selfsup/mae/config.yaml"), *cells]
stabilize_ids(nb)
with open("selfsup/mae/mae.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated selfsup/mae/mae.ipynb")
