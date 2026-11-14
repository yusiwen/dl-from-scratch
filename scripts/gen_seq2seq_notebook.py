#!/usr/bin/env python3
"""Generate the Seq2Seq Transformer notebook."""

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
# Seq2Seq Transformer: EN → DE 翻译

编码器-解码器结构 + 交叉注意力，把一句英文翻译成德文。数据集是 Multi30K。
""")

md("""\
## 背景

前面的 GPT / BERT 都是**单塔**结构：只读一种序列。翻译需要**双塔**：

1. **编码器**读入源语言（英文），产出每个位置的表示；
2. **解码器**一边自回归生成德文，一边通过**交叉注意力**回看编码器的输出。

交叉注意力是这里唯一新增的机制：Query 来自解码器，Key/Value 来自编码器 ——
"我要生成的这个词，应该看源句的哪些位置？"
""")

md(r"""\
## 数学原理

**缩放点积注意力：**

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

**自注意力 vs 交叉注意力：**

| | Query | Key / Value |
|---|---|---|
| 编码器自注意力 | 编码器输入 | 编码器输入 |
| 解码器自注意力（带因果掩码） | 解码器输入 | 解码器输入 |
| **交叉注意力** | 解码器 | **编码器输出** |

**Teacher forcing 训练：**

$$\hat{y}_t = f_\theta(y_{<t}, x), \qquad
\mathcal{L} = -\sum_{t} \log p_\theta(y_t \mid y_{<t}, x)$$

训练时把**真实**的前缀 $y_{<t}$ 喂给解码器（而不是模型自己上一步的输出），
推理时（`generate`）才逐词自回归。
""")

md("""## 数据与分词""")

code("""\
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from nlp.seq2seq.data import load_multi30k, SPECIAL
from nlp.seq2seq.model import Transformer
from nlp.seq2seq.generate import translate
from utils.device import get_device

device = get_device()
print(f"Device: {device}")

train_loader, test_loader, tokenizer = load_multi30k(
    batch_size=cfg["batch_size"], vocab_size=cfg["vocab_size"],
    max_len=cfg["max_len"], num_workers=cfg["num_workers"],
)
print(f"Train batches: {len(train_loader)}  Test batches: {len(test_loader)}")
print(f"Vocab size: {tokenizer.vocab_size:,}")
""")

md("""## 模型""")

code("""\
model = Transformer(
    vocab_size=tokenizer.vocab_size,
    d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
    d_ff=cfg["d_ff"], max_len=cfg["max_len"], dropout=cfg["dropout"],
    pad_idx=SPECIAL["[PAD]"],
).to(device)
print(f"Parameters: {model.num_params():,}")
""")

md("""\
## 训练

> ⏱ epoch / lr / 模型维度全部来自 `config.yaml`。
> `python -m nlp.seq2seq.train` 跑的是同一套配置和数据。
""")

code("""\
criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL["[PAD]"])
optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

loss_hist = []
for epoch in range(1, cfg["num_epochs"] + 1):
    model.train()
    total_loss, num_batches = 0.0, 0

    for src, tgt, src_mask in train_loader:
        src, tgt, src_mask = src.to(device), tgt.to(device), src_mask.to(device)

        # Teacher forcing: predict tgt[1:] from tgt[:-1].
        decoder_input = tgt[:, :-1]
        labels = tgt[:, 1:].contiguous()

        optimizer.zero_grad()
        logits = model(src, decoder_input, src_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    loss_hist.append(total_loss / num_batches)
    print(f"Epoch [{epoch:2d}/{cfg['num_epochs']}]  Loss: {loss_hist[-1]:.4f}")
""")

md("""## 训练曲线""")

code("""\
plt.plot(loss_hist, marker="o")
plt.xlabel("Epoch"); plt.ylabel("Cross-entropy"); plt.title("Seq2Seq training loss")
plt.grid(True); plt.show()
""")

md("""## 翻译效果（greedy decoding）""")

code("""\
samples = [
    "a man is riding a horse",
    "two dogs are running in the snow",
    "a group of people are sitting at a table",
]

model.eval()
for text in samples:
    print(f"EN: {text}")
    print(f"DE: {translate(model, tokenizer, text, max_len=cfg['max_len'], device=device)}")
    print()
""")

md("""\
## 思考题

1. 交叉注意力的 Query 为什么来自解码器、Key/Value 来自编码器？反过来会怎样？
2. 为什么解码器的自注意力要加因果掩码，编码器的不用？
3. Teacher forcing 训练和推理时的输入分布不一致（exposure bias），这会带来什么问题？
4. 用贪心解码和 beam search 会有什么区别？
""")

nb.cells = [bootstrap_cell(), config_cell("nlp/seq2seq/config.yaml"), *cells]
stabilize_ids(nb)
with open("nlp/seq2seq/seq2seq.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated nlp/seq2seq/seq2seq.ipynb")
