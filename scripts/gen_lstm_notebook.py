#!/usr/bin/env python3
"""Generate LSTM notebook."""

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
# LSTM: Long Short-Term Memory

Hand-written gates for sentiment classification on IMDB.
""")

md("""\
## 背景

LSTM（Hochreiter & Schmidhuber, 1997）解决了 RNN 的长期依赖问题。
核心机制是**门控（gating）**——通过三个门控制信息流：

- **输入门**：决定写入多少新信息到记忆
- **遗忘门**：决定丢弃多少旧记忆
- **输出门**：决定输出多少记忆

相比 RNN，LSTM 的梯度可以通过**细胞状态**直接传播，缓解梯度消失。

> 本项目中的 LSTM 实现是**手写的**（不是用 `nn.LSTM`），每个门控公式显式写出。
""")

md("""\
## 数学原理

### LSTM 单元

$$\\begin{aligned}
f_t &= \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f) \\quad \\text{(遗忘门)} \\\\
i_t &= \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i) \\quad \\text{(输入门)} \\\\
\\tilde{c}_t &= \\tanh(W_c \\cdot [h_{t-1}, x_t] + b_c) \\quad \\text{(候选细胞)} \\\\
c_t &= f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t \\quad \\text{(细胞更新)} \\\\
o_t &= \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o) \\quad \\text{(输出门)} \\\\
h_t &= o_t \\odot \\tanh(c_t) \\quad \\text{(隐藏状态)}
\\end{aligned}$$

取最后一步的隐藏状态 $h_T$ 经过全连接层做二分类（正/负面）。
""")

md("""\
## 架构

```
Input (chars) → Embedding(128) → LSTM(128→128) → FC(128→2) → logits
```

LSTM 内部每个时间步的手写门控逻辑见 `nlp/lstm/model.py`。
""")

code("""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from nlp.bert.tokenizer import CharTokenizer
from nlp.lstm.model import LSTMSentiment
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

code("""\
# 数据加载
tokenizer = CharTokenizer()
print(f"Vocabulary: {tokenizer.vocab_size}")

ds_train = load_dataset("stanfordnlp/imdb", split="train")
ds_test = load_dataset("stanfordnlp/imdb", split="test")

import random
indices = list(range(len(ds_train["text"])))
random.shuffle(indices)
train_texts = [ds_train[i]["text"] for i in indices[:5000]]
train_labels = [ds_train[i]["label"] for i in indices[:5000]]
test_texts = ds_test["text"][:1000]
test_labels = ds_test["label"][:1000]
print(f"Train: {len(train_texts)}  Test: {len(test_texts)}")
""")

code("""\
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.examples = []
        for text, label in zip(texts, labels):
            tokens, mask = tokenizer.encode(text, max_len)
            self.examples.append({
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
""")

code("""\
model = LSTMSentiment(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128, hidden_size=128, num_layers=1, num_classes=2,
).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
""")

md("""\
## 训练

> ⏱ 预估耗时：**10 epoch × ~50s/epoch ≈ 8 分钟**（M4 Max, batch_size=64）
""")

code("""\
NUM_EPOCHS = 10
LR = 5e-4

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

train_loss_hist, train_acc_hist, test_acc_hist = [], [], []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    train_correct = train_total = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = torch.max(logits, 1)
        train_correct += (pred == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    test_correct = test_total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            _, pred = torch.max(logits, 1)
            test_correct += (pred == labels).sum().item()
            test_total += labels.size(0)

    avg_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total * 100
    test_acc = test_correct / test_total * 100
    train_loss_hist.append(avg_loss)
    train_acc_hist.append(train_acc)
    test_acc_hist.append(test_acc)
    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  Loss: {avg_loss:.4f}  Train Acc: {train_acc:.1f}%  Test Acc: {test_acc:.1f}%")
""")

md("""## Loss 曲线 & 准确率""")

code("""\
import matplotlib.pyplot as plt
from utils.device import get_device

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_loss_hist, marker='o')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss"); ax1.grid(True)

ax2.plot(train_acc_hist, label='train', marker='o')
ax2.plot(test_acc_hist, label='test', marker='o')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.legend(); ax2.grid(True)

plt.tight_layout(); plt.show()
""")

md("""\
## 思考题

1. LSTM 的三个门分别控制什么信息流？没有哪个门会导致什么后果？
2. LSTM 如何缓解梯度消失？对比 RNN 的梯度路径。
3. 字符级 vs 词级 tokenization 对情感分类的影响？为什么 LSTM 用字符级表现差？
4. 把 LSTM 层数从 1 加到 2，准确率会提升吗？参数量增加多少？
""")

nb.cells = cells
out = "nlp/lstm/lstm.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Generated {out}")
