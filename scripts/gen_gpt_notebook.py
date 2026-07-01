#!/usr/bin/env python3
"""Generate GPT notebook."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12.0"},
}

cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    cells.append(nbf.v4.new_code_cell(source))


# ── Title ──
md("""\
# GPT: Decoder-Only Transformer

Autoregressive language model with Causal Self-Attention and KV Cache.
""")

# ── Background ──
md("""\
## 背景

GPT（Generative Pre-trained Transformer）是 decoder-only 的 Transformer 架构。
与 BERT 的双向注意力不同，GPT 使用 **因果掩码（causal mask）**——每个 token
只能看到自己和之前的位置，从而实现自回归生成。

训练目标：给定 tokens [t₁, t₂, ..., tₙ]，预测 [t₂, t₃, ..., tₙ₊₁]。
""")

# ── Math ──
md("""\
## 数学原理

### 自回归语言建模

$$P(x_1, ..., x_n) = \\prod_{i=1}^n P(x_i \\mid x_1, ..., x_{i-1})$$

训练损失（每个位置预测下一个 token）：

$$\\mathcal{L} = -\\sum_{i=1}^{n-1} \\log P(x_{i+1} \\mid x_1, ..., x_i)$$

### 缩放点积注意力 + 因果掩码

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V$$

因果掩码将上三角位置设为 $-\\infty$，保证位置 $i$ 只能 attend 到 $j \\leq i$。
""")

# ── Architecture ―
md("""\
## 架构

```
Token Embed → +Pos Embed → [DecoderBlock × N] → LayerNorm → LM Head → logits

DecoderBlock:
  Causal Self-Attention → Add + LayerNorm → FFN → Add + LayerNorm
```
""")

# ── Imports ──
code("""\
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from nlp.gpt.tokenizer import WordTokenizer
from nlp.gpt.model import GPT
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

# ── Tokenizer ──
code("""\
# 构建词级分词器
print("Loading text8 and building vocabulary...")
ds = load_dataset("afmck/text8", split="train")
raw = ds[0]["text"]

words = raw.lower().split()
chunk_words = 32
raw_chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]

tokenizer = WordTokenizer(vocab_size=5000)
tokenizer.build_vocab(raw_chunks)
print(f"Vocabulary size: {tokenizer.vocab_size}")
""")

# ── Dataset ──
code("""\
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=64):
        self.examples = []
        for text in texts:
            tokens, mask = tokenizer.encode(text, max_len)
            self.examples.append({
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

sentences = [s.strip() for s in raw_chunks if len(s.strip()) > 5]
sentences = sentences[:15000]  # 用 15K 子集控制训练时间
print(f"Training chunks: {len(sentences):,}")

dataset = TextDataset(sentences, tokenizer, max_len=64)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
""")

# ── Model ──
code("""\
model = GPT(
    vocab_size=tokenizer.vocab_size,
    d_model=256, n_heads=4, n_layers=4, max_len=64,
).to(device)
print(f"Parameters: {model.num_params():,}")
""")

# ── Training ──
md("""\
## 训练

> ⏱ 预估耗时：**15 epoch × ~45s/epoch ≈ 11 分钟**（M4 Max, batch_size=64）
> 如果觉得太久，可以把下面 `NUM_EPOCHS` 改小到 5 先看趋势。
""")

code("""\
NUM_EPOCHS = 15         # ← 可调
LR = 1e-3
GRAD_CLIP = 1.0

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

loss_history = []
ppl_history = []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)

        labels = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    loss_history.append(avg_loss)
    ppl_history.append(perplexity)
    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  Loss: {avg_loss:.4f}  PPL: {perplexity:.2f}")
""")

# ── Loss curve ──
md("""\
## Loss 曲线
""")

code("""\
import matplotlib.pyplot as plt
from utils.device import get_device

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(loss_history, marker='o')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss")
ax1.grid(True)

ax2.plot(ppl_history, marker='o', color='orange')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Perplexity"); ax2.set_title("Perplexity")
ax2.grid(True)

plt.tight_layout(); plt.show()
""")

# ── Generation ──
md("""\
## 文本生成

训练完成后，尝试用不同 temperature 生成文本，观察随机性变化。
""")

code("""\
def generate_text(model, tokenizer, prompt, max_new_tokens=30, temperature=0.8, top_k=40):
    tokens, _ = tokenizer.encode(prompt, max_len=64)
    input_ids = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            bad_tokens_ids=[tokenizer.sep_id, tokenizer.pad_id, tokenizer.cls_id],
        )
    full_text = tokenizer.decode(output[0].tolist())
    return full_text.strip()

prompts = [
    "the meaning of life is",
    "the future of artificial intelligence is",
]

for prompt in prompts:
    print(f"\\nPrompt: {prompt}")
    print("-" * 50)
    for temp in [0.3, 0.8, 1.2]:
        output = generate_text(model, tokenizer, prompt, temperature=temp)
        print(f"  t={temp:.1f}: {output}")
""")

# ── Temperature comparison ──
md("""\
### Temperature 的作用

- **t=0.3**：生成保守，几乎总是选概率最高的 token
- **t=0.8**：适度随机，结果多样但不散乱
- **t=1.2**：高随机性，可能偏离主题

读者可以回到训练 cell 改 `NUM_EPOCHS` 跑更久，观察更多 epoch 后生成质量的提升。
""")

# ── Thinking questions ──
md("""\
## 思考题

1. 如果去掉 causal mask，训练时每个 token 都能看到未来，模型会学到什么？推理时还能用吗？
2. KV Cache 为什么能加速推理？推理时每步的复杂度从 $O(T^2)$ 降到了多少？
3. Temperature 越高生成越随机还是越确定？从 softmax 的公式角度解释。
4. top-k 采样和 temperature 缩放各自的优缺点是什么？能同时用吗？
5. 把 `num_epochs` 从 15 改到 25，PPL 会降到多少？试试看。
""")

nb.cells = cells
out = "nlp/gpt/gpt.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Generated {out}")
