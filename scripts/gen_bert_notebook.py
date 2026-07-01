#!/usr/bin/env python3
"""Generate BERT notebook."""

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
# BERT: Bidirectional Encoder Representations from Transformers

Transformer Encoder pre-trained with Masked Language Model (MLM).
""")

md("""\
## 背景

BERT（2018）通过**掩码语言模型（MLM）**在无标注文本上预训练，
然后用少量标注数据微调下游任务。核心思想是**"熵增降噪"**：

1. **熵增**：随机掩码 15% 的 token，增加不确定性
2. **降噪**：训练 Transformer 根据上下文预测被掩码的词

与 GPT 不同，BERT 使用**双向注意力**——每个 token 可以 attend 到所有 token。
""")

md("""\
## 数学原理

### Masked Language Model

随机选择 15% 的位置进行干扰：
- 80% 替换为 `[MASK]`
- 10% 替换为随机 token
- 10% 保持不变

损失只计算被掩码的位置：

$$\\mathcal{L} = -\\sum_{i \\in \\mathcal{M}} \\log P(x_i \\mid \\mathbf{x}_{\\setminus i})$$

其中 $\\mathcal{M}$ 是被掩码的位置集合。

### 架构

```
Input → Token Embed + Segment Embed + Position Encoding
     → [EncoderBlock × N] → LayerNorm → MLM Head → vocab logits
```
""")

code("""\
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from nlp.bert.tokenizer import CharTokenizer
from nlp.bert.model import BERTForMLM
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

code("""\
# 构建字符级分词器
tokenizer = CharTokenizer()
print(f"Vocabulary size: {tokenizer.vocab_size}")

# 加载 text8
ds = load_dataset("afmck/text8", split="train")
raw = ds[0]["text"]
chunk_size = 1000
chunks = [raw[i:i + chunk_size] for i in range(0, len(raw), chunk_size)]
chunks = chunks[:5000]
print(f"Chunks: {len(chunks):,}")
""")

code("""\
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.examples = []
        for text in texts:
            tokens, _ = tokenizer.encode(text, max_len)
            self.examples.append(tokens)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = list(self.examples[idx])
        labels = list(tokens)
        for i in range(len(tokens)):
            if tokens[i] in (self.tokenizer.cls_id, self.tokenizer.sep_id, self.tokenizer.pad_id):
                continue
            if torch.rand(1).item() < self.mask_prob:
                r = torch.rand(1).item()
                if r < 0.8:  tokens[i] = self.tokenizer.mask_id
                elif r < 0.9: tokens[i] = torch.randint(5, self.tokenizer.vocab_size, (1,)).item()
            else:
                labels[i] = -100
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if t != self.tokenizer.pad_id else 0 for t in tokens], dtype=torch.long
            ),
        }

dataset = TextDataset(chunks, tokenizer, max_len=128)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
""")

code("""\
model = BERTForMLM(
    vocab_size=tokenizer.vocab_size,
    d_model=128, n_heads=4, n_layers=4, max_len=128,
).to(device)
print(f"Parameters: {model.num_params():,}")
""")

md("""\
## 训练

> ⏱ 预估耗时：**10 epoch × ~60s/epoch ≈ 10 分钟**（M4 Max, batch_size=32）
""")

code("""\
NUM_EPOCHS = 10
LR = 1e-4

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.AdamW(model.parameters(), lr=LR)

loss_history = []
ppl_history = []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    loss_history.append(avg_loss)
    ppl_history.append(perplexity)
    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  Loss: {avg_loss:.4f}  PPL: {perplexity:.2f}")
""")

md("""## Loss 曲线""")

code("""\
import matplotlib.pyplot as plt
from utils.device import get_device

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(loss_history, marker='o')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss"); ax1.grid(True)

ax2.plot(ppl_history, marker='o', color='orange')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Perplexity"); ax2.set_title("Perplexity"); ax2.grid(True)

plt.tight_layout(); plt.show()
""")

md("""## [MASK] 预测演示

训练完成后，输入一段带 `[MASK]` 的文本，观察模型预测结果。""")

code("""\
def predict_mask(text, model, tokenizer, top_k=5):
    tokens, mask = tokenizer.encode(text, max_len=128)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    mask_id = tokenizer.mask_id

    with torch.no_grad():
        logits, _ = model(input_ids)
        probs = torch.softmax(logits[0], dim=-1)

    mask_positions = [i for i, t in enumerate(tokens) if t == mask_id]
    for pos in mask_positions:
        top_probs, top_indices = torch.topk(probs[pos], top_k)
        preds = [tokenizer.id_to_word[idx.item()] for idx in top_indices]
        print(f"Position {pos}: {preds} (probs: {top_probs.tolist()})")

# 示例：输入含 [MASK] 的句子
prompts = [
    "once upon a [MASK] there was a beautiful princess",
    "the [MASK] is shining brightly in the sky today",
]

for text in prompts:
    print(f"\\nInput: {text}")
    print("-" * 50)
    predict_mask(text, model, tokenizer)
""")

md("""\
## 思考题

1. BERT 为什么用 80/10/10 的掩码策略？100% [MASK] 会怎样？
2. 双向注意力（BERT）和因果注意力（GPT）分别适合什么任务？
3. 把 `mask_prob` 改到 0.5（50% 掩码），loss 会上升还是下降？试试看。
4. BERT 的 [CLS] token 在预训练中没有明确任务，为什么能用于分类？
""")

nb.cells = cells
out = "nlp/bert/bert.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Generated {out}")
