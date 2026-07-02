#!/usr/bin/env python3
"""Generate LoRA notebook."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},"language_info": {"name": "python", "version": "3.12.0"}}

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("# LoRA: Low-Rank Adaptation\n\nParameter-efficient fine-tuning of GPT via low-rank weight updates.")

md("""## 背景

LoRA（Hu et al. 2021）解决大模型全量微调成本过高的问题。核心思想：

冻结预训练权重 $W_0 \\in \\mathbb{R}^{d \\times k}$，学习一对低秩矩阵 $B \\in \\mathbb{R}^{d \\times r}, A \\in \\mathbb{R}^{r \\times k}$，
其中 $r \\ll \\min(d, k)$。

$$h = W_0 x + \\Delta W x = W_0 x + \\frac{\\alpha}{r} BAx$$

| 方法 | 可训练参数量 | 对 GPT (5.7M) |
|---|---|---|
| Full fine-tune | $d \\times k$ | 5,693,952 |
| LoRA r=8 | $(d + k) \\times r$ | **32,772** (0.58%) |

本项目在已训练好的本地 GPT（text8）上注入 LoRA，仅微调注意力层的 Q/K/V/O 投影矩阵。
""")

md("""## 数学原理

### 低秩分解

$$\\Delta W = BA, \\quad B \\in \\mathbb{R}^{d \\times r}, A \\in \\mathbb{R}^{r \\times k}$$

推理时将 $\\Delta W$ 合并回原始权重，零额外开销：

$$W_{\\text{merged}} = W_0 + \\frac{\\alpha}{r} BA$$

### 参数效率

$$\\text{Ratio} = \\frac{r(d + k)}{d \\cdot k} \\approx \\frac{r}{k} + \\frac{r}{d}$$

当 $r=8, d=k=256$ 时：
$$\\text{Ratio} = \\frac{8 \\times 512}{256 \\times 256} = \\frac{4096}{65536} \\approx 6.25\\%$$
""")

code("""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset

from nlp.gpt.model import GPT
from nlp.gpt.tokenizer import WordTokenizer
from nlp.lora.model import LoRALayer, inject_lora, freeze_all_except_lora, lora_params_count
from utils.config import load_config
from utils.seed import set_seed
from utils.device import get_device

device = get_device()
print(f"Device: {device}")
""")

code("""\
# Load pretrained GPT
checkpoint = "nlp/gpt/gpt_text8.pt"
model = torch.load(checkpoint, map_location="cpu", weights_only=False).to(device)
full_params = sum(p.numel() for p in model.parameters())
print(f"Full model: {full_params:,} params")

# Compare different ranks
print(f"\\n{'r':>3} {'LoRA params':>12} {'Ratio':>8}")
print("-" * 25)
for r in [1, 4, 8, 16]:
    # Quick count: each of 4 Linear × 4 layers × (d*r + r*d) = 4×4×2×d×r
    d = 256
    lora_count = 4 * 4 * 2 * d * r
    print(f"{r:3d} {lora_count:>10,d} {lora_count/full_params:>7.2%}")
""")

code("""\
# Inject LoRA with r=8
freeze_all_except_lora(model)
model = inject_lora(model, r=8, alpha=16)

lora_count = lora_params_count(model)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"LoRA params: {lora_count:,} ({lora_count/full_params:.2%})")
print(f"Trainable:   {trainable:,}")
""")

md("""## 训练

> ⏱ 预估耗时：**10 epoch × ~60s/epoch ≈ 10 分钟**（M4 Max, batch_size=32）
""")

code("""\
# Load text8 subset
ds = load_dataset("afmck/text8", split="train")
raw = ds[0]["text"]
words = raw.lower().split()
raw_chunks = [" ".join(words[i:i + 32]) for i in range(0, len(words), 32)]

tokenizer = WordTokenizer(vocab_size=5000)
tokenizer.build_vocab(raw_chunks)
sentences = [s.strip() for s in raw_chunks if len(s.strip()) > 5][:5000]

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len=64):
        self.examples = []
        for text in texts:
            tokens, mask = tokenizer.encode(text, max_len)
            self.examples.append({"input_ids": torch.tensor(tokens, dtype=torch.long)})
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

dataset = TextDataset(sentences, tokenizer, max_len=64)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"Training chunks: {len(sentences):,}")
""")

code("""\
NUM_EPOCHS = 10
LR = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

loss_hist, ppl_hist = [], []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward(); optimizer.step()
        total_loss += loss.item()

    avg = total_loss / len(loader)
    ppl = __import__('math').exp(avg)
    loss_hist.append(avg)
    ppl_hist.append(ppl)
    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  Loss: {avg:.4f}  PPL: {ppl:.2f}")
""")

md("""## Loss 曲线""")

code("""\
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(loss_hist, marker='o'); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.grid(True)
ax2.plot(ppl_hist, marker='o', color='orange'); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Perplexity"); ax2.grid(True)
plt.tight_layout(); plt.show()
""")

md("""\
## 思考题

1. LoRA 的 rank $r$ 越大越好还是越小越好？分析过拟合和表达能力之间的权衡。
2. 为什么只修改注意力层的 Q/K/V/O 而不修改 FFN？
3. 推理时如何将 LoRA 权重合并回原模型，实现零额外开销？
4. 尝试将 $\\alpha$ 从 16 改到 4 或 64，观察训练速度和 loss 的变化。
""")

nb.cells = cells
with open("nlp/lora/lora.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated nlp/lora/lora.ipynb")
