#!/usr/bin/env python3
"""Generate Word2Vec notebook."""

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
# Word2Vec: Word Embeddings

CBOW + Skip-gram with Negative Sampling, trained on text8.
""")

md("""\
## 背景

Word2Vec（Mikolov 2013）学习词的分布式表示（dense vectors），
使得语义相近的词在向量空间中也相近。

两种架构：
- **CBOW**：用上下文词预测中心词
- **Skip-gram**：用中心词预测上下文词

两种加速技巧：
- **Negative Sampling**：训练时只更新少量负样本，避免完整的 softmax
- **Subsampling**：高频词（"the", "a"）被概率性丢弃，加速训练
""")

md("""\
## 数学原理

### Skip-gram with Negative Sampling

给定中心词 $w_t$ 和上下文词 $w_c$：

$$\\mathcal{L} = -\\log \\sigma(v_{w_c} \\cdot v_{w_t}) - \\sum_{k=1}^K \\log \\sigma(-v_{w_k} \\cdot v_{w_t})$$

其中 $w_k$ 是从噪声分布 $P(w) \\propto \\text{count}(w)^{0.75}$ 采样的负样本。

### CBOW

$$\\mathcal{L} = -\\log \\sigma(v_{w_t} \\cdot \\bar{v}_{\\text{ctx}}) - \\sum_{k=1}^K \\log \\sigma(-v_{w_k} \\cdot \\bar{v}_{\\text{ctx}})$$

其中 $\\bar{v}_{\\text{ctx}} = \\frac{1}{|C|} \\sum_{c \\in C} v_{w_c}$。
""")

code("""\
import torch
from torch.utils.data import DataLoader
from nlp.word2vec.train import (
    load_texts, build_vocab, subsample, generate_training_pairs,
    NoiseSampler, CBOWDataset, SkipGramDataset, train_epoch,
)
from nlp.word2vec.model import Word2Vec

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
""")

code("""\
# 加载数据
print("Loading text8...")
texts = load_texts()
word_to_id, id_to_word, vocab = build_vocab(texts, min_count=3)
vocab_size = len(word_to_id)
print(f"Vocabulary: {vocab_size:,}")
""")

code("""\
# 生成训练数据
tokenized = subsample(texts, word_to_id)
cbow_pairs, skipgram_pairs = generate_training_pairs(tokenized, window_size=2)

max_pairs = 100000
cbow_pairs = cbow_pairs[:max_pairs]
sg_pairs = skipgram_pairs[:max_pairs]

cbow_dataset = CBOWDataset(cbow_pairs, max_window=2)
sg_dataset = SkipGramDataset(sg_pairs, vocab_size)
cbow_loader = DataLoader(cbow_dataset, batch_size=128, shuffle=True)
sg_loader = DataLoader(sg_dataset, batch_size=128, shuffle=True)
noise_sampler = NoiseSampler(vocab, vocab_size)
print(f"CBOW: {len(cbow_pairs):,}  Skip-gram: {len(sg_pairs):,}")
""")

md("""## 训练

> ⏱ 预估耗时：**5 epoch × ~1min/epoch ≈ 5 分钟**（M4 Max, batch_size=128）
""")

code("""\
NUM_EPOCHS = 5
LR = 0.01
K_NEG = 5

model_sg = Word2Vec(vocab_size, embed_dim=50)
optimizer = torch.optim.Adam(model_sg.parameters(), lr=LR)

sg_loss_hist = []
for epoch in range(1, NUM_EPOCHS + 1):
    loss = train_epoch(model_sg, sg_loader, noise_sampler, optimizer, k=K_NEG, mode="skipgram")
    sg_loss_hist.append(loss)
    print(f"Skip-gram Epoch [{epoch}/{NUM_EPOCHS}]  Loss: {loss:.4f}")
""")

code("""\
import matplotlib.pyplot as plt
plt.plot(sg_loss_hist, marker='o')
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Skip-gram Training Loss"); plt.grid(True)
plt.show()
""")

md("""## 词向量相似度搜索

训练完成后，用余弦相似度查找语义相近的词。""")

code("""\
embeddings = model_sg.get_embeddings()

def similar_words(word, top_k=10):
    if word not in word_to_id:
        print(f"'{word}' not in vocabulary")
        return
    idx = word_to_id[word]
    vec = embeddings[idx]
    sims = (embeddings @ vec) / (torch.norm(embeddings, dim=1) * torch.norm(vec))
    vals, inds = torch.topk(sims, top_k + 1)
    print(f"Words similar to '{word}':")
    for val, idx in zip(vals[1:], inds[1:]):
        print(f"  {id_to_word[idx.item()]:<15} {val.item():.4f}")

for w in ["computer", "science", "king", "water"]:
    print(); similar_words(w)
""")

md("""\
## 思考题

1. Skip-gram 和 CBOW 分别擅长什么？（低频词 vs 高频词）"
2. 负样本数量 $k$ 越大越好还是越小越好？典型值是多少？
3. 为什么噪声分布要用 $\\text{count}(w)^{0.75}$ 而不是原始频率？
4. 词嵌入的维度（50）对语义质量有什么影响？试试改到 100。
""")

nb.cells = cells
out = "nlp/word2vec/word2vec.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Generated {out}")
