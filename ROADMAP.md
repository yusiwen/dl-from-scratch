# Roadmap

---

## ✅ Completed

**Basics (9):** Logistic Regression, Linear Regression, K-Means, SVM (GD + SMO), Decision Tree, Naive Bayes, PCA, k-NN, Perceptron

**Deep Learning (7):** MLP (pure NumPy), SimpleCNN, ResNet18, ResNet34, BERT, Word2Vec (CBOW + Skip-gram), LSTM (hand-written gates), GPT (Causal Attention + KV Cache)

**Image Generation (1):** DCGAN — Generator (3.5M) + Discriminator (2.8M), CelebA 64×64

**Vision Transformer (1):** ViT — Patch embedding + Transformer encoder (from BERT), CIFAR-10

**Segmentation (1):** UNet — Encoder-decoder with skip connections, Oxford-IIIT Pet

**Infrastructure (3):** Config system (YAML), TensorBoard logging, reproducibility (seed + config saving)

**Total: 22 items**

---

## 🔜 Next

| Priority | Direction | Status |
|---|---|---|
| **P0** | **Train & validate** — Run ResNet34, ViT, UNet, DCGAN to get real metrics; fill into README tables | ❌ not started |
| **P1** | **Demo scripts** — Add CV inference demos: `vit/demo.py`, `unet/demo.py` | ❌ not started |
| **P2** | **Benchmark** — CNN vs ViT on CIFAR-10: accuracy / params / convergence speed | ❌ not started |
| **P3** | **ResNet50** — Bottleneck block (3-layer), different from BasicBlock | ❌ not started |
| **P4** | **Multi-GPU / WandB / hyperparameter search** | ❌ not started |

### P0 训练耗时估算（M4 Max）

| 模型 | 数据量 | 参数量 | 每 epoch 步数 | 每 epoch 耗时 | Epochs | 总耗时 |
|---|---|---|---|---|---|---|
| ResNet34 | 162K × 224² | 21M | ~2,500 | ~10-15 min | 100 | 17-25 h |
| ViT | 50K × 32² | 807K | ~400 | ~30 sec | 50 | ~25 min |
| UNet | 3.7K × 128² | 31M | ~230 | ~1 min | 50 | ~45 min |
| DCGAN | 10K × 64² | G:3.5M+D:2.8M | ~80 | ~15 sec | 50 | ~12 min |

> ResNet34 建议先跑 10 epoch（~2h）看 loss 趋势；其余三个模型半小时内可跑完，可直接全量。

---

## 📓 Jupyter Notebooks

每个模型配备交互式 notebook，包含背景、原理、数学、训练与验证。

### 模板结构

1. **背景与动机** — 解决什么问题，历史
2. **数学原理** — 核心公式 + 直观解释
3. **架构图** — mermaid / ASCII 图
4. **代码实现** — 逐块讲解（import 已有 `.py` 模块）
5. **交互式训练** — 可调超参，小规模快速跑通
6. **可视化** — loss 曲线、生成样本、注意力图等
7. **思考题** — 引导深入理解

### 存放位置

每个模型目录下 `<model_name>.ipynb`，如 `dcgan/dcgan.ipynb`。

### 优先级

| 优先级 | 模型 | 理由 |
|---|---|---|
| **P0** | GPT, ViT, DCGAN, UNet | 最复杂，教育价值最高 |
| **P1** | ResNet18, ResNet34, CNN, BERT | 核心模型，读者最想交互调参 |
| **P2** | basics/ (9 个) | 简单但数量多，简版模板 |
| **P3** | Word2Vec, LSTM | 已有较好 docstring |

### 策略

- 直接 `import` 现有 `.py` 模块，不重复造轮子
- 训练默认用小规模（几 epoch），读者可改参跑全量
- **新模型加入时必须配套 notebook**，否则视为未完成
