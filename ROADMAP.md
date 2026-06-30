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
| **P1** | **Demo scripts** — Add CV inference demos: `vit/demo.py` (classify single image), `unet/demo.py` (segment single image) | ❌ not started |
| **P2** | **Benchmark** — CNN vs ViT on CIFAR-10: accuracy / params / convergence speed | ❌ not started |
| **P3** | **ResNet50** — Bottleneck block (3-layer), different from ResNet18/34's BasicBlock (2-layer) | ❌ not started |
| **P4** | **Multi-GPU / WandB / hyperparameter search** | ❌ not started |
