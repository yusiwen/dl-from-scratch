# Roadmap

Planned models and optimizations in priority order.

---

## ✅ Completed

- [x] **ResNet18** — CelebA face attribute classification (15 attributes, 1K samples, 91.2% val acc)
- [x] **ResNet34** — Full CelebA (40 attrs, 200K), SGD+Momentum, CosineAnnealingLR, grad accum, early stopping
- [x] **SimpleCNN** — CIFAR-10 image classification via HF datasets (Conv3+Pool3+FC2, ~620K params, ~82% test acc)
- [x] **MLP** — MNIST digit classification, pure NumPy (no autograd), 235K params, 97.9% test acc
- [x] **Logistic Regression** — single Linear layer, MNIST, 92.3% test acc
- [x] **K-Means** — unsupervised clustering, MNIST, 57.8% purity
- [x] **Linear Regression** — California Housing (Normal Equation + Gradient Descent, R²=0.583)
- [x] **SVM** — GD (primal, linear) + SMO (dual, linear/RBF), MNIST 93.3% acc
- [x] **Decision Tree** — ID3/CART, Iris, 93.3% acc (ASCII tree visualization)
- [x] **Naive Bayes** — Gaussian NB, MNIST, 53.0% (shows independence assumption gap)
- [x] **PCA** — SVD-based dimensionality reduction, MNIST 2D visualisation
- [x] **k-NN** — instance-based, MNIST, ~87% with 2000 samples
- [x] **Perceptron** — single neuron, MNIST 0v1 100% (linearly separable)
- [x] **BERT** — Transformer Encoder (Self-Attention + MLM), sentiment classification
- [x] **Word2Vec** — CBOW + Skip-gram with Negative Sampling, text8 embeddings
- [x] **LSTM** — hand-written gates (input/forget/output/cell), IMDB sentiment
- [x] **GPT** — Decoder-only Transformer with Causal Attention + KV Cache, text8 generation
- [x] **Infrastructure** — Config system (YAML), TensorBoard logging, reproducibility (seed + config saving)

---

## 📋 New Models

### 1. DCGAN — Image Generation
- **Dataset**: CelebA (already downloaded at `data/celeba/`)
- **Architecture**:
  - Generator: latent(100) → FC → 4× deconv layers → 64×64×3 (Tanh)
  - Discriminator: 4× conv layers → FC → 1 (sigmoid)
- **Training**: Adam(2e-4, 0.5), BCELoss, no BN in first conv
- **Key learnings**: transposed convolution, adversarial training, GAN training dynamics

### 2. UNet — Semantic Segmentation
- **Dataset**: `tchevrou/oxford-iiit-pet` on HF
- **Architecture**: encoder (down conv) → decoder (up conv) with skip connections
- **Key learnings**: pixel-level classification, upsampling (transposed conv / interpolation), skip connections

### 3. Vision Transformer (ViT) — Image Classification
- **Dataset**: CIFAR-10 / CIFAR-100 on HF
- **Architecture**: patch embedding → positional encoding → N× Transformer encoder (MHSA + FFN) → CLS head
- **Key learnings**: self-attention for vision, patch embedding, no convolutions

---

## 🛠 Future Directions

- More advanced ResNet variants (ResNet50 with Bottleneck blocks)
- Multi-GPU / distributed training support
- Experiment tracking comparison (WandB integration)
- Hyperparameter search (grid / random / Bayesian)
