# Roadmap

Planned models and optimizations in priority order.

---

## ✅ Completed

- [x] **ResNet18** — CelebA face attribute classification (15 attributes, 1K samples, 91.2% val acc)
- [x] **SimpleCNN** — CIFAR-10 image classification via HF datasets (Conv3+Pool3+FC2, ~620K params, ~82% test acc)
- [x] **MLP** — MNIST digit classification, pure NumPy (no autograd), 235K params, 97.9% test acc
- [x] **Logistic Regression** — single Linear layer, MNIST, 92.3% test acc
- [x] **K-Means** — unsupervised clustering, MNIST, 57.8% purity
- [x] **Linear Regression** — California Housing (Normal Equation + Gradient Descent, R²=0.583)

---

## 🔜 ResNet — Further Optimizations

- [ ] **Architecture**: Upgrade to ResNet34 ([3, 4, 6, 3] blocks, ~21M params)
- [ ] **Optimizer**: Replace Adam with SGD + Momentum (0.9, weight_decay=1e-4)
- [ ] **Full labels**: Train on all 40 CelebA attributes instead of 15
- [ ] **Full dataset**: Scale from 1K to 200K samples (multi-worker dataloader, prefetch, batch tuning)
- [ ] **LR scheduler**: CosineAnnealingLR or StepLR
- [ ] **Data augmentation**: RandomHorizontalFlip, ColorJitter, RandomRotation
- [ ] **Early stopping**: Save best model by val loss, stop on plateau
- [ ] **Gradient accumulation**: Simulate larger effective batch size
- [ ] **Loss weighting**: `pos_weight` in BCEWithLogitsLoss for imbalanced attributes
- [ ] **Metrics**: Per-attribute ROC AUC, F1 score
- [ ] **Test evaluation**: Standard CelebA test split evaluation

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

### 4. GPT-style Decoder-Only Transformer — Text Generation
- **Dataset**: `wikitext-2-raw-v1` or `tiny_shakespeare` on HF
- **Architecture**: token embedding → positional encoding → N× causal decoder block (masked self-attn + FFN) → LM head
- **Key learnings**: causal attention mask, tokenization (BPE / char-level), autoregressive generation loop

---

## 🛠 Infrastructure

- [ ] **Experiment logging**: TensorBoard / WandB for loss curves, image samples, attribute accuracy
- [ ] **Config system**: YAML / Hydra for hyperparameter management
- [ ] **Reproducibility**: Lock all training seeds, save config alongside checkpoints
