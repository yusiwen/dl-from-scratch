# Roadmap

---

## ✅ Completed

**Models (22):**
- Basics (9): Logistic Regression, Linear Regression, K-Means, SVM (GD + SMO), Decision Tree, Naive Bayes, PCA, k-NN, Perceptron
- Deep Learning (7): MLP (pure NumPy), SimpleCNN, ResNet18, ResNet34, BERT, Word2Vec (CBOW + Skip-gram), LSTM (hand-written gates), GPT (Causal Attention + KV Cache)
- Image Generation (1): DCGAN — CelebA 64×64
- Vision Transformer (1): ViT — CIFAR-10
- Segmentation (1): UNet — Oxford-IIIT Pet

**Infrastructure:**
- Config system (YAML)
- TensorBoard logging
- Reproducibility (seed + config saving)
- Device auto-detection (`utils.device.get_device`: CUDA → MPS → CPU)

**Jupyter Notebooks (15):**
- P0: GPT, ViT, DCGAN, UNet
- P1: CNN, BERT, ResNet18, ResNet34
- P2: basics/ (9)
- P3: Word2Vec, LSTM

---

## 🔜 Pending

| Priority | Direction | Status |
|---|---|---|
| **P0** | **Train & validate** — Run all models on NVIDIA GPU to get real metrics | waiting on GPU |
| **P1** | **ResNet50** — Bottleneck block (1×1→3×3→1×1) | ❌ not started |
| **P2** | **VAE** — Variational Autoencoder with reparameterization trick | ❌ not started |
| **P3** | **Seq2Seq Transformer** — Encoder-Decoder with cross-attention | ❌ not started |
| **P4** | **DDPM Diffusion** — Denoising Diffusion Probabilistic Models | ❌ not started |

---

## 📋 Candidate New Models

### 1. ResNet50 — Bottleneck Block

New concept: **Bottleneck block** — 1×1 conv to reduce→3×3→1×1 to expand (unlike BasicBlock's two 3×3). Enables much deeper networks (50/101/152 layers).

| Dataset | Params | New code |
|---|---|---|
| CelebA (existing) | ~23M | ~20 lines (`Bottleneck` class) |

Training time: ~30-40 min per 10 epoch (162K × 224², M4 Max)

### 2. VAE — Variational Autoencoder

New concepts: **Reparameterization trick**, **KL divergence**, **latent space interpolation**.

- Generator: decoder (deconv), same as DCGAN
- Encoder: conv layers → μ, logσ²
- Loss: reconstruction (MSE/BCE) + KL( N(μ,σ²) ∥ N(0,1) )

| Dataset | Params | New modules |
|---|---|---|
| CelebA (existing, reuse `dcgan/data.py`) | ~4M | `vaen/model.py` + `vaen/train.py` |

Training time: ~20 min (10K × 64², 50 epoch, M4 Max)

### 3. Seq2Seq Transformer — Encoder-Decoder

New concepts: **Cross-attention**, **Beam Search**, **full Transformer stack** (reuses BERT's EncoderBlock + GPT's DecoderBlock).

| Dataset | Params | New modules |
|---|---|---|
| Multi30k (I18N EN→FR/DE) | ~5M | `nlp/seq2seq/` (4 files) |

Training time: ~30 min (30K pairs, 30 epoch, M4 Max)

### 4. DDPM — Denoising Diffusion

New concepts: **Forward noise schedule**, **reverse denoising UNet**, **timestep embedding**, **cosine schedule**, **DDIM sampling**.

| Dataset | Params | New modules |
|---|---|---|
| CIFAR-10 (existing) | ~35M (UNet backbone) | `diffusion/` (5 files) |

Training time: ~2-3 h (50K × 32², 100 epoch, M4 Max)

---

## ⏱ Training Time Summary (M4 Max, for notebook references)

| Model | Data | Params | Epochs | Total time | Notes |
|---|---|---|---|---|---|
| ResNet50 | 162K × 224² | ~23M | 100 | 5-7 h | bottleneck, ~1.5× ResNet34 |
| VAE | 10K × 64² | ~4M | 50 | ~20 min | CelebA, reuses dcgan/data.py |
| Seq2Seq | 30K pairs | ~5M | 30 | ~30 min | Multi30k, reuses Transformer blocks |
| DDPM | 50K × 32² | ~35M | 100 | 2-3 h | CIFAR-10, UNet backbone |

> Full training of ResNet50 / DDPM on M4 Max is feasible but slow.
> Recommendation: implement & debug on small subset (2-5 epoch), then run full on NVIDIA GPU.
