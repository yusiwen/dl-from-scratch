# Roadmap

---

## ✅ Completed

**Models (26):**
- Basics (9): Logistic Regression, Linear Regression, K-Means, SVM (GD + SMO), Decision Tree, Naive Bayes, PCA, k-NN, Perceptron
- Deep Learning (11): MLP (pure NumPy), SimpleCNN, ResNet18, ResNet34, ResNet50 (Bottleneck), BERT, Word2Vec (CBOW + Skip-gram), LSTM (hand-written gates), GPT (Causal Attention + KV Cache), Seq2Seq Transformer (cross-attention), DDPM (Denoising Diffusion)
- Image Generation (1): DCGAN — CelebA 64×64
- Variational AE (1): VAE — Encoder, reparameterization trick, KL divergence, CelebA 64×64
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
- P2: basics (9)
- P3: Word2Vec, LSTM

---

## 📋 Future Ideas

- Train & validate on NVIDIA GPU
- Demo scripts for CV models (`vit/demo.py`, `unet/demo.py`)
- CNN vs ViT benchmark on CIFAR-10
- Multi-GPU / WandB / hyperparameter search
