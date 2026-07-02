# Roadmap

---

## ✅ Completed

**Models (32):**
- ML/Basics (10): Logistic Regression, Linear Regression, K-Means, SVM (GD + SMO), Decision Tree, Naive Bayes, PCA, k-NN, Perceptron, t-SNE
- Deep Learning (11): MLP (pure NumPy), SimpleCNN, ResNet18, ResNet34, ResNet50 (Bottleneck), BERT, Word2Vec (CBOW + Skip-gram), LSTM (hand-written gates), GPT (Causal Attention + KV Cache), Seq2Seq Transformer (cross-attention), DDPM (Denoising Diffusion)
- Image Generation (1): DCGAN — CelebA 64×64
- Variational AE (1): VAE — Encoder, reparameterization trick, KL divergence, CelebA 64×64
- Vision Transformer (1): ViT — CIFAR-10
- Segmentation (1): UNet — Oxford-IIIT Pet
- Graph (1): GCN — Cora citation network, spectral graph convolution
- Reinforcement Learning (1): DQN — CartPole-v1, experience replay
- Self-Supervised (1): SimCLR — CIFAR-10, NT-Xent contrastive loss
- Object Detection (1): YOLO — Pascal VOC, grid-based detection
- Efficient Architectures (1): MobileNet — CIFAR-10, depthwise separable convolution
- Parameter-Efficient Fine-Tuning (1): LoRA — GPT adapter, low-rank decomposition

**Infrastructure:**
- Config system (YAML), TensorBoard logging, Reproducibility (seed + config saving)
- Device auto-detection (`utils.device.get_device`: CUDA → MPS → CPU)

**Jupyter Notebooks (21):**
- Original 15: GPT, ViT, DCGAN, UNet, CNN, BERT, ResNet18, ResNet34, basics (9), Word2Vec, LSTM
- New 6: GCN, DQN, SimCLR, YOLO, MobileNet, LoRA

---

## 📋 Future Ideas

- Train & validate on NVIDIA GPU
- Demo scripts for CV models (`vit/demo.py`, `unet/demo.py`)
- CNN vs ViT benchmark on CIFAR-10
- Multi-GPU / WandB / hyperparameter search
