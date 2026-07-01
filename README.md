---
language: en
license: mit
tags:
  - deep-learning
  - resnet
  - celebA
  - pytorch
  - from-scratch
---
# DL From Scratch

Implement mainstream deep learning models from scratch.

## Project Structure

```
├── main.py
├── pyproject.toml
├── .gitignore
├── README.md
├── resnet18/
│   ├── README.md          # Original ResNet18 implementation
│   ├── __init__.py
│   ├── data.py            # CelebA via HF datasets (eurecom-ds/celeba)
│   ├── model.py           # ResNet18 from scratch
│   ├── train.py           # Training script (MPS + AMP)
│   ├── eval.py            # Evaluation script (per-attribute accuracy)
│   └── resnet18_celeba.pt      # (local, not tracked)
├── resnet34/
│   ├── __init__.py
│   ├── data.py            # Full CelebA (40 attrs, 200K), data augmentation
│   ├── model.py           # ResNet34 via resnet18.model.ResNet
│   ├── train.py           # SGD+Momentum + CosineAnnealingLR + grad accum + early stopping
│   └── eval.py            # Per-attribute ROC AUC, F1, test split
├── resnet50/
│   ├── __init__.py
│   ├── config.yaml        # ResNet50 hyperparameters
│   ├── model.py           # Bottleneck block (1×1→3×3→1×1) → resnet50()
│   ├── data.py            # Reuses resnet34.data (CelebA)
│   ├── train.py           # Reuses resnet34 training pattern
│   └── eval.py            # Per-attribute ROC AUC, F1
├── vae/
│   ├── __init__.py
│   ├── config.yaml        # VAE hyperparameters
│   ├── model.py           # Encoder → μ,logσ² → reparameterize → Decoder
│   ├── data.py            # CelebA images (64×64)
│   ├── train.py           # VAE training (recon + KL loss)
│   └── generate.py        # Sample generation + latent interpolation
├── dcgan/
│   ├── __init__.py
│   ├── config.yaml        # DCGAN hyperparameters
│   ├── model.py           # Generator + Discriminator
│   ├── data.py            # CelebA images (64×64, no labels)
│   ├── train.py           # Adversarial training loop (G/D alternating)
│   └── generate.py        # Generate sample grid from trained model
├── vit/
│   ├── __init__.py
│   ├── config.yaml        # ViT hyperparameters (patch_size, d_model, n_layers, etc.)
│   ├── model.py           # ViT: PatchEmbed → Transformer encoder (reused from BERT) → CLS head
│   ├── data.py            # CIFAR-10 via HF datasets
│   ├── train.py           # Training loop
│   └── eval.py            # Per-class accuracy on test split
├── unet/
│   ├── __init__.py
│   ├── config.yaml        # UNet hyperparameters
│   ├── model.py           # U-Net: encoder–decoder with skip connections
│   ├── data.py            # Oxford-IIIT Pet (image + mask) with augmentation
│   ├── train.py           # Training loop (pixel-wise CrossEntropy)
│   └── eval.py            # IoU and pixel accuracy
├── cnn/
│   ├── __init__.py
│   ├── data.py            # CIFAR-10 via HF datasets (uoft-cs/cifar10)
│   ├── model.py           # Plain CNN (Conv×3 + Pool×3 + FC×2)
│   ├── train.py           # Training script (Adam + CosineAnnealingLR)
│   └── eval.py            # Test evaluation + confusion matrix
├── mlp/
│   ├── __init__.py
│   ├── data.py            # MNIST via HF datasets (ylecun/mnist)
│   ├── model.py           # MLP — pure NumPy (Linear, ReLU, SoftmaxCrossEntropy, SGD)
│   ├── train.py           # Training script
│   └── eval.py            # Test evaluation (per-digit accuracy)
├── utils/
│   ├── __init__.py
│   ├── config.py             # YAML config loading/saving (load_config / save_config)
│   └── seed.py               # set_seed() — lock torch + numpy + random + cudnn
├── nlp/
│   ├── bert/
│   ├── word2vec/
│   ├── lstm/
│   ├── gpt/
│   └── seq2seq/
│       ├── __init__.py
│       ├── tokenizer.py       # Word-level tokenizer (5000 vocab, from text8)
│       ├── model.py           # Decoder-only Transformer (Causal Attention + KV Cache)
│       ├── train.py           # Autoregressive LM on text8
│       └── generate.py        # Text generation (temperature + top-k + [SEP] blocked)
│   └── seq2seq/
│       ├── __init__.py
│       ├── config.yaml        # Transformer hyperparameters
│       ├── model.py           # Encoder (from BERT) + Decoder (cross-attention) → Seq2Seq
│       ├── data.py            # Multi30k EN→DE, word-level tokenizer
│       ├── train.py           # Teacher forcing training
│       └── generate.py        # Greedy decoding translation demo
├── basics/
│   ├── __init__.py
│   ├── logistic_regression.py   # Single Linear layer + Softmax (92.3% on MNIST)
│   ├── linear_regression.py     # California Housing (Normal Equation + GD, R²=0.583)
│   ├── k_means.py               # Unsupervised clustering (pure NumPy)
│   ├── svm.py                   # SVM — GD (primal) + SMO (dual, Linear/RBF kernels)
│   ├── decision_tree.py          # ID3/CART on Iris (ASCII tree, ~93% acc)
│   ├── naive_bayes.py            # Gaussian NB on MNIST (generative classifier)
│   ├── pca.py                    # SVD-based dimensionality reduction (MNIST 2D visualisation)
│   ├── knn.py                    # k-Nearest Neighbors (instance-based, MNIST)
│   └── perceptron.py             # Single neuron (Rosenblatt 1958, step activation)
├── .gitattributes                 # LFS: *.zip *.pt
└── uv.lock
```

## Infrastructure

| Feature | Description |
|---|---|
| **Config system** | Each model directory has a `config.yaml` with its hyperparameters (seed, lr, batch_size, epochs, etc.). Edit the YAML to change training params without touching code. |
| **TensorBoard** | Every PyTorch training script logs loss/accuracy per epoch to `runs/{model_name}/`. Run `tensorboard --logdir runs` to visualize all experiments. |
| **Reproducibility** | `utils/seed.py` provides `set_seed()` that locks `torch` + `numpy` + `random` + `cudnn`. Called at the start of every train script. Config is saved alongside model weights (`_config.yaml`). |

### Usage

```bash
# View training curves (all models)
tensorboard --logdir runs

# Edit hyperparameters in YAML instead of code
vim resnet18/config.yaml
# then train as usual:
uv run python -m resnet18.train
```

## ResNet18

| Item | Value |
|---|---|
| Model | ResNet18 (11.2M params) |
| Dataset | CelebA via HF datasets — 1,000 images |
| Attributes | 15 binary (Smiling, Male, Young, Eyeglasses, etc.) |
| Split | 800 train / 200 val |
| Val Accuracy | **91.2%** |
| Training | MPS (Mac M4) + AMP |

## ResNet34

| Item | Value |
|---|---|
| Model | ResNet34 (~21M params, [3,4,6,3] BasicBlock) |
| Dataset | CelebA via HF datasets — full 200K |
| Attributes | All 40 binary attributes |
| Optimizer | SGD + Momentum (0.9, weight_decay=1e-4) |
| Training | CosineAnnealingLR + Gradient Accumulation + Early Stopping + Loss Weighting |

## ResNet50

| Item | Value |
|---|---|
| Model | ResNet50 (~23.6M params, [3,4,6,3] Bottleneck) |
| Dataset | CelebA via HF datasets — full 200K |
| Attributes | All 40 binary attributes |
| Optimizer | SGD + Momentum (0.9, weight_decay=1e-4) |
| Architecture | Bottleneck block: 1×1 → 3×3 → 1×1 (contrast with BasicBlock's two 3×3) |

## VAE

| Item | Value |
|---|---|
| Model | Variational Autoencoder (2.6M params) |
| Dataset | CelebA via HF datasets — 10K images (64×64) |
| Architecture | Conv Encoder → μ,logσ² → reparameterize → Deconv Decoder → Sigmoid |
| Loss | Reconstruction (BCE) + KL divergence |
| Training | Adam(lr=2e-4), 50 epoch |

## Seq2Seq Transformer

| Item | Value |
|---|---|
| Model | Encoder-Decoder Transformer (1M params) |
| Dataset | Multi30k EN→DE — 29K train / 1K test |
| Architecture | Encoder (from BERT) + Decoder (causal + cross-attention) |
| Training | Teacher forcing, weight-tying, Adam(lr=1e-4) |

## DCGAN

| Item | Value |
|---|---|
| Model | Generator (3.5M params) + Discriminator (2.8M params) |
| Dataset | CelebA via HF datasets — 10K images (64×64) |
| Architecture | Transposed conv G / Conv D, BN, LeakyReLU |
| Optimizer | Adam(lr=2e-4, β₁=0.5) — separate for G and D |
| Training | BCELoss, label smoothing, fixed noise grid for monitoring |

## ViT

| Item | Value |
|---|---|
| Model | Vision Transformer (807K params, 4 layers, 4 heads, 128-dim) |
| Dataset | CIFAR-10 via HF datasets — 50K train / 10K test |
| Architecture | PatchEmbed(4×4) → [CLS] → Transformer Encoder (from BERT) → CLS head |
| Key concept | Self-attention for vision, no convolutions, patch embeddings |

## UNet

| Item | Value |
|---|---|
| Model | U-Net (31M params, 5 encoder/decoder stages) |
| Dataset | Oxford-IIIT Pet via HF datasets — image + segmentation mask |
| Architecture | Encoder: Conv+MaxPool × 4, Decoder: UpConv+skip × 4, output: pixel-wise logits |
| Loss | CrossEntropy (ignore_index=0 for unlabeled) |
| Metrics | Pixel accuracy, mean IoU |

## CNN

| Item | Value |
|---|---|
| Model | SimpleCNN (620K params) |
| Dataset | CIFAR-10 via HF datasets — 50K images |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Test Accuracy | **82.4%** (30 epochs) |
| Training | Adam + CosineAnnealingLR |

## MLP

| Item | Value |
|---|---|
| Model | MLP (235K params, pure NumPy) |
| Dataset | MNIST via HF datasets — 60K images |
| Classes | 10 digits (0-9) |
| Test Accuracy | **97.9%** (20 epochs) |
| Framework | NumPy only (hand-written backward pass) |

## BERT

| Item | Value |
|---|---|
| Model | BERT mini (834K params, 4 layers, 4 heads, 128-dim) |
| Pre-training | MLM on text8 (90M chars, HuggingFace) |
| Fine-tuning | Sentiment classification on IMDB (HuggingFace) |
| Test Accuracy | ~50% (character-level; word-level would be higher with subword tokenization) |
| Core components | Self-Attention (semantic aggregation) + MLM (entropy increase noise reduction) |

## Word2Vec

| Item | Value |
|---|---|
| Model | Word2Vec (50-dim embeddings, 97K vocab) |
| Architectures | CBOW + Skip-gram with Negative Sampling |
| Dataset | text8 via HF datasets (~90M chars) |
| Training | Adam, 5 epochs, k=5 negative samples |
| Evaluation | Cosine similarity search in embedding space |
| Key concept | Static word embeddings from distributional semantics |

## LSTM

| Item | Value |
|---|---|
| Model | LSTM (145K params, hand-written gates) |
| Dataset | IMDB via HuggingFace (9K train / 1K test) |
| Architecture | Embedding(128) → LSTM(128→128) → FC(128→2) |
| Test Accuracy | ~50-60% (character-level, harder than word-level) |
| Key concepts | **Input/forget/output gates**, **cell state**, gradient flow through gating |

## GPT

| Item | Value |
|---|---|
| Model | Decoder-only Transformer (5.7M params, word-level) |
| Dataset | text8 via HuggingFace (15M words, 20K chunks) |
| Training | Autoregressive (predict next token), PPL 4.63 |
| Generation | Temperature + top-k sampling with **KV Cache**, [SEP] blocked |
| Key concepts | **Causal Self-Attention**, **KV Cache**, autoregressive generation, word-level tokenization |

## Basics

| Algorithm | File | Datasets | Metric |
|---|---|---|---|
| Logistic Regression | `basics/logistic_regression.py` | MNIST | 92.3% test accuracy |
| Linear Regression | `basics/linear_regression.py` | California Housing | R²=0.583 |
| K-Means | `basics/k_means.py` | MNIST | 57.8% cluster purity |
| SVM (GD + SMO) | `basics/svm.py` | MNIST 3v5 | 93.3% (RBF kernel) |
| Decision Tree | `basics/decision_tree.py` | Iris | 93.3% test acc |
| Naive Bayes | `basics/naive_bayes.py` | MNIST | 53.0% test acc |
| PCA | `basics/pca.py` | MNIST | 17.3% variance in 2 components |
| k-NN | `basics/knn.py` | MNIST | ~87% (k=5, 2000 train) |
| Perceptron | `basics/perceptron.py` | MNIST 0v1 | 100% (linearly separable) |

### SVM implementations

| Method | Type | Kernel | Notes |
|---|---|---|---|
| `SVM_GD` | Primal GD | Linear only | Fast, robust, ~80 lines |
| `SVM_SMO` | Dual SMO | Linear + RBF | Platt SMO, ~150 lines, supports kernel trick |

See [resnet18/README.md](resnet18/README.md) for details.

## Core Concepts

Every model in this project was written from scratch to teach a specific
set of ML/DL concepts. The table below maps each model to the key ideas
it demonstrates.

| Module | Model | Key concepts |
|--------|-------|-------------|
| `basics/` | Logistic Regression | Linear decision boundary, Softmax, Cross-Entropy, closed-form vs gradient descent |
| `basics/` | Linear Regression | Normal Equation, MSE, R² score, feature standardisation |
| `basics/` | K-Means | Unsupervised learning, Euclidean distance, iterative centroid refinement, cluster purity |
| `basics/` | SVM (GD) | Hinge loss, max-margin classification, L2 regularisation, primal gradient descent |
| `basics/` | SVM (SMO) | Dual formulation, Lagrange multipliers, KKT conditions, kernel trick (RBF) |
| `basics/` | Decision Tree | Entropy, Information Gain, recursive partitioning, interpretable ASCII tree |
| `basics/` | Naive Bayes | Bayes' theorem, generative vs discriminative models, Gaussian likelihood, log-space prediction |
| `basics/` | PCA | Singular Value Decomposition (SVD), eigenvalue, dimensionality reduction, variance explained |
| `basics/` | k-NN | Instance-based learning, distance metrics, curse of dimensionality, bias-variance tradeoff |
| `basics/` | Perceptron | Single neuron, step activation, online learning, Perceptron Convergence Theorem |
| `mlp/` | MLP (NumPy) | **Manual backpropagation**, chain rule, gradient descent without autograd, softmax cross-entropy |
| `cnn/` | SimpleCNN | Convolution, max-pooling, BatchNorm, Dropout, CosineAnnealing LR schedule |
| `resnet18/` | ResNet18 | **Residual connections (skip connections)**, BatchNorm in deep networks, bottleneck design, AMP |
| `resnet34/` | ResNet34 | SGD+Momentum, CosineAnnealingLR, gradient accumulation, early stopping, ROC AUC, F1 |
| `resnet50/` | ResNet50 | Bottleneck block (1×1→3×3→1×1), deeper residual networks |
| `vae/` | VAE | Reparameterization trick, KL divergence, latent space interpolation |
| `nlp/seq2seq/` | Seq2Seq Transformer | Encoder-decoder, cross-attention, teacher forcing, weight-tying |
| `dcgan/` | DCGAN | Transposed convolution, adversarial training, generator/discriminator dynamics |
| `vit/` | Vision Transformer (ViT) | Patch embedding, self-attention for vision, Transformer without convolutions |
| `unet/` | U-Net | Encoder-decoder, skip connections, pixel-wise classification, IoU metric |
| `nlp/bert/` | BERT mini | **Self-Attention** (semantic aggregation), **Masked Language Model** (entropy increase + denoising), LayerNorm, positional encoding |
| `nlp/word2vec/` | Word2Vec | **Embedding lookup tables**, **Negative Sampling**, CBOW vs Skip-gram, subsampling frequent words, cosine similarity |
| `nlp/lstm/` | LSTM | **Input/forget/output gates**, **cell state**, gradient flow through gating, sequential processing vs parallel attention |
| `nlp/gpt/` | GPT | **Causal Self-Attention**, **KV Cache**, autoregressive generation, word-level tokenizer, temperature + top-k sampling, bad-token blocking |

## Setup & Run

```bash
uv sync
```

```bash
# Train / Evaluate ResNet18
uv run python -m resnet18.train
uv run python -m resnet18.eval

# Train / Evaluate ResNet34
uv run python -m resnet34.train
uv run python -m resnet34.eval

# Train / Evaluate ResNet50
uv run python -m resnet50.train
uv run python -m resnet50.eval

# Train / Generate VAE
uv run python -m vae.train
uv run python -m vae.generate

# Train / Translate Seq2Seq
uv run python -m nlp.seq2seq.train
uv run python -m nlp.seq2seq.generate

# Train / Generate DCGAN
uv run python -m dcgan.train
uv run python -m dcgan.generate

# Train / Evaluate ViT
uv run python -m vit.train
uv run python -m vit.eval

# Train / Evaluate UNet
uv run python -m unet.train
uv run python -m unet.eval

# Train / Evaluate CNN
uv run python -m cnn.train
uv run python -m cnn.eval

# Train / Evaluate MLP (pure NumPy)
uv run python -m mlp.train
uv run python -m mlp.eval

# Basics
uv run python -m basics.logistic_regression
uv run python -m basics.k_means
uv run python -m basics.linear_regression
uv run python -m basics.svm
uv run python -m basics.decision_tree
uv run python -m basics.naive_bayes
uv run python -m basics.pca
uv run python -m basics.knn
uv run python -m basics.perceptron

# NLP
uv run python -m nlp.bert.pretrain
uv run python -m nlp.bert.finetune
uv run python -m nlp.bert.eval

# Word2Vec
uv run python -m nlp.word2vec.train
uv run python -m nlp.word2vec.eval

# LSTM
uv run python -m nlp.lstm.train
uv run python -m nlp.lstm.eval

# GPT
uv run python -m nlp.gpt.train
uv run python -m nlp.gpt.generate
```

## Models

Trained weights are **not tracked in git** (`.gitignore`'ed). Each model saves its weights
locally after training; paths are shown below for reference.

| Model | Local path | Size |
|---|---|---|
| ResNet18 (15 attrs, 1K samples) | `resnet18/resnet18_celeba.pt` | 45 MB |
| ResNet34 (40 attrs, 200K samples) | `resnet34/resnet34_celeba.pt` | ~80 MB |
| ResNet50 (40 attrs, 200K samples) | `resnet50/resnet50_celeba.pt` | ~90 MB |
| VAE (CelebA, 64×64) | `vae/vae_celeba.pt` | 10 MB |
| Seq2Seq Transformer (Multi30k) | `nlp/seq2seq/seq2seq_multi30k.pt` | 4 MB |
| DCGAN (CelebA, 64×64) | `dcgan/dcgan_celeba.pt` | ~23 MB (G+D) |
| ViT (CIFAR-10, 32×32) | `vit/vit_cifar10.pt` | 3.2 MB |
| UNet (Oxford-Pet, 128×128) | `unet/unet_oxford_pet.pt` | 119 MB |
| SimpleCNN (CIFAR-10) | `cnn/simple_cnn_cifar10.pt` | 2.4 MB |
| MLP (MNIST, NumPy) | `mlp/mlp_mnist.npz` | 0.9 MB |
| Logistic Regression | `basics/logistic_regression.npz` | 63 KB |
| K-Means centers | `basics/kmeans_centers.npz` | 32 KB |
| Linear Regression | `basics/linear_regression.npz` | 2 KB |
| SVM | `basics/svm.npz` | 45 KB |
| Decision Tree | — | N/A (no weights) |
| Naive Bayes | — | N/A (no weights) |
| PCA | — | N/A (data-dependent) |
| k-NN | — | N/A (no training) |
| Perceptron | — | N/A (no weights) |
| BERT (MLM) | `nlp/bert/bert_mlm.pt` | 3.2 MB |
| BERT (finetuned) | `nlp/bert/bert_finetuned.pt` | 3.2 MB |
| Word2Vec (SG) | `nlp/word2vec/skipgram.pt` | 19 MB |
| Word2Vec (CBOW) | `nlp/word2vec/cbow.pt` | 19 MB |
| LSTM | `nlp/lstm/lstm_sentiment.pt` | 0.6 MB |
| GPT | `nlp/gpt/gpt_text8.pt` | 3.3 MB |
