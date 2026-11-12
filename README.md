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
├── ROADMAP.md
├── ml/                    # Classical Machine Learning (pure NumPy)
│   ├── mlp/               # MLP (MNIST, manual backprop)
│   └── basics/            # 12 standalone models (lin/log reg, SVM, clustering, PCA, t-SNE, etc.)
├── cv/                    # Computer Vision
│   ├── simplecnn/         # SimpleCNN (CIFAR-10, Conv×3+Pool×3+FC×2)
│   ├── resnet18/          # ResNet18 (CelebA, 15 attrs, skip connections)
│   ├── resnet34/          # ResNet34 (CelebA, 40 attrs, [3,4,6,3] blocks)
│   ├── resnet50/          # ResNet50 (Bottleneck block 1×1→3×3→1×1)
│   ├── mobilenet/         # MobileNet (depthwise separable conv, CIFAR-10)
│   ├── vit/               # Vision Transformer (patch embed + BERT encoder, CIFAR-10)
│   ├── unet/              # UNet (Oxford-IIIT Pet segmentation)
│   └── yolo/              # YOLO (Pascal VOC object detection)
├── gen/                   # Generative Models
│   ├── autoencoder/       # Autoencoder (MNIST, reconstruction-only)
│   ├── dcgan/             # DCGAN (CelebA, transposed conv)
│   ├── vae/               # VAE (reparameterization trick, KL divergence)
│   └── ddpm/              # DDPM (CIFAR-10, denoising diffusion)
├── selfsup/               # Self-Supervised Learning
│   ├── byol/              # BYOL (CIFAR-10, EMA target, no negatives)
│   ├── deep_cluster/      # DEC (MNIST, autoencoder + KL clustering)
│   ├── mae/               # MAE (CIFAR-10, masked image modeling + linear probe)
│   └── simclr/            # SimCLR (CIFAR-10, contrastive learning + linear probe)
├── graph/                 # Graph Neural Networks
│   └── gcn/               # GCN (Cora, spectral graph convolution)
├── rl/                    # Reinforcement Learning
│   └── dqn/               # DQN (CartPole, experience replay)
├── nlp/                   # Natural Language Processing
│   ├── bert/              # BERT (MLM pretrain + classification finetune)
│   ├── gpt/               # GPT (decoder-only, causal attention, KV cache)
│   ├── lstm/              # LSTM (hand-written gates, IMDB sentiment)
│   ├── word2vec/          # Word2Vec (CBOW + Skip-gram, negative sampling)
│   ├── seq2seq/           # Seq2Seq Transformer (EN→DE translation)
│   └── lora/              # LoRA (parameter-efficient GPT fine-tuning)
├── utils/                 # Shared Infrastructure
│   ├── config.py          # YAML config loading/saving
│   ├── seed.py            # Reproducibility seed locking
│   └── device.py          # CUDA → MPS → CPU auto-detection
├── docker/                # NGC PyTorch container for the GB10 GPU
└── scripts/               # Notebook generation scripts
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
vim cv/resnet18/config.yaml
# then train as usual:
uv run python -m cv.resnet18.train
```

## CV/ResNet18

| Item | Value |
|---|---|
| Model | ResNet18 (11.2M params) |
| Dataset | CelebA via HF datasets — 1,000 images |
| Attributes | 15 binary (Smiling, Male, Young, Eyeglasses, etc.) |
| Split | 800 train / 200 val |
| Val Accuracy | **91.2%** |
| Training | MPS (Mac M4) + AMP |

## CV/ResNet34

| Item | Value |
|---|---|
| Model | ResNet34 (~21M params, [3,4,6,3] BasicBlock) |
| Dataset | CelebA via HF datasets — full 200K |
| Attributes | All 40 binary attributes |
| Optimizer | SGD + Momentum (0.9, weight_decay=1e-4) |
| Training | CosineAnnealingLR + Gradient Accumulation + Early Stopping + Loss Weighting |

## CV/ResNet50

| Item | Value |
|---|---|
| Model | ResNet50 (~23.6M params, [3,4,6,3] Bottleneck) |
| Dataset | CelebA via HF datasets — full 200K |
| Attributes | All 40 binary attributes |
| Optimizer | SGD + Momentum (0.9, weight_decay=1e-4) |
| Architecture | Bottleneck block: 1×1 → 3×3 → 1×1 (contrast with BasicBlock's two 3×3) |

## GEN/Autoencoder

| Item | Value |
|---|---|
| Model | Autoencoder (469,010 params; MLP 784→256→128→2, mirrored decoder) |
| Dataset | MNIST via HF datasets — 5,000 images, pixels in [0,1] |
| Loss | BCE reconstruction (no KL term, no sampling), Adam(lr=1e-3), 20 epochs |
| Reconstruction MSE | default 2-D: AE 0.0435 vs PCA-2 0.0556; 32-D comparison run: AE 0.0213 vs PCA-32 0.0168 |
| Measured | CPU, 17 s; subset sha256 `da4ae28e…`, identical to `ml/basics/pca.py` |

At 2 dimensions the nonlinear autoencoder beats linear PCA; at 32 dimensions PCA
still wins. The autoencoder is the step that makes the VAE's extra machinery
(a prior and a KL term) meaningful, not a replacement for PCA.

## GEN/VAE

| Item | Value |
|---|---|
| Model | Variational Autoencoder (2.6M params) |
| Dataset | CelebA via HF datasets — 10K images (64×64) |
| Architecture | Conv Encoder → μ,logσ² → reparameterize → Deconv Decoder → Sigmoid |
| Loss | Reconstruction (BCE) + KL divergence |
| Training | Adam(lr=2e-4), 50 epoch |
| FID (N=2048) | not measured — no trained checkpoint in this environment |

## NLP/Seq2Seq Transformer

| Item | Value |
|---|---|
| Model | Encoder-Decoder Transformer (1M params) |
| Dataset | Multi30k EN→DE — 29K train / 1K test |
| Architecture | Encoder (from BERT) + Decoder (causal + cross-attention) |
| Training | Teacher forcing, weight-tying, Adam(lr=1e-4) |

## GEN/DDPM

| Item | Value |
|---|---|
| Model | Denoising Diffusion (16.1M params) |
| Dataset | CIFAR-10 via HF datasets — 50K images (32×32) |
| Architecture | UNet + timestep embedding + sinusoid positional encoding |
| Training | Noise prediction (MSE), T=1000, linear β schedule |
| FID (N=2048) | not measured — no trained checkpoint in this environment |
| Sampling | Reverse diffusion (x_T → x_0), 1000 steps |

## GRAPH/GCN

| Item | Value |
|---|---|
| Model | 2-layer Graph Convolutional Network (23K params) |
| Dataset | Cora via URL — 2708 nodes, 1433 features, 7 classes |
| Architecture | GraphConv × 2: Â @ H @ W (spectral graph convolution) |
| Training | Semi-supervised (20 labels/class), CrossEntropyLoss |

## RL/DQN

| Item | Value |
|---|---|
| Model | Deep Q-Network (17K params) |
| Environment | CartPole-v1 via Gymnasium — 4-dim state, 2 actions |
| Architecture | 3-layer MLP (4→128→128→2) |
| Training | Experience replay, target network, ε-greedy decay |

## SELFSUP/SimCLR

| Item | Value |
|---|---|
| Model | SimCLR (11M params: ResNet18 encoder + MLP projector) |
| Dataset | CIFAR-10 via HF datasets — self-supervised (no labels) |
| Architecture | ResNet18 → Projector(512→256→128) → NT-Xent loss |
| Training | 100 epoch, temperature=0.5, dual random augmentation |
| Linear probe | **70.1%** test top-1 (frozen 512-d encoder features) |
| kNN (k=20, cosine) | 67.1% test top-1 (no training) |

## SELFSUP/MAE

| Item | Value |
|---|---|
| Model | Masked Autoencoder (921,200 params; ViT-tiny encoder 4 layers / 4 heads / 128-d, 2-layer decoder) |
| Dataset | CIFAR-10 via HF datasets — 50K images (32×32), 4×4 patches |
| Training | 75% patch masking, MSE on masked patches only, AdamW(lr=1.5e-4), 100 epoch |
| Linear probe | **46.8%** test top-1 (frozen 128-d CLS features) |
| kNN (k=20, cosine) | 37.0% test top-1 |

## SELFSUP/BYOL

| Item | Value |
|---|---|
| Model | BYOL (22.7M params: online + EMA-target ResNet18, projector, predictor) |
| Dataset | CIFAR-10 via HF datasets — 50K images, two augmented views |
| Training | no negative pairs, EMA target (τ=0.999), target centering (0.9), AdamW(lr=1e-4, wd=1e-6), 5-epoch warmup, 100 epoch |
| Linear probe | **55.2%** test top-1 (frozen 512-d online-encoder features) |
| kNN (k=20, cosine) | 48.3% test top-1 |

The first recipe (no centering, wd=1e-4) partially collapsed: loss fell to
~0.01 within two epochs, feature std 0.65 → 0.13, probe 28.7%. Target
centering — subtracting a running mean of the target projections (DINO-style),
plus EMA-updated BatchNorm buffers and a 5-epoch warmup — removed the
collapsed minimum: loss at epoch 2 is 1.28, final loss 0.47, feature std 1.38,
probe 55.2%.

## SELFSUP/DEC

| Item | Value |
|---|---|
| Model | Deep Embedded Clustering: MLP autoencoder (471,066 params, latent 10) + 10 trainable centres |
| Dataset | MNIST via HF datasets — the same 5,000 images as `ml/basics/pca.py` |
| Training | 20 epochs BCE reconstruction, then 50 epochs BCE + 0.1·KL(P‖Q), centres from K-Means |
| ARI (same 5,000, k=10) | pixels K-Means **0.334**, AE-latent K-Means **0.259**, DEC **0.257** |
| Latent linear probe | **84.2%** (80/20 split) |

DEC does not beat K-Means on the autoencoder latent here (0.257 vs 0.259). The
latent is linearly separable for classification (84%) but its unsupervised
clusters do not align with digits better than raw pixels do. The measured
numbers are reported as-is.

## CV/YOLO

| Item | Value |
|---|---|
| Model | Simplified YOLO (59M params) |

## NLP/LoRA

| Item | Value |
|---|---|
| Model | Low-Rank Adaptation on GPT (32K trainable / 5.7M frozen) |
| Dataset | text8 via HF datasets — 5K chunks |
| Architecture | LoRALayer: frozen Linear + low-rank B×A |
| Key concept | Parameter-efficient fine-tuning, 0.58% trainable params |
| Comparison | Full fine-tune: 5.7M vs LoRA r=8: 32K |

## CV/MobileNet

| Item | Value |
|---|---|
| Model | MobileNetV1 (135K params, width=1.0) |
| Dataset | CIFAR-10 via HF datasets — 50K train / 10K test |
| Architecture | DepthwiseSeparableConv (depthwise 3×3 + pointwise 1×1) |
| Key concept | Depthwise separable convolution, ~8.4× fewer ops than standard conv |
| Comparison | SimpleCNN 620K params → MobileNet 135K (4.6× smaller) |
| Dataset | Pascal VOC via HF datasets — 20 classes |
| Architecture | CNN backbone → FC detection head → 7×7×30 output |
| Training | YOLO loss (coord + obj + noobj + class), NMS at inference |

## DCGAN

| Item | Value |
|---|---|
| Model | Generator (3.5M params) + Discriminator (2.8M params) |
| Dataset | CelebA via HF datasets — 10K images (64×64) |
| Architecture | Transposed conv G / Conv D, BN, LeakyReLU |
| Optimizer | Adam(lr=2e-4, β₁=0.5) — separate for G and D |
| Training | BCELoss, label smoothing, fixed noise grid for monitoring |
| FID (N=2048) | **237.6** (20-epoch checkpoint) |

FID is computed by `utils/fid.py` from 2048-d torchvision InceptionV3 features,
so values are only comparable with values produced by that class — not with
published FID tables.

## CV/ViT

| Item | Value |
|---|---|
| Model | Vision Transformer (807K params, 4 layers, 4 heads, 128-dim) |
| Dataset | CIFAR-10 via HF datasets — 50K train / 10K test |
| Architecture | PatchEmbed(4×4) → [CLS] → Transformer Encoder (from BERT) → CLS head |
| Key concept | Self-attention for vision, no convolutions, patch embeddings |

## CV/UNet

| Item | Value |
|---|---|
| Model | U-Net (31M params, 5 encoder/decoder stages) |
| Dataset | Oxford-IIIT Pet via HF datasets — image + segmentation mask |
| Architecture | Encoder: Conv+MaxPool × 4, Decoder: UpConv+skip × 4, output: pixel-wise logits |
| Loss | CrossEntropy (ignore_index=0 for unlabeled) |
| Metrics | Pixel accuracy, mean IoU |

## CV/SimpleCNN

| Item | Value |
|---|---|
| Model | SimpleCNN (620K params) |
| Dataset | CIFAR-10 via HF datasets — 50K images |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Test Accuracy | **82.4%** (30 epochs) |
| Training | Adam + CosineAnnealingLR |

## ML/MLP

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
| Logistic Regression | `ml/basics/logistic_regression.py` | MNIST | 92.3% test accuracy |
| Linear Regression | `ml/basics/linear_regression.py` | California Housing | R²=0.583 |
| K-Means | `ml/basics/k_means.py` | MNIST | purity 57.8%, NMI 0.483, ARI 0.363 (need labels); silhouette 0.056 (label-free) |
| GMM (EM) | `ml/basics/gmm.py` | MNIST 5K → PCA-50 | purity 54.4%, NMI 0.550, ARI 0.398 (labels); silhouette 0.056 (label-free) |
| DBSCAN | `ml/basics/dbscan.py` | MNIST 5K → PCA-50 | 23 clusters, 81% noise; on clustered points: purity 99.4%, NMI 0.851, ARI 0.947 |
| SVM (GD + SMO) | `ml/basics/svm.py` | MNIST 3v5 | 93.3% (RBF kernel) |
| Decision Tree | `ml/basics/decision_tree.py` | Iris | 93.3% test acc |
| Naive Bayes | `ml/basics/naive_bayes.py` | MNIST | 53.0% test acc |
| PCA | `ml/basics/pca.py` | MNIST | 17.3% variance in 2 components |
| t-SNE | `ml/basics/tsne.py` | MNIST | 2-D embedding (perplexity=30), compared with PCA |
| k-NN | `ml/basics/knn.py` | MNIST | ~87% (k=5, 2000 train) |
| Perceptron | `ml/basics/perceptron.py` | MNIST 0v1 | 100% (linearly separable) |

Run the two new clustering demos with `uv run python -m ml.basics.gmm` and `uv run python -m ml.basics.dbscan`.

### SVM implementations

| Method | Type | Kernel | Notes |
|---|---|---|---|
| `SVM_GD` | Primal GD | Linear only | Fast, robust, ~80 lines |
| `SVM_SMO` | Dual SMO | Linear + RBF | Platt SMO, ~150 lines, supports kernel trick |

See [cv/resnet18/README.md](cv/resnet18/README.md) for details.

## Core Concepts

Every model in this project was written from scratch to teach a specific
set of ML/DL concepts. The table below maps each model to the key ideas
it demonstrates.

| Module | Model | Key concepts |
|--------|-------|-------------|
| `ml/basics/` | Logistic Regression | Linear decision boundary, Softmax, Cross-Entropy, closed-form vs gradient descent |
| `ml/basics/` | Linear Regression | Normal Equation, MSE, R² score, feature standardisation |
| `ml/basics/` | K-Means | Unsupervised learning, Euclidean distance, iterative centroid refinement, silhouette (label-free) vs purity/NMI/ARI (evaluation-only) |
| `ml/basics/` | GMM (EM) | Soft assignment, log-sum-exp responsibilities, full covariance, BIC/AIC model selection |
| `ml/basics/` | DBSCAN | Density-based clustering, core/border/noise points, arbitrary shapes, eps sensitivity in high dimensions |
| `ml/basics/` | SVM (GD) | Hinge loss, max-margin classification, L2 regularisation, primal gradient descent |
| `ml/basics/` | SVM (SMO) | Dual formulation, Lagrange multipliers, KKT conditions, kernel trick (RBF) |
| `ml/basics/` | Decision Tree | Entropy, Information Gain, recursive partitioning, interpretable ASCII tree |
| `ml/basics/` | Naive Bayes | Bayes' theorem, generative vs discriminative models, Gaussian likelihood, log-space prediction |
| `ml/basics/` | PCA | Singular Value Decomposition (SVD), eigenvalue, dimensionality reduction, variance explained |
| `ml/basics/` | t-SNE | Nonlinear dimensionality reduction, perplexity, KL divergence, local vs global structure |
| `gen/autoencoder/` | Autoencoder | Reconstruction-only training, bottleneck representation, nonlinear encoder/decoder, no prior or KL term |
| `ml/basics/` | k-NN | Instance-based learning, distance metrics, curse of dimensionality, bias-variance tradeoff |
| `ml/basics/` | Perceptron | Single neuron, step activation, online learning, Perceptron Convergence Theorem |
| `mlp/` | MLP (NumPy) | **Manual backpropagation**, chain rule, gradient descent without autograd, softmax cross-entropy |
| `cv/simplecnn/` | SimpleCNN | Convolution, max-pooling, BatchNorm, Dropout, CosineAnnealing LR schedule |
| `cv/resnet18/` | ResNet18 | **Residual connections (skip connections)**, BatchNorm in deep networks, bottleneck design, AMP |
| `cv/resnet34/` | ResNet34 | SGD+Momentum, CosineAnnealingLR, gradient accumulation, early stopping, ROC AUC, F1 |
| `cv/resnet50/` | ResNet50 | Bottleneck block (1×1→3×3→1×1), deeper residual networks |
| `gen/vae/` | VAE | Reparameterization trick, KL divergence, latent space interpolation |
| `nlp/seq2seq/` | Seq2Seq Transformer | Encoder-decoder, cross-attention, teacher forcing, weight-tying |
| `gen/ddpm/` | DDPM | Denoising Diffusion, UNet + timestep embedding, noise prediction |
| `gen/dcgan/` | DCGAN | Transposed convolution, adversarial training, generator/discriminator dynamics |
| `cv/vit/` | Vision Transformer (ViT) | Patch embedding, self-attention for vision, Transformer without convolutions |
| `cv/unet/` | U-Net | Encoder-decoder, skip connections, pixel-wise classification, IoU metric |
| `nlp/bert/` | BERT mini | **Self-Attention** (semantic aggregation), **Masked Language Model** (entropy increase + denoising), LayerNorm, positional encoding |
| `nlp/word2vec/` | Word2Vec | **Embedding lookup tables**, **Negative Sampling**, CBOW vs Skip-gram, subsampling frequent words, cosine similarity |
| `nlp/lstm/` | LSTM | **Input/forget/output gates**, **cell state**, gradient flow through gating, sequential processing vs parallel attention |
| `graph/gcn/` | GCN | Graph convolution, message passing, semi-supervised node classification |
| `rl/dqn/` | DQN | Q-Learning, experience replay, target network, ε-greedy |
| `selfsup/simclr/` | SimCLR | Contrastive learning, NT-Xent loss, data augmentation, frozen-encoder linear probe + kNN evaluation |
| `selfsup/mae/` | MAE | Masked image modeling, high mask ratio, asymmetric encoder/decoder, pixel reconstruction, linear probe |
| `selfsup/byol/` | BYOL | EMA target network, no negative pairs, predictor asymmetry, target centering against collapse |
| `selfsup/deep_cluster/` | DEC | Joint reconstruction + KL clustering, Student-t soft assignments, target sharpening |
| `nlp/lora/` | LoRA | Low-rank adaptation, parameter-efficient fine-tuning, GPT adapter |
| `cv/mobilenet/` | MobileNet | Depthwise separable convolution, efficient CNN, width multiplier |
| `cv/yolo/` | YOLO | Single-stage object detection, grid-based regression, NMS |
| `nlp/gpt/` | GPT | **Causal Self-Attention**, **KV Cache**, autoregressive generation, word-level tokenizer, temperature + top-k sampling, bad-token blocking |

### Unsupervised and self-supervised examples

| Family | Examples |
|---|---|
| Unsupervised (no labels) | [K-Means](ml/basics/k_means.py), [PCA](ml/basics/pca.py), [t-SNE](ml/basics/tsne.py), [GMM](ml/basics/gmm.py), [DBSCAN](ml/basics/dbscan.py), [Autoencoder](gen/autoencoder/), [VAE](gen/vae/), [DCGAN](gen/dcgan/), [DDPM](gen/ddpm/) |
| Self-supervised (labels from the data) | [SimCLR](selfsup/simclr/), [Word2Vec](nlp/word2vec/), [BERT MLM](nlp/bert/pretrain.py), [GPT](nlp/gpt/train.py) |

## Setup & Run

```bash
uv sync
```

```bash
# Train / Evaluate ResNet18
uv run python -m cv.resnet18.train
uv run python -m cv.resnet18.eval

# Train / Evaluate ResNet34
uv run python -m cv.resnet34.train
uv run python -m cv.resnet34.eval

# Train / Evaluate ResNet50
uv run python -m cv.resnet50.train
uv run python -m cv.resnet50.eval

# Train / Generate Autoencoder
uv run python -m gen.autoencoder.train
uv run python -m gen.autoencoder.generate

# Train / Generate / Evaluate (FID) VAE
uv run python -m gen.vae.train
uv run python -m gen.vae.generate
uv run python -m gen.vae.eval

# Train / Translate Seq2Seq
uv run python -m nlp.seq2seq.train
uv run python -m nlp.seq2seq.generate

# Train / Evaluate GCN
uv run python -m graph.gcn.train
uv run python -m graph.gcn.eval

# Train DQN
uv run python -m rl.dqn.train

# Train / Evaluate SimCLR (frozen-encoder linear probe + kNN)
uv run python -m selfsup.simclr.train
uv run python -m selfsup.simclr.eval

# Train / Evaluate MAE (masked image modeling + linear probe)
uv run python -m selfsup.mae.train
uv run python -m selfsup.mae.eval

# Train / Evaluate BYOL (EMA target, no negatives)
uv run python -m selfsup.byol.train
uv run python -m selfsup.byol.eval

# Train DEC (deep clustering on MNIST)
uv run python -m selfsup.deep_cluster.train

# Train YOLO
uv run python -m cv.yolo.train

# Train / Generate LoRA (requires nlp/gpt/gpt_text8.pt)
uv run python -m nlp.lora.train
uv run python -m nlp.lora.generate

# Train / Evaluate MobileNet
uv run python -m cv.mobilenet.train
uv run python -m cv.mobilenet.eval

# Train / Generate / Evaluate (FID) DDPM
uv run python -m gen.ddpm.train
uv run python -m gen.ddpm.generate
uv run python -m gen.ddpm.eval

# Train / Generate / Evaluate (FID) DCGAN
uv run python -m gen.dcgan.train
uv run python -m gen.dcgan.generate
uv run python -m gen.dcgan.eval

# Train / Evaluate ViT
uv run python -m cv.vit.train
uv run python -m cv.vit.eval

# Train / Evaluate UNet
uv run python -m cv.unet.train
uv run python -m cv.unet.eval

# Train / Evaluate CNN
uv run python -m cv.simplecnn.train
uv run python -m cv.simplecnn.eval

# Train / Evaluate MLP (pure NumPy)
uv run python -m ml.mlp.train
uv run python -m ml.mlp.eval

# Basics
uv run python -m ml.basics.logistic_regression
uv run python -m ml.basics.k_means
uv run python -m ml.basics.linear_regression
uv run python -m ml.basics.svm
uv run python -m ml.basics.decision_tree
uv run python -m ml.basics.naive_bayes
uv run python -m ml.basics.pca
uv run python -m ml.basics.knn
uv run python -m ml.basics.perceptron
uv run python -m ml.basics.tsne

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

### GPU environment (GB10 / Grace Blackwell)

The environment from `uv sync` drives GB10 directly — `torch 2.12.1+cu130` and
`torchvision 0.27.1+cu130` from PyPI:

```bash
uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# NVIDIA GB10 (12, 1)
```

Verified 2026-11-12: 48 SMs, 130.6 GB; compiled kernels for
`sm_80/90/100/110/120`; a BF16 conv/backward step and an 8K BF16 matmul
(87.7 TFLOP/s) run on the GPU.

GPU access requires opening `/dev/nvidia*`. A file sandbox that denies those
opens (some agent sandboxes do) reports `CUDA unavailable` and
`Failed to initialize NVML` even though the driver and wheel are fine — run GPU
commands outside such a sandbox.

#### Alternative: NGC PyTorch container

For a newer stack, NVIDIA's NGC image works on GB10 too:

```bash
docker build -t dlf-ngc:26.09 -f docker/ngc-pytorch.Dockerfile docker/

scripts/ngc-run.sh python -c "import torch; print(torch.cuda.get_device_name(0))"
scripts/ngc-run.sh python -m cv.simplecnn.train
```

The runner adds `--gpus all` and `--ipc=host`, mounts this repo at `/workspace`
and the HuggingFace cache at `$HF_CACHE` (default `~/.cache/huggingface`), and
runs as your `uid:gid` so checkpoints stay owned by you. The image is the official
`nvcr.io/nvidia/pytorch:26.09-py3` plus `scikit-learn`, the only dependency the
official image lacks.

Verified: `torch 2.14.0a0+b2c75dd062.nv26.09`, capability `(12, 1)`, 79.3 TFLOP/s
on the same 8K BF16 matmul, and `ml/basics/k_means` reproduces the host metrics
JSON byte-for-byte.

## Models

Trained weights are **not tracked in git** (`.gitignore`'ed). Each model saves its weights
locally after training; paths are shown below for reference.

| Model | Local path | Size |
|---|---|---|
| ResNet18 (15 attrs, 1K samples) | `cv/resnet18/resnet18_celeba.pt` | 45 MB |
| ResNet34 (40 attrs, 200K samples) | `cv/resnet34/resnet34_celeba.pt` | ~80 MB |
| ResNet50 (40 attrs, 200K samples) | `cv/resnet50/resnet50_celeba.pt` | ~90 MB |
| Autoencoder (MNIST, 2-D) | `gen/autoencoder/autoencoder_mnist.pt` | 1.9 MB |
| VAE (CelebA, 64×64) | `gen/vae/vae_celeba.pt` | 10 MB |
| Seq2Seq Transformer (Multi30k) | `nlp/seq2seq/seq2seq_multi30k.pt` | 4 MB |
| GCN (Cora) | `graph/gcn/gcn_cora.pt` | 0.1 MB |
| DQN (CartPole) | `rl/dqn/dqn_cartpole.pt` | 0.07 MB |
| SimCLR (CIFAR-10) | `selfsup/simclr/simclr_cifar10.pt` | 44 MB |
| YOLO (Pascal VOC) | `cv/yolo/yolo_voc.pt` | 226 MB |
| LoRA (GPT-adapted, text8) | `nlp/lora/lora_gpt.pt` | 0.2 MB |
| MobileNet (CIFAR-10) | `cv/mobilenet/mobilenet_cifar10.pt` | 0.5 MB |
| DDPM (CIFAR-10, 32×32) | `gen/ddpm/ddpm_cifar10.pt` | 62 MB |
| DCGAN (CelebA, 64×64) | `gen/dcgan/dcgan_celeba.pt` | ~23 MB (G+D) |
| ViT (CIFAR-10, 32×32) | `cv/vit/vit_cifar10.pt` | 3.2 MB |
| UNet (Oxford-Pet, 128×128) | `cv/unet/unet_oxford_pet.pt` | 119 MB |
| SimpleCNN (CIFAR-10) | `cv/simplecnn/simple_cnn_cifar10.pt` | 2.4 MB |
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
