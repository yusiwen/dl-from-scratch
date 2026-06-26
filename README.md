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
├── resnet/
│   ├── README.md          # Optimization roadmap & details
│   ├── __init__.py
│   ├── data.py            # CelebA via HF datasets (eurecom-ds/celeba)
│   ├── model.py           # ResNet18 from scratch
│   ├── train.py           # Training script (MPS + AMP)
│   ├── eval.py            # Evaluation script (per-attribute accuracy)
│   └── resnet18_celeba.pt      # [LFS] Trained model (45 MB)
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

## ResNet

| Item | Value |
|---|---|
| Model | ResNet18 (11.2M params) |
| Dataset | CelebA via HF datasets — 1,000 images |
| Attributes | 15 binary (Smiling, Male, Young, Eyeglasses, etc.) |
| Split | 800 train / 200 val |
| Val Accuracy | **91.2%** |
| Training | MPS (Mac M4) + AMP |

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

See [resnet/README.md](resnet/README.md) for details and optimization roadmap.

## Setup & Run

```bash
uv sync
```

```bash
# Train / Evaluate ResNet
uv run python -m resnet.train
uv run python -m resnet.eval

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
```

## Models

Trained weights are **not tracked in git** (`.gitignore`'ed). Each model saves its weights
locally after training; paths are shown below for reference.

| Model | Local path | Size |
|---|---|---|
| ResNet18 (15 attrs, 1K samples) | `resnet/resnet18_celeba.pt` | 45 MB |
| SimpleCNN (CIFAR-10) | `cnn/simple_cnn_cifar10.pt` | 2.4 MB |
| MLP (MNIST, NumPy) | `mlp/mlp_mnist.npz` | 0.9 MB |
| Logistic Regression | `basics/logistic_regression.npz` | 63 KB |
| K-Means centers | `basics/kmeans_centers.npz` | 32 KB |
| Linear Regression | `basics/linear_regression.npz` | 2 KB |
| SVM | `basics/svm.npz` | 45 KB |
| Decision Tree | — | N/A (no weights) |
| Naive Bayes | — | N/A (no weights) |
