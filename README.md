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
| Test Accuracy | ~80% (30 epochs) |
| Training | Adam + CosineAnnealingLR |

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
```

## Models

| Model | File | Size |
|---|---|---|
| ResNet18 (15 attrs, 1K samples) | `resnet/resnet18_celeba.pt` | 45 MB |
| SimpleCNN (CIFAR-10) | `cnn/simple_cnn_cifar10.pt` | 2.4 MB |
