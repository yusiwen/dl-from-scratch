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
│   ├── data.py            # CelebA zip streaming dataset
│   ├── model.py           # ResNet18 from scratch
│   ├── train.py           # Training script (MPS + AMP)
│   ├── eval.py            # Evaluation script (per-attribute accuracy)
│   └── resnet18_celeba.pt      # [LFS] Trained model (45 MB)
├── cnn/                       # Simple CNN for CIFAR-10
│   ├── __init__.py
│   ├── data.py            # CIFAR-10 dataloader (torchvision)
│   ├── model.py           # Plain CNN (Conv×3 + Pool×3 + FC×2)
│   ├── train.py           # Training script (Adam + CosineAnnealingLR)
│   └── eval.py            # Test evaluation + confusion matrix
├── data/
│   └── celeba/
│       ├── img_align_celeba.zip    # (optional, no longer tracked)
│       └── list_attr_celeba.txt    # (optional, no longer tracked)
├── .gitattributes                 # LFS: *.zip *.pt
└── uv.lock
```

## ResNet

| Item | Value |
|---|---|
| Model | ResNet18 (11.2M params) |
| Dataset | CelebA via HF datasets — 1,000 images (eurecom-ds/celeba) |
| Attributes | 15 binary (Smiling, Male, Young, Eyeglasses, etc.) |
| Split | 800 train / 200 val |
| Val Accuracy | **91.2%** |
| Training | MPS (Mac M4) + AMP |

See [resnet/README.md](resnet/README.md) for details and optimization roadmap.

## Setup & Run

```bash
uv sync
```

```bash
# Train
uv run python -m resnet.train

# Evaluate
uv run python -m resnet.eval
```

## Models

| Model | File | Size |
|---|---|---|
| ResNet18 (15 attrs, 1K samples) | `resnet/resnet18_celeba.pt` | 45 MB |
