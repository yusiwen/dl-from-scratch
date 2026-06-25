# DL From Scratch

从零手写主流深度学习模型。

## Project Structure

```
├── main.py
├── pyproject.toml
├── .gitignore
├── README.md
├── resnet/
│   ├── README.md          # ResNet 优化方向
│   ├── __init__.py
│   ├── data.py            # CelebA zip 流式读取
│   ├── model.py           # ResNet18 手写实现
│   ├── train.py           # 训练脚本 (MPS + AMP)
│   ├── eval.py            # 验证脚本 (per-attribute accuracy)
│   ├── celeba/
│   │   ├── img_align_celeba.zip    # [LFS] 200K face images
│   │   └── list_attr_celeba.txt    # 40 binary attributes
│   └── resnet18_celeba.pt      # [LFS] Trained model (45 MB)
├── .gitattributes                 # LFS: *.zip *.pt
└── uv.lock
```

## ResNet

| Item | Value |
|---|---|
| Model | ResNet18 (11.2M params) |
| Dataset | CelebA — 1,000 images (zip streaming) |
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
