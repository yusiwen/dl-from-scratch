# ResNet18 — CelebA Attribute Classification

Original baseline implementation. Kept as reference; see `resnet34/` for the
advanced version with full dataset, data augmentation, and optimized training.

| Item | Value |
|---|---|
| Model | ResNet18 (11M params) |
| Dataset | CelebA, first 1,000 images |
| Attributes | 15 binary attributes |
| Optimizer | Adam, lr=1e-3 |
| Batch size | 128 |
| Epochs | 50 |
| Split | 800 train / 200 val |
| Device | MPS (Mac M4, fp32 + AMP) |
| Input | 224×224, Normalize(ImageNet stats) |
| Storage | Zip on-the-fly, no extraction |

## Running

```bash
uv run python -m resnet18.train
```
