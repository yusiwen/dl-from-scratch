# ResNet — CelebA Attribute Classification

## Current Setup

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
uv run python resnet/train.py
```

## Future Optimization Directions

- [ ] **ResNet34**: Replace BasicBlock × 2 per stage with [3, 4, 6, 3] blocks
- [ ] **SGD + Momentum**: Replace Adam with SGD(momentum=0.9, weight_decay=1e-4), tune lr
- [ ] **Full 40 attributes**: Use all CelebA attributes as multi-label targets
- [ ] **Full dataset (200K images)**: Scale beyond 1,000 samples; needs multi-worker dataloader, prefetch, batch tuning
- [ ] **LR scheduler**: CosineAnnealingLR or StepLR
- [ ] **Data augmentation**: RandomHorizontalFlip, ColorJitter, RandomRotation
- [ ] **Early stopping + checkpoint**: Save best model by val loss, stop on plateau
- [ ] **Gradient accumulation**: Simulate larger batch sizes on limited memory
- [ ] **Label imbalance weighting**: Compute pos_weight for BCEWithLogitsLoss per attribute
- [ ] **TensorBoard / WandB**: Log loss curves, per-attribute accuracy, learning rate
- [ ] **Per-attribute metrics**: ROC AUC, F1 per attribute
- [ ] **Test set evaluation**: CelebA standard test split (last ~40K images)
