"""Simplified YOLO: CNN backbone + convolutional detection head."""

import torch
import torch.nn as nn


class YOLO(nn.Module):
    """Simplified YOLO detector (YOLOv1 layout: S x S grid, B boxes per cell).

    The head is convolutional on purpose. A ``Flatten() + Linear(1024*S*S, ...)``
    head mixes every spatial location into one vector, so a cell's box
    prediction no longer depends on *its* patch of the feature map; the grid
    structure that the target tensor encodes is thrown away. Keeping the head
    convolutional leaves the (S, S) layout intact and cuts the head from 51M to
    ~5M parameters, which matters with a few thousand training images.
    """

    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S, self.B, self.C = S, B, C

        # CNN backbone: 224 → 112 → 56 → 28 → 14 → 7.
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, 1, 1), nn.BatchNorm2d(1024), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((S, S)),
        )

        # Detection head: per-cell prediction, (B*5+C) channels, one 1x1 conv.
        self.det_head = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, B * 5 + C, 1),
        )

    def forward(self, x):
        x = self.features(x)              # (N, 1024, S, S)
        x = self.det_head(x)              # (N, B*5+C, S, S)
        return x.permute(0, 2, 3, 1)      # (N, S, S, B*5+C)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
