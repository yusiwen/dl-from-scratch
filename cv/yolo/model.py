"""Simplified YOLO: CNN backbone + detection head."""

import torch
import torch.nn as nn


class YOLO(nn.Module):
    """Simplified YOLO detector (like YOLOv1)."""

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

        # Detection head.
        self.det_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, S * S * (B * 5 + C)),
        )

    def forward(self, x):
        B = x.size(0)
        x = self.features(x)
        x = self.det_head(x)
        return x.view(B, self.S, self.S, self.B * 5 + self.C)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
