"""MobileNetV1: Depthwise Separable Convolutions."""

import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Conv3×3 → Pointwise Conv1×1 + BN + ReLU."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))


class MobileNet(nn.Module):
    """MobileNetV1 for CIFAR-10 (32×32 input)."""

    def __init__(self, num_classes=10, width_multiplier=1.0):
        super().__init__()
        w = width_multiplier

        def _ch(base):
            return max(int(base * w), 8)

        self.features = nn.Sequential(
            nn.Conv2d(3, _ch(32), 3, padding=1, bias=False),
            nn.BatchNorm2d(_ch(32)),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(_ch(32), _ch(64), stride=1),
            DepthwiseSeparableConv(_ch(64), _ch(128), stride=2),
            DepthwiseSeparableConv(_ch(128), _ch(128), stride=1),
            DepthwiseSeparableConv(_ch(128), _ch(256), stride=2),
            DepthwiseSeparableConv(_ch(256), _ch(256), stride=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(_ch(256), num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
