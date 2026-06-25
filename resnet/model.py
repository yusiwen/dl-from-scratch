import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    A BasicBlock is the building block of ResNet-18 and ResNet-34.

    Architecture:  Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+ shortcut) -> ReLU

    The core idea of ResNet is the **shortcut connection** (also called skip connection).
    Instead of learning H(x) directly, the block learns F(x) = H(x) - x (the "residual"),
    and the output is F(x) + x. This is why it's called a "residual" block.

    Why residual connections matter:
      - They allow gradients to flow directly through the shortcut during backpropagation,
        mitigating the vanishing gradient problem in deep networks.
      - They make it easier for layers to learn identity mappings (just zero out F(x)),
        so adding more layers doesn't hurt performance.
      - This enabled training of very deep networks (50+ layers) for the first time.

    The expansion = 1 means the output channels equal `planes` (unlike BottleneckBlock
    used in ResNet-50/101/152 where expansion = 4).
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # First 3x3 convolution.
        # Why 3x3? Small kernel captures local spatial patterns efficiently.
        # A stack of two 3x3 convs has the same receptive field as one 5x5 conv,
        # but with fewer parameters (2*9=18 vs 25) and more non-linearity.
        # Why bias=False? Bias is redundant when followed by BatchNorm (BN subtracts mean).
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Second 3x3 convolution (always stride=1 to preserve spatial size).
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection: when stride != 1 or channel count changes, we need a
        # 1x1 convolution to match dimensions before adding.
        #
        # Case 1: stride=1 and in_planes == planes*expansion
        #   -> identity shortcut (no extra params), just adds x directly.
        # Case 2: stride != 1 (downsampling) or channel mismatch
        #   -> 1x1 conv with stride to match spatial & channel dimensions.
        #
        # Why 1x1 conv? It changes channel count without adding spatial information,
        # acting as a lightweight "projection" between layers.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        # Conv -> BN -> ReLU is the standard pattern.
        # BN normalizes activations to have zero mean and unit variance,
        # which stabilizes training and allows higher learning rates.
        # ReLU introduces non-linearity; it's preferred over sigmoid/tanh
        # because it doesn't saturate and avoids vanishing gradients.
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # The residual connection: add the input (or projected input) to the output.
        # This creates a "highway" for gradients to flow backward.
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    Generic ResNet implementation parameterizable for different depths.

    The network has 5 stages:
      1. Initial conv7x7 + maxpool (aggressive downsampling to 56x56)
      2. Four residual stages (layer1 through layer4)
      3. Global average pooling + FC classifier

    Design rationale:
      - Initial large 7x7 conv: quickly reduces spatial size from 224x224 to 56x56
        while capturing early low-level features (edges, textures).
      - MaxPool: further downsamples 2x, keeping only the strongest activations.
      - Each stage doubles channels and halves spatial size (via strided convs in
        the first block of the stage). This is a common pattern: as spatial
        resolution decreases, representational depth increases.
      - Global average pooling: replaces multiple FC layers at the end; it's more
        robust to overfitting and has zero learnable parameters.
    """

    def __init__(self, block, num_blocks, num_classes=15):
        super().__init__()
        # Track the current number of input channels for _make_layer.
        self.in_planes = 64

        # Initial conv: 7x7 kernel, stride 2, padding 3 -> 112x112 output.
        # 3 input channels (RGB) -> 64 output channels.
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # MaxPool: 3x3 kernel, stride 2, padding 1 -> 56x56 output.
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Four residual stages with increasing channel depth.
        # ResNet18: [2, 2, 2, 2] blocks per stage.
        # numbers = block count at each stage.
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Global average pool: reduces each 7x7 feature map to 1x1.
        # Why global? It's adaptive — works for any input resolution.
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Final classifier: 512 features -> num_classes.
        # For CelebA, num_classes=15 (one logit per binary attribute).
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Create a stage consisting of `num_blocks` BasicBlocks.

        The first block may have stride > 1 (for downsampling).
        Subsequent blocks always have stride=1.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            # Update in_planes for the next block in this stage.
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stage 0: initial 7x7 conv + maxpool
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        # Stages 1-4: residual blocks.
        # Spatial progression: 56x56 -> 28x28 -> 14x14 -> 7x7.
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Global average pooling: 7x7x512 -> 1x1x512.
        out = self.avgpool(out)
        # Flatten to (batch, 512) for the FC layer.
        out = torch.flatten(out, 1)
        # Classifier: (batch, 512) -> (batch, num_classes).
        out = self.fc(out)

        # Returns raw logits (no sigmoid). BCEWithLogitsLoss in train.py
        # applies sigmoid internally — this is numerically more stable.
        return out


def resnet18(num_classes=15):
    """
    Convenience factory function for ResNet-18.

    ResNet-18 has 4 stages with [2, 2, 2, 2] BasicBlocks each,
    totaling about 11 million parameters.

    Usage:
        model = resnet18(num_classes=15)
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
