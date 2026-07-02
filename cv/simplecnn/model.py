import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A plain CNN for CIFAR-10 (32x32 RGB images, 10 classes).

    Architecture: Conv -> Pool -> Conv -> Pool -> Conv -> Pool -> FC -> FC

    Design principles:
      - Stack of 3x3 conv layers: a stack of 3x3 convs achieves the same
        receptive field as a single larger kernel (e.g. 7x7) with fewer
        parameters and more non-linearity between layers.
      - Doubling channels each stage: as spatial size halves (via pooling),
        we double the number of channels to maintain representational capacity
        (the "information preservation" heuristic).
      - BatchNorm after each conv: stabilizes training by normalizing layer
        outputs, reduces internal covariate shift, and allows higher learning rates.
      - Dropout before the final classifier: randomly drops 50% of neurons during
        training as a regularizer, preventing co-adaptation of features.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # First conv block: RGB (3 channels) -> 32 channels
        # Why 3x3? Small kernel captures local patterns (edges, textures) efficiently.
        # Why padding=1? Preserves spatial size so conv outputs 32x32.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # MaxPool halves spatial dims: 32x32 -> 16x16
        # Why MaxPool over AvgPool? MaxPool preserves the strongest activation,
        # retaining the most salient features from each local region.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second conv block: 32 -> 64 channels, spatial: 16x16 -> 8x8
        # Why 64 channels? As spatial size shrinks, we increase depth to
        # compensate for information loss, letting later layers learn more
        # abstract / compositional features.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third conv block: 64 -> 128 channels, spatial: 8x8 -> 4x4
        # After this block, the 4x4 feature map is small enough to flatten into FC layers.
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flattened size: 128 channels * 4 * 4 = 2048
        # Why 256 hidden units? A bottleneck layer between the high-dim conv output
        # and the small number of classes forces the network to learn compact representations.
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        # Dropout: randomly zeroes 50% of the 256 hidden units during each forward pass
        # during training. This prevents the model from relying too heavily on any single
        # neuron, acting as an ensemble-like regularizer.
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1: Conv -> BN -> ReLU -> Pool
        # Why ReLU? Non-linear activation; avoids vanishing gradient problem of sigmoid/tanh.
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Flatten: (B, 128, 4, 4) -> (B, 2048)
        x = torch.flatten(x, 1)

        # Classifier head
        # Why no BN/ReLU after fc1? Dropout already regularizes; we keep it simple.
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # No softmax here — CrossEntropyLoss in train.py already combines
        # LogSoftmax + NLLLoss internally. Returning raw logits is numerically
        # more stable and more flexible (e.g. for temperature scaling).
        x = self.fc2(x)
        return x


def simple_cnn(num_classes=10):
    """
    Convenience factory function.

    Usage:
        model = simple_cnn(num_classes=10)
    """
    return SimpleCNN(num_classes)
