from resnet18.model import ResNet, BasicBlock


def resnet34(num_classes=40):
    """ResNet34 with [3, 4, 6, 3] bottleneck blocks (~21M params)."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
