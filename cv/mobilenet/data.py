"""CIFAR-10 data loading (same as cv/simplecnn/data.py)."""

from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _build_transform(augment=False):
    ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else []
    ops.extend([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    return transforms.Compose(ops)


_TRAIN_TRANSFORM = _build_transform(augment=True)
_TEST_TRANSFORM = _build_transform(augment=False)


def _train_transform_batch(batch):
    batch["img"] = [_TRAIN_TRANSFORM(img.convert("RGB")) for img in batch["img"]]
    return batch


def _test_transform_batch(batch):
    batch["img"] = [_TEST_TRANSFORM(img.convert("RGB")) for img in batch["img"]]
    return batch


def load_cifar10(batch_size=128, num_workers=2):
    train_dataset = load_dataset("uoft-cs/cifar10", split="train")
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    train_dataset.set_transform(_train_transform_batch)
    test_dataset.set_transform(_test_transform_batch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
