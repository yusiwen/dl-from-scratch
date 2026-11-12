"""CIFAR-10 tensors for masked autoencoding (pixels in [0, 1])."""

import torch
from datasets import load_dataset
from torchvision import transforms


def load_cifar10_tensors(split="train"):
    """Return (images, labels); images are (n, 3, 32, 32) float32 in [0, 1]."""
    transform = transforms.ToTensor()
    dataset = load_dataset("uoft-cs/cifar10", split=split)
    images = torch.stack([transform(img.convert("RGB")) for img in dataset["img"]])
    labels = torch.tensor(dataset["label"], dtype=torch.long)
    return images, labels
