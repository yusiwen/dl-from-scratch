from torchvision import transforms
from datasets import load_dataset
import torch


def load_celeba(num_samples=10000, image_size=64):
    transform = transforms.Compose([
        transforms.Resize(image_size + 8),
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
    ])

    ds = load_dataset("eurecom-ds/celeba", split=f"train[:{num_samples}]")
    images = [transform(item["image"]).unsqueeze(0) for item in ds]
    dataset = torch.cat(images)
    return dataset
