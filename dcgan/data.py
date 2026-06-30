from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CelebADataset(Dataset):
    """CelebA images only (no labels needed for GAN training)."""

    def __init__(self, num_samples=10000, transform=None):
        self.transform = transform or self._default_transform()
        ds = load_dataset("eurecom-ds/celeba", split=f"train[:{num_samples}]")
        self.images = [item["image"] for item in ds]
        del ds

    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.Resize(72),
            transforms.RandomCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, idx):
        return self.transform(self.images[idx])

    def __len__(self):
        return len(self.images)
