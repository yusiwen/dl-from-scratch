import io
import zipfile
import torch
from torch.utils.data import Dataset
from PIL import Image

ATTRIBUTES = [
    "Smiling",
    "Male",
    "Young",
    "Eyeglasses",
    "Bald",
    "Heavy_Makeup",
    "Wearing_Lipstick",
    "Bangs",
    "Blond_Hair",
    "Black_Hair",
    "Chubby",
    "Mustache",
    "Wearing_Hat",
    "Pale_Skin",
    "Rosy_Cheeks",
]


class CelebADataset(Dataset):
    def __init__(self, zip_path, attr_path, attributes=None, num_samples=1000, transform=None):
        self.zip_path = zip_path
        self.attr_path = attr_path
        self.attributes = attributes or ATTRIBUTES
        self.transform = transform
        self._zip = None

        with open(attr_path) as f:
            _ = f.readline()
            header = f.readline().strip().split()
        self.attr_indices = [header.index(attr) for attr in self.attributes]

        self.samples = []
        with open(attr_path) as f:
            _ = f.readline()
            _ = f.readline()
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                parts = line.strip().split()
                filename = parts[0]
                labels = [
                    1.0 if parts[idx + 1] == "1" else 0.0
                    for idx in self.attr_indices
                ]
                self.samples.append((filename, torch.tensor(labels, dtype=torch.float32)))

    @property
    def zip(self):
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path, "r")
        return self._zip

    def __getitem__(self, idx):
        filename, labels = self.samples[idx]
        with self.zip.open(f"img_align_celeba/{filename}") as f:
            image = Image.open(io.BytesIO(f.read())).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, labels

    def __len__(self):
        return len(self.samples)
