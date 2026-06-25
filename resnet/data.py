from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from PIL import Image

# The 40 CelebA attributes in their canonical order.
# This matches the attribute list order used by eurecom-ds/celeba.
CELEBA_ATTR_ORDER = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young",
]

# The 15 attributes we use for training.
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

# Pre-compute column indices for our selected attributes.
_ATTR_INDICES = [CELEBA_ATTR_ORDER.index(a) for a in ATTRIBUTES]


class CelebADataset(Dataset):
    """
    CelebA dataset loaded via HuggingFace datasets library.

    The dataset is cached as Arrow files in ~/.cache/huggingface/datasets/
    (not individual images on disk). Attributes are stored as a list of
    40 ints (0 or 1), drawn from the canonical order above.

    Only the first `num_samples` images are materialized.
    """

    def __init__(self, attributes=None, num_samples=1000, transform=None):
        self.attributes = attributes or ATTRIBUTES
        self.transform = transform
        # Load only the first num_samples rows from the HF dataset.
        # Split syntax "train[:N]" reads N rows without loading the full dataset.
        ds = load_dataset("eurecom-ds/celeba", split=f"train[:{num_samples}]")

        # Pre-extract labels for our selected attributes.
        # This avoids repeated list lookups in __getitem__.
        self.samples = []
        for item in ds:
            labels = [
                1.0 if item["attributes"][idx] == 1 else 0.0
                for idx in _ATTR_INDICES
            ]
            self.samples.append(((item["image"]), torch.tensor(labels, dtype=torch.float32)))

        # Release the HF dataset reference so it can be garbage-collected.
        del ds

    def __getitem__(self, idx):
        image_pil, labels = self.samples[idx]
        if self.transform:
            image = self.transform(image_pil)
        else:
            image = image_pil
        return image, labels

    def __len__(self):
        return len(self.samples)
