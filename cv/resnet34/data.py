from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from torchvision import transforms


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

TRAIN_SIZE = 162770
VAL_SIZE = 19867
TEST_SIZE = 19962


class CelebADataset(Dataset):
    def __init__(self, split="train", transform=None, num_samples=None):
        self.transform = transform
        self.attr_indices = list(range(40))

        # eurecom-ds/celeba exposes train / validation / test as separate splits.
        # Slicing "train[N:M]" does not work: the train split ends at TRAIN_SIZE,
        # so a slice into the validation range raises "corresponds to no data".
        if split == "train":
            hf_split = "train"
        elif split == "val":
            hf_split = "validation"
        elif split == "test":
            hf_split = "test"
        else:
            raise ValueError(f"Unknown split: {split}")

        # ``num_samples`` bounds the length: the notebooks train on a 10K slice
        # instead of the full 162,770-image split.
        if num_samples is not None:
            hf_split = f"{hf_split}[:{num_samples}]"

        # Keep the Arrow table and decode one row per __getitem__. Walking the
        # split up front decodes a PIL image for every row -- 162,770 for train,
        # ~19,900 for test -- and holds them all in RAM, which is what made
        # `python -m cv.resnet50.train` unstartable on a 16 GB machine. The
        # sliced split (``train[:N]``) bounds the length, not the footprint.
        self.dataset = load_dataset("eurecom-ds/celeba", split=hf_split)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        labels = torch.tensor(
            [1.0 if item["attributes"][i] == 1 else 0.0 for i in self.attr_indices],
            dtype=torch.float32,
        )
        image_pil = item["image"]
        return self.transform(image_pil) if self.transform else image_pil, labels

    def __len__(self):
        return len(self.dataset)


def train_transform():
    """Training transforms with data augmentation."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05),
        transforms.RandomRotation(5),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def eval_transform():
    """Standard transforms without augmentation."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
