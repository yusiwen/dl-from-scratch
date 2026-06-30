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
    def __init__(self, split="train", transform=None):
        self.transform = transform
        self.attr_indices = list(range(40))

        if split == "train":
            slice_ = slice(0, TRAIN_SIZE)
        elif split == "val":
            slice_ = slice(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE)
        elif split == "test":
            slice_ = slice(TRAIN_SIZE + VAL_SIZE, TRAIN_SIZE + VAL_SIZE + TEST_SIZE)
        else:
            raise ValueError(f"Unknown split: {split}")

        ds = load_dataset("eurecom-ds/celeba", split=f"train[{slice_.start}:{slice_.stop}]")

        self.samples = []
        for item in ds:
            labels = [1.0 if item["attributes"][idx] == 1 else 0.0 for idx in self.attr_indices]
            self.samples.append((item["image"], torch.tensor(labels, dtype=torch.float32)))

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
