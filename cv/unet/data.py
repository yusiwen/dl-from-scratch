import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from datasets import load_dataset
import random


class PetDataset(Dataset):
    """Oxford-IIIT Pet with image-mask pairs for segmentation."""

    # Mask values: 0=outside bbox, 1=foreground, 2=background, 3=contour
    # We map to 3 classes: 0→ignore, 1→foreground, 2→background, 3→contour
    # CrossEntropyLoss ignore_index=0 handles unlabeled pixels.
    MASK_CLASSES = 3

    def __init__(self, split="train", image_size=128, augment=False):
        self.image_size = image_size
        self.augment = augment and split == "train"

        ds = load_dataset("tchevrou/oxford-iiit-pet", split=split)
        self.images = [item["image"].convert("RGB") for item in ds]
        self.masks = [item["label"] for item in ds]
        del ds

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Resize.
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        # Data augmentation.
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask = torch.tensor(list(mask.getdata()), dtype=torch.long).view(self.image_size, self.image_size)
        # Map mask values: 0 stays 0 (ignored by loss), 1→1, 2→2, 3→3.
        # No remapping needed since values already match our classes.

        return image, mask


def visualize_sample(image, mask, num_classes=3):
    """Convert tensors back to display format for debugging."""
    import numpy as np
    img = image.cpu().permute(1, 2, 0).numpy()
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img = np.clip(img, 0, 1)
    mask_img = (mask.cpu().numpy() / num_classes * 255).astype(np.uint8)
    return img, mask_img
