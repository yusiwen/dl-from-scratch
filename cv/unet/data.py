"""Oxford-IIIT Pet segmentation dataset backed by torchvision.

The Hugging Face id this module used to rely on (``tchevrou/oxford-iiit-pet``)
no longer exists, so we read the official Oxford archive through torchvision,
which ships the trimap annotations needed for segmentation training.
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF


# Trimap values in the Oxford annotations: 1 = pet, 2 = background, 3 = boundary.
# The model has num_classes=3 and the loss ignores index 0, so we remap
# 1 -> 1 (pet), 2 -> 2 (background) and 3 -> 0 (boundary, ignored by the loss).
TRIMAP_TO_CLASS = {1: 1, 2: 2, 3: 0}

# 256-entry lookup table: much faster than a per-pixel dict lookup.
_LUT = np.zeros(256, dtype=np.int64)
for _src, _dst in TRIMAP_TO_CLASS.items():
    _LUT[_src] = _dst

DATA_ROOT = "data/oxford-pet"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class PetDataset(Dataset):
    """Oxford-IIIT Pet with image-mask pairs for segmentation.

    The archive is ~800 MB and is downloaded once into DATA_ROOT (gitignored).
    Samples are decoded lazily in __getitem__, so memory stays flat instead of
    holding every image of the split as a PIL object.
    """

    # 3 classes: 0 = ignored boundary pixels, 1 = pet, 2 = background.
    MASK_CLASSES = 3

    def __init__(self, split="train", image_size=128, augment=False, max_samples=None):
        self.image_size = image_size
        self.augment = augment and split == "train"

        # torchvision exposes trainval/test; "train" and "val" both come from
        # the trainval archive, "test" is the held-out set.
        tv_split = "test" if split == "test" else "trainval"
        self.dataset = OxfordIIITPet(
            root=DATA_ROOT,
            split=tv_split,
            # torchvision returns (image, mask) for this target type; the image
            # is always included, "image" is not a valid target type.
            target_types="segmentation",
            download=True,
        )

        self.indices = list(range(len(self.dataset)))
        if max_samples is not None:
            self.indices = self.indices[:max_samples]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, mask = self.dataset[self.indices[idx]]
        image = image.convert("RGB")

        # Resize to an explicit square: passing a single int resizes the
        # shorter side and keeps the aspect ratio, which breaks the UNet skip
        # connections (they need matching H/W and sizes divisible by the
        # number of downsampling stages).
        size = (self.image_size, self.image_size)
        image = TF.resize(image, size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size, interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=list(MEAN), std=list(STD))

        # Remap trimap values: 3 (boundary) becomes 0 so CrossEntropyLoss
        # (ignore_index=0) skips it.
        mask = torch.from_numpy(_LUT[np.array(mask, dtype=np.uint8)].copy())
        return image, mask


def visualize_sample(image, mask, num_classes=3):
    """Convert tensors back to display format for debugging."""
    img = image.cpu().permute(1, 2, 0).numpy()
    img = img * list(STD) + list(MEAN)
    img = np.clip(img, 0, 1)
    mask_img = (mask.cpu().numpy() / num_classes * 255).astype(np.uint8)
    return img, mask_img
