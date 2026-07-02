"""Pascal VOC dataset for YOLO."""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def _build_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def _voc_to_yolo(annotation, S, B, C, image_size=224):
    """Convert VOC annotation to YOLO target tensor (S, S, B*5+C)."""
    target = torch.zeros(S, S, B * 5 + C)

    for obj in annotation:
        try:
            label = obj["category"]
            bbox = obj["bbox"]  # [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = bbox
            w = xmax - xmin
            h = ymax - ymin

            # Skip invalid boxes.
            if w <= 0 or h <= 0:
                continue

            # Center and normalize.
            x_center = (xmin + xmax) / 2 / image_size
            y_center = (ymin + ymax) / 2 / image_size
            w_norm = w / image_size
            h_norm = h / image_size

            # Grid cell.
            col = int(x_center * S)
            row = int(y_center * S)
            if col >= S or row >= S:
                continue

            # Relative position within cell.
            x_cell = x_center * S - col
            y_cell = y_center * S - row

            class_idx = label
            if class_idx < 0 or class_idx >= C:
                continue

            # Assign to first available box.
            for b in range(B):
                box_start = b * 5
                if target[row, col, box_start + 4] == 0:  # confidence unused
                    target[row, col, box_start:box_start + 5] = torch.tensor([
                        x_cell, y_cell, w_norm, h_norm, 1.0,
                    ])
                    target[row, col, B * 5 + class_idx] = 1.0
                    break
        except (KeyError, TypeError, IndexError):
            continue

    return target


class VOCDataset(Dataset):
    def __init__(self, split="train", image_size=224, S=7, B=2, C=20):
        self.S, self.B, self.C = S, B, C
        self.image_size = image_size
        self.transform = _build_transform(image_size)

        ds = load_dataset("widerface/pascal_voc", split="train")
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        image = self.transform(image)

        try:
            objects = item["objects"]
        except (KeyError, TypeError):
            objects = []

        target = _voc_to_yolo(objects, self.S, self.B, self.C, self.image_size)
        return image, target


def load_voc(batch_size=32, image_size=224, S=7, B=2, C=20, num_workers=4):
    train_dataset = VOCDataset(split="train", image_size=image_size, S=S, B=B, C=C)
    test_dataset = VOCDataset(split="test", image_size=image_size, S=S, B=B, C=C)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
