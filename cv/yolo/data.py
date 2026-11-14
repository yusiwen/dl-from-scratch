"""Pascal VOC 2007 detection dataset for YOLO.

The Hugging Face id this module used to point at (``widerface/pascal_voc``) no
longer exists, so we read the official VOC2007 archive through torchvision.
The ``val`` image list ships inside the same ``trainval`` archive, so both
splits come from a single ~460 MB download.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import VOCDetection


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

CLASS_TO_IDX = {name: i for i, name in enumerate(VOC_CLASSES)}
DATA_ROOT = "data/VOC"


def _build_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def _as_list(value):
    """VOC XML parsing returns a dict for single-object images, a list otherwise."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _scaled_boxes(annotation, image_size=224):
    """Return (boxes (N, 4), labels) in the coordinate frame of the resized image.

    VOC boxes are in pixels of the *original* image, so they are scaled to the
    resized square first. Without that step every box lands outside the grid
    and the target is all zeros.
    """
    size = annotation.get("size", {}) or {}
    width = float(size.get("width") or image_size)
    height = float(size.get("height") or image_size)

    boxes, labels = [], []
    for obj in _as_list(annotation.get("object")):
        class_idx = CLASS_TO_IDX.get(obj.get("name"))
        if class_idx is None:
            continue

        box = obj.get("bndbox", {}) or {}
        try:
            xmin = float(box["xmin"]) / width * image_size
            ymin = float(box["ymin"]) / height * image_size
            xmax = float(box["xmax"]) / width * image_size
            ymax = float(box["ymax"]) / height * image_size
        except (KeyError, TypeError, ValueError):
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_idx)

    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), labels


def annotation_to_boxes(annotation, image_size=224):
    """Public wrapper used by the mAP evaluation code."""
    return _scaled_boxes(annotation, image_size)


def _voc_to_yolo(annotation, S, B, C, image_size=224):
    """Convert a VOC annotation dict to a YOLO target tensor (S, S, B*5+C)."""
    boxes, labels = _scaled_boxes(annotation, image_size)
    return _boxes_to_yolo(boxes, labels, S, B, C, image_size)


def _boxes_to_yolo(boxes, labels, S, B, C, image_size=224):
    """Rasterise boxes in the resized image frame onto the S x S target grid."""
    target = torch.zeros(S, S, B * 5 + C)

    for (xmin, ymin, xmax, ymax), class_idx in zip(boxes.tolist(), labels):
        if class_idx >= C:
            continue

        # Center and size normalised to the resized square.
        x_center = (xmin + xmax) / 2 / image_size
        y_center = (ymin + ymax) / 2 / image_size
        w_norm = (xmax - xmin) / image_size
        h_norm = (ymax - ymin) / image_size

        # Grid cell and the offset of the center inside that cell.
        col = min(int(x_center * S), S - 1)
        row = min(int(y_center * S), S - 1)
        x_cell = x_center * S - col
        y_cell = y_center * S - row

        # Assign to the first free box slot in this cell.
        for b in range(B):
            box_start = b * 5
            if target[row, col, box_start + 4] == 0:
                target[row, col, box_start:box_start + 5] = torch.tensor(
                    [x_cell, y_cell, w_norm, h_norm, 1.0]
                )
                target[row, col, B * 5 + class_idx] = 1.0
                break

    return target


# Light photometric jitter: brightness/contrast/saturation only, so the boxes
# stay valid. Applied per sample by the training dataset.
_COLOR_JITTER = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)


def _augment(image, boxes, labels, image_size=224, p_flip=0.5):
    """Random horizontal flip (boxes mirrored) plus colour jitter.

    VOC objects are roughly left-right symmetric, so a flip is a free extra
    sample -- but the boxes have to be mirrored with the image, otherwise the
    target stops pointing at the object. Colour jitter leaves geometry alone.
    Without this the 11M-parameter detector overfits 3,000 images: mAP@0.5
    peaks at 0.0339 around epoch 25 and is down to 0.0127 by epoch 50.
    """
    if boxes.numel() and torch.rand(1).item() < p_flip:
        image = torch.flip(image, dims=[2])
        x1 = boxes[:, 0].clone()
        boxes[:, 0] = image_size - boxes[:, 2]
        boxes[:, 2] = image_size - x1
    return _COLOR_JITTER(image), boxes, labels


class VOCDataset(Dataset):
    def __init__(self, split="trainval", image_size=224, S=7, B=2, C=20,
                 max_samples=None, offset=0, augment=False):
        self.S, self.B, self.C = S, B, C
        self.image_size = image_size
        self.augment = augment
        self.transform = _build_transform(image_size)

        self.dataset = VOCDetection(
            root=DATA_ROOT,
            year="2007",
            image_set=split,
            download=True,
        )

        # ``offset`` lets load_voc carve disjoint train/test slices out of one
        # split, which is how we avoid the trainval/val overlap.
        indices = list(range(len(self.dataset)))[offset:]
        if max_samples is not None:
            indices = indices[:max_samples]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, target = self.dataset[self.indices[idx]]
        image = self.transform(image.convert("RGB"))
        annotation = target.get("annotation", {}) if isinstance(target, dict) else {}
        boxes, labels = _scaled_boxes(annotation, self.image_size)
        if self.augment:
            image, boxes, labels = _augment(image, boxes, labels, self.image_size)
        return image, _boxes_to_yolo(boxes, labels, self.S, self.B, self.C, self.image_size)

    def annotation(self, idx):
        """Raw VOC annotation dict, parsed straight from XML.

        The mAP evaluation needs the original boxes, which the YOLO target grid
        no longer carries. Reading the XML avoids decoding the image twice.
        ``parse_voc_xml`` wraps everything under the root tag, so the inner
        "annotation" dict is what callers want.
        """
        import xml.etree.ElementTree as ET

        path = self.dataset.annotations[self.indices[idx]]
        parsed = self.dataset.parse_voc_xml(ET.parse(path).getroot())
        return parsed.get("annotation", {})


def load_voc(batch_size=32, image_size=224, S=7, B=2, C=20, num_workers=4,
             max_train=None, max_test=None, val_fraction=0.2):
    """Build train/test loaders from VOC2007 trainval.

    The official VOC2007 ``val`` list is a *subset of trainval*, so training on
    trainval and evaluating on val leaks most of the evaluation images into
    training. We instead train on the head of trainval and evaluate on a
    disjoint tail, which keeps a clean held-out set without a second download.
    """
    total = len(VOCDataset(split="trainval", image_size=image_size, S=S, B=B, C=C))
    n_train = max_train if max_train is not None else int(total * (1 - val_fraction))
    n_test = max_test if max_test is not None else total - n_train

    train_dataset = VOCDataset(split="trainval", image_size=image_size, S=S, B=B, C=C,
                               max_samples=n_train, offset=0, augment=True)
    test_dataset = VOCDataset(split="trainval", image_size=image_size, S=S, B=B, C=C,
                              max_samples=n_test, offset=n_train, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
