from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# CIFAR-10 per-channel mean and std (pre-computed over the training set).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Class names in CIFAR-10, in order of the label index (0-9).
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _build_transform(augment=False):
    """
    Build a torchvision transform pipeline.

    When augment=True, include RandomCrop + RandomHorizontalFlip for training.
    Both pipelines normalize with CIFAR-10 statistics.
    """
    ops = []
    if augment:
        ops.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return transforms.Compose(ops)


def _apply_transform(batch, transform_fn):
    """
    HuggingFace dataset transform: applies `transform_fn` to each image.

    Called by set_transform() — receives a dict-of-lists, returns a dict-of-lists.
    """
    batch["img"] = [transform_fn(img.convert("RGB")) for img in batch["img"]]
    return batch


# Pre-built transform pipelines (module-level for multiprocessing compatibility).
_TRAIN_TRANSFORM = _build_transform(augment=True)
_TEST_TRANSFORM = _build_transform(augment=False)


def _train_transform_batch(batch):
    """Train transform: augmentation + normalize. Pickle-safe (module-level function)."""
    batch["img"] = [_TRAIN_TRANSFORM(img.convert("RGB")) for img in batch["img"]]
    return batch


def _test_transform_batch(batch):
    """Test transform: normalize only. Pickle-safe (module-level function)."""
    batch["img"] = [_TEST_TRANSFORM(img.convert("RGB")) for img in batch["img"]]
    return batch


def load_cifar10(batch_size=128, num_workers=2):
    """
    Load CIFAR-10 via HuggingFace datasets library.

    The dataset is cached as Arrow files in ~/.cache/huggingface/datasets/
    (not individual images on disk). Transforms are applied lazily via
    set_transform, so no pre-processing happens at load time.
    """
    train_dataset = load_dataset("uoft-cs/cifar10", split="train")
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")

    train_dataset.set_transform(_train_transform_batch)
    test_dataset.set_transform(_test_transform_batch)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader
