import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CIFAR-10 per-channel mean and std (pre-computed over the training set).
# Why normalize? Neural networks train more reliably when input features
# have zero mean and unit variance — it keeps gradients in a well-behaved range
# and avoids saturating activation functions.
# These specific values are computed from the entire CIFAR-10 training set.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Class names in CIFAR-10, in order of the label index (0-9).
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_cifar10(batch_size=128, num_workers=2):
    """
    Load CIFAR-10 with optional data augmentation.

    Data augmentation (train only):
      - RandomHorizontalFlip: flips the image horizontally with 50% probability.
        Why? Real-world objects don't have a preferred left/right orientation;
        this teaches invariance to mirroring.
      - RandomCrop with 4px padding: crops a random 32x32 patch from a 40x40 padded image.
        Why? Introduces small translation shifts, making the model robust to
        slight positional changes of the object.

    Test transforms:
      - Only ToTensor + Normalize. No augmentation — we want deterministic evaluation.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # torchvision.datasets.CIFAR10 downloads the dataset automatically on first use.
    # It stores files in the specified root directory (~150MB for the full dataset).
    train_dataset = torchvision.datasets.CIFAR10(
        root="data/cifar10", train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="data/cifar10", train=False, download=True, transform=test_transform
    )

    # Why num_workers=2? Loading images and applying transforms is CPU-bound;
    # using 2 background workers overlaps I/O with GPU computation.
    # For MPS on Mac, pin_memory is not supported (silently ignored), so we omit it.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader
