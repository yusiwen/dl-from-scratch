"""CIFAR-10 with SimCLR augmentations (two random views per image)."""

from torchvision import transforms
from torch.utils.data import Dataset
from datasets import load_dataset


class SimCLRTransform:
    """Random augmentation for SimCLR (crop + color jitter + flip + grayscale)."""

    def __init__(self, image_size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def load_cifar10_simclr(batch_size=256, num_workers=4):
    transform = SimCLRTransform()

    def transform_batch(batch):
        images = [img.convert("RGB") for img in batch["img"]]
        views1, views2 = zip(*[transform(img) for img in images])
        batch["view1"] = list(views1)
        batch["view2"] = list(views2)
        return batch

    ds = load_dataset("uoft-cs/cifar10", split="train")
    ds.set_transform(transform_batch)

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return loader
