from torchvision import transforms
from datasets import load_dataset


def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def transform_batch(batch):
        batch["img"] = [transform(img.convert("RGB")) for img in batch["img"]]
        return batch

    ds = load_dataset("uoft-cs/cifar10", split="train")
    ds.set_transform(transform_batch)
    return ds
