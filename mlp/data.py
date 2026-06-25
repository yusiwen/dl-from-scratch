from datasets import load_dataset
import numpy as np


MNIST_CLASSES = list(range(10))


def _to_numpy(batch):
    """
    Convert an HF batch (PIL images + int labels) to numpy arrays.

    Images are flattened to 784-dim vectors and normalized to [0, 1].
    Labels are one-hot encoded for use with SoftmaxCrossEntropy.
    """
    imgs = np.stack([np.array(img, dtype=np.float32).reshape(-1) / 255.0 for img in batch["image"]])
    labels = np.array(batch["label"], dtype=np.int32)
    one_hot = np.zeros((len(labels), 10), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return imgs, labels, one_hot


def load_mnist(batch_size=64):
    """
    Load MNIST via HuggingFace datasets.

    Returns train_loader and test_loader, each yielding
    (images, labels, one_hot_labels) as numpy arrays.

    The dataset is cached as Arrow files in ~/.cache/huggingface/datasets/.
    """
    ds = load_dataset("ylecun/mnist", split="train")
    ds_test = load_dataset("ylecun/mnist", split="test")

    def batch_generator(dataset, bs, shuffle):
        """Manual mini-batch generator from HF dataset."""
        indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(dataset), bs):
            batch_indices = indices[start:start + bs]
            batch = dataset[batch_indices.tolist()]
            yield _to_numpy(batch)

    def train_loader():
        return batch_generator(ds, batch_size, shuffle=True)

    def test_loader():
        return batch_generator(ds_test, batch_size, shuffle=False)

    return train_loader, test_loader
