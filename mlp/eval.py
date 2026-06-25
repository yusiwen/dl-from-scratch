import numpy as np
from mlp.data import load_mnist, MNIST_CLASSES
from mlp.model import MLP


def load_weights(model, path="mlp/mlp_mnist.npz"):
    """Load saved numpy weights into model layers."""
    data = np.load(path)
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'W'):
            layer.W = data[f"layer_{i}_W"]
            layer.b = data[f"layer_{i}_b"]


def evaluate():
    _, test_loader = load_mnist(batch_size=64)

    model = MLP([784, 256, 128, 10])
    try:
        load_weights(model)
        print("Loaded saved weights from mlp/mlp_mnist.npz")
    except FileNotFoundError:
        print("No saved weights found. Training from scratch first...")
        from mlp.train import train as do_train
        do_train()
        load_weights(model)

    all_preds = []
    all_labels = []
    for images, labels, _ in test_loader():
        logits = model.forward(images)
        predictions = np.argmax(logits, axis=1)
        all_preds.append(predictions)
        all_labels.append(labels)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    correct = (all_preds == all_labels).sum()
    total = len(all_labels)
    overall_acc = correct / total * 100
    print(f"\nTest Accuracy: {overall_acc:.2f}%\n")

    print(f"{'Digit':<8} {'Accuracy':>8}  {'Correct':>6}/{6:<6}")
    print("-" * 35)
    for i in MNIST_CLASSES:
        mask = all_labels == i
        cls_total = mask.sum()
        cls_correct = (all_preds[mask] == all_labels[mask]).sum()
        cls_acc = cls_correct / cls_total * 100 if cls_total > 0 else 0
        print(f"{i:<8} {cls_acc:>7.1f}%  {cls_correct:>5}/{cls_total:<5}")
    print("-" * 35)
    print(f"{'All':<8} {overall_acc:>7.1f}%  {correct:>5}/{total:<5}")

    # Confusion matrix.
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    print(f"\n{'Confusion Matrix (rows=true, cols=pred)':^60}")
    print(f"{'':>7}", end="")
    for c in range(10):
        print(f"{c:>5}", end="")
    print()
    for i in range(10):
        print(f"{i:>7}", end="")
        for j in range(10):
            print(f"{cm[i, j]:>5}", end="")
        print()

    print("\nSample predictions (first 10 test images):")
    for images, labels, _ in test_loader():
        logits = model.forward(images)
        preds = np.argmax(logits, axis=1)
        for i in range(min(10, len(images))):
            marker = "✓" if preds[i] == labels[i] else "✗"
            print(f"  {marker} true={labels[i]}, pred={preds[i]}")
        break


if __name__ == "__main__":
    evaluate()
