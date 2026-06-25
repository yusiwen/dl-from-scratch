import torch
import torch.nn.functional as F
import numpy as np

from cnn.data import load_cifar10, CIFAR10_CLASSES
from cnn.model import SimpleCNN


def evaluate():
    """
    Evaluate the trained model on the CIFAR-10 test set.

    Produces:
      - Overall test accuracy
      - Per-class accuracy (useful for identifying weak classes)
      - Confusion matrix (visualizes which classes are commonly confused)
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = "cnn/simple_cnn_cifar10.pt"
    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    _, test_loader = load_cifar10(batch_size=128, num_workers=2)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["img"].to(device)
            labels = batch["label"]
            outputs = model(images)
            # softmax converts logits to probabilities (class-wise, sum to 1).
            # argmax picks the class with the highest probability.
            predicted = torch.argmax(F.softmax(outputs, dim=1), dim=1).cpu()
            all_preds.append(predicted)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # --- Overall Accuracy ---
    correct = (all_preds == all_labels).sum().item()
    total = all_labels.size(0)
    overall_acc = correct / total * 100
    print(f"\nTest Accuracy: {overall_acc:.2f}%\n")

    # --- Per-class Accuracy ---
    # Why per-class? Overall accuracy can hide a model that performs well on
    # majority classes but poorly on minority ones. Per-class breakdown reveals
    # which categories need improvement.
    print(f"{'Class':<15} {'Accuracy':>8}  {'Correct':>6}/{6:<6}")
    print("-" * 40)
    for i, cls in enumerate(CIFAR10_CLASSES):
        mask = all_labels == i
        cls_total = mask.sum().item()
        cls_correct = (all_preds[mask] == all_labels[mask]).sum().item()
        cls_acc = cls_correct / cls_total * 100 if cls_total > 0 else 0
        print(f"{cls:<15} {cls_acc:>7.1f}%  {cls_correct:>5}/{cls_total:<5}")
    print("-" * 40)
    print(f"{'Overall':<15} {overall_acc:>7.1f}%  {correct:>5}/{total:<5}")

    # --- Confusion Matrix ---
    # A 10x10 matrix where entry (i, j) = number of test samples with true
    # class i that were predicted as class j.
    # Why? Shows systematic confusions (e.g. cat often mistaken for dog).
    n_classes = len(CIFAR10_CLASSES)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_labels.numpy(), all_preds.numpy()):
        cm[t, p] += 1

    print(f"\n{'Confusion Matrix (rows=true, cols=pred)':^75}")
    print(f"{'':>9}", end="")
    for cls in CIFAR10_CLASSES:
        print(f"{cls:>7}", end="")
    print()
    for i, cls in enumerate(CIFAR10_CLASSES):
        print(f"{cls:>9}", end="")
        for j in range(n_classes):
            print(f"{cm[i, j]:>7}", end="")
        print()


if __name__ == "__main__":
    evaluate()
