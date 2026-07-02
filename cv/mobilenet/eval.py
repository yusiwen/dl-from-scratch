import torch

from cv.mobilenet.model import MobileNet
from cv.mobilenet.data import load_cifar10, CIFAR10_CLASSES
from utils.config import load_config


def evaluate():
    cfg = load_config("cv/mobilenet/config.yaml")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MobileNet(num_classes=cfg["num_classes"], width_multiplier=cfg["width_multiplier"])
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {cfg['model_path']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    _, test_loader = load_cifar10(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    correct = total = 0
    class_correct = [0] * cfg["num_classes"]
    class_total = [0] * cfg["num_classes"]

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if pred[i].item() == label:
                    class_correct[label] += 1

    print(f"\nTest Accuracy: {correct / total:.1%} ({correct}/{total})")
    print(f"\nPer-class accuracy:")
    for i, name in enumerate(CIFAR10_CLASSES):
        acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"  {name:<12} {acc:.1%}")

    # Compare with SimpleCNN
    from cv.simplecnn.model import SimpleCNN
    cnn = SimpleCNN(num_classes=10)
    cnn_params = sum(p.numel() for p in cv.simplecnn.parameters())
    print(f"\n--- Comparison ---")
    print(f"MobileNet:  {sum(p.numel() for p in model.parameters()):,} params")
    print(f"SimpleCNN:  {cnn_params:,} params")
    ratio = cnn_params / sum(p.numel() for p in model.parameters())
    print(f"MobileNet is {ratio:.1f}× smaller than SimpleCNN")


if __name__ == "__main__":
    evaluate()
