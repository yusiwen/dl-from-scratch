import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from cv.resnet18.data import CelebADataset, ATTRIBUTES
from utils.config import load_config
from utils.device import get_device


def evaluate():
    cfg = load_config("cv/resnet18/config.yaml")

    device = get_device()
    print(f"Device: {device}")

    model_path = cfg["model_path"]
    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = CelebADataset(ATTRIBUTES, num_samples=cfg["num_samples"], transform=transform)

    _, val_dataset = torch.utils.data.random_split(
        full_dataset, [cfg["train_split"], cfg["val_split"]],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    # --- Inference ---
    # Collect all predictions and labels for metric computation.
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            # Sigmoid converts logits to probabilities (range 0-1).
            # Threshold at 0.5: probability > 0.5 means the attribute is present.
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # --- Overall accuracy ---
    correct = (all_preds == all_labels).sum().item()
    total = all_labels.numel()
    overall_acc = correct / total * 100

    print()
    print("Per-attribute accuracy:")
    print(f"  {'Attribute':<20} {'Accuracy':>8}")
    print(f"  {'-'*20} {'-'*8}")
    for i, attr in enumerate(ATTRIBUTES):
        attr_correct = (all_preds[:, i] == all_labels[:, i]).sum().item()
        attr_total = all_labels.size(0)
        attr_acc = attr_correct / attr_total * 100
        print(f"  {attr:<20} {attr_acc:>7.1f}%")
    print(f"  {'-'*20} {'-'*8}")
    print(f"  {'Overall':<20} {overall_acc:>7.1f}%")

    # --- Sample predictions ---
    # Show 5 examples with their true vs. predicted attribute sets.
    # This helps qualitatively assess model behavior: is it making
    # reasonable mistakes (e.g. predicting Young when not) or nonsense ones?
    print()
    print("Sample predictions (filename, true vs predicted):")
    val_indices = val_dataset.indices
    for offset in range(5):
        idx = val_indices[offset]
        filename = full_dataset.samples[idx][0]
        true_lbl = all_labels[offset]
        pred_lbl = all_preds[offset]
        true_attrs = [ATTRIBUTES[i] for i, v in enumerate(true_lbl) if v == 1]
        pred_attrs = [ATTRIBUTES[i] for i, v in enumerate(pred_lbl) if v == 1]
        print(f"  {filename}")
        print(f"    True:  {true_attrs}")
        print(f"    Pred:  {pred_attrs}")


if __name__ == "__main__":
    evaluate()
