import torch

from gcn.model import GCN
from gcn.data import load_cora
from utils.config import load_config


def evaluate():
    cfg = load_config("gcn/config.yaml")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    features, adj_norm, labels, _, _, test_mask, class_mapping = load_cora()
    features, adj_norm, labels = features.to(device), adj_norm.to(device), labels.to(device)
    test_mask = test_mask.to(device)

    model = GCN(
        in_features=features.size(1),
        hidden_dim=cfg["hidden_dim"],
        num_classes=labels.max().item() + 1,
        dropout=cfg["dropout"],
    )
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {cfg['model_path']}")

    with torch.no_grad():
        output = model(features, adj_norm)
        pred = output[test_mask].argmax(dim=1)
        correct = (pred == labels[test_mask]).sum().item()
        total = test_mask.sum().item()
        acc = correct / total

    print(f"Test Accuracy: {acc:.2%} ({correct}/{total})")

    id_to_name = {v: k for k, v in class_mapping.items()}
    print(f"\nPer-class accuracy:")
    pred_all = output.argmax(dim=1)
    for c, name in sorted(id_to_name.items()):
        mask = test_mask & (labels == c)
        if mask.sum() > 0:
            acc_c = (pred_all[mask] == labels[mask]).float().mean().item()
            print(f"  {name:<15} {acc_c:.2%}")


if __name__ == "__main__":
    evaluate()
