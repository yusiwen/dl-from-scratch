import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

from cv.resnet34.data import CelebADataset, CELEBA_ATTR_ORDER, eval_transform
from utils.config import load_config
from utils.device import get_device


def evaluate():
    cfg = load_config("cv/resnet34/config.yaml")

    device = get_device()
    print(f"Device: {device}")

    model_path = cfg["model_path"]
    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    transform = eval_transform()
    test_dataset = CelebADataset(split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"])

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = torch.sigmoid(model(images)).cpu()
            all_preds.append(outputs)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    print()
    print(f"Test set: {len(all_labels)} samples, 40 attributes")
    print()
    print(f"  {'Attribute':<22} {'ROC AUC':>8} {'F1':>8} {'Acc':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")

    aucs, f1s, accs = [], [], []
    for i, attr in enumerate(CELEBA_ATTR_ORDER):
        y_true = all_labels[:, i]
        y_pred_bin = (all_preds[:, i] > 0.5).astype(float)
        # ROC AUC requires at least one positive and one negative sample.
        if y_true.sum() > 0 and (1 - y_true).sum() > 0:
            auc = roc_auc_score(y_true, all_preds[:, i])
        else:
            auc = float("nan")
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        acc = (y_pred_bin == y_true).mean()
        aucs.append(auc)
        f1s.append(f1)
        accs.append(acc)
        print(f"  {attr:<22} {auc:>7.3f}  {f1:>7.3f}  {acc:>7.1%}")

    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Average':<22} {np.nanmean(aucs):>7.3f}  {np.mean(f1s):>7.3f}  {np.mean(accs):>7.1%}")

    return aucs, f1s, accs


if __name__ == "__main__":
    import numpy as np
    evaluate()
