"""YOLO loss and NMS utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def yolo_loss(pred, target, S=7, B=2, C=20, coord_scale=5, noobj_scale=0.5):
    """YOLO loss: coord + obj + noobj + class."""
    pred = pred.view(-1, S, S, B * 5 + C)
    target = target.view(-1, S, S, B * 5 + C)

    obj_mask = target[..., 4] > 0  # (N, S, S) — first box has obj

    # ── Box coordinates loss (only for obj cells, best box) ──
    coord_loss = torch.tensor(0.0, device=pred.device)
    if obj_mask.any():
        pred_box = pred[obj_mask]
        target_box = target[obj_mask]
        # Take first box for simplicity (would IoU match in real YOLO).
        coord_loss = F.mse_loss(pred_box[:, :4], target_box[:, :4], reduction="sum")

    # ── Confidence loss ──
    obj_conf = pred[..., 4]
    target_conf = target[..., 4]
    obj_loss = F.mse_loss(obj_conf[obj_mask], target_conf[obj_mask], reduction="sum")
    noobj_loss = F.mse_loss(obj_conf[~obj_mask], target_conf[~obj_mask], reduction="sum")

    # ── Class loss ──
    class_loss = torch.tensor(0.0, device=pred.device)
    if obj_mask.any():
        class_loss = F.mse_loss(
            pred[obj_mask][:, B * 5:],
            target[obj_mask][:, B * 5:],
            reduction="sum",
        )

    N = obj_mask.numel()
    return (coord_scale * coord_loss + obj_loss + noobj_scale * noobj_loss + class_loss) / N


def nms(predictions, conf_threshold=0.3, iou_threshold=0.5):
    """Non-Maximum Suppression — simplified for single image."""
    boxes, scores, labels = [], [], []
    S, B, C = 7, 2, 20

    for row in range(S):
        for col in range(S):
            for b in range(B):
                box = predictions[row, col, b * 5:(b + 1) * 5]
                conf = box[4].item()
                if conf < conf_threshold:
                    continue
                x, y, w, h = box[:4].tolist()
                # Convert to absolute coordinates.
                x_abs = (col + x) / S * 224
                y_abs = (row + y) / S * 224
                w_abs = w * 224
                h_abs = h * 224
                x1 = x_abs - w_abs / 2
                y1 = y_abs - h_abs / 2
                x2 = x_abs + w_abs / 2
                y2 = y_abs + h_abs / 2

                class_probs = predictions[row, col, B * 5:].softmax(dim=0)
                class_score, class_idx = class_probs.max(dim=0)

                score = conf * class_score.item()
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                labels.append(class_idx.item())

    if not boxes:
        return [], [], []

    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    labels = torch.tensor(labels)

    # Sort by score.
    _, order = scores.sort(descending=True)
    boxes, scores, labels = boxes[order], scores[order], labels[order]

    keep = []
    while boxes.size(0) > 0:
        keep.append(0)
        if boxes.size(0) == 1:
            break
        ious = _compute_iou(boxes[0:1], boxes[1:])
        mask = ious < iou_threshold
        boxes, scores, labels = boxes[1:][mask], scores[1:][mask], labels[1:][mask]

    return boxes[keep], scores[keep], labels[keep]


def _compute_iou(box1, box2):
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1 + area2 - inter).clamp(min=1e-8)
