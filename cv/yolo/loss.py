"""YOLO loss and NMS utilities."""

import torch
import torch.nn.functional as F


def yolo_loss(pred, target, S=7, B=2, C=20, coord_scale=5, noobj_scale=0.5):
    """YOLOv1 loss over a whole (N, S, S, B*5+C) head output.

    Three terms, each normalised the way the paper does it:

    * **coordinates** — squared error over every slot that owns a target box,
      averaged over objects and weighted by ``coord_scale``.
    * **objectness** — binary cross entropy on *every* slot's confidence logit:
      1 where the slot owns a box, 0 elsewhere. The background half is scaled by
      ``noobj_scale`` and averaged over the empty slots.
    * **class** — cross entropy over the cells that contain an object, so the
      class logits are trained as logits (the evaluation already applies a
      softmax to them).

    Slot ``b`` is responsible for its cell's object when the target marks its
    confidence channel, which is why both slots are supervised. The previous
    version read only slot 0: slot 1's boxes and confidence never received a
    gradient, and the class term regressed one-hot labels with MSE.
    """
    pred = pred.view(-1, S, S, B * 5 + C)
    target = target.view(-1, S, S, B * 5 + C)

    conf_channels = [b * 5 + 4 for b in range(B)]
    tconf = torch.stack([target[..., i] for i in conf_channels], dim=-1)   # (N, S, S, B)
    pconf = torch.stack([pred[..., i] for i in conf_channels], dim=-1)     # logits

    has_obj = tconf.sum(dim=-1) > 0                                       # (N, S, S)

    # ── Box coordinates, for each slot that carries a target box ──
    # w and h are compared in square-root space, as in the paper: the same box
    # error in a small box then costs more than in a large one, which keeps the
    # loss from being dominated by big objects.
    coord_loss = pred.new_zeros(())
    for b in range(B):
        owns = tconf[..., b] > 0
        if owns.any():
            xy_pred, wh_pred = pred[..., b * 5:b * 5 + 2][owns], pred[..., b * 5 + 2:b * 5 + 4][owns]
            xy_true, wh_true = target[..., b * 5:b * 5 + 2][owns], target[..., b * 5 + 2:b * 5 + 4][owns]
            coord_loss = coord_loss + F.mse_loss(xy_pred, xy_true, reduction="sum")
            coord_loss = coord_loss + F.mse_loss(
                wh_pred.clamp(min=1e-6).sqrt(), wh_true.clamp(min=1e-6).sqrt(), reduction="sum"
            )

    # ── Objectness, positive and negative halves normalised separately ──
    positive = tconf > 0
    obj_loss = F.binary_cross_entropy_with_logits(pconf[positive],
                                                  tconf[positive], reduction="sum")
    noobj_loss = F.binary_cross_entropy_with_logits(pconf[~positive],
                                                    tconf[~positive], reduction="sum")

    # ── Class, cross entropy on the cells that hold an object ──
    class_loss = pred.new_zeros(())
    if has_obj.any():
        class_loss = F.cross_entropy(
            pred[..., B * 5:][has_obj],
            target[..., B * 5:][has_obj].argmax(dim=-1),
            reduction="sum",
        )

    n_obj = positive.sum().clamp(min=1)
    n_bg = (~positive).sum().clamp(min=1)
    n_cells = has_obj.sum().clamp(min=1)

    return (
        coord_scale * coord_loss / n_obj
        + obj_loss / n_obj
        + noobj_scale * noobj_loss / n_bg
        + class_loss / n_cells
    )


def nms(predictions, conf_threshold=0.3, iou_threshold=0.5):
    """Non-Maximum Suppression — simplified for single image."""
    boxes, scores, labels = [], [], []
    S, B, C = 7, 2, 20

    for row in range(S):
        for col in range(S):
            for b in range(B):
                box = predictions[row, col, b * 5:(b + 1) * 5]
                conf = box[4].sigmoid().item()
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
