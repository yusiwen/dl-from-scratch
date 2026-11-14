"""IoU matching and mAP evaluation for the simplified YOLO detector.

The training loss alone does not say whether the detections are any good: most
of the S x S grid is empty, so the loss can shrink while the model predicts
nothing useful. IoU measures how well a predicted box overlaps a ground-truth
box, and mAP turns that into a single ranking-quality number per class.
"""

import torch

from cv.yolo.data import VOC_CLASSES, annotation_to_boxes
from cv.yolo.loss import _compute_iou


def decode_predictions(pred, S=7, B=2, C=20, image_size=224,
                       conf_threshold=0.05, max_detections=20):
    """Turn one (S, S, B*5+C) head output into boxes, scores and labels.

    Boxes come back in pixel coordinates of the ``image_size`` square, matching
    the coordinate frame that ``annotation_to_boxes`` returns.

    The head emits raw logits: the class block is trained with cross entropy
    (softmax here), and each box's confidence is trained as a logit with binary
    cross entropy, so it is squashed with a sigmoid before it becomes a score.
    A detection's score is confidence times its best class probability, which is
    what the precision/recall curve needs.
    """
    device = pred.device
    rows, cols = torch.meshgrid(
        torch.arange(S, device=device), torch.arange(S, device=device), indexing="ij"
    )

    class_prob = torch.softmax(pred[..., B * 5:], dim=-1)   # (S, S, C)
    class_score, class_idx = class_prob.max(dim=-1)         # (S, S)

    boxes_per_slot, scores_per_slot = [], []
    for b in range(B):
        raw = pred[..., b * 5:(b + 1) * 5]                  # (S, S, 5)
        conf = raw[..., 4].sigmoid()
        # Clamp before decoding: an untrained or early-epoch head can emit a
        # centre outside its cell or a negative size, which would turn into an
        # inverted box (x2 < x1) and a meaningless IoU.
        cx = (cols + raw[..., 0].clamp(0.0, 1.0)) / S * image_size
        cy = (rows + raw[..., 1].clamp(0.0, 1.0)) / S * image_size
        w = raw[..., 2].clamp(min=0.0) * image_size
        h = raw[..., 3].clamp(min=0.0) * image_size
        boxes_per_slot.append(
            torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        )
        scores_per_slot.append(conf * class_score)

    boxes = torch.stack(boxes_per_slot, dim=-2).reshape(-1, 4)   # (S*S*B, 4)
    scores = torch.stack(scores_per_slot, dim=-1).reshape(-1)    # (S*S*B,)
    labels = class_idx.unsqueeze(-1).expand(-1, -1, B).reshape(-1)

    keep = scores >= conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if scores.numel() > max_detections:
        scores, order = scores.sort(descending=True)
        boxes, labels = boxes[order], labels[order]
        boxes, scores, labels = boxes[:max_detections], scores[:max_detections], labels[:max_detections]

    return boxes, scores, labels


def batched_nms(boxes, scores, labels, iou_threshold=0.5):
    """Class-aware greedy NMS; returns the indices of the boxes to keep."""
    keep = []
    for cls in labels.unique().tolist():
        idx = torch.nonzero(labels == cls, as_tuple=False).flatten()
        idx = idx[torch.argsort(scores[idx], descending=True)]
        while idx.numel() > 0:
            keep.append(int(idx[0]))
            if idx.numel() == 1:
                break
            ious = _compute_iou(boxes[idx[0:1]], boxes[idx[1:]])
            idx = idx[1:][ious < iou_threshold]
    return keep


def average_precision(detections, ground_truths, num_classes, iou_threshold=0.5):
    """VOC-style all-point AP per class.

    ``detections[i]`` is a list of (class, score, box) and ``ground_truths[i]``
    a list of (class, box) for the same image. A detection counts as a true
    positive when it matches an unused ground-truth box of the same class with
    IoU >= ``iou_threshold``.
    """
    results = {}
    for cls in range(num_classes):
        n_pos = sum(1 for per_img in ground_truths for gt_cls, _ in per_img if gt_cls == cls)
        if n_pos == 0:
            continue

        entries = []
        for img_dets, img_gts in zip(detections, ground_truths):
            cls_dets = sorted((d for d in img_dets if d[0] == cls), key=lambda d: -d[1])
            matched = [False] * len(img_gts)

            for _, score, box in cls_dets:
                best_iou, best_j = 0.0, -1
                for j, (gt_cls, gt_box) in enumerate(img_gts):
                    if gt_cls != cls or matched[j]:
                        continue
                    iou = float(_compute_iou(box[None], gt_box[None])[0])
                    if iou > best_iou:
                        best_iou, best_j = iou, j

                if best_j >= 0 and best_iou >= iou_threshold:
                    matched[best_j] = True
                    entries.append((score, 1))
                else:
                    entries.append((score, 0))

        if not entries:
            continue

        entries.sort(key=lambda e: -e[0])

        tp = fp = 0
        recalls, precisions = [], []
        for _, is_tp in entries:
            tp += is_tp
            fp += 1 - is_tp
            precisions.append(tp / (tp + fp))
            recalls.append(tp / n_pos)

        # All-point interpolation: sweep right-to-left so each recall step uses
        # the best precision achievable at that recall or higher.
        ap = 0.0
        best = 0.0
        for i in range(len(recalls) - 1, -1, -1):
            best = max(best, precisions[i])
            prev_recall = recalls[i - 1] if i > 0 else 0.0
            ap += best * (recalls[i] - prev_recall)

        results[cls] = ap

    return results


def evaluate_map(model, dataset, device, image_size=224, batch_size=32,
                 conf_threshold=0.05, nms_iou=0.5,
                 iou_thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
                 max_images=None):
    """Run the model over ``dataset`` and report mAP@0.5 and mAP@0.5:0.95.

    ``dataset`` is a ``VOCDataset``: its ``annotation`` method supplies the raw
    ground-truth boxes that the YOLO target grid no longer carries.
    """
    model.eval()
    S, B, C = model.S, model.B, model.C
    n_images = len(dataset) if max_images is None else min(max_images, len(dataset))

    detections, ground_truths = [], []
    with torch.no_grad():
        for start in range(0, n_images, batch_size):
            idxs = list(range(start, min(start + batch_size, n_images)))
            images = torch.stack([dataset[i][0] for i in idxs]).to(device)
            preds = model(images)

            for k, i in enumerate(idxs):
                boxes, scores, labels = decode_predictions(
                    preds[k], S, B, C, image_size, conf_threshold
                )
                keep = batched_nms(boxes, scores, labels, nms_iou)
                detections.append(
                    [(int(labels[j]), float(scores[j]), boxes[j].cpu()) for j in keep]
                )

                gt_boxes, gt_labels = annotation_to_boxes(dataset.annotation(i), image_size)
                ground_truths.append(list(zip(gt_labels, gt_boxes)))

    aps_per_threshold = [
        average_precision(detections, ground_truths, C, threshold)
        for threshold in iou_thresholds
    ]

    def _mean(values):
        return sum(values.values()) / len(values) if values else 0.0

    ap50 = aps_per_threshold[0]
    return {
        "mAP@0.5": _mean(ap50),
        "mAP@0.5:0.95": sum(_mean(aps) for aps in aps_per_threshold) / len(iou_thresholds),
        "AP@0.5": {VOC_CLASSES[cls]: ap for cls, ap in sorted(ap50.items(), key=lambda x: -x[1])},
        "num_images": n_images,
        "num_ground_truths": sum(len(g) for g in ground_truths),
    }
