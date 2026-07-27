"""COCO-style mAP evaluation (single-class), no pycocotools dependency."""
from __future__ import annotations
import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a [N,4], b [M,4] xyxy -> [N,M] IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def average_precision(scores: np.ndarray, matches: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP for one IoU threshold."""
    if n_gt == 0:
        return float('nan')
    order = np.argsort(-scores)
    tp = matches[order].astype(np.float64)
    fp = 1 - tp
    ctp = np.cumsum(tp)
    cfp = np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    xs = np.linspace(0, 1, 101)
    ap = 0.0
    for x in xs:
        m = rec >= x
        ap += (prec[m].max() if m.any() else 0.0)
    return ap / 101.0


def compute_map(pred_boxes, gt_boxes, iou_thrs=np.arange(0.5, 1.0, 0.05),
                min_size=0.0, max_size=1e9):
    """pred_boxes/gt_boxes: lists per image of [N,5] (xyxy+score) / [M,4] (xyxy).

    Returns dict with mAP, mAP50, per-threshold APs. Optional size filter on GT
    (for small/large slices): GT outside [min_size,max_size) pixel-area treated
    as ignore (neither TP nor FP target).
    """
    per_thr = {t: ([], [], 0) for t in iou_thrs}
    for pb, gb in zip(pred_boxes, gt_boxes):
        pb = np.asarray(pb).reshape(-1, 5)
        gb = np.asarray(gb).reshape(-1, 4)
        if len(gb):
            areas = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
            keep = (areas >= min_size) & (areas < max_size)
            gb_eval = gb[keep]
        else:
            gb_eval = gb
        ious = iou_matrix(pb[:, :4] if len(pb) else np.zeros((0, 4)), gb_eval)
        for t in iou_thrs:
            scores_l, match_l, ng = per_thr[t]
            matched_gt = np.zeros(len(gb_eval), bool)
            m = np.zeros(len(pb))
            if len(pb) and len(gb_eval):
                for di in np.argsort(-pb[:, 4]):
                    if ious.shape[1] == 0:
                        break
                    j = int(np.argmax(ious[di]))
                    if ious[di, j] >= t and not matched_gt[j]:
                        matched_gt[j] = True
                        m[di] = 1
            per_thr[t] = (scores_l + [pb[:, 4] if len(pb) else np.empty(0)],
                          match_l + [m], ng + len(gb_eval))
    aps = {}
    for t in iou_thrs:
        s, m, ng = per_thr[t]
        s = np.concatenate(s) if s else np.empty(0)
        m = np.concatenate(m) if m else np.empty(0)
        aps[round(float(t), 2)] = average_precision(s, m, ng)
    valid = [v for v in aps.values() if not np.isnan(v)]
    return dict(mAP=float(np.mean(valid)) if valid else 0.0,
                mAP50=aps.get(0.5, 0.0), per_thr=aps)
