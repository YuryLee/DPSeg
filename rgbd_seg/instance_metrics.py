"""Class-agnostic visible-instance metrics; see docs/metric_protocol.md.

Independent implementation of the UOIS/UCN overlap/boundary definitions. Values
returned in metric dictionaries are percentages. Label 0 alone is background.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment

METRIC_NAMES = ('overlap_precision', 'overlap_recall', 'overlap_f1',
                'boundary_precision', 'boundary_recall', 'boundary_f1',
                'object_recall_f_gt075', 'object_recall_iou_ge075')


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _label_map(value):
    array = _numpy(value)
    if array.ndim != 2 or array.dtype.kind not in 'iub' or np.any(array < 0) or 0 in array.shape:
        raise ValueError('Instance maps must be nonnegative integer nonempty HxW arrays')
    labels, inverse = np.unique(array, return_inverse=True)
    foreground = labels[labels != 0]
    # Reserve zero even when no background pixels occur.
    offset = int(labels[0] != 0)
    return (inverse.reshape(array.shape) + offset).astype(np.int32), foreground.tolist()


def _boundary(mask):
    """DAVIS-style boundary at pixel transitions; exterior image edge is ignored."""
    result = np.zeros(mask.shape, dtype=bool)
    horizontal = mask[:, :-1] != mask[:, 1:]
    vertical = mask[:-1, :] != mask[1:, :]
    result[:, :-1] |= horizontal
    result[:-1, :] |= vertical
    result[:-1, :-1] |= mask[:-1, :-1] != mask[1:, 1:]
    return result


def _near_boundary(boundary, radius):
    if not boundary.any():
        return np.zeros(boundary.shape, dtype=bool)
    # Euclidean distance <= radius equals dilation by a discrete radius-r disk.
    return distance_transform_edt(~boundary) <= radius


def _prf(tp_precision, denom_precision, tp_recall, denom_recall):
    p = tp_precision / denom_precision if denom_precision else 1.0
    r = tp_recall / denom_recall if denom_recall else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _from_counts(counts):
    ng, npred = counts['gt_objects'], counts['pred_objects']
    op, ore, of = _prf(counts['matched_intersection_pixels'], counts['pred_pixels'],
                       counts['matched_intersection_pixels'], counts['gt_pixels'])
    if not ng or not npred:
        # Match the reference convention for empty prediction/GT images.
        bp, br, bf = op, ore, of
    else:
        bp, br, bf = _prf(counts['matched_pred_boundary_pixels'], counts['pred_boundary_pixels'],
                          counts['matched_gt_boundary_pixels'], counts['gt_boundary_pixels'])
    empty_fraction = float(npred == 0)
    f75 = counts['matched_f_gt075'] / ng if ng else empty_fraction
    iou75 = counts['matched_iou_ge075'] / ng if ng else empty_fraction
    return dict(zip(METRIC_NAMES, (100 * x for x in (op, ore, of, bp, br, bf, f75, iou75))))


def instance_metrics(prediction, ground_truth, boundary_tolerance=0.003):
    """Evaluate two disjoint visible instance maps using one-to-one matching.

    Reference matching maximizes summed pairwise Dice/F1. F@.75 uses strictly
    F1>0.75. The separately named IoU diagnostic uses its own summed-IoU match.
    Boundary tolerance <1 is a fraction of image diagonal, otherwise pixels.
    """
    if not math.isfinite(boundary_tolerance) or boundary_tolerance <= 0:
        raise ValueError('boundary_tolerance must be finite and positive')
    pred, pred_ids = _label_map(prediction)
    gt, gt_ids = _label_map(ground_truth)
    if pred.shape != gt.shape:
        raise ValueError('Prediction and GT dimensions differ')
    ng, npred = len(gt_ids), len(pred_ids)
    radius = math.ceil(boundary_tolerance * math.hypot(*gt.shape)) if boundary_tolerance < 1 else math.ceil(boundary_tolerance)
    histogram = np.bincount((gt.astype(np.int64) * (npred + 1) + pred).ravel(), minlength=(ng + 1) * (npred + 1)).reshape(ng + 1, npred + 1)
    intersection = histogram[1:, 1:]
    ga = histogram[1:, :].sum(axis=1)
    pa = histogram[:, 1:].sum(axis=0)
    counts = {'gt_objects': ng, 'pred_objects': npred,
              'gt_pixels': int(ga.sum()), 'pred_pixels': int(pa.sum()),
              'matched_intersection_pixels': 0, 'gt_boundary_pixels': 0,
              'pred_boundary_pixels': 0, 'matched_pred_boundary_pixels': 0,
              'matched_gt_boundary_pixels': 0, 'matched_f_gt075': 0,
              'matched_iou_ge075': 0}
    gt_boundaries = [_boundary(gt == i + 1) for i in range(ng)]
    pred_boundaries = [_boundary(pred == j + 1) for j in range(npred)]
    counts['gt_boundary_pixels'] = sum(int(x.sum()) for x in gt_boundaries)
    counts['pred_boundary_pixels'] = sum(int(x.sum()) for x in pred_boundaries)
    matches = []
    if ng and npred:
        dice = 2 * intersection / (ga[:, None] + pa[None, :])
        iou = intersection / (ga[:, None] + pa[None, :] - intersection)
        rows, cols = linear_sum_assignment(-dice)
        counts['matched_intersection_pixels'] = int(intersection[rows, cols].sum())
        counts['matched_f_gt075'] = int((dice[rows, cols] > 0.75).sum())
        iou_rows, iou_cols = linear_sum_assignment(-iou)
        counts['matched_iou_ge075'] = int((iou[iou_rows, iou_cols] >= 0.75).sum())
        for row, col in zip(rows, cols):
            gb, pb = gt_boundaries[row], pred_boundaries[col]
            counts['matched_pred_boundary_pixels'] += int((pb & _near_boundary(gb, radius)).sum())
            counts['matched_gt_boundary_pixels'] += int((gb & _near_boundary(pb, radius)).sum())
            matches.append({'gt_label_id': int(gt_ids[row]), 'pred_label_id': int(pred_ids[col]),
                            'pair_f1': float(dice[row, col]), 'pair_iou': float(iou[row, col])})
    return {'metrics': _from_counts(counts), 'counts': counts, 'matches_f1': matches,
            'boundary_tolerance_pixels': radius}


def predictions_to_label_map(prediction, shape, score_threshold=0.5, mask_threshold=0.5):
    """Give overlapping pixels to highest-scoring retained mask (stable ties).

    This is an explicit adapter for overlapping Mask R-CNN predictions. Empty or
    fully hidden masks disappear from the resulting visible instance label map.
    """
    for value in (score_threshold, mask_threshold):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Thresholds must be finite in [0,1]')
    masks, scores = _numpy(prediction['masks']), _numpy(prediction['scores'])
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[1:] != tuple(shape) or scores.shape != (len(masks),):
        raise ValueError('Prediction masks/scores have inconsistent shapes')
    if not np.isfinite(masks).all() or not np.isfinite(scores).all() or np.any((masks < 0) | (masks > 1)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Masks/scores must be finite probabilities in [0,1]')
    result = np.zeros(shape, dtype=np.int32)
    kept = 0
    for index in np.argsort(-scores, kind='stable'):
        if scores[index] < score_threshold:
            continue
        kept += 1
        region = (masks[index] >= mask_threshold) & (result == 0)
        result[region] = int(index) + 1
    visible = int(len(np.unique(result[result > 0])))
    return result, {'raw_predictions': len(masks), 'score_retained_predictions': kept,
                    'visible_predictions': visible, 'empty_or_fully_occluded_predictions': kept - visible}


def evaluate_extra_predictions(dataset, predictions, score_threshold=0.5, mask_threshold=0.5, boundary_tolerance=0.003):
    """Macro means follow per-image averaging; micro counts are a separate diagnostic."""
    if len(dataset) != len(predictions):
        raise ValueError('Exactly one prediction is required per dataset image')
    rows = []
    for index, prediction in enumerate(predictions):
        if (hasattr(dataset, 'samples') and dataset.samples[index].get('instance_path') is None
                and dataset.samples[index].get('instance_mask_paths') is None):
            raise ValueError('Evaluation requires original instance annotations')
        sample, target = dataset[index]
        shape = tuple(sample['image'].shape[-2:])
        gt_masks = _numpy(target['masks'])
        if gt_masks.ndim != 3 or gt_masks.shape[1:] != shape:
            raise ValueError('GT masks must match original image dimensions')
        if np.any((gt_masks != 0) & (gt_masks != 1)) or np.any(gt_masks.sum(axis=0) > 1):
            raise ValueError('Visible GT masks must be binary and disjoint')
        gt = np.zeros(shape, dtype=np.int32)
        for mask_index, mask in enumerate(gt_masks):
            gt[mask.astype(bool)] = mask_index + 1
        pred, adapter_counts = predictions_to_label_map(prediction, shape, score_threshold, mask_threshold)
        row = instance_metrics(pred, gt, boundary_tolerance)
        row.update({'image_id': int(_numpy(target['image_id']).reshape(-1)[0]),
                    'sample_id': dataset.samples[index]['id'] if hasattr(dataset, 'samples') else str(index),
                    'prediction_adapter': adapter_counts})
        rows.append(row)
    totals = {key: sum(row['counts'][key] for row in rows) for key in rows[0]['counts']} if rows else {}
    return {'protocol': {'name': 'uois_visible_f1_matching_v1', 'class_agnostic': True,
                         'score_threshold': score_threshold, 'mask_threshold': mask_threshold,
                         'overlap_resolution': 'highest score wins; input order breaks tied scores',
                         'boundary_tolerance': boundary_tolerance, 'evaluation_resolution': 'original image',
                         'macro_aggregation': 'arithmetic mean of each per-image metric, including documented empty cases',
                         'micro_aggregation': 'ratios from summed matched pixel/boundary/object counts; diagnostic, not the published macro protocol',
                         'f075_definition': 'GT fraction whose sum-F1 Hungarian match has pairwise F1 strictly >0.75',
                         'iou075_definition': 'Separate diagnostic: GT fraction whose sum-IoU Hungarian match has IoU >=0.75',
                         'units': 'metric values 0–100 percent; pair_f1/pair_iou 0–1',
                         'not_directly_comparable_to_published_scores': 'Different dataset and original resolution; explicit score-based mask overlap adapter differs from author last-mask-overwrite adapter'},
            'metrics_macro': {key: float(np.mean([row['metrics'][key] for row in rows])) if rows else None for key in METRIC_NAMES},
            'metrics_micro': _from_counts(totals) if rows else {key: None for key in METRIC_NAMES},
            'totals': totals, 'per_image': rows}
