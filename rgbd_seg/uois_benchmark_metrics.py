"""Original DA-Fusion/UOIS metrics without importing the model or detectron2.

Reference: teamROBI/DA-Fusion, commit c5bf9e0395c11ed1c2fbb2b88e7d0589a90df570,
eval/compute_PRF.py (NVIDIA copyright/license notice remains in that source).
Joint-label counting and matched-boundary evaluation follow this workspace's
dpseg_generalization/generalization/fast_uois.py. The pinned original Munkres,
boundary functions and arithmetic order preserve ties and boundary conventions.

All ratios are fractions, not percentages. The pinned evaluator uses F > .75;
F >= .75 is a separately named diagnostic. Inputs are original-resolution,
mutually exclusive instance ID maps with only documented background mapped to 0.
This module never selects annotation_fixed, removes small GT, or resizes GT.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


REFERENCE_COMMIT = 'c5bf9e0395c11ed1c2fbb2b88e7d0589a90df570'
_VENDOR = Path(__file__).resolve().parents[1] / 'third_party' / 'DA-Fusion'
_EVAL_REL = 'dafusion/dafusion/eval/'
_REFERENCE_SHA256 = {
    '__init__.py': '6745d1fcadb77ced64860312e852588b18c2353bb2e528e638094e676e2a0fa8',
    'compute_PRF.py': '7f6406eb127aff14d54846f2275e4dd7083e0eaaf40b30e927e42a4d55dc4479',
    'munkres.py': '1ad7629fdb82d4b4178b437a1070fe75b1b9053b6c7519cd6ce7a8f3ce28567d',
}
PAPER_METRIC_NAMES = (
    'Objects Precision', 'Objects Recall', 'Objects F-measure',
    'Boundary Precision', 'Boundary Recall', 'Boundary F-measure',
    'obj_detected_075_percentage',
)
METRIC_NAMES = PAPER_METRIC_NAMES + ('obj_detected', 'obj_detected_075', 'obj_gt')
DIAGNOSTIC_NAMES = (
    'obj_detected_ge075', 'obj_detected_ge075_percentage', 'obj_matched_exactly075',
)


def reference_provenance():
    """Check the three actually imported files against fixed hashes and receipt."""
    receipt = json.loads((_VENDOR / 'source_receipt.json').read_text())
    if receipt['commit'] != REFERENCE_COMMIT:
        raise RuntimeError('DA-Fusion source receipt does not match the pinned commit')
    files = {}
    for name, expected in _REFERENCE_SHA256.items():
        relative = _EVAL_REL + name
        path = _VENDOR / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected or receipt['files'][relative]['sha256'] != expected:
            raise RuntimeError(f'Changed UOIS reference source: {path}')
        files[name] = {'path': str(path), 'sha256': actual}
    return {'repository': receipt['repository'], 'commit': REFERENCE_COMMIT,
            'files': files}


@lru_cache(maxsize=1)
def load_reference():
    """Import only the verified evaluation package, without DA-Fusion __init__."""
    reference_provenance()
    folder = _VENDOR / _EVAL_REL
    name = '_dpseg_uois_reference_c5bf9e0395c1'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, folder / '__init__.py', submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(name + '.compute_PRF')


def _label_map(value, name):
    value = np.asarray(value)
    if value.ndim != 2 or not all(value.shape):
        raise ValueError(f'{name} must be a nonempty H x W instance ID map')
    if value.dtype.kind not in 'uib' or np.any(value < 0):
        raise ValueError(f'{name} must contain nonnegative integer instance IDs')
    return value


def predictions_to_label_map(predictions, shape, score_threshold=.5):
    """Decode ordered COCO RLE records; later retained masks overwrite earlier ones.

    Returns an int32 array, not a tuple. Each record needs ``score`` and
    ``segmentation`` (one compressed or uncompressed COCO RLE). The list is never
    sorted. As in benchmark.py, only score > threshold is retained. Category IDs
    do not affect this class-agnostic metric. No mask postprocessing is applied.
    """
    from pycocotools import mask as mask_utils

    if (len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in shape)):
        raise ValueError('shape must be positive integer (height, width)')
    shape = tuple(int(n) for n in shape)
    if not np.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
        raise ValueError('score_threshold must be finite and between 0 and 1')
    labels = np.zeros(shape, dtype=np.int32)
    retained = 0
    for record in predictions:
        score = float(record['score'])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Prediction score must be finite and between 0 and 1')
        if score <= score_threshold:
            continue
        rle = dict(record['segmentation'])
        if tuple(rle['size']) != shape:
            raise ValueError('Prediction RLE dimensions differ from the original GT')
        if isinstance(rle['counts'], list):
            counts = rle['counts']
            if (any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts)
                    or sum(counts) != shape[0] * shape[1]):
                raise ValueError('Invalid uncompressed COCO RLE run lengths')
            rle = mask_utils.frPyObjects(rle, *shape)
        elif isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('ascii')
        elif not isinstance(rle['counts'], bytes):
            raise ValueError('Expected one compressed or uncompressed COCO RLE')
        mask = mask_utils.decode(rle)
        if mask.shape != shape:
            raise ValueError('Expected a single mask at original GT resolution')
        retained += 1
        labels[mask > 0] = retained
    return labels


def evaluate_label_maps(prediction, ground_truth):
    """Return exact native metrics, separate >= .75 diagnostics and assignments.

    Neither input is changed. The returned assignment pairs are [GT ID, pred ID]
    in native Munkres order. Native NaN boundary P/R for whole-image masks remain
    NaN (boundary F is 0); they are not silently replaced with perfect scores.
    """
    prediction = _label_map(prediction, 'prediction')
    truth = _label_map(ground_truth, 'ground_truth')
    if prediction.shape != truth.shape:
        raise ValueError('Prediction and GT must have identical original dimensions')
    native = load_reference()
    truth_values, truth_inverse = np.unique(truth.ravel(), return_inverse=True)
    pred_values, pred_inverse = np.unique(prediction.ravel(), return_inverse=True)
    truth_positions = np.flatnonzero(truth_values != 0)
    pred_positions = np.flatnonzero(pred_values != 0)
    truth_labels, pred_labels = truth_values[truth_positions], pred_values[pred_positions]
    if not len(truth_labels) or not len(pred_labels):
        metrics, assignments = native.multilabel_metrics(
            prediction, truth, None, None, return_assign=True)
        diagnostics = {'obj_detected_ge075': 0,
                       'obj_detected_ge075_percentage': metrics['obj_detected_075_percentage'],
                       'obj_matched_exactly075': 0}
        return {'metrics': {k: float(v) for k, v in metrics.items()},
                'diagnostics': diagnostics, 'assignments': assignments}

    joint = np.bincount(truth_inverse * len(pred_values) + pred_inverse,
                        minlength=len(truth_values) * len(pred_values)).reshape(
                            len(truth_values), len(pred_values))
    intersections = joint[np.ix_(truth_positions, pred_positions)].astype(np.float64)
    truth_areas, pred_areas = joint.sum(1)[truth_positions], joint.sum(0)[pred_positions]
    scores = np.zeros_like(intersections)
    for i in range(len(truth_labels)):
        for j in range(len(pred_labels)):
            # Preserve native floating-point order: algebraic Dice shortcuts can
            # alter a Hungarian tie or move a value across the strict .75 endpoint.
            overlap = np.int64(intersections[i, j])
            precision, recall = overlap / pred_areas[j], overlap / truth_areas[i]
            if precision + recall > 0:
                scores[i, j] = 2 * precision * recall / (precision + recall)
    assignments = native.Munkres().compute(scores.max() - scores.copy())
    indices = tuple(np.array(assignments).T)
    precision = np.sum(intersections[indices]) / np.count_nonzero(prediction)
    recall = np.sum(intersections[indices]) / np.count_nonzero(truth)
    with np.errstate(divide='ignore', invalid='ignore'):
        overlap_f = 2 * precision * recall / (precision + recall)
    if np.isnan(overlap_f):
        overlap_f = 0.

    boundary_pred = 0.
    for label in pred_labels:
        boundary_pred += np.sum(native.seg2bmap(prediction == label))
    boundary_gt = 0.
    for label in truth_labels:
        boundary_gt += np.sum(native.seg2bmap(truth == label))
    boundary_counts = np.zeros((len(assignments), 2))
    for k, (i, j) in enumerate(assignments):
        boundary_counts[k] = native.boundary_overlap(
            i, j, prediction == pred_labels[j], truth == truth_labels[i], None, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        boundary_precision = np.sum(boundary_counts[:, 0]) / boundary_pred
        boundary_recall = np.sum(boundary_counts[:, 1]) / boundary_gt
        boundary_f = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
    if np.isnan(boundary_f):
        boundary_f = 0.
    detected = int(np.count_nonzero(scores[indices] > .75))
    inclusive = int(np.count_nonzero(scores[indices] >= .75))
    metrics = {
        'Objects F-measure': overlap_f, 'Objects Precision': precision, 'Objects Recall': recall,
        'Boundary F-measure': boundary_f, 'Boundary Precision': boundary_precision,
        'Boundary Recall': boundary_recall, 'obj_detected': len(pred_labels),
        'obj_detected_075': detected, 'obj_gt': len(truth_labels),
        'obj_detected_075_percentage': detected / len(truth_labels),
    }
    return {
        'metrics': {k: float(v) for k, v in metrics.items()},
        'diagnostics': {'obj_detected_ge075': inclusive,
                        'obj_detected_ge075_percentage': inclusive / len(truth_labels),
                        'obj_matched_exactly075': int(np.count_nonzero(scores[indices] == .75))},
        'assignments': [[int(truth_labels[i]), int(pred_labels[j])] for i, j in assignments],
    }


def macro_average(rows):
    """Arithmetic mean across images; retain native undefined values as NaN.

    Count metrics are also per-image means, not pooled totals. Undefined-value
    counts are exposed so callers can encode NaN as JSON null without losing why
    it occurred. Empty input is rejected rather than reported as an evaluation.
    """
    rows = list(rows)
    if not rows:
        raise ValueError('Cannot average an empty benchmark')
    result = {'images': len(rows), 'aggregation': 'arithmetic mean over images',
              'units': 'fractions (counts are mean counts per image)', 'undefined_images': {}}
    for group, names in [('metrics', METRIC_NAMES), ('diagnostics', DIAGNOSTIC_NAMES)]:
        result[group] = {}
        for name in names:
            values = np.asarray([row[group][name] for row in rows], dtype=np.float64)
            result[group][name] = float(np.mean(values))
            undefined = int(np.count_nonzero(~np.isfinite(values)))
            if undefined:
                result['undefined_images'][name] = undefined
    return result
