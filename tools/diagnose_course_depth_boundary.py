#!/usr/bin/env python3
"""Describe remaining mask geometry and training-only depth boundary signal."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_depth_data import load_depth_record
from tools.select_course_resolution import read, sha, require

THRESHOLDS_M = [0.0001, 0.0005, 0.001, 0.002, 0.005]
RANGES = {'small': (1, 1024), '1_64': (1, 64),
          '65_256': (65, 256), '257_1024': (257, 1024)}


def adjacent_depth_pairs(mask, depth, valid):
    """Count undirected horizontal/vertical boundary and foreground pairs.

    Invalid depths never contribute a difference. Outside-image neighbors are
    excluded. Each pair occurs once; diagonal adjacency is not used.
    """
    require(mask.ndim == 2 and mask.shape == depth.shape == valid.shape,
            'Aligned 2D arrays required')
    require(mask.dtype == bool and valid.dtype == bool, 'Boolean masks required')
    require(np.isfinite(depth[valid]).all() and (depth[valid] > 0).all(),
            'Valid depths must be positive and finite')
    values = {'boundary': [], 'interior': []}
    totals = {'boundary': 0, 'interior': 0}
    for first, second in ((np.s_[:-1, :], np.s_[1:, :]),
                          (np.s_[:, :-1], np.s_[:, 1:])):
        usable = valid[first] & valid[second]
        difference = np.abs(depth[first].astype(np.float64) - depth[second])
        for name, selected in [('boundary', mask[first] != mask[second]),
                               ('interior', mask[first] & mask[second])]:
            totals[name] += int(selected.sum())
            values[name].append(difference[selected & usable])
    return {name: (totals[name], np.concatenate(values[name])) for name in values}


def rle(value):
    return {'size': value['size'], 'counts': value['counts'].encode('ascii')}


def geometry(rows):
    values = np.array([x['best_iou'] for x in rows])
    return {'instances': len(rows), 'no_overlap_fraction': float((values == 0).mean()),
            'best_iou_ge_05_fraction': float((values >= .5).mean()),
            'best_iou_ge_075_fraction': float((values >= .75).mean()),
            'best_iou_ge_09_fraction': float((values >= .9).mean()),
            'median_best_iou': float(np.median(values))}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    decision = read(args.decision)
    require(decision['status'] == 'completed' and not decision['test_used'], 'Completed comparison required')
    method = decision['recommended_method_for_next_stage']
    require(method == 'native', 'This diagnosis is scoped to the selected native1024 model')
    selected = decision['baseline']
    folder = Path(selected['best_checkpoint']).parent
    require(sha(folder/'run.json') == selected['run_sha256'] and
            sha(folder/'selection.json') == selected['selection_sha256'], 'Selected run identity changed')
    run = read(folder/'run.json')
    require(run['status'] == 'completed' and run['resolution'] == 1024, 'Selected run incomplete')
    parent_decision = read(ROOT/'reports/course_resolution_ddp/decision.json')
    require(sha(ROOT/'reports/course_resolution_ddp/decision.json') == decision['resolution_decision_sha256'],
            'Resolution decision changed')
    protocol_path = Path(parent_decision['protocol_path'])
    require(sha(protocol_path) == parent_decision['protocol_sha256'], 'Protocol changed')
    protocol = read(protocol_path)
    validation = folder/f"val_epoch_{selected['best_epoch']:02d}"
    receipt = read(validation/'evaluation_run.json')
    require(receipt['status'] == 'completed' and receipt['images'] == 2584
            and receipt['split'] == 'val' and receipt['scope'] == 'full_split', 'Full validation required')
    predictions_path = validation/'predictions.jsonl'
    require(sha(predictions_path) == receipt['predictions_sha256'], 'Predictions changed')
    require(not args.output.exists(), 'Refusing to overwrite a diagnosis')
    args.output.mkdir(parents=True)
    plan = {'status': 'frozen', 'scope': 'Descriptive diagnosis; no model or postprocessing selection',
            'decision_sha256': sha(args.decision), 'script_sha256': sha(__file__),
            'predictions_sha256': receipt['predictions_sha256'],
            'validation': 'All2584 images; each small GT independently chooses highest-IoU saved mask',
            'training_sample': '128 lowest SHA256(depth-boundary-v1:official-image-id) images, independent of labels',
            'depth_pairs': '4-neighbor undirected; compare cross-GT-boundary with both-inside-GT',
            'thresholds_m': THRESHOLDS_M, 'invalid_pairs': 'Excluded from differences; coverage reported',
            'test_accessed': False, 'new_inference': False, 'gpu_used': False}
    dump(args.output/'diagnostic_plan.json', plan)
    report = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': sha(args.output/'diagnostic_plan.json'),
              'limitations': ['Best-IoU candidates may be reused; this is not COCO matching or AP.',
                             'Depth statistics describe 128 training images, not the whole population.',
                             'Edge contrast does not prove an inference algorithm improves masks.',
                             'Pixel-pair aggregates weight long boundaries more; per-instance results are saved.']}
    dump(args.output/'diagnostics.json', report)
    try:
        with CourseArchive(protocol['archive'], 'val') as dataset:
            truth = dataset.coco_ground_truth()
            grouped = defaultdict(list)
            for annotation in truth['annotations']:
                if 0 < annotation['area'] <= 1024:
                    require(not annotation.get('iscrowd', 0), 'Crowd unsupported in geometry diagnosis')
                    grouped[annotation['image_id']].append(annotation)
            require(len(dataset.image_ids) == 2584 and sum(map(len, grouped.values())) == 10999,
                    'Validation coverage differs')
            rows = []
            with predictions_path.open() as stream:
                count = 0
                for index, line in enumerate(stream):
                    item = json.loads(line)
                    require(item['image_id'] == dataset.image_ids[index], 'Prediction order differs')
                    count += 1
                    anns, preds = grouped[item['image_id']], item['predictions']
                    if not anns:
                        continue
                    if preds:
                        pr = [rle(p['segmentation']) for p in preds]
                        ious = mask_utils.iou(pr, [rle(a['segmentation']) for a in anns], [0]*len(anns))
                        indices, areas = ious.argmax(axis=0), mask_utils.area(pr)
                    for j, ann in enumerate(anns):
                        k = int(indices[j]) if preds else None
                        rows.append({'image_id': item['image_id'], 'annotation_id': ann['id'],
                            'gt_area': ann['area'], 'best_iou': float(ious[k, j]) if preds else 0.,
                            'best_prediction_index': k, 'best_prediction_area': int(areas[k]) if preds else 0})
                    if count % 200 == 0:
                        dump(args.output/'progress.json', {'phase': 'validation_geometry', 'images': count})
            require(count == 2584 and len(rows) == 10999, 'Incomplete geometry diagnosis')
            with (args.output/'validation_per_gt.jsonl').open('x') as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
            report['validation_geometry'] = {key: geometry([r for r in rows if lo <= r['gt_area'] <= hi])
                                             for key, (lo, hi) in RANGES.items()}
        dump(args.output/'progress.json', {'phase': 'training_depth_signal'})
        with CourseArchive(protocol['archive'], 'train') as dataset:
            require(len(dataset.image_ids) == 14432, 'Training image count differs')
            chosen = sorted(dataset.image_ids, key=lambda i: hashlib.sha256(
                f'depth-boundary-v1:{i}'.encode()).hexdigest())[:128]
            dump(args.output/'training_image_ids.json', chosen)
            chosen_set = set(chosen)
            grouped = defaultdict(list)
            for ann in dataset.coco_ground_truth()['annotations']:
                if ann['image_id'] in chosen_set and 0 < ann['area'] <= 1024:
                    require(not ann.get('iscrowd', 0), 'Crowd unsupported')
                    grouped[ann['image_id']].append(ann)
            records = dict(zip(dataset.image_ids, dataset.images))
            values = {key: [] for key in ('boundary', 'interior')}
            totals = {key: 0 for key in values}
            instance_rows = []
            for i, image_id in enumerate(chosen):
                depth, valid = load_depth_record(dataset._image_archive(), records[image_id])
                depth, valid = depth[0].numpy(), valid[0].numpy()
                for ann in grouped[image_id]:
                    mask = mask_utils.decode(rle(ann['segmentation'])).astype(bool)
                    yy, xx = np.nonzero(mask)
                    require(len(yy) == ann['area'], 'Annotation area differs from mask')
                    region = np.s_[max(0, yy.min()-1):min(mask.shape[0], yy.max()+2),
                                   max(0, xx.min()-1):min(mask.shape[1], xx.max()+2)]
                    pairs = adjacent_depth_pairs(mask[region], depth[region], valid[region])
                    row = {'image_id': image_id, 'annotation_id': ann['id'], 'gt_area': ann['area']}
                    for key, (total, differences) in pairs.items():
                        totals[key] += total
                        values[key].append(differences)
                        row[key] = {'pairs': total, 'valid_pairs': len(differences),
                            'median_difference_m': float(np.median(differences)) if differences.size else None,
                            'fraction_ge_threshold': [float((differences >= t).mean()) for t in THRESHOLDS_M]
                                if differences.size else None}
                    instance_rows.append(row)
                if (i+1) % 16 == 0:
                    dump(args.output/'progress.json', {'phase': 'training_depth_signal', 'images': i+1})
            with (args.output/'training_depth_per_gt.jsonl').open('x') as stream:
                for row in instance_rows:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
            report['training_depth_signal'] = {'images': 128, 'small_instances': len(instance_rows),
                'images_with_small_instances': sum(bool(v) for v in grouped.values()),
                'thresholds_m': THRESHOLDS_M, 'pairs': {}}
            for key in values:
                differences = np.concatenate(values[key]) if values[key] else np.array([])
                report['training_depth_signal']['pairs'][key] = {'pairs': totals[key],
                    'valid_pairs': len(differences),
                    'valid_fraction': len(differences)/totals[key] if totals[key] else None,
                    'difference_m_quantiles_10_50_90': np.quantile(differences, [.1, .5, .9]).tolist()
                        if differences.size else None,
                    'fraction_ge_threshold': [float((differences >= t).mean()) for t in THRESHOLDS_M]
                        if differences.size else None}
        require(sha(predictions_path) == receipt['predictions_sha256'], 'Predictions changed during diagnosis')
        report.update(status='passed', finished_at_utc=datetime.now(timezone.utc).isoformat(),
            validation_per_gt_sha256=sha(args.output/'validation_per_gt.jsonl'),
            training_per_gt_sha256=sha(args.output/'training_depth_per_gt.jsonl'),
            training_image_ids_sha256=sha(args.output/'training_image_ids.json'))
        print(json.dumps(report, allow_nan=False), flush=True)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        dump(args.output/'diagnostics.json', report)


if __name__ == '__main__':
    main()
