#!/usr/bin/env python3
"""Compare saved validation mask geometry; best overlap is not a COCO match."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_seg.course_archive import CourseArchive
from tools.summarize_course_finetune import read, sha256, check_hash, require


def rle(value):
    return {'size': value['size'], 'counts': value['counts'].encode('ascii')}


def summarize(rows):
    overlap = [row for row in rows if row['best_iou'] > 0]
    ratios = [row['best_prediction_area'] / row['gt_area'] for row in overlap]
    return {'gt_instances': len(rows), 'any_overlap_instances': len(overlap),
            'median_best_iou': float(np.median([row['best_iou'] for row in rows])),
            'best_iou_ge_05_fraction': sum(row['best_iou'] >= .5 for row in rows) / len(rows),
            'prediction_to_gt_area_ratio_quantiles_on_overlap':
                np.quantile(ratios, [.1, .5, .9]).tolist() if ratios else None,
            'area_ratio_ge_2_fraction_on_overlap':
                sum(r >= 2 for r in ratios) / len(ratios) if ratios else None,
            'area_ratio_le_half_fraction_on_overlap':
                sum(r <= .5 for r in ratios) / len(ratios) if ratios else None}


def main():
    base = ROOT / 'reports/course_small_target'
    source = base / 'validation_diagnostics/diagnostics.json'
    diagnosis = read(source)
    require(diagnosis['status'] == 'pass', 'Complete selected-validation diagnosis required')
    output = base / 'validation_geometry'
    require(not output.exists(), 'Refusing to overwrite geometry diagnosis')
    protocol = read(base / 'protocol.json')
    with CourseArchive(protocol['archive'], 'val') as dataset:
        truth = dataset.coco_ground_truth()
        grouped = defaultdict(list)
        for ann in truth['annotations']:
            if 0 < ann['area'] <= 1024:
                require(not ann.get('iscrowd', 0), 'Crowd requires a different diagnostic')
                grouped[ann['image_id']].append(ann)
        require(sum(map(len, grouped.values())) == 10999, 'Small GT coverage differs')
        output.mkdir()
        report = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                  'source_diagnosis_sha256': sha256(source), 'script_sha256': sha256(__file__),
                  'scope': 'Full validation, selected weights, saved candidates with score>=0.01 and max100',
                  'new_inference': False, 'test_accessed': False, 'models': {},
                  'interpretation_limit': 'Each GT independently chooses its highest-IoU saved candidate; candidates may be reused. This is an optimistic geometry diagnostic, not COCO one-to-one matching or a causal explanation. Area ratios condition on nonzero overlap; ties use first saved candidate.'}
        try:
            for name, model in diagnosis['models'].items():
                path = Path(model['predictions_path'])
                check_hash(path, model['predictions_sha256'])
                rows = []
                with path.open() as stream:
                    count = 0
                    for index, line in enumerate(stream):
                        item = json.loads(line)
                        require(item['image_id'] == dataset.image_ids[index], 'Prediction ID differs')
                        count += 1
                        anns = grouped[item['image_id']]
                        if not anns:
                            continue
                        preds = item['predictions']
                        if preds:
                            pr = [rle(p['segmentation']) for p in preds]
                            gt = [rle(a['segmentation']) for a in anns]
                            areas = mask_utils.area(pr)
                            ious = mask_utils.iou(pr, gt, [0] * len(gt))
                            require(np.isfinite(ious).all(), 'Nonfinite mask IoU')
                            indices = ious.argmax(axis=0)
                        for j, ann in enumerate(anns):
                            best = int(indices[j]) if preds else None
                            rows.append({'image_id': item['image_id'], 'annotation_id': ann['id'],
                                'gt_area': ann['area'], 'best_iou': float(ious[best, j]) if preds else 0.,
                                'best_prediction_area': int(areas[best]) if preds else 0})
                require(count == 2584 and len(rows) == 10999, 'Incomplete validation geometry')
                check_hash(path, model['predictions_sha256'])
                bins = {}
                for label, (lo, hi) in diagnosis['area_ranges'].items():
                    part = [row for row in rows if lo <= row['gt_area'] <= hi]
                    require(len(part) == diagnosis['gt_counts'][label], 'Area-group coverage differs')
                    bins[label] = summarize(part)
                raw_path = output / f'{name}_per_gt.jsonl'
                with raw_path.open('x') as raw:
                    for row in rows:
                        raw.write(json.dumps(row, allow_nan=False) + '\n')
                report['models'][name] = {'selected_epoch': model['selected_epoch'], 'bins': bins,
                    'predictions_sha256': model['predictions_sha256'], 'per_gt_sha256': sha256(raw_path)}
                print(json.dumps({'model': name, 'bins': bins}), flush=True)
            report.update(status='pass', completed_at_utc=datetime.now(timezone.utc).isoformat())
        except BaseException as error:
            report.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            (output / 'geometry.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
