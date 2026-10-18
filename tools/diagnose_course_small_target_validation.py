#!/usr/bin/env python3
"""Diagnose saved validation predictions by area and IoU, without model execution.

The first area range reproduces the existing standard COCO small AP/AR. Other
ranges are descriptive diagnostics and never select checkpoints or alter runs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '4'

import argparse
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg import course_submission as submission
from tools.summarize_course_finetune import read, sha256, check_hash, require, selected_epoch

RANGES = [('small', [0, 1024]), ('1_64', [1, 64]),
          ('65_256', [65, 256]), ('257_1024', [257, 1024])]


def average_valid(values):
    valid = values[values >= 0]
    return float(np.mean(valid)) if valid.size else None


def selected_validation(folder):
    selection = read(folder / 'selection.json')
    epoch = selected_epoch(selection['epochs'])
    require(epoch == selection['best_epoch'], 'Validation selection differs')
    require(selection['test_used_for_selection'] is False, 'Unexpected test selection')
    path = Path(selection['epochs'][epoch - 1]['output']).resolve()
    require(path.parent == folder, 'Validation path outside formal run')
    evaluation = read(path / 'evaluation.json')
    receipt = read(path / 'evaluation_run.json')
    require(evaluation['split'] == receipt['split'] == 'val', 'Only validation allowed')
    require(evaluation['status'] == receipt['status'] == 'completed', 'Validation incomplete')
    require(evaluation['images'] == 2584 and receipt['scope'] == 'full_split', 'Incomplete split')
    check_hash(path / 'evaluation.json', receipt['evaluation_sha256'])
    check_hash(path / 'predictions.jsonl', receipt['predictions_sha256'])
    require(evaluation['metrics'] == selection['epochs'][epoch - 1]['metrics'], 'Metric binding differs')
    return epoch, path, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'reports/course_small_target/validation_diagnostics')
    args = parser.parse_args()
    base = ROOT / 'reports/course_small_target'
    protocol = read(base / 'protocol.json')
    for name, digest in protocol['source_sha256'].items():
        check_hash(ROOT / name, digest)
    folders = {'original': Path(protocol['baseline_comparison']['baseline_run']),
               'sampling': base / 'formal_v1'}
    selected = {key: selected_validation(folder.resolve()) for key, folder in folders.items()}
    require(not args.output.exists(), 'Refusing to overwrite diagnostic results')
    torch.set_num_threads(4)
    with CourseArchive(protocol['archive'], 'val') as dataset:
        truth = dataset.coco_ground_truth()
        truth.setdefault('info', {})
        require(len(dataset.image_ids) == 2584, 'Validation image count differs')
        require(len(truth['annotations']) == 73421, 'Validation annotation count differs')
        require(all(a['area'] == int(a['area']) and a['area'] > 0
                    for a in truth['annotations']), 'Area bins require positive integer mask areas')
        gt_counts = {label: sum(lo <= a['area'] <= hi for a in truth['annotations'])
                     for label, (lo, hi) in RANGES}
        require(gt_counts['small'] == 10999, 'Small-instance coverage differs')
        spec = submission._Spec(dataset.image_ids, (1,), (1024, 1024), 100)
        args.output.mkdir(parents=True)
        report = {'status': 'running', 'scope': 'Post-hoc validation diagnosis only',
                  'started_at_utc': datetime.now(timezone.utc).isoformat(),
                  'script_sha256': sha256(__file__), 'protocol_sha256': sha256(base / 'protocol.json'),
                  'new_inference': False, 'test_accessed': False, 'gpu_used': False,
                  'checkpoint_selection_changed': False, 'images': 2584,
                  'area_ranges': dict(RANGES), 'gt_counts': gt_counts, 'models': {}}
        try:
            for name, (epoch, path, reference) in selected.items():
                predictions = []
                with (path / 'predictions.jsonl').open() as stream:
                    count = 0
                    for index, line in enumerate(stream):
                        record = json.loads(line, object_pairs_hook=submission._unique_object)
                        submission._check_id(record['image_id'], index, spec)
                        values, _ = submission._predictions(record['predictions'], spec, decode=False)
                        predictions.extend({'image_id': record['image_id'], **p} for p in values)
                        count += 1
                require(count == 2584, 'Missing validation prediction rows')
                require(len(predictions) == reference['predicted_instances'], 'Prediction count differs')
                with (args.output / f'{name}_coco.log').open('x') as log, contextlib.redirect_stdout(log):
                    coco = COCO()
                    coco.dataset = truth
                    coco.createIndex()
                    detections = coco.loadRes(predictions)
                    evaluator = COCOeval(coco, detections, 'segm')
                    evaluator.params.imgIds = dataset.image_ids
                    evaluator.params.catIds = [1]
                    evaluator.params.maxDets = [100]
                    evaluator.params.areaRng = [bounds for _, bounds in RANGES]
                    evaluator.params.areaRngLbl = [label for label, _ in RANGES]
                    evaluator.evaluate()
                    evaluator.accumulate()
                bins = {}
                for index, (label, _) in enumerate(RANGES):
                    precision = evaluator.eval['precision'][:, :, :, index, 0]
                    recall = evaluator.eval['recall'][:, :, index, 0]
                    bins[label] = {'ap': average_valid(precision), 'ar': average_valid(recall),
                                   'ap_by_iou': [average_valid(row) for row in precision],
                                   'recall_by_iou': [average_valid(row) for row in recall]}
                require(abs(bins['small']['ap'] - reference['metrics']['mask_ap_small']) < 1e-12,
                        'Diagnostic small AP does not reproduce saved official COCO result')
                require(abs(bins['small']['ar'] - reference['metrics']['mask_ar_small']) < 1e-12,
                        'Diagnostic small AR does not reproduce saved official COCO result')
                check_hash(path / 'predictions.jsonl', reference['predictions_sha256'])
                report['iou_thresholds'] = evaluator.params.iouThrs.tolist()
                report['models'][name] = {'selected_epoch': epoch, 'bins': bins,
                    'standard_small_ap_ar_reproduced': True,
                    'predictions_path': str(path / 'predictions.jsonl'),
                    'predictions_sha256': sha256(path / 'predictions.jsonl'),
                    'evaluation_sha256': sha256(path / 'evaluation.json')}
                print(json.dumps({'completed_model': name, 'bins': bins}), flush=True)
            report['delta_sampling_minus_original'] = {
                label: {metric: report['models']['sampling']['bins'][label][metric]
                        - report['models']['original']['bins'][label][metric]
                        for metric in ('ap', 'ar')} for label, _ in RANGES}
            require(not torch.cuda.is_initialized(), 'Unexpected GPU initialization')
            for name, digest in protocol['source_sha256'].items():
                check_hash(ROOT / name, digest)
            report.update(status='pass', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                          interpretation='Descriptive size and IoU breakdown; does not establish a causal failure mechanism.')
        except BaseException as error:
            report.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            (args.output / 'diagnostics.json').write_text(
                json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
