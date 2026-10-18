#!/usr/bin/env python3
"""Bounded first-epoch validation diagnostic from existing saved predictions.

No model is loaded and no prediction is generated. The original frozen COCO
scoring rules are retained; only per-IoU small-object AP/AR are extracted.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile

# Restrict this diagnostic process before importing numerical libraries.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/course_dafusion_metric_plot_cache')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.cocoeval as coco_eval_source

from rgbd_seg.course_archive import CourseArchive
from rgbd_seg import course_submission as submission


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def checked_sources(mapping, *, evaluation_names=False):
    actual = {}
    for name, expected in mapping.items():
        path = ROOT / ('rgbd_seg' if evaluation_names and '/' not in name else '') / name
        digest = sha256(path)
        if digest != expected:
            raise ValueError(f'Source changed from recorded evaluation/protocol: {name}')
        actual[str(path.relative_to(ROOT))] = digest
    return actual


def annotation_digest(dataset):
    digest = hashlib.sha256()
    with ZipFile(dataset.zip_path) as archive:
        for member in sorted(dataset.annotation_members):
            digest.update(member.encode() + b'\0' + archive.read(member))
    return digest.hexdigest()


def valid_mean(values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError('COCO accumulation contains nonfinite values')
    available = values[values > -1]
    if not available.size or not np.isfinite(available).all():
        raise ValueError('Expected finite defined small-object AP/AR values')
    return float(available.mean())


def score_one(dataset, prediction_path, saved, log_path):
    """Same ordered predictions, native RLE guards, GT and COCO settings as frozen scorer."""
    ids = dataset.image_ids
    spec = submission._Spec(ids, (1,), (1024, 1024), 100)
    results, rows = [], 0
    with prediction_path.open() as stream:
        for line in stream:
            record = json.loads(line, object_pairs_hook=submission._unique_object)
            submission._check_id(record['image_id'], rows, spec)
            values, _ = submission._predictions(record['predictions'], spec, decode=False)
            # Deliberately omit boxes: COCO computes prediction area from each RLE.
            results.extend({'image_id': ids[rows], **value} for value in values)
            rows += 1
    if rows != len(ids):
        raise ValueError('Missing prediction rows before COCO evaluation')
    truth = dataset.coco_ground_truth()
    if {image['id'] for image in truth['images']} != set(ids):
        raise ValueError('Ground-truth image IDs differ from evaluated split')
    if {category['id'] for category in truth['categories']} != {1}:
        raise ValueError('Expected component category 1')
    if (len(truth['annotations']) != saved['gt_instances']
            or len(results) != saved['predicted_instances']):
        raise ValueError('GT/prediction count differs from saved evaluation')
    truth.setdefault('info', {})
    started = time.monotonic()
    with log_path.open('x') as log, contextlib.redirect_stdout(log):
        coco = COCO()
        coco.dataset = truth
        coco.createIndex()
        if results:
            detections = coco.loadRes(results)
        else:
            detections = COCO()
            detections.dataset = {'info': {}, 'images': truth['images'],
                                  'categories': truth['categories'], 'annotations': []}
            detections.createIndex()
        evaluator = COCOeval(coco, detections, 'segm')
        evaluator.params.imgIds = list(ids)
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    thresholds = evaluator.params.iouThrs
    if not np.array_equal(thresholds, np.asarray(saved['iou_thresholds'])):
        raise ValueError('IoU thresholds differ from saved evaluation')
    if evaluator.params.areaRng != saved['area_ranges']:
        raise ValueError('Area ranges differ from saved evaluation')
    small_index = evaluator.params.areaRngLbl.index('small')
    max_index = evaluator.params.maxDets.index(100)
    precision = evaluator.eval['precision']
    recall = evaluator.eval['recall']
    if precision.shape != (10, 101, 1, 4, 3) or recall.shape != (10, 1, 4, 3):
        raise ValueError('Unexpected COCO accumulation axes')
    ap_curve = [valid_mean(precision[t, :, :, small_index, max_index]) for t in range(10)]
    ar_curve = [valid_mean(recall[t, :, small_index, max_index]) for t in range(10)]
    ap_mean, ar_mean = float(np.mean(ap_curve)), float(np.mean(ar_curve))
    for name, current, stats_index in (('mask_ap_small', ap_mean, 3), ('mask_ar_small', ar_mean, 9)):
        if (abs(current - saved['metrics'][name]) > 1e-12
                or abs(current - float(evaluator.stats[stats_index])) > 1e-12):
            raise ValueError(f'{name} ten-threshold mean differs from original result: '
                             f'{current} vs {saved["metrics"][name]}')
    result = {
        'iou_thresholds': thresholds.tolist(), 'small_ap_by_iou': ap_curve,
        'small_ar_max100_by_iou': ar_curve, 'small_ap_mean': ap_mean,
        'small_ar_max100_mean': ar_mean,
        'saved_small_ap': saved['metrics']['mask_ap_small'],
        'saved_small_ar_max100': saved['metrics']['mask_ar_small'],
        'small_ap_mean_absolute_error': abs(ap_mean - saved['metrics']['mask_ap_small']),
        'small_ar_mean_absolute_error': abs(ar_mean - saved['metrics']['mask_ar_small']),
        'agreement_absolute_tolerance': 1e-12,
        'images': rows, 'gt_instances': len(truth['annotations']),
        'gt_small_instances': sum(0 <= ann['area'] <= 1024 for ann in truth['annotations']),
        'predicted_instances': len(results), 'elapsed_seconds': time.monotonic() - started,
    }
    return result


def plot_curves(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    styles = [('mask2former_rgb', 'Mask2Former RGB', '#2463a4', 'o', '--'),
              ('dafusion_rgbd', 'DA-Fusion RGB-D', '#b75418', 's', '-')]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    figure.suptitle('Small-object masks: first-epoch validation', fontsize=15, weight='bold', y=.98)
    figure.text(.5, .91, '2,584 images  |  original mask area ≤ 1,024 pixels  |  maxDets = 100',
                ha='center', fontsize=10, color='#444444')
    for axis, metric, title, mean_key in zip(axes,
            ['small_ap_by_iou', 'small_ar_max100_by_iou'],
            ['Average precision (APₛ)', 'Recall (ARₛ at each IoU)'],
            ['small_ap_mean', 'small_ar_max100_mean']):
        for key, label, color, marker, linestyle in styles:
            model = report['models'][key]
            axis.plot(model['iou_thresholds'], model[metric], color=color, marker=marker,
                      linestyle=linestyle, linewidth=2, markersize=4.5,
                      label=f'{label} · mean {model[mean_key]:.4f}')
        axis.set_title(title, fontsize=12, pad=10)
        axis.set_xlabel('Required mask IoU')
        axis.set_xlim(.485, .965)
        axis.set_ylim(0, 1)
        axis.set_xticks(np.arange(.5, 1., .05))
        axis.set_yticks(np.arange(0, 1.01, .2))
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(loc='upper right', frameon=False, fontsize=8.5)
    axes[0].set_ylabel('Score (0–1)')
    figure.text(.5, .035, 'Saved predictions re-scored with COCO. Validation diagnostic; final test results remain pending.',
                ha='center', fontsize=9, color='#444444')
    figure.subplots_adjust(left=.075, right=.985, bottom=.17, top=.80, wspace=.10)
    figure.savefig(output, dpi=180, facecolor='white')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'reports/course_dafusion/protocol.json')
    parser.add_argument('--baseline', type=Path, default=ROOT/'reports/course_finetune/mask2former/val_epoch_01')
    parser.add_argument('--dafusion', type=Path, default=ROOT/'reports/course_dafusion/formal_v1/val_epoch_01')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/course_dafusion/interim_small_ap_epoch01')
    args = parser.parse_args()
    if any(path.name != 'val_epoch_01' for path in (args.baseline, args.dafusion)):
        raise ValueError('This bounded diagnostic accepts first-epoch validation artifacts only')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Diagnostic output must be new or empty')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    receipt = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'first_epoch_full_validation_diagnostic_from_saved_predictions',
               'gpu_used': False, 'model_loaded': False, 'inference_run': False,
               'test_data_accessed': False, 'hyperparameters_changed': False,
               'cpu_threads': 4, 'torch_interop_threads': 1,
               'script_sha256': sha256(__file__),
               'cocoeval_source_sha256': sha256(coco_eval_source.__file__),
               'versions': {name: importlib.metadata.version(name)
                            for name in ('numpy', 'pycocotools', 'torch', 'matplotlib')}}
    write_json(args.output/'receipt.json', receipt)
    try:
        protocol = json.loads(args.protocol.read_text())
        receipt['protocol'] = {'path': str(args.protocol.resolve()), 'sha256': sha256(args.protocol)}
        formal_run = json.loads((args.dafusion.parent/'run.json').read_text())
        if (formal_run['protocol_sha256'] != receipt['protocol']['sha256']
                or formal_run['source_sha256'] != protocol['source_sha256']):
            raise ValueError('Protocol provenance differs from the running formal experiment')
        receipt['formal_run_provenance_snapshot'] = {
            'path': str((args.dafusion.parent/'run.json').resolve()),
            'protocol_sha256': formal_run['protocol_sha256'],
            'source_sha256': formal_run['source_sha256'],
            'note': 'Only immutable provenance fields captured; live training progress is not used'}
        receipt['frozen_sources_sha256'] = checked_sources(protocol['source_sha256'])
        audit_path = Path(protocol['data_audit']['path'])
        if sha256(audit_path) != protocol['data_audit']['sha256']:
            raise ValueError('Data audit changed from frozen protocol')
        audit = json.loads(audit_path.read_text())
        expected_annotation_sha = audit['splits']['val']['annotation_member_name_and_bytes_sha256']
        receipt['data_audit'] = {'path': str(audit_path), 'sha256': sha256(audit_path)}
        receipt['archive'] = {'path': protocol['archive'],
            'full_archive_sha256_recorded_in_training_protocol': protocol['archive_sha256_from_completed_download_receipt'],
            'verification_this_diagnostic': 'All validation annotation member names and compressed bytes rehashed; no RGB/depth image read'}
        receipt['inputs'] = {}
        report = {'status': 'completed', 'scope': receipt['scope'], 'split': 'val', 'epoch': 1,
            'metric_scale': '0-1', 'small_area': 'original ground-truth mask area 0 through 1024 inclusive',
            'iou_type': 'segm', 'max_detections': [1, 10, 100], 'category_ids': [1],
            'ap_extraction': 'precision[T,R,K,A,M]; mean of defined 101-recall precision entries per IoU, K=1, A=small, M=100',
            'ar_extraction': 'recall[T,K,A,M]; recall at maxDets=100 per IoU, K=1, A=small',
            'models': {}, 'limitations': [
                'First-epoch validation diagnostic; neither final test results nor independent model reproduction.',
                'Same saved predictions rescored; no new training/inference or hyperparameter selection.',
                'Differences at stricter IoU thresholds describe mask prediction performance; they do not establish depth or a boundary-module causal effect.']}
        with CourseArchive(protocol['archive'], split='val') as dataset:
            if len(dataset) != 2584:
                raise ValueError('Expected the complete 2,584-image validation split')
            actual_annotation_sha = annotation_digest(dataset)
            if actual_annotation_sha != expected_annotation_sha:
                raise ValueError('Validation annotation bytes differ from frozen data audit')
            receipt['archive']['validation_annotation_sha256'] = actual_annotation_sha
            for key, folder in [('mask2former_rgb', args.baseline), ('dafusion_rgbd', args.dafusion)]:
                saved_path, prediction_path = folder/'evaluation.json', folder/'predictions.jsonl'
                saved = json.loads(saved_path.read_text())
                if saved['split'] != 'val' or saved['images'] != 2584 or saved['scope'] != 'full_split':
                    raise ValueError('Expected complete saved validation results')
                checked_sources(saved['source_sha256'], evaluation_names=True)
                prediction_sha = sha256(prediction_path)
                if prediction_sha != saved['predictions_sha256']:
                    raise ValueError('Saved prediction SHA differs from original evaluation')
                receipt['inputs'][key] = {'evaluation': str(saved_path.resolve()), 'evaluation_sha256': sha256(saved_path),
                    'predictions': str(prediction_path.resolve()), 'predictions_sha256': prediction_sha}
                write_json(args.output/'receipt.json', receipt)
                print(json.dumps({'status': 'scoring', 'model': key, 'images': len(dataset)}), flush=True)
                report['models'][key] = score_one(dataset, prediction_path, saved, args.output/f'{key}_coco.log')
                if sha256(prediction_path) != prediction_sha or sha256(saved_path) != receipt['inputs'][key]['evaluation_sha256']:
                    raise ValueError('Input artifact changed during scoring')
                print(json.dumps({'status': 'scored', 'model': key, 'small_ap': report['models'][key]['small_ap_mean']}), flush=True)
                gc.collect()
            if annotation_digest(dataset) != actual_annotation_sha:
                raise ValueError('Validation annotation inputs changed during diagnostic')
        baseline, dafusion = (report['models'][name] for name in ('mask2former_rgb', 'dafusion_rgbd'))
        report['dafusion_minus_rgb'] = {
            'small_ap_by_iou': [b-a for a, b in zip(baseline['small_ap_by_iou'], dafusion['small_ap_by_iou'])],
            'small_ar_max100_by_iou': [b-a for a, b in zip(baseline['small_ar_max100_by_iou'], dafusion['small_ar_max100_by_iou'])],
            'small_ap_mean': dafusion['small_ap_mean'] - baseline['small_ap_mean'],
            'small_ar_max100_mean': dafusion['small_ar_max100_mean'] - baseline['small_ar_max100_mean']}
        checked_sources(protocol['source_sha256'])
        if torch.cuda.is_initialized():
            raise RuntimeError('Diagnostic unexpectedly initialized CUDA')
        write_json(args.output/'small_ap_by_iou.json', report)
        plot_curves(report, args.output/'small_ap_by_iou.png')
        receipt.update(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.monotonic()-started,
                       outputs_sha256={name: sha256(args.output/name) for name in
                           ('small_ap_by_iou.json', 'small_ap_by_iou.png', 'mask2former_rgb_coco.log', 'dafusion_rgbd_coco.log')})
        write_json(args.output/'receipt.json', receipt)
        print(json.dumps({'status': 'completed', 'output': str(args.output.resolve()),
                          'small_ap_delta': report['dafusion_minus_rgb']['small_ap_mean']}), flush=True)
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error),
                       elapsed_seconds=time.monotonic()-started)
        write_json(args.output/'receipt.json', receipt)
        raise


if __name__ == '__main__':
    main()
