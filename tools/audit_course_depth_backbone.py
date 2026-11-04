#!/usr/bin/env python3
"""Independently score exported CSVs and inspect completed variant checkpoints.

This CPU audit never trains, infers, changes predictions, or selects a model.
It deliberately reads the submitted CSV rather than the evaluator's JSONL.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_submission import _Spec, _check_id, _predictions, _unique_object
from tools.queue_course_depth_backbone import ARMS, completed_base, check_results


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checkpoint_architecture(path, arm):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    metadata, state = checkpoint['model_metadata'], checkpoint['model']
    embed, depth, encoding = (96, 6, 'depth') if arm == 'swint_depth_no_k' else (192, 18, 'xyz')
    if (metadata['depth_encoding'] != encoding or metadata['camera_intrinsics_used'] != (encoding == 'xyz')
            or checkpoint['epoch'] != 15):
        raise ValueError('Checkpoint does not implement the requested arm')
    for branch in ('swin_rgb', 'swin_d'):
        prefix = 'model.model.pixel_level_module.encoder.'+branch+'.'
        projection = state[prefix+'embeddings.patch_embeddings.projection.weight']
        blocks = {int(match.group(1)) for name in state
                  if (match := re.match(re.escape(prefix)+r'encoder\.layers\.2\.blocks\.(\d+)\.', name))}
        if tuple(projection.shape) != (embed, 3, 4, 4) or blocks != set(range(depth)):
            raise ValueError('Actual Swin tensor shapes/depth disagree with the requested backbone')
    return {'embedding_channels': embed, 'third_stage_blocks': depth,
            'depth_encoding': encoding, 'camera_intrinsics_used': encoding == 'xyz',
            'base_epoch': 15, 'parameters': metadata['parameters']}


def csv_detections(path, prediction_path, ids):
    """Check CSV/JSONL semantic identity and produce fresh COCO detections."""
    spec = _Spec(ids, (1,), (1024, 1024), 100)
    previous = csv.field_size_limit()
    csv.field_size_limit(max(previous, spec.max_json_chars))
    detections, rows = [], 0
    try:
        with path.open(newline='', encoding='utf-8-sig') as stream, prediction_path.open() as raw:
            reader = csv.reader(stream, strict=True)
            if next(reader, None) != ['image_id', 'PredictionString']:
                raise ValueError('Unexpected CSV columns')
            for row in reader:
                if len(row) != 2:
                    raise ValueError('CSV row width differs')
                _check_id(row[0], rows, spec)
                values = json.loads(row[1], object_pairs_hook=_unique_object)
                values, _ = _predictions(values, spec, decode=False)
                record = json.loads(next(raw), object_pairs_hook=_unique_object)
                if record['image_id'] != ids[rows] or record['predictions'] != values:
                    raise ValueError('Exported CSV differs from the scored prediction JSONL')
                detections.extend({'image_id': ids[rows], **value} for value in values)
                rows += 1
            if rows != len(ids) or next(raw, None) is not None:
                raise ValueError('CSV/JSONL coverage differs from the official test IDs')
    finally:
        csv.field_size_limit(previous)
    return detections


def rescore(coco, detections, ids, log_path):
    with log_path.open('x') as log, contextlib.redirect_stdout(log):
        if detections:
            predicted = coco.loadRes(detections)
        else:
            predicted = COCO()
            predicted.dataset = {'info': {}, 'images': coco.dataset['images'],
                'categories': coco.dataset['categories'], 'annotations': []}
            predicted.createIndex()
        evaluator = COCOeval(coco, predicted, 'segm')
        evaluator.params.imgIds = ids
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {'mask_ap': float(evaluator.stats[0]), 'mask_ap_small': float(evaluator.stats[3])}


def audit_arm(root, arm, coco, ids, train_ids):
    folder = root/arm
    output = folder/'independent_audit.json'
    if output.exists():
        raise FileExistsError(output)
    protocol = read(folder/'protocol.json')
    completed_base(folder/'base15', protocol)
    expected = check_results(folder/'final_evaluation')
    architecture = checkpoint_architecture(folder/'base15/final.pt', arm)
    selected = read(folder/'points6/selection.json')
    if (selected['status'] != 'locked' or selected['test_used'] or selected['epochs'] != 6
            or not selected['base_unchanged'] or not selected['base_gradients_absent']
            or selected['global_steps'] != 1804*6 or selected['training_images'] != len(train_ids)):
        raise ValueError('Point training/validation selection is incomplete')
    for epoch in range(1, 7):
        coverage = [iid for rank in range(2) for iid in
                    read(folder/f'points6/coverage_epoch{epoch:02d}_rank{rank}.json')]
        if len(coverage) != len(train_ids) or set(coverage) != train_ids:
            raise ValueError('Point training did not cover the exact train split once per epoch')
    for epoch in range(1, 16):
        order = read(folder/f'base15/train_order_epoch_{epoch:02d}.json')
        if len(order['image_ids']) != len(train_ids) or set(order['image_ids']) != train_ids:
            raise ValueError('Base training image coverage differs')
        batches = [index for rank in order['rank_batches'] for batch in rank for index in batch]
        if len(batches) != len(train_ids) or set(batches) != set(range(len(train_ids))):
            raise ValueError('Distributed base batches duplicate or omit training images')
    head_path = Path(selected['selected']['checkpoint'])
    if sha(head_path) != selected['selected_head_sha256']:
        raise ValueError('Selected head checkpoint changed')
    head = torch.load(head_path, map_location='cpu', weights_only=True)
    fine = architecture['embedding_channels']+256
    if (head['fine_channels'] != fine or head['base_checkpoint_sha256'] != sha(folder/'base15/final.pt')
            or head['heads']['plain.layers.0.weight'].shape[1] != fine+1
            or head['heads']['depth.layers.0.weight'].shape[1] != fine+10):
        raise ValueError('Selected head architecture/base identity differs')
    result = {'status': 'running', 'arm': arm, 'architecture': architecture, 'rows': [],
              'scope': 'full_test_2112_csv_rescoring', 'gpu_used': False,
              'auditor_sha256': sha(__file__), 'test_used_for_selection': False}
    for variant in ('native', 'full'):
        path = folder/f'final_evaluation/test_{variant}'
        csv_path = path/'submission.csv'
        before = sha(csv_path)
        detections = csv_detections(csv_path, path/'predictions.jsonl', ids)
        metrics = rescore(coco, detections, ids, folder/f'independent_csv_{variant}_coco.log')
        reference = expected['test'][variant]['metrics']
        differences = {key: abs(value-reference[key]) for key, value in metrics.items()}
        if max(differences.values()) > 1e-12 or sha(csv_path) != before:
            raise ValueError('Independent CSV score differs from reported AP/APsmall or input changed')
        result['rows'].append({'variant': variant, 'csv': str(csv_path), 'sha256': before,
                              'instances': len(detections), 'metrics': metrics, 'absolute_differences': differences})
    result.update(status='passed', finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  exact_training_coverage=True, selected_head_epoch=selected['selected']['epoch'])
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'reports/course_depth_backbone_v1')
    parser.add_argument('--arm', choices=(*ARMS, 'all'), default='all')
    args = parser.parse_args()
    arms = ARMS if args.arm == 'all' else (args.arm,)
    torch.set_num_threads(4)
    # Fail before reading GT or doing expensive scoring if formal runs are unfinished.
    for arm in arms:
        if read(args.root/arm/'final_evaluation/run.json')['status'] != 'completed':
            raise ValueError('Formal evaluation is still incomplete: '+arm)
    protocol = read(args.root/arms[0]/'protocol.json')
    with CourseArchive(protocol['archive'], 'train') as data:
        train_ids = set(data.image_ids)
    with CourseArchive(protocol['archive'], 'test') as data:
        ids = data.image_ids
        if ids != [record['id'] for record in read(args.root/'test_images.json')['images']]:
            raise ValueError('Official test ID/order mismatch')
        coco = COCO()
        coco.dataset = data.coco_ground_truth()
        coco.createIndex()
    results = [audit_arm(args.root, arm, coco, ids, train_ids) for arm in arms]
    if torch.cuda.is_initialized():
        raise ValueError('CPU audit unexpectedly initialized CUDA')
    print(json.dumps({'status': 'passed', 'arms': [r['arm'] for r in results]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
