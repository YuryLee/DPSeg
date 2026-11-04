#!/usr/bin/env python3
"""Find qualitative examples in frozen S0-S4 validation or local test predictions."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import heapq
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pycocotools import mask as coco_mask
from rgbd_seg.course_archive import CourseArchive

ARMS = ('mask2former_finetune', 'rgbd_fusion', 'depth_point',
        'depth_point_tiled', 'depth_point_tiled_probability')
LABELS = ('S0 · Mask2Former', 'S1 · + RGB-D 融合', 'S2 · + 深度点细化',
          'S3 · + 切片推理', 'S4 · + 概率图深度细化')
EVALUATION = ROOT / 'reports/course_progressive_ablation/v1/evaluation'
ARCHIVE = Path('/data/yinli/workspace/downloads/buaa_image_segmentation/blender.zip')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def paths(split='val'):
    return {f'S{i}': EVALUATION / ('test' if split == 'test' else ('rgb_val' if i == 0 else 'rgbd_val'))
            / arm / 'predictions.jsonl' for i, arm in enumerate(ARMS)}


def scan(output, threshold=.5, split='val'):
    output.mkdir(parents=True, exist_ok=False)
    bindings = {}
    count = 2112 if split == 'test' else 2584
    for stage, path in paths(split).items():
        digest = sha(path)
        receipt = json.loads((path.parent / 'evaluation.json').read_text())
        if receipt['images'] != count or receipt['split'] != split or receipt['predictions_sha256'] != digest:
            raise ValueError('Predictions are not the completed scored outputs')
        bindings[stage] = dict(path=str(path), sha256=digest)
    pools = defaultdict(list)
    serial = 0

    def add(kind, priority, row):
        nonlocal serial
        serial += 1
        entry = (float(priority), serial, dict(kind=kind, priority=float(priority), **row))
        heapq.heappush(pools[kind], entry)
        if len(pools[kind]) > 240:
            heapq.heappop(pools[kind])

    began = time.monotonic()
    with CourseArchive(ARCHIVE, split) as data, ExitStack() as stack:
        truth = data.coco_ground_truth()
        annotations = defaultdict(list)
        for ann in truth['annotations']:
            if ann['area'] > 0 and not ann.get('iscrowd', 0):
                annotations[ann['image_id']].append(ann)
        streams = {stage: stack.enter_context(path.open()) for stage, path in paths(split).items()}
        for index, image_id in enumerate(data.image_ids):
            gt = annotations[image_id]
            gtrle = [a['segmentation'] for a in gt]
            predictions, matrices, best, argbest = {}, {}, {}, {}
            for stage, stream in streams.items():
                record = json.loads(next(stream))
                if record['image_id'] != image_id:
                    raise ValueError('Unexpected prediction order')
                values = record['predictions']
                predictions[stage] = values
                m = (coco_mask.iou([v['segmentation'] for v in values], gtrle, [0]*len(gt))
                     if gt and values else np.zeros((len(values), len(gt))))
                matrices[stage] = m
                visible = np.array([v['score'] >= threshold for v in values])
                filtered = m.copy()
                filtered[~visible] = 0
                best[stage] = filtered.max(0) if len(values) else np.zeros(len(gt))
                argbest[stage] = filtered.argmax(0) if len(values) else np.full(len(gt), -1)
            for g, ann in enumerate(gt):
                area = ann['area']
                if area < 64 or area > 6000:
                    continue
                ious = {stage: float(v[g]) for stage,v in best.items()}
                indices = {stage: int(v[g]) if best[stage][g] > 0 else None
                           for stage,v in argbest.items()}
                row = dict(image_id=image_id, image_index=index, annotation_id=ann['id'],
                           gt_area=area, bbox=ann['bbox'], ious=ious,
                           prediction_indices=indices, file_name=data.images[index]['file_name'])
                fill = area / max(1, ann['bbox'][2] * ann['bbox'][3])
                if 400 <= area <= 1024 and min(ious.values()) > .90 and ious['S4'] > .98 and fill > .5:
                    add('success_small', ious['S4'] + min(ious.values())*.1 + area/1024*.01, row)
                if 2000 <= area <= 6000 and min(ious.values()) > .90 and ious['S4'] > .98 and fill > .5:
                    add('success_component', ious['S4'] + min(ious.values())*.1 + area/6000*.01, row)
                if area <= 1024 and ious['S0'] < .15 and ious['S4'] > .8:
                    add('small_baseline', ious['S4']-ious['S0'], row)
                if area <= 1024 and ious['S2'] < .2 and ious['S3'] > .7:
                    add('small_tiled', ious['S3']-ious['S2'], row)
                if area <= 4096 and .35 < ious['S1'] < .8 and ious['S2'] > .85:
                    if indices['S1'] == indices['S2']:
                        add('point_boundary', ious['S2']-ious['S1'], row)
                if .45 < ious['S3'] < .85 and ious['S4'] > .9:
                    if indices['S3'] == indices['S4']:
                        add('probability_boundary', ious['S4']-ious['S3'], row)
                if area <= 4096 and ious['S3'] > .5 and ious['S4'] < ious['S3'] - .15:
                    if indices['S3'] == indices['S4']:
                        add('regression', ious['S3']-ious['S4'], row)
                if 128 <= area <= 1024 and ious['S0'] < .1 and ious['S4'] < .3:
                    add('unresolved', min(area, 512)/512 - ious['S4'], row)
            for pred_index, pred in enumerate(predictions['S0']):
                iou = float(matrices['S0'][pred_index].max()) if gt else 0.
                area = int(coco_mask.area(pred['segmentation']))
                if pred['score'] >= .65 and iou < .1 and 32 <= area <= 16000:
                    row = dict(image_id=image_id, image_index=index, prediction_index=pred_index,
                        score=pred['score'], prediction_area=area, max_gt_iou=iou,
                        bbox=coco_mask.toBbox(pred['segmentation']).tolist(),
                        file_name=data.images[index]['file_name'])
                    add('false_positive', pred['score'] * min(1., area/150), row)
            if index == 0 or (index+1) % 100 == 0 or index+1 == len(data):
                progress = dict(images=index+1, expected=len(data),
                                seconds=time.monotonic()-began,
                                candidate_counts={k:len(v) for k,v in pools.items()})
                (output / 'progress.json').write_text(json.dumps(progress))
                print(json.dumps(progress), flush=True)
        if any(next(stream, None) is not None for stream in streams.values()):
            raise ValueError('Extra prediction records')
    result = dict(status='completed', split=split, images=count, display_threshold=threshold,
        selection='post-hoc qualitative screening; not a random sample or a new AP evaluation',
        matching='best mask IoU among predictions above the common display threshold; not COCO AP assignment',
        source_predictions=bindings, archive=str(ARCHIVE),
        candidates={kind:[v[2] for v in sorted(pool, reverse=True)] for kind,pool in pools.items()},
        elapsed_seconds=time.monotonic()-began, source_sha256=sha(__file__))
    (output / 'candidates.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'completed':str(output/'candidates.json')}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--split', choices=('val','test'), default='val')
    args = parser.parse_args()
    scan(args.output, args.threshold, args.split)
