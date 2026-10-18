#!/usr/bin/env python3
"""Audit finished paired outputs and report prespecified object-size bins."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha
from rgbd_seg.course_tiled_probability import VARIANTS


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def positive_mean(array):
    value = array[array >= 0]
    return None if not value.size else float(value.mean())


def size_bins(protocol, folder, name):
    predictions = records(folder/name/'predictions.jsonl')
    results = [{'image_id': item['image_id'], **prediction}
               for item in predictions for prediction in item['predictions']]
    with CourseArchive(protocol['archive'], 'val') as data:
        truth = data.coco_ground_truth()
        ids = data.image_ids
    truth.setdefault('info', {})
    with (folder/name/'size_bins.log').open('x') as stream, contextlib.redirect_stdout(stream):
        coco = COCO()
        coco.dataset = truth
        coco.createIndex()
        detections = coco.loadRes(results)
        evaluator = COCOeval(coco, detections, 'segm')
        evaluator.params.imgIds = ids
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.params.areaRng = [[0, 1024], [0, 64], [64.0000001, 256], [256.0000001, 1024]]
        evaluator.params.areaRngLbl = ['small', 'area_1_64', 'area_65_256', 'area_257_1024']
        evaluator.evaluate()
        evaluator.accumulate()
    rows = {}
    for area, label in enumerate(evaluator.params.areaRngLbl):
        lo, hi = evaluator.params.areaRng[area]
        precision = evaluator.eval['precision'][:, :, :, area, -1]
        recall = evaluator.eval['recall'][:, :, area, -1]
        rows[label] = {'mask_ap': positive_mean(precision), 'mask_ar': positive_mean(recall),
            'mask_ap50': positive_mean(precision[np.isclose(evaluator.params.iouThrs, .5)]),
            'mask_ap75': positive_mean(precision[np.isclose(evaluator.params.iouThrs, .75)]),
            'gt_instances_including_crowd': sum(lo <= item['area'] <= hi for item in truth['annotations']),
            'area_range': [lo, hi]}
    original = read(folder/name/'evaluation.json')['metrics']['mask_ap_small']
    if abs(original-rows['small']['mask_ap']) > 1e-10:
        raise ValueError('Independent area-bin evaluator failed to reproduce primary APsmall')
    write_json(folder/name/'size_bins.json', {'status': 'completed', 'images': len(ids),
        'metrics': rows, 'predictions_sha256': file_sha(folder/name/'predictions.jsonl'),
        'scope': 'COCO one-to-one matching; original GT area; independent RLE; maxDets100'})


def audit(protocol_path, folder):
    protocol = read(protocol_path)
    run = read(folder/'run.json')
    comparison = read(folder/'comparison.json')
    if run['status'] != 'completed' or comparison['scope'] != 'full_validation':
        raise ValueError('Require completed full validation')
    if run['protocol_sha256'] != file_sha(protocol_path) or read(folder/'exit.json')['exit_code'] != 0:
        raise ValueError('Protocol/exit binding failed')
    ids = read(folder/'image_ids.json')
    if len(ids) != 2584 or len(set(ids)) != 2584:
        raise ValueError('Full validation coverage missing')
    loaded = {}
    for name in VARIANTS:
        receipt = read(folder/name/'scored.json')
        evaluation = read(folder/name/'evaluation.json')
        path = folder/name/'predictions.jsonl'
        if (receipt['evaluation_sha256'] != file_sha(folder/name/'evaluation.json')
                or evaluation['predictions_sha256'] != file_sha(path)
                or evaluation['images'] != 2584):
            raise ValueError('Output identity differs')
        loaded[name] = records(path)
        if [item['image_id'] for item in loaded[name]] != ids:
            raise ValueError('Prediction coverage/order differs')
    preservation = {}
    for name in VARIANTS:
        if 'depth' not in name:
            continue
        base = loaded['tiled' if name.startswith('tiled') else 'native']
        count, changed = 0, 0
        for before, after in zip(base, loaded[name]):
            a, b = before['predictions'], after['predictions']
            if [(p['category_id'], p['score']) for p in a] != [(p['category_id'], p['score']) for p in b]:
                raise ValueError('Depth refinement changed score/category/order/count')
            count += len(a)
            changed += sum(p['segmentation'] != q['segmentation'] for p, q in zip(a, b))
        preservation[name] = {'images': 2584, 'instances': count, 'changed_masks': changed,
                               'scores_categories_order_count_preserved': True}
    historical = records(protocol['baseline_predictions']['path'])
    mismatch = {'images_with_mask_or_order_difference': 0, 'images_with_count_difference': 0,
                'max_score_difference': 0.}
    for before, after in zip(historical, loaded['native']):
        if before['image_id'] != after['image_id']:
            raise ValueError('Historical baseline coverage differs')
        a, b = before['predictions'], after['predictions']
        mismatch['images_with_count_difference'] += len(a) != len(b)
        mismatch['images_with_mask_or_order_difference'] += (len(a) != len(b) or any(
            p['segmentation'] != q['segmentation'] for p, q in zip(a, b)))
        mismatch['max_score_difference'] = max(mismatch['max_score_difference'],
            max((abs(p['score']-q['score']) for p, q in zip(a, b)), default=0.))
    ranks = [read(folder/f'rank{i}.json') for i in range(len(protocol['gpus']))]
    native_seconds = sum(r['timing']['native_seconds'] for r in ranks)
    extra_seconds = sum(r['timing']['tile_extra_seconds'] for r in ranks)
    timings = {}
    for name in VARIANTS:
        total = native_seconds+(extra_seconds if name.startswith('tiled') else 0)
        total += sum(r['timing']['refinement_seconds'].get(name, 0.) for r in ranks)
        timings[name] = total/2584
    report = {'status': 'passed', 'images': 2584, 'preservation_checks': preservation,
        'historical_native_comparison': mismatch, 'mean_seconds_per_image': timings,
        'images_at_100_instance_cap': {name: sum(len(item['predictions']) == 100 for item in loaded[name])
                                      for name in VARIANTS},
        'timing_scope': 'synchronized FP32 model/preprocessing/RLE/merge/refinement; excludes ZIP I/O and scoring',
        'gpu_peak_allocated_GiB': {str(r['physical_gpu']): r['peak_cuda_allocated_bytes']/2**30 for r in ranks},
        'protocol_sha256': file_sha(protocol_path), 'run_sha256': file_sha(folder/'run.json'),
        'comparison_sha256': file_sha(folder/'comparison.json'), 'auditor_sha256': file_sha(__file__),
        'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'test_accessed': False}
    write_json(folder/'audit.json', report)
    return report


def finish(protocol_path, folder):
    report = audit(protocol_path, folder)
    bins = {name: read(folder/name/'size_bins.json')['metrics'] for name in VARIANTS}
    write_json(folder/'size_comparison.json', bins)
    labels = {'native': '整图原始预测', 'native_binary_depth': '整图＋二值深度',
              'native_probability_depth': '整图＋概率深度', 'tiled': '整图＋切片',
              'tiled_binary_depth': '切片＋二值深度', 'tiled_probability_depth': '切片＋概率深度'}
    with (folder/'comparison.md').open('a') as stream:
        stream.write('\n## 面积分项与耗时\n\n| 配置 | AP 1–64 px | AP 65–256 px | AP 257–1024 px | 单图平均耗时（秒） |\n|---|---:|---:|---:|---:|\n')
        for name in VARIANTS:
            b = bins[name]
            stream.write(f"| {labels[name]} | {b['area_1_64']['mask_ap']:.6f} | {b['area_65_256']['mask_ap']:.6f} | {b['area_257_1024']['mask_ap']:.6f} | {report['mean_seconds_per_image'][name]:.3f} |\n")
        stream.write('\n这些分项分别按 COCO 规则计算，不能取平均得到 APₛ。耗时为五张 GPU 的单图测量平均，包含预处理、模型、掩码编码、去重和细化，不含读取压缩包及评测；并行吞吐率与单图耗时不同。\n')
    print(json.dumps({'status': 'passed', 'images': report['images'], 'audit': str(folder/'audit.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size-bin', choices=VARIANTS)
    args = parser.parse_args()
    if args.size_bin:
        size_bins(read(args.protocol), args.output, args.size_bin)
    else:
        finish(args.protocol, args.output)


if __name__ == '__main__':
    main()
