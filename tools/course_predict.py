#!/usr/bin/env python3
"""Run the selected RGB-D model and build a self-contained course demo.

The default is a small validation preview. --all --evaluate reuses the original
COCO evaluator on the complete validation split; preview images have no AP.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import base64
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_depth_data import load_inference_sample
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha
from rgbd_seg.course_portable import DEFAULT_MANIFEST, VARIANTS, load_model, predict_variants, read_json

LABELS = {
    'native': 'DA-Fusion 原始输出',
    'native_probability_depth': '概率图深度细化',
    'point': '普通点头',
    'point_probability_depth': '普通点头＋概率图深度细化',
    'point_depth': '深度点头',
    'point_depth_probability_depth': '深度点头＋概率图深度细化',
}


def embedded_png(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def depth_preview(sample):
    from matplotlib import colormaps
    depth = sample['depth_m'][0].cpu().numpy()
    valid = sample['depth_valid'][0].cpu().numpy()
    normalized = np.zeros_like(depth)
    if valid.any():
        low, high = np.percentile(depth[valid], [2, 98])
        normalized[valid] = np.clip((depth[valid] - low) / max(float(high - low), 1e-6), 0, 1)
    colors = (colormaps['viridis'](normalized)[..., :3] * 255).astype(np.uint8)
    colors[~valid] = 0
    return Image.fromarray(colors)


def write_gallery(output, cases, bundle, split):
    payload = {'cases': cases, 'labels': LABELS, 'validation': bundle['validation'],
               'split': split, 'base_epoch': bundle['base']['epoch'],
               'head_epoch': bundle['head']['epoch']}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    encoded = encoded.replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    template = (ROOT / 'web/course.html').read_text(encoding='utf-8')
    (output / 'index.html').write_text(template.replace('__COURSE_DATA__', encoded), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--base-weights', type=Path, required=True)
    parser.add_argument('--head-weights', type=Path, required=True)
    parser.add_argument('--coco-checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--indices', help='Comma-separated preview indices, at most eight')
    parser.add_argument('--all', action='store_true', help='Predict every official image in the split')
    parser.add_argument('--evaluate', action='store_true', help='Score a complete validation run')
    parser.add_argument('--device', default='auto', help='auto, cpu, or cuda:0 after setting visible devices')
    args = parser.parse_args()
    if args.evaluate and (not args.all or args.split != 'val'):
        parser.error('--evaluate requires --all --split val')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    bundle = read_json(args.manifest)
    state = {'status': 'loading', 'scope': 'full_split' if args.all else 'demo_subset',
             'split': args.split, 'ground_truth_in_prediction': False,
             'new_metrics_computed': False, 'started_at_unix': time.time()}
    write_json(output / 'run.json', state)
    try:
        adapter, heads, receipt = load_model(
            base_weights=args.base_weights, head_weights=args.head_weights,
            coco_checkpoint=args.coco_checkpoint, manifest=args.manifest, device=device,
        )
        write_json(output / 'model_receipt.json', receipt)
        cases, counts, ids = [], {name: 0 for name in VARIANTS}, []
        with CourseArchive(args.archive, args.split) as data, ExitStack() as stack:
            expected = bundle['dataset'][args.split + '_images']
            if len(data) != expected:
                raise ValueError('Official split size differs from the course manifest')
            previews = ([int(value) for value in args.indices.split(',')]
                        if args.indices else [0, len(data) // 4, len(data) // 2, len(data) - 1])
            if (not previews or len(previews) > 8 or len(set(previews)) != len(previews)
                    or any(value < 0 or value >= len(data) for value in previews)):
                raise ValueError('Provide one to eight distinct valid preview indices')
            indices = range(len(data)) if args.all else previews
            streams = {}
            for name in VARIANTS:
                (output / name).mkdir()
                streams[name] = stack.enter_context((output / name / 'predictions.jsonl').open('x'))
            for position, index in enumerate(indices, 1):
                image_id = data.image_ids[index]
                sample = load_inference_sample(data, image_id)
                predictions, timing = predict_variants(adapter, heads, sample)
                ids.append(image_id)
                for name, values in predictions.items():
                    counts[name] += len(values)
                    streams[name].write(json.dumps({'image_id': image_id, 'predictions': values}, allow_nan=False) + '\n')
                    streams[name].flush()
                if index in previews:
                    cases.append({'index': index, 'image_id': image_id,
                                  'rgb': embedded_png(data.load_rgb(index)),
                                  'depth': embedded_png(depth_preview(sample)),
                                  'predictions': predictions})
                if position == 1 or position % 25 == 0 or position == len(indices):
                    state.update(status='inferring', images=position, expected_images=len(indices))
                    write_json(output / 'run.json', state)
                    print(json.dumps({'phase': 'inferring', 'images': position, 'total': len(indices)}), flush=True)
            if args.all and ids != data.image_ids:
                raise ValueError('Complete inference image coverage or order differs')
            for stream in streams.values():
                stream.close()
            if args.evaluate:
                from rgbd_seg.course_dafusion_eval import score_predictions
                evaluations = {}
                for name in VARIANTS:
                    evaluations[name] = score_predictions(data, output / name / 'predictions.jsonl', output / name)['metrics']
                write_json(output / 'evaluation_summary.json', evaluations)
                bundle['validation']['metrics'] = evaluations
                selected_metrics = evaluations[bundle['validation']['selected_variant']]
                bundle['validation'].update(images=len(data), mask_ap=selected_metrics['mask_ap'],
                                            mask_ap_small=selected_metrics['mask_ap_small'])
                state['new_metrics_computed'] = True
        write_json(output / 'image_ids.json', ids)
        cases.sort(key=lambda item: previews.index(item['index']))
        write_gallery(output, cases, bundle, args.split)
        state.update(status='completed', images=len(ids), instances=counts,
                     source_sha256={'tools/course_predict.py': file_sha(__file__),
                                    'rgbd_seg/course_portable.py': file_sha(ROOT / 'rgbd_seg/course_portable.py'),
                                    'web/course.html': file_sha(ROOT / 'web/course.html')},
                     prediction_sha256={name: file_sha(output / name / 'predictions.jsonl') for name in VARIANTS},
                     finished_at_unix=time.time())
        write_json(output / 'run.json', state)
        print(json.dumps({'status': 'completed', 'images': len(ids),
                          'demo': str(output / 'index.html'), 'new_metrics_computed': state['new_metrics_computed']}))
    except BaseException as exc:
        state.update(status='failed', error=str(exc), finished_at_unix=time.time())
        write_json(output / 'run.json', state)
        raise


if __name__ == '__main__':
    main()
