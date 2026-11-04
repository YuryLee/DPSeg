#!/usr/bin/env python3
"""Run frozen course DPSeg on OSD/OCID as an explicitly labelled transfer test.

This does not train on UOAIS-Sim or reproduce DA-Fusion paper results. Saved
native-size RLE rows are consumed by score_uois_benchmark.py independently.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_seg.uois_benchmark_data import BenchmarkDataset, file_sha256


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--base-weights', type=Path, required=True)
    parser.add_argument('--head-weights', type=Path, required=True)
    parser.add_argument('--model-manifest', type=Path, default=ROOT / 'configs/course_e15_point.json')
    parser.add_argument('--coco-init', type=Path, default=ROOT / 'pretrained/mask2former_swin_tiny_coco_instance')
    parser.add_argument('--physical-gpu', type=int, required=True)
    parser.add_argument('--stages', choices=('base', 'all'), default='all')
    parser.add_argument('--max-images', type=int, help='Engineering subset only; recorded as partial coverage')
    parser.add_argument('--max-seconds', type=int, default=14400)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; use a new directory')
    if args.physical_gpu < 0 or args.max_seconds <= 0 or (args.max_images is not None and args.max_images <= 0):
        parser.error('Invalid GPU, image limit or timeout')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.physical_gpu)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['WANDB_MODE'] = 'disabled'
    import torch
    from rgbd_seg.course_portable import load_model
    from rgbd_seg.course_progressive_ablation import predict_rgbd
    from rgbd_seg.uois_benchmark_geometry import letterbox_sample, restore_predictions

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly the declared physical GPU must be visible')
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = BenchmarkDataset(args.manifest)
    count = len(data) if args.max_images is None else min(args.max_images, len(data))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    progress_path = args.output / 'progress.json'
    write_json(progress_path, {'status': 'loading_model', 'images': count, 'completed_images': 0})
    try:
        model, heads, receipt = load_model(
            base_weights=args.base_weights, head_weights=args.head_weights,
            coco_checkpoint=args.coco_init, manifest=args.model_manifest, device='cuda:0')
        stages = ['rgbd_fusion'] if args.stages == 'base' else [
            'rgbd_fusion', 'depth_point', 'depth_point_tiled', 'depth_point_tiled_probability']
        source_files = [Path(__file__), ROOT / 'rgbd_seg/uois_benchmark_data.py',
                        ROOT / 'rgbd_seg/uois_benchmark_geometry.py',
                        ROOT / 'rgbd_seg/course_progressive_ablation.py']
        protocol = {
            'scope': 'course-trained frozen model transfer; NOT UOAIS-Sim-trained comparison',
            'training_dataset': 'course synthetic RGB-D components',
            'dataset': data.manifest['dataset'], 'images': count, 'manifest_images': len(data),
            'coverage': 'complete manifest' if count == len(data) else 'engineering subset',
            'image_ids': [r['id'] for r in data.records[:count]],
            'manifest': str(args.manifest.resolve()), 'manifest_sha256': file_sha256(args.manifest),
            'models': receipt, 'stages': stages,
            'calibration': data.calibration, 'annotation': data.manifest['annotation_policy'],
            'input_geometry': 'aspect-preserving center letterbox 1024; RGB bilinear, depth/valid nearest-exact; K half-pixel affine',
            'output_geometry': 'crop padding; nearest-exact binary masks to original resolution; remove empty only; preserve score and order',
            'selection': 'no fitting, checkpoint selection or parameter selection on test images',
            'physical_gpu': args.physical_gpu, 'gpu': torch.cuda.get_device_name(0),
            'torch': torch.__version__, 'started_utc': datetime.now(timezone.utc).isoformat(),
            'command': sys.argv, 'max_seconds': args.max_seconds,
            'sources_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in source_files},
        }
        write_json(args.output / 'protocol.json', protocol)
        removed = {stage: 0 for stage in stages}
        torch.cuda.reset_peak_memory_stats()
        with ExitStack() as stack, torch.inference_mode():
            streams = {}
            for stage in stages:
                directory = args.output / stage
                directory.mkdir()
                streams[stage] = stack.enter_context((directory / 'predictions.jsonl').open('x'))
            for index in range(count):
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError('Declared inference timeout exceeded')
                sample, geometry = letterbox_sample(data.inference_sample(index))
                if args.stages == 'base':
                    values = {'rgbd_fusion': model.predict_sample(sample)}
                else:
                    values, _ = predict_rgbd(model, heads, sample)
                for stage in stages:
                    restored = restore_predictions(values[stage], geometry)
                    removed[stage] += len(values[stage]) - len(restored)
                    streams[stage].write(json.dumps({'image_id': data.records[index]['id'],
                                                    'predictions': restored}, allow_nan=False) + '\n')
                    streams[stage].flush()
                progress = {'status': 'running', 'completed_images': index + 1,
                            'images': count, 'seconds': time.monotonic() - started}
                write_json(progress_path, progress)
                print(json.dumps(progress), flush=True)
        torch.cuda.synchronize()
        completion = {
            'status': 'completed', 'images': count, 'manifest_images': len(data),
            'coverage': protocol['coverage'], 'scope': protocol['scope'],
            'seconds_including_load_and_serialization': time.monotonic() - started,
            'peak_allocated_bytes_after_load': torch.cuda.max_memory_allocated(),
            'removed_empty_after_unpadding': removed,
            'predictions_sha256': {stage: file_sha256(args.output / stage / 'predictions.jsonl') for stage in stages},
            'protocol_sha256': file_sha256(args.output / 'protocol.json'),
        }
        write_json(args.output / 'completion.json', completion)
        write_json(progress_path, completion)
        print(json.dumps(completion), flush=True)
    except BaseException as error:
        previous = json.loads(progress_path.read_text())
        write_json(progress_path, {**previous, 'status': 'failed', 'error': str(error),
                                   'seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
