#!/usr/bin/env python3
"""Predict complete OSD/OCID manifests with a validation-selected UOAIS base.

Only completed training runs are accepted. This process never calls ground_truth;
score its native-size RLE output separately with score_uois_benchmark.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.physical_gpu < 0 or args.output.exists():
        parser.error('Use a nonnegative GPU index and a new output directory')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.physical_gpu)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['WANDB_MODE'] = 'disabled'
    import torch
    from tools.train_uoais_benchmark import sha, write_json, source_hashes, now
    from rgbd_seg.uoais_benchmark_training import build_model
    from rgbd_seg.uois_benchmark_data import BenchmarkDataset
    from rgbd_seg.uois_benchmark_geometry import letterbox_sample, restore_predictions
    run = json.loads((args.training_run/'run.json').read_text())
    protocol_path = args.training_run/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if sha(protocol_path) != run['protocol_sha256']:
        raise ValueError('Training protocol snapshot changed')
    if run['status'] != 'training_completed' or run['scope'] != 'full_train_val':
        raise ValueError('Final test inference requires completed full UOAIS training')
    if (run['state']['epoch'] != protocol['training']['epochs']
            or len(run['state']['validation']) != protocol['training']['epochs']):
        raise ValueError('Training epochs or full validation history are incomplete')
    hashes = source_hashes()
    if hashes != run['source_sha256']:
        raise ValueError('Model/geometry/metric source changed since training')
    weights = args.training_run/'best.pt'
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    if (checkpoint['scope'] != 'full_train_val' or checkpoint['family'] != run['family']
            or checkpoint['protocol_sha256'] != run['protocol_sha256']
            or checkpoint['source_sha256'] != hashes
            or checkpoint['selection']['scope'] != 'full_validation'
            or checkpoint['selection']['images'] != 5000):
        raise ValueError('Best checkpoint lacks the required validation selection receipt')
    data = BenchmarkDataset(args.manifest)
    expected = {'osd': 111, 'ocid': 2390}
    if len(data) != expected.get(data.manifest['dataset']):
        raise ValueError('This runner requires the complete original OSD or OCID manifest')
    torch.set_num_threads(4)
    torch.manual_seed(protocol['training']['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    family = run['family']
    model, metadata = build_model(family, protocol['models'][family]['pretrained'], protocol['training']['seed'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to('cuda').eval().requires_grad_(False)
    del checkpoint
    args.output.mkdir(parents=True)
    receipt = {'scope': 'UOAIS-Sim-trained base; complete original real test manifest',
               'family': family, 'model_metadata': metadata, 'training_run': str(args.training_run.resolve()),
               'training_run_sha256': sha(args.training_run/'run.json'),
               'training_protocol_sha256': run['protocol_sha256'],
               'selected_checkpoint': str(weights.resolve()), 'weights_sha256': sha(weights),
               'manifest': str(args.manifest.resolve()), 'manifest_sha256': sha(args.manifest),
               'dataset': data.manifest['dataset'], 'images': len(data),
               'score_saved_above': .01, 'fixed_evaluation_threshold': .5,
               'overlap_rule': 'ascending confidence; later retained mask overwrites',
               'precision': 'float32; same as the paired point/system evaluation',
               'test_used_for_selection': False,
               'source_sha256': {**hashes, str(Path(__file__).relative_to(ROOT)): sha(__file__)},
               'physical_gpu': args.physical_gpu, 'gpu': torch.cuda.get_device_name(),
               'started_utc': now()}
    write_json(args.output/'protocol.json', receipt)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode(), (args.output/'predictions.jsonl').open('x') as stream:
            for index, record in enumerate(data.records):
                sample = data.inference_sample(index)
                geometry = None
                if family == 'dpseg':
                    sample, geometry = letterbox_sample(sample)
                predictions = model.predict_sample(sample, score_threshold=.01, max_detections=100)
                if geometry is not None:
                    predictions = restore_predictions(predictions, geometry)
                predictions = sorted(predictions, key=lambda row: row['score'])
                stream.write(json.dumps({'image_id': record['id'], 'predictions': predictions}, allow_nan=False)+'\n')
                stream.flush()
                if (index+1) % 25 == 0 or index+1 == len(data):
                    progress = {'status': 'predicting', 'done': index+1, 'images': len(data),
                                'elapsed_seconds': time.monotonic()-started, 'updated_utc': now()}
                    write_json(args.output/'progress.json', progress)
                    print(json.dumps(progress), flush=True)
        write_json(args.output/'completion.json', {'status': 'completed', 'images': len(data),
                   'elapsed_seconds': time.monotonic()-started,
                   'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                   'predictions_sha256': sha(args.output/'predictions.jsonl')})
    except BaseException as error:
        write_json(args.output/'failure.json', {'status': 'failed', 'error': str(error), 'time': now()})
        raise


if __name__ == '__main__':
    main()
