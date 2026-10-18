#!/usr/bin/env python3
"""Evaluate a frozen checkpoint on separate confirmation data, globally/by scene."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pycocotools import mask as mask_utils
import torch
from rgbd_seg.data import RGBDDataset, dataset_fingerprint
from rgbd_seg.evaluation import evaluate_predictions
from rgbd_seg.model import load_model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SelectedImages:
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices
        self.classes = dataset.classes
        self.samples = [dataset.samples[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite results: {output}')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    dataset = RGBDDataset(args.data, 'test')
    if not len(dataset):
        raise ValueError('Confirmation test split is empty')
    if dataset.manifest['provenance'] != 'public_tless_subset':
        raise ValueError('This frozen confirmation protocol requires the separate T-LESS subset')
    fingerprint = dataset_fingerprint(args.data)
    checkpoint_hash = sha256(args.checkpoint)
    run = json.loads((Path(args.checkpoint).parent/'run.json').read_text())
    if run['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Checkpoint differs from completed training record')
    model, checkpoint = load_model(args.checkpoint, args.device)
    if checkpoint['classes'] != dataset.classes:
        raise ValueError('Checkpoint and dataset classes must match exactly')
    if (run.get('exit_status') != 'completed' or run['config'] != checkpoint['config']
            or run['manifest_sha256'] != checkpoint['manifest_sha256']
            or run['provenance'] != 'public_graspnet_subset'
            or checkpoint['provenance'] != 'public_graspnet_subset'
            or run['dataset_sha256'] == fingerprint):
        raise ValueError('Expected a completed, matching GraspNet training record and distinct test data')
    predictions, exported, times, scene_indices = [], [], [], {}
    cuda = args.device.startswith('cuda')
    with torch.inference_mode():
        warm = {k:v.to(args.device) for k,v in dataset[0][0].items()}
        for _ in range(3):
            model([warm])
        for index in range(len(dataset)):
            sample, _ = dataset[index]
            if cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            pred = model([{k:v.to(args.device) for k,v in sample.items()}])[0]
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
            pred = {k:v.cpu() for k,v in pred.items()}
            # COCO only consumes masks thresholded at 0.5. Preserve that exact
            # binary decision, scores and labels; avoid retaining float masks.
            pred['masks'] = pred['masks'] >= .5
            predictions.append(pred)
            record = dataset.samples[index]
            scene = record.get('scene_id', record.get('source_scene', record.get('scene')))
            if scene is None:
                raise ValueError('Confirmation samples must carry an explicit scene identifier')
            scene = str(scene)
            scene_indices.setdefault(scene, []).append(index)
            instances = []
            for mask, score, label in zip(pred['masks'][:,0], pred['scores'], pred['labels']):
                encoded = mask_utils.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))
                instances.append({'score':float(score), 'label':int(label),
                                  'rle':{'size':encoded['size'], 'counts':encoded['counts'].decode('ascii')}})
            exported.append({'sample_id':record['id'], 'instances':instances})
            print(json.dumps({'completed_images':index+1, 'total':len(dataset)}), flush=True)
    result = {'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'checkpoint':str(Path(args.checkpoint).resolve()), 'checkpoint_sha256':checkpoint_hash,
        'training_dataset_sha256':run['dataset_sha256'], 'config':checkpoint['config'],
        'training_provenance':run['provenance'], 'training_manifest_sha256':run['manifest_sha256'],
        'manifest_sha256':sha256(Path(args.data)/'manifest.json'), 'dataset_sha256':fingerprint,
        'provenance':dataset.manifest['provenance'], 'split':'test', 'samples':len(dataset),
        'metrics':evaluate_predictions(dataset,predictions), 'per_scene':{},
        'protocol':{'type':'Frozen-weight cross-dataset stress test; no fitting or threshold tuning',
            'classes':dataset.classes, 'gt':'Original separate visible instance masks, including overlaps',
            'mask_threshold':.5, 'candidate_score_threshold':checkpoint['config']['score_threshold'],
            'max_detections':checkpoint['config']['max_detections'],
            'metric':'COCO segmentation AP/AR, 0–100. Global scores are recomputed across all images, not averaged scene AP.',
            'small':'COCO area <=1024 at original image resolution; includes occluded visible fragments.',
            'comparison_limit':'Separate from GraspNet and official identity-aware BOP scores; correlated views are not independent replicates.',
            'extra_metrics':'Not computed: converting overlapping GT into one label map would discard annotation pixels.'},
        'timing':{'device':args.device, 'threads':4, 'warmup':3, 'batch':1, 'precision':'float32',
            'mean_ms':1000*float(np.mean(times)),
            'scope':'In-memory tensors through transfer, model, transforms and postprocessing; excludes IO, RLE and evaluation.'}}
    for scene, indices in scene_indices.items():
        subset = SelectedImages(dataset,indices)
        areas = [float(area) for i in indices for area in dataset[i][1]['area']]
        result['per_scene'][scene] = {'images':len(indices),'instances':len(areas),
            'small_instances':sum(0<area<=1024 for area in areas),
            'metrics':evaluate_predictions(subset,[predictions[i] for i in indices])}
    if dataset_fingerprint(args.data) != fingerprint or sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError('Input bytes changed during evaluation')
    prediction_file = output/'predictions_rle.json'
    prediction_file.write_text(json.dumps({'dataset_sha256':fingerprint,'checkpoint_sha256':checkpoint_hash,
        'mask_threshold':.5,'samples':exported},separators=(',',':'))+'\n')
    result['predictions_sha256'] = sha256(prediction_file)
    result['source_sha256'] = {str(p.relative_to(ROOT)):sha256(p) for p in
        (Path(__file__).resolve(),ROOT/'rgbd_seg/data.py',ROOT/'rgbd_seg/evaluation.py',ROOT/'rgbd_seg/model.py')}
    (output/'evaluation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'metrics':result['metrics'],'output':str(output)},indent=2))


if __name__ == '__main__':
    main()
