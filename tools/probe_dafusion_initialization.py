#!/usr/bin/env python3
"""Check every transferred native DA-Fusion weight and whitelist fresh modules.

CPU only, no training, downloads, inference scores or external logging.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--conversion-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['HF_HUB_OFFLINE'] = '1'
    vendor = ROOT / 'third_party/DA-Fusion'
    source = json.loads((vendor / 'source_receipt.json').read_text())
    for name, record in source['files'].items():
        if sha(vendor / name) != record['sha256']:
            raise ValueError('Changed native model source: ' + name)
    sys.path.insert(0, str(vendor / 'dafusion'))
    import torch
    from PIL import Image
    if not hasattr(Image, 'LINEAR'):
        Image.LINEAR = Image.Resampling.BILINEAR
    import dafusion
    from dafusion.config import add_dafusion_config
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    torch.set_num_threads(4)
    torch.manual_seed(17)
    receipt = json.loads(args.conversion_receipt.read_text())
    if receipt['status'] != 'passed' or sha(args.weights) != receipt['native_initialization']['sha256']:
        raise ValueError('Initialization differs from its verified conversion receipt')
    cfg = get_cfg()
    add_dafusion_config(cfg)
    config = vendor / 'dafusion/configs/dafusion_rgbd_uoais_stage1.yaml'
    cfg.merge_from_file(str(config))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = ''
    cfg.freeze()
    model = build_model(cfg)
    initial = torch.load(args.weights, map_location='cpu', weights_only=True)['model']
    if getattr(initial, '_metadata', {}).get('sem_seg_head', {}).get('version') != 2:
        raise ValueError('Version 2 state-dict metadata is required to avoid the legacy head conversion')
    class AuditedCheckpointer(DetectionCheckpointer):
        def _load_model(self, checkpoint):
            incompatible = super()._load_model(checkpoint)
            self.incompatible = incompatible
            return incompatible

    checkpointer = AuditedCheckpointer(model)
    checkpointer.load(str(args.weights.resolve()))
    result = checkpointer.incompatible
    fresh_prefixes = ('backbone.fusions.', 'backbone.depth_adapter.',
                     'sem_seg_head.predictor.class_embed.',
                     'sem_seg_head.predictor.query_feat.', 'sem_seg_head.predictor.query_embed.')
    unexplained = [key for key in result.missing_keys
                   if key != 'criterion.empty_weight' and not key.startswith(fresh_prefixes)]
    if result.unexpected_keys or unexplained:
        raise ValueError(f'Unexplained load mismatch: missing={unexplained}, unexpected={result.unexpected_keys}')
    actual = model.state_dict()
    for key, value in initial.items():
        if not torch.equal(actual[key], value):
            raise ValueError('Native tensor not loaded exactly: ' + key)
    branches = 0
    for key, value in actual.items():
        if key.startswith('backbone.swin_rgb.'):
            other = 'backbone.swin_d.' + key.removeprefix('backbone.swin_rgb.')
            if not torch.equal(value, actual[other]) or value.data_ptr() == actual[other].data_ptr():
                raise ValueError('Branches must start equal in independent storage: ' + key)
            branches += 1
    output = {'status': 'passed', 'scope': 'COCO initialization load, no UOAIS training or benchmark',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'source_commit': source['commit'], 'config_sha256': sha(config),
              'initialization_sha256': sha(args.weights),
              'conversion_receipt_sha256': sha(args.conversion_receipt),
              'script_sha256': sha(Path(__file__)), 'torch': torch.__version__,
              'load_route': 'Detectron2 DetectionCheckpointer.load (same route as native training)',
              'model_parameters': sum(p.numel() for p in model.parameters()),
              'exactly_loaded_tensors': len(initial), 'each_backbone_tensors': branches,
              'fresh_tensors': list(result.missing_keys), 'unexplained_missing': unexplained,
              'unexpected_tensors': list(result.unexpected_keys),
              'branches_equal_and_independent': True, 'formal_training_started': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({key: output[key] for key in ('status', 'exactly_loaded_tensors', 'each_backbone_tensors', 'branches_equal_and_independent')}))


if __name__ == '__main__':
    main()
