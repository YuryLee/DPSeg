#!/usr/bin/env python3
"""Check reverse conversion against the pinned official forward converter.

Reads local files only. The selected reference class is parsed from the exact
official Git blob; the reference module's imports/CLI/network code never run.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import logging
from pathlib import Path
from pprint import pformat
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import load_file
from rgbd_seg.mask2former_native_init import hf_to_native, duplicate_native_backbones

REFERENCE_BLOB = 'ea1c578509f60bb6fcb07a373d82635188444dc8'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference_class(path):
    source = path.read_bytes()
    git_blob = hashlib.sha1(b'blob ' + str(len(source)).encode() + b'\0' + source).hexdigest()
    if git_blob != REFERENCE_BLOB:
        raise ValueError('Reference is not the pinned official v4.37.2 converter')
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
             and node.name == 'OriginalMask2FormerCheckpointToOursConverter']
    if len(nodes) != 1:
        raise ValueError('Expected one official mapping class')
    module = ast.Module(body=[*ast.parse('from __future__ import annotations').body, nodes[0]], type_ignores=[])
    namespace = {'torch': torch, 'logger': logging.getLogger('reference_converter'), 'pformat': pformat}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[nodes[0].name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--save-native-init', type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.save_native_init and args.save_native_init.exists()):
        raise FileExistsError('Use new output paths')
    config_path = args.checkpoint / 'config.json'
    weights_path = args.checkpoint / 'model.safetensors'
    config = json.loads(config_path.read_text())
    state = load_file(weights_path)
    native, receipt = hf_to_native(state, config)
    converter = reference_class(args.reference)(None, SimpleNamespace(
        backbone_config=SimpleNamespace(depths=config['backbone_config']['depths']),
        encoder_layers=config['encoder_layers'], decoder_layers=config['decoder_layers']))
    remaining, core, classes = dict(native), {}, {}
    converter.replace_pixel_module(core, remaining)
    converter.replace_transformer_module(core, remaining)
    converter.replace_universal_segmentation_module(classes, remaining)
    if remaining:
        raise ValueError('Official forward conversion left unused native tensors')
    restored = {'model.' + name: value for name, value in core.items()}
    restored.update(classes)
    expected = set(state) - {'criterion.empty_weight'}
    if set(restored) != expected:
        raise ValueError('Round-trip tensor coverage differs')
    for name in restored:
        if not torch.equal(restored[name], state[name]):
            raise ValueError('Official round trip changed tensor: ' + name)
    initialized, policy = duplicate_native_backbones(native, drop_queries=True)
    result = {'status': 'passed', 'scope': 'initialization conversion, not DA-Fusion training or benchmark',
              'checkpoint': str(args.checkpoint.resolve()), 'weights_sha256': sha(weights_path),
              'config_sha256': sha(config_path), 'reference_git_blob': REFERENCE_BLOB,
              'reference_sha256': sha(args.reference), 'source_sha256': sha(ROOT / 'rgbd_seg/mask2former_native_init.py'),
              'script_sha256': sha(Path(__file__)), 'official_roundtrip_exact_tensors': len(restored),
              'conversion': receipt, 'initialization_policy': policy,
              'native_model_load_validated': False}
    if args.save_native_init:
        args.save_native_init.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': initialized, 'initialization_receipt': {
            'hf_weights_sha256': result['weights_sha256'], 'reference_git_blob': REFERENCE_BLOB,
            'drop_queries': True, 'trained_on_UOAIS': False}}, args.save_native_init)
        result['native_initialization'] = {'path': str(args.save_native_init.resolve()),
                                          'sha256': sha(args.save_native_init)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'official_roundtrip_exact_tensors', 'native_model_load_validated')}))


if __name__ == '__main__':
    main()
