#!/usr/bin/env python3
"""Independent crop or Focal arm using the frozen resolution training engine."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import run_course_resolution_ddp as engine
from rgbd_seg.course_crop import CROP_POLICY, CourseCropTrainingDataset
from rgbd_seg.course_focal import FOCAL_POLICY
from rgbd_seg.course_optimized import build_trainable as build_optimized

BASE_SOURCES = engine.sources


def sources():
    result = BASE_SOURCES()
    for name in ('rgbd_seg/course_crop.py', 'rgbd_seg/course_focal.py',
                 'rgbd_seg/course_optimized.py', 'tools/run_course_crop_focal_ddp.py',
                 'tools/launch_course_crop_focal_ddp.py'):
        result[name] = engine.sha256(ROOT/name)
    return result


def install_audit(adapter, *, method, output, rank):
    """Record actual successful training forwards, without adding model state."""
    counter = {'mask_calls_before': 0}

    def before(module, args):
        counter['mask_calls_before'] = getattr(module.model.criterion, 'mask_loss_calls', 0)

    def after(module, args, result):
        if not module.training:
            return
        samples, targets = args
        rows = []
        for sample, target in zip(samples, targets):
            crop = target.get('crop_info', {'applied': False})
            if method == 'crop' and 'crop_info' not in target:
                raise RuntimeError('Crop arm did not use the crop training dataset')
            if method == 'focal' and crop['applied']:
                raise RuntimeError('Focal-only arm received a cropped image')
            rows.append({'image_id': int(target['image_id'].item()),
                'input_hw': list(sample['image'].shape[-2:]),
                'instances': len(target['labels']), 'crop': crop})
        calls = getattr(module.model.criterion, 'mask_loss_calls', 0)-counter['mask_calls_before']
        if method == 'focal' and calls < 1:
            raise RuntimeError('Focal criterion was not used in the training forward')
        record = {'rank': rank, 'method': method, 'images': rows, 'focal_mask_loss_calls': calls}
        with (output/f'method_audit_rank{rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')

    adapter.register_forward_pre_hook(before)
    adapter.register_forward_hook(after)


def configure(method, output, rank):
    if method not in ('crop', 'focal'):
        raise ValueError('Choose exactly one intervention: crop or focal')

    def build(*args, **kwargs):
        adapter, metadata = build_optimized(*args, method=method, **kwargs)
        install_audit(adapter, method=method, output=output, rank=rank)
        return adapter, metadata

    if method == 'crop':
        # Only this worker process is changed. The class itself is top-level and
        # importable, so DataLoader spawn workers can unpickle it safely.
        import rgbd_seg.course_depth_data as data
        data.CourseDepthTrainingDataset = CourseCropTrainingDataset
    engine.build_trainable = build
    engine.sources = sources


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, required=True)
    parser.add_argument('--smoke-steps', type=int, default=0)
    args, _ = parser.parse_known_args()
    protocol = json.loads(args.protocol.read_text())
    method = protocol['intervention']['method']
    expected = {'crop': CROP_POLICY, 'focal': FOCAL_POLICY}.get(method)
    if expected is None or protocol['intervention']['policy'] != expected:
        raise ValueError('Protocol intervention does not match the implementation')
    if not args.smoke_steps:
        selection = protocol['resolution_selection']
        decision_path = Path(selection['path'])
        if engine.sha256(decision_path) != selection['sha256']:
            raise ValueError('Resolution decision changed')
        decision = json.loads(decision_path.read_text())
        if decision['status'] != 'completed' or decision['selected_resolution'] != args.resolution:
            raise ValueError('Complete resolution comparison before either formal intervention')
    configure(method, args.output.resolve(), int(os.environ['RANK']))
    return engine.main()


if __name__ == '__main__':
    raise SystemExit(main())
