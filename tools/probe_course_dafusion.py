#!/usr/bin/env python3
"""Bounded GPU capacity/gradient check using training images only; no checkpoint."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rgbd_seg.course_dafusion import build_trainable
from rgbd_seg.course_depth_data import CourseDepthTrainingDataset
from tools.run_course_dafusion import epoch_order, make_optimizer, runtime_source_sha256


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '6':
        raise ValueError('Engineering probe is restricted to the checked idle physical GPU 6')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.85)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(17)
    report = {'scope': 'engineering capacity and gradient check, training images only',
              'microbatch': 4, 'source_sha256': runtime_source_sha256(), 'steps': []}
    path = ROOT / 'reports/course_dafusion/gpu_capacity.json'
    try:
        adapter, metadata = build_trainable(device='cuda:0')
        optimizer, _ = make_optimizer(adapter, metadata, .0002)
        data = CourseDepthTrainingDataset('/data/yinli/workspace/downloads/buaa_image_segmentation/blender.zip')
        order = epoch_order(len(data), 17, 0)
        for step in range(3):
            pairs = [data[index] for index in order[step*4:(step+1)*4]]
            samples, targets = zip(*pairs)
            torch.cuda.synchronize()
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses = adapter(samples, targets)
                loss = sum(losses.values())
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
            offset_gradients = {name: float(p.grad.float().norm()) for name, p in adapter.named_parameters()
                               if 'offset_net' in name and p.grad is not None}
            if not offset_gradients or not all(v > 0 for v in offset_gradients.values()):
                raise RuntimeError('Expected nonzero gradients for all learned sampling offsets')
            optimizer.step()
            torch.cuda.synchronize()
            report['steps'].append({'loss': float(loss), 'grad_norm': float(norm),
                'seconds_without_data': time.perf_counter()-began,
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                'offset_parameter_gradients': offset_gradients})
            print(json.dumps({k: v for k, v in report['steps'][-1].items() if k != 'offset_parameter_gradients'}), flush=True)
        data.close()
        report['status'] = 'pass'
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
