#!/usr/bin/env python3
"""Reuse the established initial/continuation training loops for an RGB control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rgbd_seg.course_optimized import file_sha
from rgbd_seg.course_progressive_rgb import build_trainable, check_parent_checkpoint
from rgbd_seg.course_distributed_eval import evaluate, write_json
from tools import run_course_resolution_ddp as initial
from tools import run_course_longer_training as continuation

_INITIAL_SOURCES = initial.sources
_CONTINUATION_SOURCES = continuation.sources


def sources():
    result = {**_INITIAL_SOURCES(), **_CONTINUATION_SOURCES()}
    for name in ('rgbd_seg/course_progressive_rgb.py', 'tools/run_course_progressive_rgb.py',
                 'tools/supervise_course_progressive_rgb.py'):
        result[name] = file_sha(ROOT / name)
    return result


def evaluate_rgb(adapter, archive, output, *, rank, **kwargs):
    result = evaluate(adapter, archive, output, rank=rank, **kwargs)
    if rank == 0:
        path = Path(output) / 'evaluation_run.json'
        record = json.loads(path.read_text())
        record.update(depth_used=False, input_modality='RGB only; shared loader depth fields ignored')
        write_json(path, record)
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--phase', choices=('initial', 'continuation'), required=True)
    args, remaining = parser.parse_known_args()
    runner = initial if args.phase == 'initial' else continuation
    runner.build_trainable = build_trainable
    runner.sources = sources
    runner.evaluate = evaluate_rgb
    if args.phase == 'continuation':
        runner.check_parent_checkpoint = check_parent_checkpoint
    sys.argv = [sys.argv[0], *remaining]
    runner.main()


if __name__ == '__main__':
    main()
