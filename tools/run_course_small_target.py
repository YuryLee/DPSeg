#!/usr/bin/env python3
"""Run the sampling-only variant through the unchanged full DA-Fusion runner."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_course_dafusion as base
from rgbd_seg.course_small_target import build_trainable

_base_sources = base.runtime_source_sha256


def runtime_source_sha256():
    result = _base_sources()
    for name in ('rgbd_seg/course_small_target.py', 'tools/run_course_small_target.py',
                 'tools/launch_course_small_target.py'):
        result[name] = base.sha256(ROOT/name)
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    known, _ = parser.parse_known_args()
    def build_adapter(name, checkpoint, device, seed):
        if name != 'dafusion':
            raise ValueError('Sampling variant requires DA-Fusion')
        adapter, metadata = build_trainable(checkpoint, device=device, seed=seed)
        adapter.sampling_audit_path = known.output.resolve()/'sampling_log.jsonl'
        return adapter, metadata
    # Local module entry points only; no mutation of on-disk baseline sources.
    base.build_adapter = build_adapter
    base.runtime_source_sha256 = runtime_source_sha256
    base.main()


if __name__ == '__main__':
    main()
