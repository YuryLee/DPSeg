#!/usr/bin/env python3
"""Verify original OSD/OCID files and create a versioned evaluation manifest."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgbd_seg.uois_benchmark_data import build_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['osd', 'ocid'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--calibration', required=True, type=Path,
                        help='JSON with camera_K, depth_scale_m, source, status and optional image_size')
    parser.add_argument('--expected-images', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; keep prior dataset receipts unchanged')
    if args.expected_images <= 0:
        parser.error('expected-images must be positive')
    manifest = build_manifest(args.dataset, args.root, json.loads(args.calibration.read_text()),
                              expected_images=args.expected_images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: manifest[k] for k in ('dataset', 'images', 'objects', 'status')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
