#!/usr/bin/env python3
"""Audit one completed 5000-image paired point validation epoch without a GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rgbd_seg.uoais_point_validation_audit import audit_point_epoch, file_sha, load_point_validation_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point-run', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.epoch <= 5 or args.output.exists():
        parser.error('Use epoch1..5 and a new output file')
    point = args.point_run.resolve()
    if not (point / f'val_epoch{args.epoch:02d}' / 'summary.json').is_file():
        raise ValueError('Completed paired validation summary is not available yet')
    run = json.loads((point / 'run.json').read_text())
    protocol = json.loads((point / 'protocol.json').read_text())
    canonical = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (run['scope'] != 'full_train_val' or protocol['scope'] != 'full_train_val'
            or run['expected_steps'] != 56250 or protocol['training']['epochs'] != 5
            or protocol['training']['global_batch'] != 4 or protocol['training']['world_size'] != 2
            or protocol['test_used'] is not False or canonical != run['protocol_sha256']
            or protocol['source_sha256'] != run['source_sha256']):
        raise ValueError('Expected the identified complete-budget paired point experiment')
    for name, expected in run['source_sha256'].items():
        if file_sha(ROOT / name) != expected:
            raise ValueError(f'Frozen training source changed: {name}')
    base = Path(protocol['base']['base_run'])
    if base.resolve() != (point.parent / 'uoais_dpseg_swint_20ep').resolve():
        raise ValueError('Point run belongs to another benchmark base directory')
    if file_sha(base / 'best.pt') != protocol['base']['checkpoint_sha256']:
        raise ValueError('Point base checkpoint changed')
    if file_sha(base / 'protocol.json') != protocol['base']['protocol_sha256']:
        raise ValueError('Point base protocol changed')
    ids, cohort = load_point_validation_cohort(base.parent)
    receipt = audit_point_epoch(point / f'val_epoch{args.epoch:02d}', ids, args.epoch)
    receipt.update(protocol_canonical_sha256=canonical, cohort=cohort,
                   base_checkpoint_sha256=protocol['base']['checkpoint_sha256'],
                   audit_sources_sha256={str(path): file_sha(path) for path in
                       [Path(__file__).resolve(), ROOT / 'rgbd_seg/uoais_point_validation_audit.py']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': receipt['status'], 'epoch': args.epoch, 'images': receipt['images'],
                      'max_difference': receipt['maximum_summary_difference'], 'receipt': str(args.output.resolve())}))


if __name__ == '__main__':
    main()
