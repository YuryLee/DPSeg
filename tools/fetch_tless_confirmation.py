#!/usr/bin/env python3
"""Fetch a pinned T-LESS confirmation archive from the mirror under a 1 GB cap."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from xet_range import RangeReader, RangeReadError

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/tless_download'
REPORT = ROOT / 'reports/v3/tless_download.json'
PLAN = ROOT / 'reports/v3/tless_confirmation_plan.json'
REPO = 'bop-benchmark/tless'
REV = '5fd309a04476a842d93abfb584fba9ee7caecdf1'
BUDGET = 1_000_000_000
FILES = [
    ('tless_base.zip', 49597, 'dd70ca884b7c471a530a952f70c5ab2c212f3d2c2f371be86397442b97d70a7e',
     'd7539b66c7bc60c9c7791879a5217bed8d153f12da744f9614ad554380af4bb8'),
    ('tless_test_primesense_bop19.zip', 825276992, '1a18f6bbfb5ac4ced8529f7a35225adfed88c0f62ef38067933e2b541ef1d00b',
     'a6c47ad5ccfcdd3c8f731ff3a033df931072866d6775c179a6d00e2ea7cabdf8'),
]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    if not PLAN.exists():
        PLAN.write_text(json.dumps({
            'registered_utc': datetime.now(timezone.utc).isoformat(),
            'purpose': 'External T-LESS confirmation/stress test of frozen GraspNet models. No training or threshold tuning on this data.',
            'repository': REPO, 'revision': REV, 'mirror': 'https://hf-mirror.com',
            'network_budget_bytes': BUDGET, 'proxy': False, 'tls_verification': True,
            'split': 'test only; separate dataset, never pool with GraspNet scores',
            'selection_levels_frames_per_scene': [5, 10, 20, 50, 100, 'all'],
            'frame_order': 'ascending original available frame IDs per scene',
            'selection_within_scene': 'numpy.linspace(0, n-1, min(k,n), dtype=int); include all frames at the final level',
            'stopping_rule': 'First level with at least 50 small visible instance observations, contributed by at least 3 scene IDs and 3 object IDs; otherwise retain all available and report target not reached.',
            'small_definition': '0 < original visible mask pixel area < 32**2; original resolution unchanged',
            'annotation_inclusion': 'Keep every nonempty visible instance mask in every selected frame; no selection of individual objects.',
            'selection_information': 'Original archive metadata/labels only. No model predictions or evaluation scores.',
            'dependence_note': 'Repeated viewpoints of one scene/object remain correlated; report scenes and model-object IDs, never label views independent physical objects.',
            'depth': 'Read each scene_camera.json depth_scale (raw to millimetres), then multiply by 0.001 for metres. Verify matching image dimensions.',
            'mask_overlap': 'Visible masks must be disjoint for exact conversion to one ID PNG; refuse conversion if conflicts would discard pixels.',
        }, indent=2) + '\n')
    state = json.loads(REPORT.read_text()) if REPORT.exists() else {
        'repository': REPO, 'revision': REV, 'mirror': 'https://hf-mirror.com',
        'proxy': False, 'tls_verification': True, 'network_budget_bytes': BUDGET,
        'downloaded_bytes': 0, 'files': {}, 'status': 'started',
        'source': 'https://bop.felk.cvut.cz/datasets/', 'license': 'CC BY 4.0',
    }
    base_used = state['downloaded_bytes']
    cumulative = 0
    started = time.monotonic()
    reader = None

    def save(status):
        state.update(status=status, downloaded_bytes=base_used+cumulative+(reader.downloaded_bytes if reader else 0),
                     updated_utc=datetime.now(timezone.utc).isoformat())
        REPORT.write_text(json.dumps(state, indent=2) + '\n')

    try:
        for name, size, expected, xet_hash in FILES:
            target, partial = DEST / name, DEST / (name + '.partial')
            if not target.exists():
                reader = RangeReader(REPO, REV, xet_hash, size,
                                     max_download_bytes=BUDGET-base_used-cumulative,
                                     max_read_bytes=16*1024**2)
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > size:
                    raise ValueError('Unexpected existing partial file size')
                with partial.open('ab') as handle:
                    while offset < size:
                        for attempt in range(3):
                            try:
                                data = reader.read_range(offset, min(16*1024**2, size-offset))
                                break
                            except RangeReadError:
                                save('incomplete')
                                if attempt == 2:
                                    raise
                                time.sleep(2)
                        handle.write(data)
                        handle.flush()
                        offset += len(data)
                        state['files'][name] = {'status': 'downloading', 'bytes': offset,
                                                'expected_bytes': size, 'expected_sha256': expected}
                        save('downloading')
                        print(json.dumps({'file': name, 'bytes': offset, 'network_bytes': state['downloaded_bytes']}), flush=True)
                cumulative += reader.downloaded_bytes
                reader.close()
                reader = None
                partial.rename(target)
            with target.open('rb') as handle:
                actual = hashlib.file_digest(handle, 'sha256').hexdigest()
            if target.stat().st_size != size or actual != expected:
                raise ValueError('Archive size/SHA256 mismatch; do not extract')
            state['files'][name] = {'status': 'verified', 'bytes': size, 'sha256': actual, 'xet_hash': xet_hash}
            save('downloading')
        state['elapsed_this_run_s'] = time.monotonic() - started
        save('completed')
    except BaseException:
        save('incomplete')
        raise
    finally:
        if reader:
            reader.close()


if __name__ == '__main__':
    main()
