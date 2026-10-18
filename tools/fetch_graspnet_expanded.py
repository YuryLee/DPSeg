#!/usr/bin/env python3
"""Expand a pinned, scene-disjoint GraspNet subset through bounded mirror ranges."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import time

import numpy as np
from PIL import Image
from scipy.io import loadmat

from xet_range import RangeReader, RangeReadError

ROOT = Path(__file__).resolve().parents[1]
REPO = 'DravenALG/GraspNet-1Billion'
REV = '1f9db26044cd2a7e629bbb465be3665364cbf2bf'
SIZE = 142475571200
XET = '051a80be3773d2bf40b98f0121f0fd20e5038dc2728d65fb40f887ada2aa1a69'
TRAIN = [2, 20]
TRAIN_FRAMES = list(range(0, 256, 8))
TEST_FRAMES = list(range(0, 256, 32))


def names(scene, frames):
    return {f'scenes/scene_{scene:04d}/realsense/{kind}/{frame:04d}.{suffix}'
            for frame in frames for kind, suffix in
            [('rgb', 'png'), ('depth', 'png'), ('label', 'png'), ('meta', 'mat')]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget-mb', type=int, default=4096)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--additional-tests', action='store_true',
                        help='Protocol amendment: add fixed 8-frame test scenes 114 and 128')
    args = parser.parse_args()
    raw = ROOT / 'data/graspnet_expanded_raw'
    report_prefix = 'graspnet_expansion_additional' if args.additional_tests else 'graspnet_expansion'
    receipt = ROOT / f'reports/{report_prefix}_download.json'
    original = json.loads((ROOT / 'reports/graspnet_labels_download.json').read_text())
    plan = {
        'train_scenes': TRAIN, 'train_frames': TRAIN_FRAMES,
        'development_scenes': [17], 'development_frames': [0, 64, 128, 192],
        'heldout_rule': 'First official test scene (100-189) encountered after the original scan position, with no model evaluation used for selection.',
        'heldout_frames': TEST_FRAMES, 'original_next_offset': original['next_offset'],
        'network_budget_bytes': args.budget_mb * 1024**2,
        'proxy': False, 'tls_verification': True,
    }
    if args.additional_tests:
        base_receipt = json.loads((ROOT / 'reports/graspnet_expansion_download.json').read_text())
        if base_receipt['status'] != 'completed':
            raise ValueError('Complete the original expansion before adding held-out scenes')
        if args.budget_mb > 2048 or args.budget_mb * 1024**2 + base_receipt['downloaded_total_bytes'] > 4096 * 1024**2:
            raise ValueError('Additional traffic must stay below 2 GiB and cumulative traffic below 4 GiB')
        plan = {'additional_test_scenes': [114, 128], 'additional_test_partition': 'seen',
                'frames': TEST_FRAMES, 'retained_test_scenes': [138],
                'selection_rule': 'First two official test scenes encountered in the original archive order, fixed uniformly spaced frames; selected before any expanded-model training or evaluation.',
                'amendment_reason': 'Scene 138 contains no COCO-small instances. Add independent scenes without selecting frames by mask size or model score; small-object coverage is not guaranteed.',
                'network_budget_bytes': args.budget_mb * 1024**2,
                'previous_network_bytes': base_receipt['downloaded_total_bytes'],
                'proxy': False, 'tls_verification': True}
    plan_path = ROOT / f'reports/{report_prefix}_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Existing selection plan differs; refuse to alter registered selection')
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    previous = json.loads(receipt.read_text()) if args.resume and receipt.exists() else {}
    if receipt.exists() and not args.resume:
        raise ValueError('Receipt exists; use --resume to preserve cumulative network budget')
    state = previous or {'status': 'running', 'repository': REPO, 'revision': REV,
        'archive_xet_hash': XET, 'archive_size': SIZE, 'proxy': False,
        'files': {}, 'stages': {}, 'encountered_scenes': [], 'selected_test_scene': None,
        'integrity': 'Pinned Xet archive identity, HTTPS/TLS, checked tar headers, PNG/MAT validation, per-file SHA256. Full archive digest is not recomputed.'}
    prior_download = state.get('downloaded_total_bytes', 0)
    remaining_budget = plan['network_budget_bytes'] - prior_download
    reader = RangeReader(REPO, REV, XET, SIZE, max_download_bytes=remaining_budget,
                         max_read_bytes=16 * 1024**2, max_cache_bytes=64 * 1024**2)
    window = b''
    window_start = -1
    started = time.time()
    prior_seconds = state.get('elapsed_total_s', 0)

    def save(status='running'):
        state.update(status=status, downloaded_total_bytes=prior_download + reader.downloaded_bytes,
                     elapsed_total_s=prior_seconds + time.time() - started)
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(receipt)

    def read(start, length):
        nonlocal window, window_start
        if start < window_start or start + length > window_start + len(window):
            for attempt in range(3):
                try:
                    new_window = reader.read_range(start, min(max(length, 16 * 1024**2), SIZE - start))
                    break
                except RangeReadError:
                    save()
                    if attempt == 2:
                        raise
                    time.sleep(2)
            window_start, window = start, new_window
        return window[start-window_start:start-window_start+length]

    def validate(name, data):
        if name.endswith('.png'):
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            array = np.asarray(Image.open(io.BytesIO(data)))
            if array.shape[:2] != (720, 1280):
                raise ValueError('Unexpected original image shape')
            if '/rgb/' in name and (array.shape != (720, 1280, 3) or array.dtype != np.uint8):
                raise ValueError('Invalid original RGB')
            if '/depth/' in name and (array.ndim != 2 or array.dtype not in (np.uint16, np.int32)):
                raise ValueError('Invalid original raw depth')
            if '/label/' in name and (array.ndim != 2 or np.max(array) > 88):
                raise ValueError('Invalid original instance IDs')
        else:
            metadata = loadmat(io.BytesIO(data))
            factor = float(np.asarray(metadata['factor_depth']).squeeze())
            if factor != 1000.0:
                raise ValueError('Unexpected depth scale')

    def store(name, data, *, tar_offset, reused=False):
        validate(name, data)
        target = raw / name
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(data).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError('Existing local file differs')
        if not target.exists():
            target.write_bytes(data)
        state['files'][name] = {'sha256': digest, 'bytes': len(data),
                                'tar_offset': tar_offset, 'reused_original_subset': reused}

    # Reuse verified, exact original bytes, including the untouched development scene.
    for name, source in original['files'].items():
        data = (ROOT / 'data/graspnet_original_subset' / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != source['sha256']:
            raise ValueError('Original subset no longer matches receipt')
        store(name, data, tar_offset=source['tar_offset'], reused=True)
    save()

    def find_depth_start(scene):
        known_positions = [value['tar_offset'] for name, value in original['files'].items()
                           if f'scene_{scene:04d}/realsense/depth/' in name]
        # Inspect all aligned candidate headers in a bounded prefix ending at a known
        # header; only accept the exact depth directory. Subsequent walking follows
        # checked tar sizes, so no directory's unrecorded contents are skipped.
        if known_positions:
            start = max(0, min(known_positions) - 512 - 16 * 1024**2)
        else:
            # Rounded directory positions recorded during the original tar walk.
            # They are search hints only: accept an exact checked directory header.
            hints = {114: 725_000_000, 128: 1_463_000_000}
            start = max(0, (hints[scene] // 512) * 512 - 8 * 1024**2)
        prefix = read(start, min(16 * 1024**2, SIZE-start))
        expected = f'scenes/scene_{scene:04d}/realsense/depth'
        found = []
        for position in range(0, len(prefix) - 511, 512):
            block = prefix[position:position+512]
            if expected.encode() not in block:
                continue
            try:
                info = tarfile.TarInfo.frombuf(block, 'utf-8', 'strict')
            except (tarfile.HeaderError, ValueError):
                continue
            if info.name.lstrip('./').rstrip('/') == expected and info.isdir() and info.size == 0:
                found.append(start + position)
        if len(found) != 1:
            raise ValueError('Could not uniquely locate checked depth directory header')
        return found[0]

    def walk(stage, start, scene=None, discover_test=False):
        progress = state['stages'].setdefault(stage, {'next_offset': start, 'completed': False})
        if progress['completed']:
            return progress['next_offset']
        offset = progress['next_offset']
        longname = None
        count = 0
        selected = scene if scene is not None else state['selected_test_scene']
        wanted = names(selected, TEST_FRAMES if discover_test else TRAIN_FRAMES) if selected is not None else set()
        wanted -= set(state['files'])
        previous_scene = None
        while offset < SIZE:
            header = read(offset, 512)
            if header == b'\0' * 512:
                raise ValueError('Reached archive end before completing selection')
            info = tarfile.TarInfo.frombuf(header, encoding='utf-8', errors='strict')
            payload = offset + 512
            next_offset = payload + ((info.size + 511)//512)*512
            if not offset < next_offset <= SIZE:
                raise ValueError('Invalid tar payload extent')
            if info.type == tarfile.GNUTYPE_LONGNAME:
                longname = read(payload, info.size).rstrip(b'\0').decode()
            elif info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
                override = re.search(r'\d+ path=([^\n]+)', read(payload, info.size).decode())
                if override:
                    longname = override[1]
            else:
                name = (longname or info.name).lstrip('./')
                longname = None
                match = re.match(r'scenes/scene_(\d{4})(?:/|$)', name)
                current_scene = int(match[1]) if match else None
                if current_scene is not None and current_scene != previous_scene:
                    previous_scene = current_scene
                    if current_scene not in state['encountered_scenes']:
                        state['encountered_scenes'].append(current_scene)
                    print(json.dumps({'stage': stage, 'scene': current_scene, 'offset': offset,
                                      'download_mb': round((prior_download+reader.downloaded_bytes)/1e6, 1)}), flush=True)
                if discover_test and selected is None and current_scene is not None and 100 <= current_scene <= 189:
                    selected = current_scene
                    state['selected_test_scene'] = selected
                    state['test_partition'] = 'seen' if selected < 130 else ('similar' if selected < 160 else 'novel')
                    wanted = names(selected, TEST_FRAMES) - set(state['files'])
                    save()
                    print(json.dumps({'selected_heldout_scene': selected, 'partition': state['test_partition']}), flush=True)
                if name in wanted:
                    if not info.isfile() or not 0 < info.size < 10_000_000:
                        raise ValueError('Unexpected selected file entry')
                    store(name, read(payload, info.size), tar_offset=payload)
                    wanted.remove(name)
                    print(json.dumps({'fetched': name, 'remaining_in_stage': len(wanted)}), flush=True)
            offset = next_offset
            progress['next_offset'] = offset
            count += 1
            if count % 50 == 0:
                save()
            if selected is not None and not wanted:
                progress['completed'] = True
                save()
                return offset
        raise ValueError('Archive exhausted')

    try:
        if args.additional_tests:
            state['additional_test_scenes'] = [114, 128]
            state['test_partition'] = 'seen'
            for scene in [114, 128]:
                stage = f'test_scene_{scene:04d}'
                start = state['stages'].get(stage, {}).get('next_offset')
                if start is None:
                    start = find_depth_start(scene)
                walk(stage, start, scene=scene, discover_test=True)
            save('completed')
            return
        final_train_offset = original['next_offset']
        for scene in [20, 2]:
            stage = f'train_scene_{scene:04d}'
            start = state['stages'].get(stage, {}).get('next_offset')
            if start is None:
                start = find_depth_start(scene)
            finish = walk(stage, start, scene=scene)
            final_train_offset = max(final_train_offset, finish)
            print(json.dumps({'training_scene_ready': scene, 'frame_count': len(TRAIN_FRAMES)}), flush=True)
        state['training_data_ready'] = True
        save()
        walk('heldout', final_train_offset, discover_test=True)
        save('completed')
    except BaseException:
        save('incomplete')
        raise
    finally:
        reader.close()


if __name__ == '__main__':
    main()
