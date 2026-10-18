#!/usr/bin/env python3
"""Prepare frozen T-LESS RGB-D confirmation data without flattening visible masks."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from zipfile import ZipFile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_seg.data import audit_dataset


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    output = ROOT / 'data/tless_confirmation'
    report_dir = ROOT / 'reports/v3'
    receipt = json.loads((report_dir / 'tless_download.json').read_text())
    plan = json.loads((report_dir / 'tless_confirmation_plan.json').read_text())
    if receipt['status'] != 'completed':
        raise ValueError('Archive acquisition has not completed')
    if output.exists():
        raise FileExistsError('Refuse to overwrite the frozen confirmation dataset')
    for name, row in receipt['files'].items():
        path = ROOT / 'data/tless_download' / name
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if path.stat().st_size != row['bytes'] or digest != row['sha256']:
            raise ValueError('Downloaded archive no longer matches its verified receipt')
    archive_path = ROOT / 'data/tless_download/tless_test_primesense_bop19.zip'
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Archive contains duplicate member names')
        available = defaultdict(list)
        for name in names:
            match = re.fullmatch(r'test_primesense/(\d{6})/rgb/(\d{6})\.png', name)
            if match:
                available[int(match[1])].append(int(match[2]))
        metadata = {}
        metadata_bytes = {}
        for scene in sorted(available):
            metadata[scene] = {}
            for kind in ['scene_camera', 'scene_gt', 'scene_gt_info']:
                name = f'test_primesense/{scene:06d}/{kind}.json'
                metadata_bytes[name] = archive.read(name)
                metadata[scene][kind] = json.loads(metadata_bytes[name])
        levels = []
        selected = None
        for level in plan['selection_levels_frames_per_scene']:
            candidate = {}
            for scene, frames in sorted(available.items()):
                frames = sorted(frames)
                count = len(frames) if level == 'all' else min(level, len(frames))
                candidate[scene] = [frames[i] for i in np.linspace(0, len(frames)-1, count, dtype=int)]
            small, scene_ids, object_ids = 0, set(), set()
            for scene, frames in candidate.items():
                for frame in frames:
                    gt = metadata[scene]['scene_gt'][str(frame)]
                    info = metadata[scene]['scene_gt_info'][str(frame)]
                    if len(gt) != len(info):
                        raise ValueError('GT and GT-info instance counts disagree')
                    for index, obj in enumerate(info):
                        if 0 < obj['px_count_visib'] < 1024:
                            small += 1
                            scene_ids.add(scene)
                            object_ids.add(gt[index]['obj_id'])
            count = sum(map(len, candidate.values()))
            reached = small >= 50 and len(scene_ids) >= 3 and len(object_ids) >= 3
            levels.append({'frames_per_scene': level, 'images': count, 'small_by_upstream_metadata': small,
                           'small_scenes': len(scene_ids), 'small_object_ids': sorted(object_ids), 'target_reached': reached})
            if count > plan['maximum_selected_images_before_root_review']:
                (report_dir / 'tless_selection_needs_review.json').write_text(json.dumps(levels, indent=2) + '\n')
                raise ValueError('Selection would exceed registered frame cap; report before conversion')
            if reached or level == 'all':
                selected = candidate
                break
        if selected is None:
            raise ValueError('No selection level was completed')
        selection = {'registered_plan': 'reports/v3/tless_confirmation_plan.json',
                     'frozen_utc': datetime.now(timezone.utc).isoformat(), 'levels_considered': levels,
                     'selected_frames_by_scene': selected, 'selection_used_model_outputs': False,
                     'representation_amendment': 'Original visible masks overlap. Preserve each original binary PNG in instance_masks; do not flatten or change pixel areas.'}
        (report_dir / 'tless_frozen_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
        staging = Path(tempfile.mkdtemp(prefix='.tless_confirmation-', dir=output.parent))
        provenance = {'archives': receipt['files'], 'selected_member_files': {}, 'converted_files': {},
                      'metadata_area_discrepancies': [], 'empty_visible_instances_omitted': [],
                      'depth_scale_raw_to_mm': [], 'original_shapes': [], 'per_scene': {},
                      'small_observations': [], 'original_mask_overlap': {'frames': 0, 'overlap_pixel_count': 0}}
        def copy_member(member, relative):
            data = archive.read(member)  # ZipFile also checks CRC32 on decompression.
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            provenance['selected_member_files'][member] = {'sha256': sha(data), 'bytes': len(data), 'copied_to': relative}
            provenance['converted_files'][relative] = {'sha256': sha(data), 'bytes': len(data)}
            return data
        manifest = {'provenance': 'public_tless_subset', 'source_dataset': 'T-LESS / BOP19 PrimeSense subset',
                    'source_url': 'https://bop.felk.cvut.cz/datasets/',
                    'original_dataset_url': 'https://cmp.felk.cvut.cz/t-less/',
                    'mirror': 'https://hf-mirror.com/datasets/bop-benchmark/tless',
                    'source_revision': receipt['revision'], 'license': 'CC BY 4.0',
                    'classes': ['object'], 'class_scheme': 'category_agnostic',
                    'split_protocol': 'External confirmation only: frozen original test frames; no training or tuning',
                    'annotation_format': 'instance_masks: list of original visible binary PNGs. Overlap retained exactly; zero-area invisible instances omitted.',
                    'area_note': 'COCO uses decoded visible mask area at original resolution. The registered selection uses strict area <1024; report also includes exactly 1024 and COCO <=1024 counts.',
                    'dependence_note': 'Viewpoints from the same static scene remain correlated; model-object identity is not an independent physical-object count.',
                    'selection_record': 'reports/v3/tless_frozen_selection.json',
                    'selected_frames_by_scene': selected, 'samples': []}
        scale_values, shapes = set(), set()
        try:
            for scene, frames in selected.items():
                summary = {'images': len(frames), 'instances': 0, 'small_strict_lt1024': 0,
                           'exactly1024': 0, 'coco_small_le1024': 0, 'object_ids': set()}
                provenance['per_scene'][str(scene)] = summary
                for kind in ['scene_camera', 'scene_gt', 'scene_gt_info']:
                    member = f'test_primesense/{scene:06d}/{kind}.json'
                    copy_member(member, f'source_metadata/{scene:06d}/{kind}.json')
                for frame in frames:
                    sid = f'tless_{scene:06d}_{frame:06d}'
                    prefix = f'test_primesense/{scene:06d}'
                    camera = metadata[scene]['scene_camera'][str(frame)]
                    factor_mm = float(camera['depth_scale'])
                    if not np.isfinite(factor_mm) or factor_mm <= 0:
                        raise ValueError('Invalid depth unit metadata')
                    scale_values.add(factor_mm)
                    rgb_name, depth_name = f'{prefix}/rgb/{frame:06d}.png', f'{prefix}/depth/{frame:06d}.png'
                    rgb_path, depth_path = f'rgb/{sid}.png', f'depth/{sid}.png'
                    rgb_bytes = copy_member(rgb_name, rgb_path)
                    depth_bytes = copy_member(depth_name, depth_path)
                    rgb = np.asarray(Image.open(io.BytesIO(rgb_bytes)))
                    depth = np.asarray(Image.open(io.BytesIO(depth_bytes)))
                    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                        raise ValueError('Unexpected original RGB format')
                    shape = tuple(rgb.shape[:2])
                    if depth.shape != shape or depth.dtype != np.uint16:
                        raise ValueError('Raw depth must be aligned uint16 at original RGB resolution')
                    shapes.add(shape)
                    sample = {'id': sid, 'split': 'test', 'rgb': rgb_path, 'depth': depth_path,
                              'scene_id': scene, 'source_scene': scene, 'source_frame': frame,
                              'camera_K': camera['cam_K'], 'instance_masks': []}
                    coverage = np.zeros(shape, np.uint16)
                    gt = metadata[scene]['scene_gt'][str(frame)]
                    info = metadata[scene]['scene_gt_info'][str(frame)]
                    for index, obj in enumerate(gt):
                        member = f'{prefix}/mask_visib/{frame:06d}_{index:06d}.png'
                        mask_path = f'masks/{sid}/{index:06d}.png'
                        mask_bytes = copy_member(member, mask_path)
                        mask = np.asarray(Image.open(io.BytesIO(mask_bytes)))
                        if mask.shape != shape or mask.ndim != 2 or not np.isin(mask, [0, 255]).all():
                            raise ValueError('Unexpected original binary visible-mask format')
                        binary = mask > 0
                        area = int(binary.sum())
                        if area != info[index]['px_count_visib']:
                            provenance['metadata_area_discrepancies'].append({'scene': scene, 'frame': frame,
                                'instance_index': index, 'metadata_px_count_visib': info[index]['px_count_visib'],
                                'decoded_original_mask_area': area})
                        coverage += binary
                        if not area:
                            provenance['empty_visible_instances_omitted'].append({'scene': scene, 'frame': frame, 'instance_index': index})
                            continue
                        object_id = int(obj['obj_id'])
                        if not 1 <= object_id <= 30:
                            raise ValueError('Unknown T-LESS identity')
                        sample['instance_masks'].append({'path': mask_path, 'label': 1,
                            'object_id': object_id, 'source_instance_index': index,
                            'source_visible_fraction': info[index]['visib_fract'], 'original_area': area})
                        summary['instances'] += 1
                        summary['object_ids'].add(object_id)
                        summary['small_strict_lt1024'] += area < 1024
                        summary['exactly1024'] += area == 1024
                        summary['coco_small_le1024'] += area <= 1024
                        if area < 1024:
                            provenance['small_observations'].append({'scene': scene, 'frame': frame,
                                'source_instance_index': index, 'object_id': object_id, 'pixels': area,
                                'visible_fraction': info[index]['visib_fract']})
                    overlap = int((coverage > 1).sum())
                    provenance['original_mask_overlap']['frames'] += overlap > 0
                    provenance['original_mask_overlap']['overlap_pixel_count'] += overlap
                    manifest['samples'].append(sample)
                summary['object_ids'] = sorted(summary['object_ids'])
            if len(scale_values) != 1:
                raise ValueError('Manifest requires one depth scale; original per-frame scales differ')
            manifest['depth_scale'] = next(iter(scale_values)) * .001
            manifest['depth_scale_source'] = 'scene_camera.json depth_scale is raw-to-mm; multiply by 0.001 to obtain metres'
            provenance['depth_scale_raw_to_mm'] = sorted(scale_values)
            provenance['original_shapes'] = [list(shape) for shape in sorted(shapes)]
            provenance['small_instance_count_strict_lt1024'] = len(provenance['small_observations'])
            provenance['exactly1024_count'] = sum(x['exactly1024'] for x in provenance['per_scene'].values())
            provenance['coco_small_count_le1024'] = sum(x['coco_small_le1024'] for x in provenance['per_scene'].values())
            provenance['small_scene_ids'] = sorted({x['scene'] for x in provenance['small_observations']})
            provenance['small_object_ids'] = sorted({x['object_id'] for x in provenance['small_observations']})
            provenance['small_scene_object_pairs'] = len({(x['scene'], x['object_id']) for x in provenance['small_observations']})
            provenance['small_visible_fraction_below_0_1'] = sum(x['visible_fraction'] < .1 for x in provenance['small_observations'])
            provenance['registered_coverage_target_reached_from_decoded_masks'] = (
                len(provenance['small_observations']) >= 50 and len(provenance['small_scene_ids']) >= 3
                and len(provenance['small_object_ids']) >= 3)
            if not provenance['registered_coverage_target_reached_from_decoded_masks']:
                raise ValueError('Original mask counts fail selected level target; review metadata discrepancy before proceeding')
            with ZipFile(ROOT / 'data/tless_download/tless_base.zip') as base:
                for name in ['tless/dataset_info.md', 'tless/camera_primesense.json']:
                    path = staging / 'source_metadata' / Path(name).name
                    path.write_bytes(base.read(name))
            (staging / 'SOURCE.md').write_text(
                '# T-LESS confirmation subset\n\n'
                'Source: official BOP/T-LESS repository, mirrored through hf-mirror.com. '
                'License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). '
                'Credit: Tomas Hodan, Pavel Haluza, Stepan Obdrzalek, Jiri Matas, Manolis Lourakis, Xenophon Zabulis. '
                'T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects, WACV 2017.\n\n'
                'Original dataset: https://cmp.felk.cvut.cz/t-less/\n'
                'BOP distribution: https://bop.felk.cvut.cz/datasets/\n\n'
                'RGB, depth and mask PNGs are copied exactly from the verified upstream BOP archive. '
                'Only mask_visib files are targets; amodal mask files are not used. '
                'Overlapping visible masks remain separate instances. Zero-area invisible masks are retained on disk as provenance but omitted from evaluation targets. '
                'All nonempty visible masks are included, including highly occluded fragments. '
                'This custom category-agnostic COCO evaluation is not an official BOP leaderboard score.\n')
            (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            staging.rename(output)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    (report_dir / 'tless_data_integrity.json').write_text(json.dumps(provenance, indent=2) + '\n')
    audit = audit_dataset(output)
    audit['coco_small_count_le1024'] = provenance['coco_small_count_le1024']
    audit['exactly1024_count'] = provenance['exactly1024_count']
    audit['small_count_selection_lt1024'] = provenance['small_instance_count_strict_lt1024']
    (report_dir / 'tless_confirmation_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'images': len(manifest['samples']),
                      'instances': audit['instances'], 'small': provenance['coco_small_count_le1024'],
                      'small_scenes': len(provenance['small_scene_ids']),
                      'small_object_ids': len(provenance['small_object_ids']),
                      'depth_scale_metres': manifest['depth_scale'],
                      'dataset_sha256': audit['dataset_sha256']}, indent=2))


if __name__ == '__main__':
    main()
