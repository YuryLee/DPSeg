#!/usr/bin/env python3
"""Verify and publish the seven-row current DPSeg baseline/removal comparison.

Incomplete runs stay null. This command never trains, predicts, changes a
protocol, submits to Kaggle, or changes the guarded640 default. Use --publish
to update the main table and its documentation after all available audits pass.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RGB_PROTOCOL_SHA = 'df6abf969308e031d1ee31eee78f875457c5513988743ecbe377887b388dbfcd'
METRICS = ('mask_ap_small', 'mask_ar_small', 'mask_ap')
SEMANTICS = ('iou_type', 'max_detections', 'iou_thresholds', 'area_ranges',
             'ground_truth_area', 'prediction_area', 'gt_instances', 'gt_small_including_crowd')
SPECS = (
    ('mask2former', 'Mask2Former', 'baseline', 'RGB 单分支基线，不带点细化、概率图细化或切片。'),
    ('dafusion', 'DA-Fusion', 'baseline', '本项目适配的 RGB-D 融合基座，不带两种掩码细化或切片。'),
    ('without_rgbd_fusion', 'w/o RGB-D 融合', 'independent_removal',
     '移除双分支 RGB-D 特征融合，复用 RGB 第15轮基座；保留两种深度细化、四切片及当前合并，另训 RGB 专属点头。'),
    ('without_point', 'w/o 深度感知点细化', 'independent_removal',
     '保留 RGB-D 融合、当前四切片合并和概率图细化，仅跳过可学习点细化。'),
    ('without_probability', 'w/o 深度引导概率图细化', 'independent_removal',
     '保留 RGB-D 融合、当前四切片合并和点细化，仅跳过概率图细化。'),
    ('without_tiles', 'w/o 切片增强推理', 'independent_removal',
     '仅使用整图，保留 RGB-D 融合和两种深度细化；不涉及跨视图合并。'),
    ('full', 'DPSeg（完整模型）', 'full',
     'RGB-D 融合、两种深度细化、整图＋四个640切片，以及整图优先＋截断处理＋一致性合并。'),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def receipt(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def verify_artifact(item):
    require(sha(item['path']) == item['sha256'], 'Artifact changed: ' + item['path'])


def check_sources(mapping):
    for name, expected in mapping.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path if (ROOT / path).is_file() else ROOT / 'rgbd_seg' / path
        require(sha(path) == expected, 'Source changed: ' + name)


def check_geometry(config):
    require(config['strategy'] == 'global_guard_edge_consensus'
            and config['tile_size_px'] == 640 and config['tiles'] == 4
            and config['include_whole_image'] and config['views_including_whole'] == 5
            and config['overlap'] == .4 and config['point_refinement']
            and config['probability_depth_refinement'] and config['max_detections'] == 100,
            'The current guarded640 definition changed')


def bind_csv(csv_path, prediction_path, ids):
    """Recheck every ID, category, score and compressed mask against scored input."""
    previous = csv.field_size_limit()
    rows = instances = 0
    try:
        csv.field_size_limit(max(previous, 64 * 1024 * 1024))
        with Path(csv_path).open(encoding='utf-8-sig', newline='') as values, Path(prediction_path).open() as predictions:
            reader = csv.reader(values, strict=True)
            require(next(reader, None) == ['image_id', 'PredictionString'], 'Unexpected CSV columns')
            for row, line in zip_longest(reader, predictions):
                require(row is not None and line is not None and rows < len(ids), 'CSV/prediction row count differs')
                item = json.loads(line)
                require(len(row) == 2 and row[0] == str(item['image_id']) == str(ids[rows]), 'CSV image/order differs')
                require(json.loads(row[1]) == item['predictions'], 'CSV masks or scores differ from scored input')
                require(len(item['predictions']) <= 100, 'Too many instances')
                instances += len(item['predictions'])
                rows += 1
    finally:
        csv.field_size_limit(previous)
    require(rows == len(ids), 'Incomplete CSV binding')
    return rows, instances


def verify_prediction_ids(path, ids):
    seen, instances = [], 0
    with Path(path).open() as stream:
        for line in stream:
            item = json.loads(line)
            seen.append(item['image_id'])
            instances += len(item['predictions'])
    require(seen == ids, 'Scored split ID/order differs')
    return instances


def checked_result(row, ids, expected_semantics, *, depth_used):
    require(row['images'] == len(ids) == 2112, 'Only complete test results may enter the table')
    for name in ('evaluation', 'predictions', 'csv'):
        require(sha(row[name]) == row[name + '_sha256'], name + ' changed')
    evaluated = read(row['evaluation'])
    require(evaluated['status'] == 'completed' and evaluated['split'] == 'test'
            and evaluated['images'] == len(ids) and evaluated['metrics'] == row['metrics']
            and evaluated['predictions_sha256'] == row['predictions_sha256']
            and evaluated['predicted_instances'] == row['predicted_instances']
            and evaluated['depth_used'] is depth_used, 'Evaluation/result identity differs')
    require({key: evaluated[key] for key in SEMANTICS} == expected_semantics, 'COCO settings or GT statistics differ')
    require(all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
                for value in row['metrics'].values()), 'Invalid full-test metric')
    check_sources(evaluated['source_sha256'])
    parent = Path(row['evaluation']).parent
    candidates = [parent / 'csv_validation.json', parent / 'submission_validation.json']
    valid_path = next((p for p in candidates if p.is_file()), None)
    require(valid_path is not None, 'Missing full RLE validation receipt')
    valid = read(valid_path)
    require(valid['sha256'] == row['csv_sha256'] and valid['rows'] == 2112
            and valid['predictions'] == row['predicted_instances'] and valid['id_order_verified']
            and valid['rle_runs_verified'] and valid['rle_native_decode_verified']
            and valid['image_size'] == [1024, 1024] and valid['category_ids'] == [1]
            and valid['max_detections'] == 100, 'Invalid CSV validation receipt')
    check_sources(valid.get('source_sha256', {}))
    count, instances = bind_csv(row['csv'], row['predictions'], ids)
    require(count == row['csv_rows_identical_to_scored_predictions'] == 2112
            and instances == row['predicted_instances'], 'Incomplete CSV-to-score binding')
    return receipt(valid_path)


def pending_state(run):
    state_path = run / 'status.json'
    state = read(state_path) if state_path.exists() else {'status': 'not_started'}
    require(state.get('status') != 'completed' or (run / 'results.json').is_file(),
            'A completed run has no final results')
    phase = state.get('status', 'unknown')
    label = ('failed' if phase == 'failed' else 'pending' if phase == 'not_started' else 'in_progress')
    return {'status': label, 'run_phase': phase, 'metrics': None, 'images': None, 'artifacts': None,
            'progress_observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'run_directory': str(run), 'status_receipt': receipt(state_path) if state_path.exists() else None}


def audit_rgb_training(run, ids, semantics, current, checkpoints):
    """Audit the complete six-epoch RGB-specific head, not engineering weights."""
    import torch

    protocol_path = run / 'protocol.json'
    require(sha(protocol_path) == RGB_PROTOCOL_SHA, 'RGB head protocol differs from the authorized frozen run')
    plan, result, state = read(protocol_path), read(run / 'results.json'), read(run / 'status.json')
    require(state['status'] == 'completed' and state['exit_code'] == 0 and result['status'] == 'completed',
            'RGB head/test controller has not completed')
    for name in ('protocol', 'base', 'head', 'head_selection'):
        verify_artifact(result[name])
    check_sources(plan['system_source_sha256'])
    for name in ('base_checkpoint', 'rgb_base_protocol', 'base_training_receipt',
                 'current_default_manifest', 'reference_system_protocol', 'native_val_reference'):
        verify_artifact(plan[name])
    require(plan['base_checkpoint']['sha256'] == checkpoints['rgb']['sha256']
            and plan['base_checkpoint_sha256'] == checkpoints['rgb']['sha256']
            and result['base'] == plan['base_checkpoint']
            and not plan['new_base_training'] and plan['completed_training_reused'], 'RGB base was not reused')
    check_geometry(plan['inference'])
    require(plan['inference'] == current['default'] and plan['image_ids']['test'] == ids
            and len(plan['image_ids']['train']) == 14432 and len(plan['image_ids']['val']) == 2584
            and plan['training']['epochs'] == 6 and plan['training']['global_batch'] == 8
            and plan['training']['batch_per_rank'] == 4 and plan['training']['seed'] == 17
            and plan['training']['precision'] == 'float32' and not plan['evaluation']['test_used_for_selection']
            and not plan['model']['rgbd_feature_fusion'] and plan['model']['base_modality'] == 'RGB'
            and plan['model']['depth_used_in_point_refinement'] and plan['model']['depth_used_in_probability_refinement']
            and not plan['model']['camera_intrinsics_used'], 'RGB ablation/training policy differs')
    base_run = read(plan['base_training_receipt']['path'])
    require(base_run['status'] == 'completed' and base_run['epoch'] == 15
            and base_run['training_image_exposures'] == 14432 * 15
            and base_run['final_checkpoint_sha256'] == checkpoints['rgb']['sha256'], 'RGB base training incomplete')
    for phase in ('engineering', 'training', 'evaluation'):
        require(read(run / f'{phase}_exit.json')['exit_code'] == 0, phase + ' did not exit successfully')
    gate = read(run / 'engineering_gate.json')
    require(gate['status'] == 'passed' and gate['protocol_sha256'] == RGB_PROTOCOL_SHA, 'RGB engineering gate differs')
    verify_artifact(gate['training_smoke'])
    smoke = read(gate['training_smoke']['path'])
    require(smoke['epochs'] == 1 and smoke['global_steps'] == 2 and smoke['training_images'] == 16,
            'Engineering smoke/formal training separation differs')
    require(len(gate['native_checks']) == 4, 'Missing engineering views')
    for item in gate['native_checks']:
        stats = item['stats']
        require(item['native_exact'] and item['zero_head_exact'] and stats['model_views'] == 5
                and stats['geometry']['tile_size_px'] == 640 and stats['actual_model_input_size_verified']
                and stats['merge_strategy'] == 'global_guard_edge_consensus', 'RGB engineering geometry mismatch')
    points = run / 'points6'
    training_run, selection = read(points / 'run.json'), read(points / 'selection.json')
    require(training_run['status'] == 'completed' and training_run['phase'] == 'training'
            and training_run['protocol_sha256'] == RGB_PROTOCOL_SHA, 'Formal training run incomplete')
    require(selection['status'] == 'locked' and not selection['test_used'] and selection['epochs'] == 6
            and selection['training_images'] == 14432 and selection['global_steps'] == 10824
            and selection['base_unchanged'] and selection['base_gradients_absent']
            and selection['head_replicas_equal'] and selection['base_checkpoint_sha256'] == checkpoints['rgb']['sha256'],
            'RGB point training did not meet the full six-epoch contract')
    history = read(points / 'validation_history.json')
    require([row['epoch'] for row in history] == list(range(1, 7)), 'Missing full validation epochs')
    evidence = []
    for epoch, row in enumerate(history, 1):
        require(row['global_step'] == 1804 * epoch, 'Unexpected head-training update budget')
        coverage = []
        for rank in range(2):
            cover_path = points / f'coverage_epoch{epoch:02d}_rank{rank}.json'
            cover = read(cover_path)
            require(len(cover) == len(set(cover)) == 7216, 'Training rank coverage incomplete')
            coverage.extend(cover)
            log_path = points / f'train_epoch{epoch:02d}_rank{rank}.jsonl'
            count = 0
            with log_path.open() as stream:
                for step, line in enumerate(stream, 1):
                    log = json.loads(line)
                    require(log['epoch'] == epoch and log['epochs'] == 6 and log['step'] == step
                            and log['steps_per_epoch'] == 1804 and log['global_step'] == (epoch - 1) * 1804 + step
                            and log['expected_images'] == 7216 and log['images'] == 4 * step
                            and all(math.isfinite(log[key]) for key in ('point_loss', 'depth_point_loss')),
                            'Training steps/loss/coverage differ')
                    count += 1
            require(count == 1804, 'Missing optimizer steps in training log')
            evidence.extend((receipt(cover_path), receipt(log_path)))
        require(len(coverage) == 14432 and set(coverage) == set(plan['image_ids']['train']), 'Epoch coverage differs')
        folder = points / f'val_epoch_{epoch:02d}'
        ev, ev_run = read(folder / 'evaluation.json'), read(folder / 'evaluation_run.json')
        pred_sha = sha(folder / 'predictions.jsonl')
        require(ev['status'] == 'completed' and ev['split'] == 'val' and ev['scope'] == 'full_split'
                and ev['images'] == ev['full_split_images'] == 2584 and ev['metrics'] == row['metrics']
                and ev['predictions_sha256'] == pred_sha == ev_run['predictions_sha256']
                and ev_run['evaluation_sha256'] == sha(folder / 'evaluation.json')
                and ev_run['status'] == 'completed' and ev_run['images'] == 2584
                and ev_run['scope'] == 'full_split' and not ev_run['ground_truth_in_inference'],
                'Head selection must use a full validation result for every epoch')
        check_sources(ev['source_sha256'])
        require(verify_prediction_ids(folder / 'predictions.jsonl', plan['image_ids']['val']) == ev['predicted_instances'],
                'Validation prediction count differs')
        require(Path(row['checkpoint']).resolve() == (points / f'heads_epoch_{epoch:02d}.pt').resolve(),
                'Validation history points to a nonformal checkpoint')
        evidence.extend((receipt(folder / 'evaluation.json'), receipt(folder / 'evaluation_run.json'),
                         receipt(folder / 'predictions.jsonl'), receipt(row['checkpoint'])))
    chosen = max(history, key=lambda row: (row['metrics']['mask_ap_small'] or 0., row['metrics']['mask_ap'] or 0., -row['epoch']))
    require(selection['selected'] == chosen and result['selected_epoch'] == chosen['epoch'], 'Validation-selected epoch differs')
    require(Path(result['head']['path']).resolve() == Path(chosen['checkpoint']).resolve()
            and Path(result['head_selection']['path']).resolve() == (points / 'selection.json').resolve(),
            'Final result refers to a different selected head')
    require(sha(chosen['checkpoint']) == selection['selected_head_sha256'] == result['head']['sha256'], 'Selected head changed')
    head = torch.load(chosen['checkpoint'], map_location='cpu', weights_only=True)
    require(head['epoch'] == chosen['epoch'] and head['fine_channels'] == 448
            and head['protocol_sha256'] == RGB_PROTOCOL_SHA and head['source_sha256'] == plan['system_source_sha256']
            and head['base_checkpoint_sha256'] == checkpoints['rgb']['sha256'], 'Point-head pairing/provenance differs')
    del head
    identity_path = run / 'final_evaluation/model_identity.json'
    identity = read(identity_path)
    require(identity['base_modality'] == 'RGB' and not identity['rgbd_feature_fusion']
            and identity['depth_used_in_refinement'] and not identity['camera_intrinsics_used']
            and identity['selected_epoch'] == chosen['epoch'] and identity['inference'] == current['default']
            and identity['model_input_size'] == [1024, 1024] and identity['metrics'] == result['metrics']
            and identity['head'] == result['head'] and identity['base'] == result['base'], 'Test model identity differs')
    require(result['variant'] == 'without_rgbd_fusion' and not result['rgbd_feature_fusion']
            and result['depth_used_in_point_refinement'] and result['depth_used_in_probability_refinement']
            and not result['camera_intrinsics_used'] and not result['test_used_for_selection']
            and result['inference'] == current['default'] and result['protocol']['sha256'] == RGB_PROTOCOL_SHA,
            'RGB final result definition differs')
    test_folder = run / 'final_evaluation/test'
    for name, filename in (('evaluation', 'evaluation.json'), ('predictions', 'predictions.jsonl')):
        require(Path(result[name]).resolve() == (test_folder / filename).resolve(), 'RGB test result came from a different run')
    test_run = read(test_folder / 'evaluation_run.json')
    require(test_run['status'] == 'completed' and test_run['split'] == 'test'
            and test_run['scope'] == 'full_split' and test_run['images'] == 2112
            and test_run['evaluation_sha256'] == result['evaluation_sha256']
            and test_run['predictions_sha256'] == result['predictions_sha256']
            and not test_run['ground_truth_in_inference'], 'RGB test run receipt differs')
    valid = checked_result(result, ids, semantics, depth_used=True)
    return result, {'protocol': receipt(protocol_path), 'result': receipt(run / 'results.json'),
                    'selection': receipt(points / 'selection.json'), 'selected_epoch': chosen['epoch'],
                    'training_epochs': 6, 'training_image_exposures': 14432 * 6,
                    'global_steps': 10824, 'validation_images_each_epoch': 2584,
                    'base_unchanged': True, 'base_gradients_absent': True, 'head_replicas_equal': True,
                    'engineering_weights_reused_for_formal_training': False,
                    'per_epoch_artifacts': evidence, 'test_identity': receipt(identity_path), 'csv_validation': valid}


def completed_row(identifier, source, verification):
    row = copy.deepcopy(source)
    return {'status': 'completed', 'metrics': row['metrics'], 'images': row['images'],
            'artifacts': {key: row[key] for kind in ('evaluation', 'predictions', 'csv')
                          for key in (kind, kind + '_sha256')},
            'predicted_instances': row['predicted_instances'],
            'csv_rows_identical_to_scored_predictions': 2112, 'verification': verification}


def audit(args):
    from rgbd_seg.course_archive import CourseArchive

    snapshot = read(args.snapshot)
    current = read(ROOT / 'configs/dpseg_current.json')
    check_geometry(current['default'])
    check_sources(current['source_sha256'])
    require(snapshot['images'] == 2112 and snapshot['current_default']['variant'] == current['default']['strategy']
            and snapshot['current_default']['tile_size_px'] == 640, 'Current result snapshot differs')
    for item in snapshot['checkpoints'].values():
        verify_artifact(item)
    system = read(ROOT / current['system_protocol']['path'])
    with CourseArchive(system['archive'], split='test') as data:
        ids = data.image_ids
    require(len(ids) == len(set(ids)) == 2112, 'Unexpected test images')
    semantics = snapshot['semantics']
    require(semantics['iou_type'] == 'segm' and semantics['max_detections'] == [1, 10, 100]
            and semantics['area_ranges'][1] == [0, 1024]
            and len(semantics['iou_thresholds']) == 10
            and all(abs(x - (.5 + .05 * i)) < 1e-12 for i, x in enumerate(semantics['iou_thresholds'])),
            'Unexpected COCO metric settings')
    sources, available = {'snapshot': receipt(args.snapshot), 'current_manifest': receipt(ROOT / 'configs/dpseg_current.json')}, {}
    for name, stage in (('mask2former', 0), ('dafusion', 1), ('without_tiles', 3), ('full', 4)):
        row = snapshot['results'][stage]
        require(row['stage'] == f'S{stage}', 'Snapshot row order changed')
        valid = checked_result(row, ids, semantics, depth_used=name != 'mask2former')
        available[name] = completed_row(name, row, {'source_stage': f'S{stage}', 'csv_validation': valid})
    require(available['full']['metrics'] == current['test_evidence']['metrics'], 'Full row is not the adopted default')

    removal_summary = args.removal_run / 'summary.json'
    if removal_summary.is_file() and (args.removal_run / 'summary_exit.json').is_file():
        require(read(args.removal_run / 'summary_exit.json')['exit_code'] == 0, 'Removal summary did not succeed')
        summary, plan, result, state = (read(args.removal_run / name) for name in
                                      ('summary.json', 'protocol.json', 'results.json', 'status.json'))
        require(summary['status'] == 'verified' and summary['images'] == 2112
                and summary['full_without_tiles_native_exact_images'] == 2112
                and summary['candidate_scores_order_exact_images'] == 2112
                and state['status'] == 'completed' and state['exit_code'] == 0
                and result['status'] == 'completed' and not summary['training']
                and not summary['default_changed'], 'Removal run has not been verified completely')
        verify_artifact(summary['protocol'])
        require(result['protocol_sha256'] == summary['protocol']['sha256'] == sha(args.removal_run / 'protocol.json'),
                'Removal protocol/result differs')
        verify_artifact(summary['current_default_manifest'])
        check_sources(summary['source_sha256'])
        require(summary['publisher_sha256'] == sha(ROOT / 'tools/summarize_course_current_removal.py'), 'Removal publisher changed')
        check_sources(summary['publisher_dependencies'])
        require(summary['checkpoints']['base']['sha256'] == snapshot['checkpoints']['rgbd']['sha256']
                and summary['checkpoints']['heads']['sha256'] == snapshot['checkpoints']['points']['sha256'],
                'Removal checkpoint pairing differs')
        for item in summary['checkpoints'].values():
            verify_artifact(item)
        require(plan['geometry']['tile_size_px'] == 640 and plan['geometry']['views_including_whole'] == 5
                and plan['geometry']['tiles'] == 4 and plan['merge_strategy'] == 'global_guard_edge_consensus',
                'Removal windows/merging differ')
        for phase, expected in summary['phase_exits'].items():
            require(read(args.removal_run / f'{phase}_exits.json') == expected
                    and all(code == 0 for code in expected.values()), 'Removal phase exit differs')
        by_name = {row['variant']: row for row in summary['results']}
        require(by_name['full']['metrics'] == available['full']['metrics']
                and by_name['full']['predictions_sha256'] == available['full']['artifacts']['predictions_sha256']
                and by_name['without_tiles']['metrics'] == available['without_tiles']['metrics'], 'Removal references differ')
        for name in ('without_point', 'without_probability'):
            row = by_name[name]
            require(row['metrics'] == result['variants'][name]['metrics'], 'Removal summary metric differs')
            valid = checked_result(row, ids, semantics, depth_used=True)
            available[name] = completed_row(name, row, {'summary': receipt(removal_summary), 'csv_validation': valid})
        sources['current_removal'] = receipt(removal_summary)
    else:
        for name in ('without_point', 'without_probability'):
            available[name] = pending_state(args.removal_run)
            available[name]['pending_reason'] = '当前640合并的完整推理、评分或最终核验尚未全部完成。'
    rgb_state = read(args.rgb_run / 'status.json') if (args.rgb_run / 'status.json').is_file() else {}
    if (args.rgb_run / 'results.json').is_file() and rgb_state.get('status') == 'completed':
        row, verified = audit_rgb_training(args.rgb_run, ids, semantics, current, snapshot['checkpoints'])
        available['without_rgbd_fusion'] = completed_row('without_rgbd_fusion', row, verified)
        sources['rgb_head'] = receipt(args.rgb_run / 'results.json')
    else:
        available['without_rgbd_fusion'] = pending_state(args.rgb_run)
        available['without_rgbd_fusion']['pending_reason'] = '复用RGB第15轮基座，专属点头训练6轮后，用完整验证集选轮；仅最终锁定权重进入完整测试。'
    rows = [dict(id=name, method=method, kind=kind, definition=definition, **available[name])
            for name, method, kind, definition in SPECS]
    full = available['full']['metrics']
    for row in rows:
        row['full_minus_row'] = ({key: full[key] - row['metrics'][key] for key in METRICS}
                                 if row['metrics'] is not None else None)
    count = sum(row['status'] == 'completed' for row in rows)
    selection_provenance = copy.deepcopy(snapshot['selection_provenance'])
    selection_provenance['point_epoch_applies_to'] = 'RGB-D-backed point-refinement rows only'
    selection_provenance['without_rgbd_fusion'] = {
        'trained_epochs_required': 6,
        'selected_epoch': available['without_rgbd_fusion'].get('verification', {}).get('selected_epoch'),
        'selection': 'full validation APsmall, then AP, then earlier epoch',
        'status': available['without_rgbd_fusion']['status'],
        'base_training_reused': True,
        'point_head_retrained_for_rgb_features': True,
    }
    return {'schema': 'dpseg-current-baselines-and-independent-removal-v2',
            'status': 'completed' if count == len(rows) else 'partial',
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            'split': 'local_test', 'images': 2112, 'backbone': 'Swin-Large', 'camera_intrinsics_used': False,
            'comparison': 'baselines_and_independent_module_removal', 'current_default': snapshot['current_default'],
            'completed_rows': count, 'pending_rows': len(rows) - count, 'results': rows,
            'semantics': semantics, 'selection_provenance': selection_provenance,
            'source_snapshots': sources, 'checkpoints': snapshot['checkpoints'],
            'verification_scope': 'All completed rows: file hashes, full RLE receipts and CSV-to-scored-prediction binding. '
                                  'RGB removal additionally requires all six full training/validation epochs and locked head.',
            'publisher': receipt(Path(__file__)),
            'pending_measurements': ['full_validation_of_current_complete_system', 'Kaggle_of_current_merge',
                                     'independent_inference_timing_of_current_merge']}


def render(artifact, prefix=''):
    rows = artifact['results']
    lines = ['| 方法 | APₛ ↑ | ARₛ ↑ | AP ↑ |', '|:---|---:|---:|---:|']
    for row in rows:
        cells = [row['method'], *([f"{row['metrics'][key]:.6f}" for key in METRICS] if row['metrics'] else ['—'] * 3)]
        if row['id'] == 'full':
            cells = ['**' + cell + '**' for cell in cells]
        lines.append('| ' + ' | '.join(cells) + ' |')
    table = '\n'.join(lines)
    completed, pending = artifact['completed_rows'], artifact['pending_rows']
    status = f'当前已核验 **{completed}/7 行**完整测试结果。' + (
        f'其余 {pending} 行进行中或待测，**— 表示尚无完整结果，不代表 0**。' if pending else '七行均通过完整测试及提交文件核验。')
    by_name = {row['id']: row for row in rows}
    rgb = by_name['without_rgbd_fusion']
    if rgb['metrics']:
        rgb_note = f"专属点头已完成6轮训练，完整验证集选取第 **{rgb['verification']['selected_epoch']}** 轮用于测试。"
    else:
        phase = {'training': '训练', 'engineering': '工程检查', 'evaluation': '完整测试',
                 'failed': '运行失败，需处理后继续', 'prepared': '已准备', 'not_started': '待开始'}.get(rgb.get('run_phase'), '结果核验')
        rgb_note = f'专属点头流程当前处于**{phase}**；完成6轮后按每轮完整验证结果选择，未发布部分测试分数。'
    remove_ready = all(by_name[key]['metrics'] is not None for key in ('without_point', 'without_probability'))
    remove_note = ('两组均已按当前640合并完成完整测试，候选身份、类别、分数及次序与完整版一致，仅改变掩码细化路径。'
                   if remove_ready else '正在按当前640合并补齐完整推理和评分；旧分数优先合并的结果不填入本表。')
    comparisons = []
    for row in rows:
        if row['kind'] == 'independent_removal' and row['metrics']:
            delta = row['full_minus_row']
            comparisons.append(f"- 完整模型相对 **{row['method']}**：APₛ **{delta['mask_ap_small']:+.6f}**、ARₛ **{delta['mask_ar_small']:+.6f}**、AP **{delta['mask_ap']:+.6f}**。")
    d = by_name['dafusion']['full_minus_row']
    return f'''统一采用 **Swin-Large、1024×1024 模型输入、完整本地测试集 2,112 张图**，分数范围为 0–1。每个 `w/o` 配置从完整 DPSeg 独立去掉相应模块；完整版保持 **640×640、4个切片、40%重叠率**，保留整图并使用整图优先＋截断处理＋跨视图一致性合并。

{table}

{status}

- **Mask2Former、DA-Fusion：** 分别为RGB基线与本项目适配的RGB-D融合基座，均使用既有第15轮权重，不带两种掩码细化或切片。
- **w/o RGB-D融合：** 移除双分支跨模态特征融合，复用已完成的RGB第15轮基座，冻结后训练专属448维特征点头；保留局部深度点线索、深度引导概率图细化和当前切片合并，仍使用深度。{rgb_note} 没有重新训练基座，也不直接沿用RGB-D点头。
- **w/o点细化、w/o概率图细化：** {remove_note} 保留点头时使用原第6轮RGB-D配套点头。
- **w/o切片增强推理：** 保留RGB-D融合与两种细化，只运行整图；不涉及跨视图合并，复用已核验的完整整图结果。

完整DPSeg相对DA-Fusion的APₛ、ARₛ、AP分别变化 **{d['mask_ap_small']:+.6f}、{d['mask_ar_small']:+.6f}、{d['mask_ap']:+.6f}**。独立去除的描述性差值如下，正号表示完整版数值更高：

{chr(10).join(comparisons)}

RGB专属点头和原RGB-D点头分别按各自完整验证集结果选轮；移除融合同时改变基座表征、参数量和点头训练，不能将其差值解释为固定权重下关闭单个算子的效果。其他去除组使用固定权重。实验仅一次训练种子，不作统计显著性结论，也不预设每个模块一定使所有指标提高。当前合并由用户根据已查看的本地测试对比指定，尚无该完整配置的新验证集、Kaggle或独立速度结果。默认继续保持640切片。

[完整定义、核验与CSV路径]({prefix}course_baselines_removal_current.md) · [指标CSV]({prefix}course_baselines_removal_current.csv) · [来源JSON]({prefix}course_baselines_removal_current.json)。原[累计表]({prefix}course_ablation_swinl_current.md)与[旧合并去除表]({prefix}course_removal_swinl.md)保留作历史记录。
'''


def status_sentence(artifact):
    pending = [row['method'] for row in artifact['results'] if row['metrics'] is None]
    return (f"基线与独立消融总表已核验 {artifact['completed_rows']}/7 行；"
            + ('、'.join(pending) + '的完整结果尚未全部核验。' if pending else '七行完整测试结果均已核验。'))


def insert_status(text, sentence):
    start, end = '<!-- current-comparison-status:start -->', '<!-- current-comparison-status:end -->'
    block = start + '\n' + sentence + '\n' + end
    if start in text:
        require(text.count(start) == text.count(end) == 1, 'Ambiguous comparison status markers')
        return text[:text.index(start)] + block + text[text.index(end) + len(end):]
    first = text.index('\n\n')
    return text[:first] + '\n\n' + block + text[first:]


def publish(artifact, docs):
    docs.mkdir(parents=True, exist_ok=True)
    name = 'course_baselines_removal_current'
    cumulative_selected = '<!-- cumulative-main:start -->' in (ROOT / 'README.md').read_text()
    title = '补充实验：基线对比与独立模块去除' if cumulative_selected else 'DPSeg 基线对比与独立模块消融'
    intro = ('当前论文主表采用[五组累计式消融](course_ablation_swinl_current.md)，'
             '本页保留七组独立去除模块的补充结果。\n\n' if cumulative_selected else '')
    publish_scope = '本补充表，保留当前累计主表' if cumulative_selected else '本表、README和关联文档'
    paths = []
    for row in artifact['results']:
        paths.append(f"- **{row['method']}：** `{row['artifacts']['csv']}`" if row['artifacts']
                     else f"- **{row['method']}：** 尚无完成且核验通过的CSV，状态：`{row['status']}`。")
    contents = f'''# {title}

{intro}## Material Passport

- 来源：已有累计结果、当前640独立细化去除实验，以及RGB专属点头流程；原始实验与协议均保留。
- 状态：**{artifact['status'].upper()}**，完成并核验 {artifact['completed_rows']}/7 行。
- 核验：每个完成行核对权重、评分/预测/CSV文件哈希、2112行覆盖、全量RLE校验收据，以及CSV与实际评分输入逐行一致。
- 范围：完整本地测试。RGB去除组在六轮训练、逐轮完整验证及锁定点头的完整测试全部通过后入表。

## 对比表

{render(artifact)}
## 各配置提交CSV

{chr(10).join(paths)}

## 更新与复核

```bash
.venv/bin/python tools/summarize_course_baselines_removal.py
.venv/bin/python tools/summarize_course_baselines_removal.py --publish
```

第一条仅核验当前已完成结果并显示状态；第二条在核验通过后更新{publish_scope}。缺结果会保留进行中/待测，已有但校验失败的结果会报错，不能降级成已完成。原累计与旧合并实验快照不被覆盖；不训练、不推理、不改默认配置、不提交Kaggle。
'''
    if cumulative_selected and (docs / 'course_efficiency_swinl_current.md').exists():
        contents = contents.replace('尚无该完整配置的新验证集、Kaggle或独立速度结果。',
                    '尚无该完整配置的新验证集精度和Kaggle结果；五组累计配置的独立速度见[当前效率报告](course_efficiency_swinl_current.md)。')
    (docs / f'{name}.md').write_text(contents)
    (docs / f'{name}.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    with (docs / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['方法', 'AP_s', 'AR_s', 'AP', '状态', 'CSV路径', 'CSV_SHA256'])
        for row in artifact['results']:
            writer.writerow([row['method'], *([row['metrics'][key] for key in METRICS] if row['metrics'] else ['', '', '']),
                             row['status'], row['artifacts']['csv'] if row['artifacts'] else '',
                             row['artifacts']['csv_sha256'] if row['artifacts'] else ''])
    if cumulative_selected:
        return
    readme = ROOT / 'README.md'
    text = readme.read_text()
    start, end = '<!-- baseline-removal:start -->', '<!-- baseline-removal:end -->'
    require(text.count(start) == text.count(end) == 1, 'README baseline-removal markers missing or ambiguous')
    block = start + '\n### 基线对比与独立模块消融结果\n\n' + render(artifact, 'docs/results/') + end
    text = text[:text.index(start)] + block + text[text.index(end) + len(end):]
    text = re.sub(r'当前主表采用基线对比与独立去除模块的形式(?:；[^。]+。|。基线与独立消融总表已核验[^。]+。)',
                  '当前主表采用基线对比与独立去除模块的形式。' + status_sentence(artifact), text)
    readme.write_text(text)
    index = ROOT / 'docs/README.md'
    text = index.read_text()
    text = re.sub(r'^- \[当前完整版：基线对比与独立模块消融、CSV[^\]]*\]',
                  f"- [当前完整版：基线对比与独立模块消融、CSV（已核验{artifact['completed_rows']}/7行）]", text, flags=re.M)
    index.write_text(text)
    for name in ('methods', 'course_metric', 'historical_results'):
        path = ROOT / f'docs/{name}.md'
        text = path.read_text()
        text = text.replace('；去除 RGB-D 融合、点细化和概率图细化三行待补测。', '；独立消融的最新完成范围见本页状态及主表。')
        text = insert_status(text, status_sentence(artifact) + '\n\n[当前总表及CSV路径](results/course_baselines_removal_current.md)。'
                            'w/o RGB-D融合保留深度掩码细化，复用RGB基座并训练专属点头；各完整模型的验证/Kaggle成绩另行列报。')
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=ROOT / 'docs/results/course_ablation_swinl_current.json')
    parser.add_argument('--removal-run', type=Path, default=ROOT / 'reports/course_swinl_current_removal_v1')
    parser.add_argument('--rgb-run', type=Path, default=ROOT / 'reports/course_swinl_without_rgbd_v1')
    parser.add_argument('--docs', type=Path, default=ROOT / 'docs/results')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    artifact = audit(args)
    if args.publish:
        publish(artifact, args.docs)
    print(json.dumps({'status': artifact['status'], 'completed_rows': artifact['completed_rows'],
                      'pending_rows': artifact['pending_rows'], 'published': args.publish,
                      'rows': {row['id']: row['status'] for row in artifact['results']}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
