#!/usr/bin/env python3
"""Audit the frozen v5 12-training/28-evaluation matrix without rerunning models."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VARIANTS = ('baseline', 'bce2', 'bace', 'freqfusion')
SEEDS = (17, 23, 41)
COCO = ('mask_ap', 'mask_ap50', 'mask_ap75', 'mask_ap_small', 'mask_ap_medium', 'mask_ap_large',
        'mask_ar_1', 'mask_ar_10', 'mask_ar_100', 'mask_ar_small', 'mask_ar_medium', 'mask_ar_large')
TRAIN_KEYS = {(seed, variant) for seed in SEEDS for variant in VARIANTS}
EVAL_KEYS = {('graspnet', seed, variant, split) for seed, variant in TRAIN_KEYS for split in ('val', 'test')}
EVAL_KEYS |= {('tless', 17, variant, 'test') for variant in VARIANTS}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def describe(values):
    valid = [x for x in values if x is not None]
    require(all(math.isfinite(x) for x in valid), 'Nonfinite metric')
    return {'n': len(valid), 'missing': len(values)-len(valid), 'values': values,
            'mean': statistics.mean(valid) if valid else None,
            'sample_std_ddof1': statistics.stdev(valid) if len(valid) > 1 else None}


def metric_delta(a, b):
    return None if a is None or b is None else a-b


def paired(rows, variant, reference, split):
    by_key = {(r['dataset'], r['seed'], r['variant'], r['split']): r for r in rows}
    pairs = []
    for seed in SEEDS:
        a = by_key.get(('graspnet', seed, variant, split))
        b = by_key.get(('graspnet', seed, reference, split))
        if a is None or b is None:
            continue
        pairs.append({'seed': seed, 'variant_metrics': a['metrics'], 'reference_metrics': b['metrics'],
                      'delta_pp': {k: metric_delta(a['metrics'][k], b['metrics'][k]) for k in COCO},
                      'variant_mean_ms': a['timing']['mean_ms'], 'reference_mean_ms': b['timing']['mean_ms']})
    return {'variant': variant, 'reference': reference, 'split': split, 'pairs': pairs,
            'descriptive_delta_pp': {k: describe([p['delta_pp'][k] for p in pairs]) for k in COCO}}


def development_gate(rows, complete):
    if not complete:
        return {'status': 'withheld_incomplete_matrix', 'preferred_optional_variant': None,
                'reason': 'Final development gate is withheld until all 12 formal training runs and 28 evaluations pass audit.'}
    choices = {}
    for variant in VARIANTS[1:]:
        result = paired(rows, variant, 'baseline', 'val')
        require(len(result['pairs']) == 3, 'Gate needs all three development seed pairs')
        small = result['descriptive_delta_pp']['mask_ap_small']
        ap = result['descriptive_delta_pp']['mask_ap']
        require(small['n'] == ap['n'] == 3, 'Gate requires available AP and APsmall for every seed')
        conditions = {'mean_small_delta_ge_2pp': small['mean'] >= 2,
                      'at_least_two_positive_seeds': sum(v > 0 for v in small['values']) >= 2,
                      'mean_ap_delta_ge_minus1pp': ap['mean'] >= -1}
        choices[variant] = {**result, 'conditions': conditions, 'passed': all(conditions.values()),
                            'positive_small_seeds': sum(v > 0 for v in small['values']),
                            'mean_val_ap_small': statistics.mean(p['variant_metrics']['mask_ap_small'] for p in result['pairs']),
                            'mean_val_ap': statistics.mean(p['variant_metrics']['mask_ap'] for p in result['pairs']),
                            'mean_val_ms': statistics.mean(p['variant_mean_ms'] for p in result['pairs'])}
    eligible = sorted([v for v in choices if choices[v]['passed']],
                      key=lambda v: (-choices[v]['mean_val_ap_small'], -choices[v]['mean_val_ap'], choices[v]['mean_val_ms']))
    return {'status': 'complete', 'selection_data': 'Only GraspNet E2 development seeds17/23/41',
            'thresholds': {'mean_ap_small_delta_min_pp': 2, 'positive_seed_min': 2, 'mean_ap_delta_min_pp': -1},
            'choices': choices, 'eligible_ranked': eligible, 'preferred_optional_variant': eligible[0] if eligible else None,
            'online_default_changes': False, 'bce2_interpretation': 'Simple mask-loss reweighting control; equally eligible under the fixed engineering gate.',
            'tie_break': 'Higher development mean APsmall, then higher development mean AP, then lower development mean inference ms; no test/TLESS selection.'}


def audit_loss_trace(loss_path, sampling_path, steps=800, variant=None):
    losses = [json.loads(line) for line in Path(loss_path).read_text().splitlines()]
    traces = [json.loads(line) for line in Path(sampling_path).read_text().splitlines()]
    require(len(losses) == len(traces) == steps, f'Expected {steps} logged steps: {loss_path}')
    trace_keys = ('batch_indices', 'sample_ids', 'augmentation_rng_before_sha256', 'augmentation_rng_after_sha256')
    totals = []
    for index, (loss, trace) in enumerate(zip(losses, traces), 1):
        require(loss['step'] == trace['step'] == index, 'Nonconsecutive training steps')
        require(all(loss[k] == trace[k] for k in trace_keys), 'Loss and sampling trace disagree')
        numeric = [v for k, v in loss.items() if k == 'loss' or k.startswith('loss_') or k in ('grad_norm', 'elapsed_s')]
        numeric += loss['lr_used'] + loss['lr_next']
        require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in numeric), 'Nonfinite training loss/gradient/lr/time')
        require(len(loss['batch_indices']) == len(loss['sample_ids']) == 2, 'Batch size trace differs from protocol')
        require(math.isclose(loss['loss'], sum(v for k, v in loss.items() if k.startswith('loss_')), rel_tol=1e-5, abs_tol=1e-5), 'Total loss does not match component sum')
        if variant == 'bce2':
            require(loss['loss_mask_aux'] == loss['loss_mask'], 'BCE2 auxiliary must equal mask BCE')
        if variant in ('bce2', 'bace'):
            require('loss_mask_aux' in loss, 'Missing prescribed auxiliary loss')
        if variant in ('baseline', 'freqfusion'):
            require('loss_mask_aux' not in loss, 'Unexpected auxiliary loss')
        totals.append(loss['loss'])
    return {'steps': len(losses), 'all_logged_losses_gradients_and_lrs_finite': True,
            'first_loss': totals[0], 'last_loss': totals[-1], 'min_loss': min(totals), 'max_loss': max(totals),
            'mean_last_50_loss': statistics.mean(totals[-50:]),
            'interpretation': 'Total losses contain different objectives; do not rank methods by raw loss magnitude.'}


def check_pairing(records):
    checks = {}
    for seed in SEEDS:
        group = [r for r in records if r['seed'] == seed]
        by_variant = {r['variant']: r for r in group}
        comparisons = {}
        for key in ('sampling_trace_sha256', 'shared_initialization', 'initial_torch_cpu_rng_sha256', 'initial_torch_cuda_rng_sha256',
                    'dataset_sha256', 'manifest_sha256', 'fixed_training', 'base_config', 'init_checkpoint_sha256'):
            values = [r[key] for r in group]
            require(not values or all(v == values[0] for v in values), f'Same-seed pairing differs: seed{seed}, {key}')
            comparisons[key] = {'matches': True, 'value': values[0] if values else None}
        checks[str(seed)] = {'variants_present': list(by_variant), 'complete_four_variants': set(by_variant) == set(VARIANTS),
                             'checks': comparisons}
    return checks


def audit(training_root, evaluation_root, protocol_path, allow_partial=False):
    import torch
    from rgbd_seg.data import RGBDDataset, dataset_fingerprint
    from tools.run_v5 import source_snapshot, FIXED_TRAINING, validate_prediction_ids
    from tools.evaluate_multiscale import area_counts
    training_root, evaluation_root = Path(training_root), Path(evaluation_root)
    protocol_path = Path(protocol_path).resolve()
    protocol = {'path': str(protocol_path), 'sha256': sha(protocol_path)}
    sources = source_snapshot()
    missing, runs, rows = [], [], []
    data_cache, training_by_key = {}, {}
    runtime_reference, training_signature = None, None
    def checked_data(root, expected_hash, expected_manifest):
        root = Path(root).resolve()
        if str(root) not in data_cache:
            data_cache[str(root)] = {'dataset_sha256': dataset_fingerprint(root), 'manifest_sha256': sha(root/'manifest.json')}
        require(data_cache[str(root)]['dataset_sha256'] == expected_hash and data_cache[str(root)]['manifest_sha256'] == expected_manifest, f'Dataset changed: {root}')
    def check_sources(record):
        require(record['source_sha256'] == sources, 'Frozen source SHA mismatch')
        require(record['protocol'] == protocol, 'Frozen protocol path/SHA mismatch')
    def check_runtime(record):
        nonlocal runtime_reference
        current = {k: record[k] for k in ('device', 'device_name', 'cuda_visible_devices', 'threads', 'libraries', 'cuda_version', 'cudnn_version', 'cuda_memory_fraction')}
        if runtime_reference is None:
            runtime_reference = current
        require(current == runtime_reference, 'Training/evaluation hardware, threads or software differ')
    for seed in SEEDS:
        for variant in VARIANTS:
            folder = training_root/f'seed{seed}'/variant
            path = folder/'run.json'
            if not path.exists() or read(path).get('exit_status') != 'completed':
                missing.append({'kind': 'training', 'seed': seed, 'variant': variant, 'path': str(path),
                                'status': read(path).get('exit_status') if path.exists() else 'missing'})
                continue
            run = read(path)
            require(run['seed'] == seed and run['variant'] == variant and run['phase'] == 'formal', f'Wrong run identity/phase: {path}')
            require(run['steps'] == run['completed_steps'] == 800 and run['classes'] == ['object'], 'Formal training must have800 steps and object class')
            require(run['training_samples'] == 64 and run['depth_scale'] == .001, 'Training sample count/depth units changed')
            require(run['fixed_training'] == FIXED_TRAINING and run['resume_supported'] is False, 'Training budget/settings/resume differ')
            require(run['base_config']['seed'] == seed and run['base_config']['mode'] == 'E2', 'Base configuration seed/mode differs')
            check_sources(run); check_runtime(run)
            checked_data(run['data_root'], run['dataset_sha256'], run['manifest_sha256'])
            for record_key, file_name in (('checkpoint_sha256', 'model.pt'), ('weights_sha256', 'weights.pt'),
                                           ('training_state_sha256', 'training_state.pt'), ('loss_trace_sha256', 'loss.jsonl'), ('sampling_trace_sha256', 'sampling.jsonl')):
                require(sha(folder/file_name) == run[record_key], f'{file_name} hash mismatch: {path}')
            require(sha(run['config_file']['path']) == run['config_file']['sha256'], 'Config file changed')
            require(sha(run['init_checkpoint']['path']) == run['init_checkpoint']['sha256'] == run['rgb_initialization']['sha256'], 'COCO initialization file changed')
            checkpoint = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
            require(checkpoint['format'] == 'rgbd_v5_v1', 'Wrong checkpoint format')
            for key in ('base_config', 'config', 'variant', 'provenance', 'classes', 'depth_scale', 'manifest_sha256', 'dataset_sha256', 'protocol', 'source_sha256', 'phase', 'seed', 'steps'):
                require(checkpoint[key] == run[key], f'Checkpoint/run mismatch: {key}')
            require(all(torch.isfinite(tensor).all() for tensor in checkpoint['state_dict'].values() if tensor.is_floating_point()), 'Nonfinite final checkpoint tensor')
            del checkpoint
            losses = audit_loss_trace(folder/'loss.jsonl', folder/'sampling.jsonl', variant=variant)
            groups = run['optimizer']['groups']
            require(len(groups) == 2 and [g['name'] for g in groups] == ['pretrained_rgb', 'heads_depth_new_fusion'], 'Unexpected optimizer groups')
            require(groups[0]['lr'] == .00002 and groups[1]['lr'] == .0002, 'Optimizer learning rates differ')
            group_names = [name for group in groups for name in group['parameter_names']]
            require(len(group_names) == len(set(group_names)), 'Repeated optimizer parameter')
            new_names = [name for name in group_names if '.fpn.fusions.' in name]
            require((variant == 'freqfusion') == bool(new_names), 'FreqFusion parameter groups do not match variant')
            require(all(name in groups[1]['parameter_names'] for name in new_names), 'New fusion received backbone LR')
            signature = {'base_config': {k: v for k,v in run['base_config'].items() if k != 'seed'},
                         'fixed_training': run['fixed_training'], 'dataset_sha256': run['dataset_sha256'],
                         'init_sha256': run['init_checkpoint']['sha256'], 'selection': run['selection']}
            if training_signature is None:
                training_signature = signature
            require(signature == training_signature, 'Shared training protocol differs beyond variant/seed')
            record = {k: run[k] for k in ('variant', 'seed', 'base_config', 'dataset_sha256', 'manifest_sha256', 'fixed_training',
                'sampling_trace_sha256', 'shared_initialization', 'initial_torch_cpu_rng_sha256', 'initial_torch_cuda_rng_sha256',
                'parameters', 'trainable_parameters', 'weight_bytes', 'training_duration_s', 'duration_s',
                'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes', 'checkpoint_sha256')}
            record.update({'run_path': str(path.resolve()), 'run_sha256': sha(path), 'init_checkpoint_sha256': run['init_checkpoint']['sha256'],
                           'checkpoint_path': str((folder/'model.pt').resolve()), 'loss_audit': losses,
                           'checkpoint_metadata_and_finite_state_verified': True,
                           'checkpoint_bytes': (folder/'model.pt').stat().st_size})
            runs.append(record); training_by_key[(seed, variant)] = record
    pairing = check_pairing(runs)
    sample_cache = {}
    for dataset_name, seed, variant, split in sorted(EVAL_KEYS):
        folder = evaluation_root/dataset_name/f'seed{seed}'/variant/split
        path = folder/'evaluation.json'
        if not path.exists() or read(path).get('exit_status') != 'completed':
            metadata = folder/'metadata.json'
            missing.append({'kind': 'evaluation', 'dataset': dataset_name, 'seed': seed, 'variant': variant, 'split': split,
                            'path': str(path), 'status': read(metadata).get('exit_status') if metadata.exists() else 'missing'})
            continue
        require((seed, variant) in training_by_key, 'Completed evaluation lacks verified completed training run')
        result, train = read(path), training_by_key[(seed, variant)]
        require(result['training_seed'] == seed and result['variant'] == variant and result['split'] == split and result['phase'] == 'formal', 'Evaluation identity mismatch')
        check_sources(result); check_runtime(result)
        require(result['training_run_sha256'] == train['run_sha256'] and Path(result['checkpoint']).resolve() == Path(train['checkpoint_path']), 'Evaluation training run/checkpoint reference differs')
        require(result['checkpoint_sha256'] == train['checkpoint_sha256'] and result['training_dataset_sha256'] == train['dataset_sha256'], 'Evaluation checkpoint/training data SHA mismatch')
        require(result['training_protocol'] == protocol and result['training_steps'] == 800 and result['parameters'] == train['parameters'], 'Evaluation model/protocol/budget differs')
        require(result['method'] == 'full' and result['mask_threshold'] == .5 and result['score_threshold'] == .01, 'Nonprescribed inference settings')
        require(result['model']['max_detections'] == 100 and result['classes'] == ['object'], 'Inference classes/candidate count differ')
        require(result['evaluation_depth_scale'] == (.001 if dataset_name == 'graspnet' else .0001), 'Evaluation depth units differ')
        checked_data(result['data_root'], result['dataset_sha256'], result['manifest_sha256'])
        if dataset_name == 'graspnet':
            require(result['dataset_sha256'] == train['dataset_sha256'], 'GraspNet eval uses different manifest from training')
        key = (result['data_root'], split)
        if key not in sample_cache:
            dataset = RGBDDataset(result['data_root'], split)
            sample_cache[key] = (dataset, area_counts(dataset))
        dataset, areas = sample_cache[key]
        require(result['samples'] == len(dataset) == (4 if split == 'val' else 24 if dataset_name == 'graspnet' else 100), 'Unexpected evaluation image count')
        require(result['original_gt_areas'] == areas, 'Original GT area counts/IDs differ')
        require(areas['totals']['small'] == (4 if split == 'val' else 1 if dataset_name == 'graspnet' else 52), 'Unexpected original small GT count')
        pred_path = folder/'predictions.jsonl'
        require(sha(pred_path) == result['prediction_sha256'], 'Saved prediction SHA mismatch')
        validate_prediction_ids(pred_path, dataset)
        require(set(result['metrics']) == set(COCO), 'Expected twelve COCO statistics')
        require(all(v is None or isinstance(v,(int,float)) and math.isfinite(v) and 0 <= v <= 100 for v in result['metrics'].values()), 'Invalid COCO metric')
        timing = result['timing']
        require(timing['warmup'] == 3 and timing['batch'] == 1 and timing['precision'] == 'float32', 'Timing protocol differs')
        for key in ('mean_ms', 'median_ms', 'p95_ms', 'fps'):
            require(math.isfinite(timing[key]) and timing[key] > 0, 'Invalid timing statistic')
        require(len(result['per_frame']) == len(dataset), 'Missing per-frame timings')
        require([r['sample_id'] for r in result['per_frame']] == [r['id'] for r in dataset.samples], 'Timing sample order differs')
        require(math.isclose(statistics.mean(r['time_ms'] for r in result['per_frame']), timing['mean_ms'], rel_tol=1e-8), 'Mean timing differs from per-frame measurements')
        metadata = read(folder/'metadata.json')
        require(all(result.get(k) == v for k,v in metadata.items()), 'Evaluation/metadata receipts disagree')
        rows.append({'dataset': dataset_name, 'seed': seed, 'variant': variant, 'split': split,
                     'metrics': result['metrics'], 'timing': timing,
                     'peak_cuda_allocated_bytes': result['peak_cuda_allocated_bytes'], 'peak_cuda_reserved_bytes': result['peak_cuda_reserved_bytes'],
                     'parameters': result['parameters'], 'checkpoint_bytes': result['checkpoint_bytes'],
                     'checkpoint_sha256': result['checkpoint_sha256'], 'dataset_sha256': result['dataset_sha256'],
                     'samples': result['samples'], 'original_gt_areas': areas['totals'],
                     'evaluation_path': str(path.resolve()), 'evaluation_sha256': sha(path), 'predictions_sha256': result['prediction_sha256'],
                     'prediction_ids_and_original_dimensions_verified': True})
    complete = len(runs) == 12 and len(rows) == 28 and not missing
    require(allow_partial or complete, f'Matrix incomplete: {len(runs)}/12 training, {len(rows)}/28 evaluations; use --partial for progress only')
    # Cost ratios always compare the same seed, data and split in this trial.
    row_index = {(r['dataset'],r['seed'],r['variant'],r['split']): r for r in rows}
    for row in rows:
        baseline = row_index.get((row['dataset'],row['seed'],'baseline',row['split']))
        row['latency_ratio_vs_paired_baseline'] = row['timing']['mean_ms']/baseline['timing']['mean_ms'] if baseline else None
        row['delta_vs_paired_baseline_pp'] = {k: metric_delta(row['metrics'][k], baseline['metrics'][k]) for k in COCO} if baseline else None
    for run in runs:
        baseline = training_by_key.get((run['seed'],'baseline'))
        run['training_time_ratio_vs_paired_baseline'] = run['training_duration_s']/baseline['training_duration_s'] if baseline else None
        if baseline:
            require(run['parameters']-baseline['parameters'] == (314070 if run['variant']=='freqfusion' else 0), 'Unexpected added parameter count')
    grouped = {}
    for split in ('val','test'):
        grouped[split] = {}
        for variant in VARIANTS:
            selected = sorted([r for r in rows if r['dataset']=='graspnet' and r['variant']==variant and r['split']==split],key=lambda r:r['seed'])
            grouped[split][variant] = {'seeds':[r['seed'] for r in selected],
                'metrics': {k:describe([r['metrics'][k] for r in selected]) for k in COCO},
                'mean_inference_ms': describe([r['timing']['mean_ms'] for r in selected])}
    return {'status': 'complete_and_verified' if complete else 'partial_no_final_gate',
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol': protocol, 'source_sha256': sources,
            'summarizer_sha256': sha(__file__), 'runtime': runtime_reference,
            'counts': {'completed_training': len(runs), 'expected_training':12,'completed_evaluations':len(rows),'expected_evaluations':28},
            'missing_or_incomplete': missing, 'training': runs, 'same_seed_pairing': pairing, 'evaluations': rows,
            'graspnet_three_seed_descriptive': grouped,
            'paired_vs_baseline': {split:{v:paired(rows,v,'baseline',split) for v in VARIANTS[1:]} for split in ('val','test')},
            'bace_minus_bce2': {split:paired(rows,'bace','bce2',split) for split in ('val','test')},
            'development_gate': development_gate(rows,complete),
            'limits': ['AP/AR use 0–100 units; differences are percentage points; sample SD uses ddof=1 across training seeds, not views.',
                       'All evaluation datasets were previously observed; this is exploratory reuse, not unseen confirmation.',
                       'Training64 images/two scenes; development4 small GT; GraspNet test1 small GT; TLESS52 small GT with related views.',
                       'Initial shared/RNG digests are training receipts, not reconstruction of initial tensors from final checkpoints. Matching traces prove input/augmentation schedules match, not identical internal ROI/RPN sampling after method-dependent proposals.',
                       'BACE minus BCE2 is reported separately; a gain over baseline alone cannot isolate the boundary formula from increased mask-loss weight.',
                       'FreqFusion changes the entire fusion core and adds full-LR parameters; no claim isolates one filter submechanism.',
                       'CUDA allocated peaks and reserved cache exclude other processes and do not equal whole-device memory; no test on the course-specified24GB GPU. No significance claims.']}


def save_outputs(result, prefix, overwrite=False):
    prefix = Path(prefix)
    destinations = [prefix.with_suffix(ext) for ext in ('.json','.csv')] + [prefix.with_name(prefix.name+'_ap_small').with_suffix(ext) for ext in ('.png','.pdf')]
    require(overwrite or not any(p.exists() for p in destinations), 'Summary artifact exists; use a fresh --output-prefix or explicit --overwrite')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    colors = ('#1f77b4','#d55e00','#009e73')
    for split, ax in zip(('val','test'),axes):
        for seed,color in zip(SEEDS,colors):
            selected={(r['variant']):r for r in result['evaluations'] if r['dataset']=='graspnet' and r['split']==split and r['seed']==seed}
            values=[selected[v]['metrics']['mask_ap_small'] if v in selected else float('nan') for v in VARIANTS]
            ax.plot(range(4),values,'o-',label=f'seed {seed}',color=color,linewidth=1.4)
        ax.set_xticks(range(4),VARIANTS,rotation=15);ax.set_ylabel('Mask AP small (0–100)')
        ax.set_title(f"{'Development: 4' if split=='val' else 'Test: 1'} small GT; paired seeds")
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=9)
    if result['status']!='complete_and_verified':
        fig.suptitle('PARTIAL — no final method selection')
    for destination in destinations[2:]:
        fig.savefig(destination,dpi=180)
    plt.close(fig)
    with destinations[1].open('w',newline='') as stream:
        fields=['dataset','split','seed','variant',*COCO,'mean_ms','median_ms','p95_ms','latency_ratio_vs_paired_baseline',
                'peak_cuda_allocated_GiB','peak_cuda_reserved_GiB','parameters','checkpoint_bytes','small_GT',
                'training_duration_s','training_peak_allocated_GiB','training_time_ratio_vs_paired_baseline','checkpoint_sha256','dataset_sha256']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        trains={(r['seed'],r['variant']):r for r in result['training']}
        for row in result['evaluations']:
            train=trains[(row['seed'],row['variant'])]
            record={k:row[k] for k in ('dataset','split','seed','variant','parameters','checkpoint_bytes','checkpoint_sha256','dataset_sha256','latency_ratio_vs_paired_baseline')}
            record.update(row['metrics']);record.update({k:row['timing'][k] for k in ('mean_ms','median_ms','p95_ms')})
            record.update({'peak_cuda_allocated_GiB':row['peak_cuda_allocated_bytes']/2**30 if row['peak_cuda_allocated_bytes'] is not None else None,
                           'peak_cuda_reserved_GiB':row['peak_cuda_reserved_bytes']/2**30 if row['peak_cuda_reserved_bytes'] is not None else None,
                           'small_GT':row['original_gt_areas']['small'],'training_duration_s':train['training_duration_s'],
                           'training_peak_allocated_GiB':train['peak_cuda_allocated_bytes']/2**30 if train['peak_cuda_allocated_bytes'] is not None else None,
                           'training_time_ratio_vs_paired_baseline':train['training_time_ratio_vs_paired_baseline']})
            writer.writerow(record)
    destinations[0].write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')



def attach_integrity_audit(result, path):
    evidence = read(path)
    require(result['status'] == 'complete_and_verified' and evidence.get('status') == 'pass', 'Only complete passing matrices can attach independent integrity evidence')
    require(evidence['source_sha256'] == result['source_sha256'] and evidence['protocol_sha256'] == result['protocol']['sha256'], 'Independent audit source/protocol mismatch')
    require(len(evidence['training']) == 12 and len(evidence['evaluation']) == 28, 'Independent audit must cover all12+28')
    train = {(row['seed'], row['variant']): row for row in evidence['training']}
    evaluations = {(row['dataset'], row['seed'], row['variant'], row['split']): row for row in evidence['evaluation']}
    require(set(train) == TRAIN_KEYS and set(evaluations) == EVAL_KEYS, 'Independent audit matrix identities differ')
    for row in result['training']:
        require(train[(row['seed'], row['variant'])]['run_sha256'] == row['run_sha256'], 'Independent audit training receipt differs')
    for row in result['evaluations']:
        item = evaluations[(row['dataset'], row['seed'], row['variant'], row['split'])]
        require(item['evaluation_sha256'] == row['evaluation_sha256'] and item['predictions_sha256'] == row['predictions_sha256'], 'Independent audit evaluation artifacts differ')
    result['independent_integrity_audit'] = {'path': str(Path(path).resolve()), 'report_sha256': sha(path), 'status': 'pass',
                                           'scope': 'Source/protocol, complete identities, and every run/evaluation/prediction SHA cross-checked against this summary.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root',default=str(ROOT/'outputs/paper_v5'))
    parser.add_argument('--evaluation-root',default=str(ROOT/'reports/v5/evaluation'))
    parser.add_argument('--protocol',default=str(ROOT/'docs/training_methods_protocol.md'))
    parser.add_argument('--output-prefix',default=str(ROOT/'reports/v5/trial_summary'))
    parser.add_argument('--integrity-audit', help='Optional completed independent audit JSON to cross-check and cite')
    parser.add_argument('--partial',action='store_true',help='Audit completed artifacts and show missing work, withholding final gate')
    parser.add_argument('--overwrite',action='store_true',help='Explicitly replace only this summary prefix artifacts')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(4)
    result=audit(args.training_root,args.evaluation_root,args.protocol,args.partial)
    result['command']=sys.argv
    if args.integrity_audit:
        attach_integrity_audit(result,args.integrity_audit)
    save_outputs(result,args.output_prefix,args.overwrite)
    print(json.dumps({'status':result['status'],'counts':result['counts'],'development_gate':result['development_gate'],'output_prefix':args.output_prefix},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
