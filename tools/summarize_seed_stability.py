#!/usr/bin/env python3
"""Describe matched-seed E0/E2 results without treating image views as replicates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

METRICS = ('mask_ap', 'mask_ap_small', 'mask_ap75', 'boundary_f1', 'overlap_f1', 'object_recall_f_gt075')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values):
    if any(value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)) for value in values):
        raise ValueError('Metric values must be finite or null')
    valid = [value for value in values if value is not None]
    return {'n': len(valid), 'missing': len(values)-len(valid),
            'mean': statistics.mean(valid) if valid else None,
            'sample_std_ddof1': statistics.stdev(valid) if len(valid) > 1 else None,
            'min': min(valid) if valid else None, 'max': max(valid) if valid else None}


def training_signature(run):
    return {'config': {key: value for key, value in run['config'].items() if key not in ('seed', 'mode')},
            **{key: run[key] for key in ('dataset_sha256', 'manifest_sha256', 'steps', 'batch_size', 'optimizer', 'augmentation', 'selection')},
            'rgb_initialization_sha256': run['rgb_initialization']['sha256'],
            'rgb_head_initialization': run['rgb_initialization']['head_initialization']}


def summarize(roots, source_receipt=None, source_amendment=None, evaluation_regression=None):
    source_check = None
    regression_check = None
    if evaluation_regression is not None:
        regression = read_json(evaluation_regression)
        project = Path(__file__).resolve().parents[1]
        before_path, after_path = project/regression['before'], project/regression['after']
        before, after = read_json(before_path), read_json(after_path)
        if (len(before['metrics']) != 12 or before['metrics'] != after['metrics'] or
            before['checkpoint_sha256'] != after['checkpoint_sha256'] or
            before['dataset_sha256'] != after['dataset_sha256'] or
            before['split'] != after['split'] or before['samples'] != after['samples']):
            raise ValueError('Evaluation regression does not preserve all12 COCO metrics and input identities')
        regression_check = {'report': str(Path(evaluation_regression).resolve()),
                            'report_sha256': sha256(evaluation_regression),
                            'before_sha256': sha256(before_path), 'after_sha256': sha256(after_path),
                            'all12_metrics_and_input_identities_verified_equal': True,
                            'scope': 'One recorded fixed-checkpoint regression; not a training reproducibility claim.',
                            'evidence': regression}
    if source_amendment is not None and source_receipt is None:
        raise ValueError('A source amendment requires the original source receipt')
    if source_receipt is not None:
        receipt = read_json(source_receipt)
        reference = receipt.get('critical_source_sha256', receipt.get('code_sha256'))
        if not isinstance(reference, dict):
            raise ValueError('Source receipt must contain a per-file code_sha256 mapping')
        project = Path(__file__).resolve().parents[1]
        comparisons = {name: {'reference_sha256': digest, 'current_sha256': sha256(project/name)} for name, digest in reference.items()}
        amendment = read_json(source_amendment) if source_amendment else None
        changes = amendment.get('files', {}) if amendment else {}
        if not isinstance(changes, dict):
            raise ValueError('Source amendment must contain a files mapping')
        # Only changes anchored to the original critical receipt are accepted;
        # unrelated data-loader changes need their own separate receipt.
        for name, change in changes.items():
            current = sha256(project/name)
            if (not isinstance(change.get('reason'), str) or not change['reason'].strip() or
                change.get('after_sha256') != current):
                raise ValueError(f'Invalid amendment reason or after SHA for {name}')
            if name in reference and change.get('before_sha256') != reference[name]:
                raise ValueError(f'Amendment before SHA differs from source receipt for {name}')
            if name not in reference:
                raise ValueError(f'Amendment file {name} has no original critical-source SHA; use a separate data-change receipt')
        for name, item in comparisons.items():
            item['matches'] = item['reference_sha256'] == item['current_sha256']
            item['documented_amendment'] = changes.get(name)
            item['accepted'] = item['matches'] or name in changes
        source_check = {'receipt': str(Path(source_receipt).resolve()), 'receipt_sha256': sha256(source_receipt), 'files': comparisons,
                        'all_match': all(item['matches'] for item in comparisons.values()),
                        'all_accepted': all(item['accepted'] for item in comparisons.values()),
                        'amendment': amendment, 'amendment_sha256': sha256(source_amendment) if source_amendment else None,
                        'limit': 'Compares current critical sources to a prior receipt and explicitly supplied amendments; this is not an immutable per-run snapshot. Amendment evidence paths are recorded, not automatically interpreted as proof of equivalence.'}
        if not source_check['all_accepted']:
            raise ValueError('Critical source files differ without a verified source amendment')
    files = sorted({file.resolve() for root in roots for file in Path(root).rglob('run.json')})
    records = {}
    signature = None
    protocols, scene_counts = {}, {}
    for file in files:
        run = read_json(file)
        mode = run.get('config', {}).get('mode')
        if mode not in ('E0', 'E2'):
            continue
        if run.get('exit_status') != 'completed':
            raise ValueError(f'Incomplete run: {file}')
        seed = run['seed']
        if run['config']['seed'] != seed:
            raise ValueError(f'Seed mismatch: {file}')
        if (seed, mode) in records:
            raise ValueError(f'Duplicate seed/mode {seed}/{mode}; provide only one completed run per pair')
        current_signature = training_signature(run)
        if signature is None:
            signature = current_signature
        elif current_signature != signature:
            raise ValueError(f'Training protocols differ beyond mode/seed: {file}')
        checkpoint_hash = sha256(file.parent/'model.pt')
        if checkpoint_hash != run['checkpoint_sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {file}')
        entry = {'seed': seed, 'mode': mode, 'run': str(file), 'checkpoint_sha256': checkpoint_hash, 'recorded_code_sha256': run['code_sha256'], 'splits': {}}
        for split, eval_name in (('val', 'evaluation.json'), ('test', 'evaluation_test.json')):
            evaluation = read_json(file.parent/eval_name)
            extra = read_json(file.parent/f'extra_{split}.json')
            if (evaluation['checkpoint_sha256'] != checkpoint_hash or extra['run']['input_sha256'] != checkpoint_hash or
                evaluation['dataset_sha256'] != run['dataset_sha256'] or extra['run']['dataset_sha256'] != run['dataset_sha256'] or
                evaluation['manifest_sha256'] != run['manifest_sha256'] or evaluation['split'] != split or extra['run']['split'] != split):
                raise ValueError(f'Evaluation provenance mismatch: {file}, {split}')
            if evaluation['model'] != run['config'] or extra['run']['model_config'] != run['config']:
                raise ValueError(f'Evaluation config mismatch: {file}, {split}')
            image_ids = [row['sample_id'] for row in extra['per_image']]
            if len(image_ids) != evaluation['samples'] or len(set(image_ids)) != len(image_ids):
                raise ValueError(f'Invalid evaluation sample identities: {file}, {split}')
            protocol = {'extra_protocol': extra['protocol'], 'ordered_sample_ids': image_ids,
                        'gt_objects': extra['totals']['gt_objects']}
            if split in protocols and protocols[split] != protocol:
                raise ValueError(f'Evaluation protocol/GT mismatch: {file}, {split}')
            protocols[split] = protocol
            # Scene identity is explicit in this project's GraspNet sample IDs.
            scenes = sorted({sid.split('_realsense_')[0].split('_kinect_')[0] for sid in image_ids})
            scene_counts[split] = {'images': len(image_ids), 'scenes': scenes, 'scene_count': len(scenes),
                                  'gt_objects': extra['totals']['gt_objects']}
            metrics = {key: (evaluation['metrics'][key] if key.startswith('mask_') else extra['metrics_macro'][key]) for key in METRICS}
            if any(value is not None and (not math.isfinite(value) or not 0 <= value <= 100) for value in metrics.values()):
                raise ValueError(f'Invalid metrics: {file}, {split}')
            entry['splits'][split] = metrics
        records[seed, mode] = entry
    if not records:
        raise ValueError('No completed E0/E2 runs found')
    seeds = sorted({seed for seed, _ in records})
    for seed in seeds:
        if (seed, 'E0') not in records or (seed, 'E2') not in records:
            raise ValueError(f'Unpaired seed {seed}; both E0 and E2 are required')
    per_seed = [{'seed': seed, 'methods': {mode: records[seed, mode] for mode in ('E0', 'E2')},
                 'paired_E2_minus_E0_pp': {split: {key: (records[seed, 'E2']['splits'][split][key]-records[seed, 'E0']['splits'][split][key]
                    if records[seed, 'E2']['splits'][split][key] is not None and records[seed, 'E0']['splits'][split][key] is not None else None)
                    for key in METRICS} for split in ('val', 'test')}} for seed in seeds]
    summary = {split: {'methods': {mode: {key: describe([records[seed, mode]['splits'][split][key] for seed in seeds])
                    for key in METRICS} for mode in ('E0', 'E2')},
                    'paired_E2_minus_E0_pp': {key: describe([row['paired_E2_minus_E0_pp'][split][key] for row in per_seed]) for key in METRICS}}
               for split in ('val', 'test')}
    return {'status': 'descriptive_only', 'matched_seeds': seeds, 'matched_seed_count': len(seeds),
            'units': 'All metrics percent 0–100; paired differences in percentage points.',
            'uncertainty_unit': 'Training seed, holding the same fixed scene split constant. Sample standard deviation uses ddof=1 across seeds, including paired differences. Views and objects are not independent replicates.',
            'not_estimated': 'No p-values, significance claims, confidence intervals, or uncertainty over unseen scenes. Three seeds do not establish population-level reliability; one test small object remains one object at every seed.',
            'selection_policy': 'This report does not select a method, seed or checkpoint. Preserve the development-only AP_small then AP rule; test values are descriptive.',
            'missing_value_policy': 'Unavailable metrics remain null; descriptive statistics include an explicit count and missing count. Standard deviation is null when fewer than two values exist.',
            'critical_source_verification': source_check, 'evaluation_regression_verification': regression_check,
            'common_training_protocol': signature, 'code_provenance': {'distinct_recorded_hashes': sorted({entry['recorded_code_sha256'] for entry in records.values()}), 'note': 'Recorded code hash covers all rgbd_seg/*.py and tools/*.py. Adding an analysis tool changes it without changing training. Distinct hashes require separate review of relevant model/training source; protocol matching alone does not prove code identity.'}, 'evaluation_counts': scene_counts,
            'evaluation_protocols': protocols, 'per_seed': per_seed, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, help='Roots searched recursively for run.json; do not mix old and V2 runs')
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-receipt', help='Prior JSON with critical_source_sha256 or code_sha256 per-file mapping; only listed critical files are compared')
    parser.add_argument('--source-amendment', help='Explicit reviewed files mapping with before_sha256, after_sha256 and reason for each changed critical file')
    parser.add_argument('--evaluation-regression', help='JSON with before/after evaluation paths; verify all12 COCO metrics and input identities exactly agree')
    args = parser.parse_args()
    for root in args.runs:
        if not Path(root).is_dir():
            parser.error(f'Run root does not exist: {root}')
    result = summarize(args.runs, args.source_receipt, args.source_amendment, args.evaluation_regression)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'output': str(output.resolve()), 'matched_seeds': result['matched_seeds'], 'summary': result['summary']}, indent=2))


if __name__ == '__main__':
    main()
