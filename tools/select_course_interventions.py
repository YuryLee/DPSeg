#!/usr/bin/env python3
"""Compare completed independent crop/Focal arms with the paired native baseline."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.select_course_resolution import audit_trial, metric_key, read, require, sha
from tools.launch_course_crop_focal_ddp import load_protocols
from rgbd_seg.course_resolution import rank_batches


def choose_method(baseline, candidates, gates):
    """Apply each independent gate before ranking eligible models by APs/AP."""
    base_small, base_ap = metric_key(baseline)
    eligible = {'native': baseline}
    comparisons = {}
    for method in ('crop', 'focal'):
        small, ap = metric_key(candidates[method])
        gate = gates[method]
        gain, drop = small-base_small, base_ap-ap
        accepted = gain >= gate['APsmall_gain_at_least'] and drop <= gate['AP_drop_at_most']
        comparisons[method] = {'adopt': accepted, 'APsmall_gain': gain, 'AP_drop': drop, 'gate': gate}
        if accepted:
            eligible[method] = candidates[method]
    # Stable order resolves an exact tie: native, then crop, then focal.
    selected = max(eligible, key=lambda method: metric_key(eligible[method]))
    return {'recommended_method_for_next_stage': selected, 'comparisons': comparisons}


def check_baseline_settings(protocol, baseline):
    for field in ('hardware', 'archive', 'checkpoint', 'input_sha256'):
        require(protocol[field] == baseline[field], f'Baseline differs in {field}')
    # The test-policy prose names a different stage; both actual runs must
    # separately prove test_used=False through audit_trial.
    actual_eval = {k:v for k,v in protocol['evaluation'].items() if k != 'test_policy'}
    baseline_eval = {k:v for k,v in baseline['evaluation'].items() if k != 'test_policy'}
    require(actual_eval == baseline_eval, 'Baseline evaluation settings differ')
    # Schedule prose was clarified after baseline freezing; both runs use
    # the same frozen schedule implementation and numerical parameters.
    for key, value in baseline['training'].items():
        if key not in ('augmentation', 'schedule'):
            require(protocol['training'][key] == value, f'Baseline training differs in {key}')


def audit_actual_training(run, protocol, expected_path):
    method = protocol['intervention']['method']
    orders = [read(run/f'train_order_epoch_{epoch:02d}.json')['image_ids'] for epoch in range(1, 6)]
    expected = {}
    if method == 'crop':
        coverage = read(expected_path.parent/'coverage.json')
        require(sha(expected_path) == coverage['expected_manifest_sha256'], 'Expected crop manifest changed')
        require(coverage['seed'] == protocol['training']['seed'] and coverage['epochs'] == 5
                and coverage['policy'] == protocol['intervention']['policy'], 'Expected crop policy differs')
        with expected_path.open() as stream:
            expected = {(int(x['epoch']), int(x['image_id'])): bool(int(x['expected_crop']))
                        for x in csv.DictReader(stream)}
    counts = [{'epoch':e, 'images':0, 'cropped':0} for e in range(1, 6)]
    hashes = {}
    for rank in range(7):
        positions = rank_batches(list(range(14432)), rank=rank,
                                batch_sizes=protocol['training']['per_rank_batch_sizes'])
        path = run/f'method_audit_rank{rank}.jsonl'
        hashes[path.name] = sha(path)
        forwards = 0
        with path.open() as stream:
            for index, line in enumerate(stream):
                epoch, step = divmod(index, 438)
                require(epoch < 5, 'Extra actual training forwards')
                row = json.loads(line)
                require(row['method'] == method and row['rank'] == rank, 'Actual method/rank differs')
                require(row['focal_mask_loss_calls'] == (10 if method == 'focal' else 0),
                        'Actual mask criterion differs')
                require([x['image_id'] for x in row['images']] == [orders[epoch][i] for i in positions[step]],
                        'Actual training image order or rank batch differs')
                for item in row['images']:
                    cropped = item['crop']['applied']
                    want = expected[(epoch+1, item['image_id'])] if method == 'crop' else False
                    require(cropped == want, 'Actual crop differs from frozen deterministic policy')
                    require(item['input_hw'] == ([512, 512] if cropped else [1024, 1024]),
                            'Actual input shape differs')
                    counts[epoch]['images'] += 1
                    counts[epoch]['cropped'] += cropped
                forwards += 1
        require(forwards == 2190, 'Missing actual training forwards')
    require(all(x['images'] == 14432 for x in counts), 'Actual epoch coverage differs')
    return {'status':'passed', 'counts':counts, 'actual_forward_log_sha256':hashes,
            'expected_crop_manifest_sha256':sha(expected_path) if method == 'crop' else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocols', type=Path, nargs=2, required=True)
    p.add_argument('--trials', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--expected-crops', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite a comparison')
    protocols = load_protocols(args.protocols)
    protocol_hashes = [sha(x) for x in args.protocols]
    terminal = read(args.control/'exit.json')
    require(terminal['status'] == 'completed' and terminal['exit_code'] == 0
            and terminal['protocol_sha256'] == protocol_hashes, 'Supervisor not successfully complete')
    require([x['method'] for x in terminal['trials']] == ['crop', 'focal']
            and all(x['status'] == 'completed' and x['exit_code'] == 0 for x in terminal['trials']),
            'Both independent trials must complete first')
    decision_path = Path(protocols[0]['resolution_selection']['path'])
    resolution_decision = read(decision_path)
    resolution = resolution_decision['selected_resolution']
    baseline_record = resolution_decision['trials'][str(resolution)]
    baseline_path = Path(baseline_record['best_checkpoint']).parent
    baseline_protocol_path = Path(resolution_decision['protocol_path'])
    baseline_protocol = read(baseline_protocol_path)
    require(sha(baseline_protocol_path) == resolution_decision['protocol_sha256'], 'Baseline protocol changed')
    baseline = audit_trial(baseline_path, baseline_protocol, sha(baseline_protocol_path), resolution)
    require(baseline == baseline_record, 'Baseline differs from frozen resolution decision')
    trials, actual_audits, gates = {}, {}, {}
    for path, protocol, protocol_hash in zip(args.protocols, protocols, protocol_hashes):
        method = protocol['intervention']['method']
        run = args.trials.resolve()/method
        check_baseline_settings(protocol, baseline_protocol)
        trials[method] = audit_trial(run, protocol, protocol_hash, resolution)
        require(trials[method]['initial_model_sha256'] == baseline['initial_model_sha256'], 'Different initialization')
        for epoch in range(1, 6):
            name = f'train_order_epoch_{epoch:02d}.json'
            require(sha(run/name) == sha(baseline_path/name), 'Different training image order or batches')
        actual_audits[method] = audit_actual_training(run, protocol, args.expected_crops)
        gates[method] = protocol['adoption_gate']
    result = choose_method(baseline['metrics'], {m:r['metrics'] for m,r in trials.items()}, gates)
    result.update(status='completed', scope='Completed independent crop/Focal validation comparison',
        finished_at_utc=datetime.now(timezone.utc).isoformat(), resolution=resolution, baseline=baseline,
        trials=trials, actual_training_audits=actual_audits, test_used=False,
        resolution_decision_sha256=sha(decision_path), protocol_sha256=protocol_hashes,
        supervisor_exit_sha256=sha(args.control/'exit.json'), checker_sha256=sha(__file__),
        limitations=['Single seed; no significance claim',
                    'Depth-boundary consideration and final candidate/test/demo work remain separate',
                    'Crop changes visible image content despite equal image exposures'])
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status', 'recommended_method_for_next_stage', 'comparisons')}))


if __name__ == '__main__':
    main()
