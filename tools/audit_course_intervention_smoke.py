#!/usr/bin/env python3
"""Audit actual crop/Focal seven-GPU smoke execution and prediction reload."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_seg.course_optimized import file_sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(run, protocol_path, reload_path):
    run = run.resolve()
    protocol = json.loads(protocol_path.read_text())
    method = protocol['intervention']['method']
    require(method in ('crop', 'focal'), 'Unknown method')
    state = json.loads((run/'run.json').read_text())
    require(state['status'] == 'completed' and state['scope'] == 'engineering_smoke'
            and state['world_size'] == 7 and not state['test_used'], 'Incomplete/non-engineering run')
    require(state['optimizer_steps'] == 3 and state['training_image_exposures'] == 77,
            'Wrong engineering budget')
    require(state['protocol_sha256'] == file_sha(protocol_path)
            and state['source_sha256'] == protocol['source_sha256'], 'Protocol/source mismatch')
    for name, digest in state['source_sha256'].items():
        require(file_sha(ROOT/name) == digest, f'Source changed: {name}')
    decision_ref = protocol['resolution_selection']
    require(file_sha(decision_ref['path']) == decision_ref['sha256'], 'Resolution decision changed')
    decision = json.loads(Path(decision_ref['path']).read_text())
    require(decision['status'] == 'completed' and state['resolution'] == decision['selected_resolution'],
            'Wrong selected resolution')
    require(state['initial_model_sha256'] == decision['trials'][str(state['resolution'])]['initial_model_sha256'],
            'Initialization differs from paired baseline')
    rows = [json.loads(x) for x in (run/'training_log.jsonl').read_text().splitlines()]
    require([x['global_step'] for x in rows] == [1, 2, 3]
            and [x['global_batch_images'] for x in rows] == [33, 33, 11], 'Wrong step/tail coverage')
    require(all(math.isfinite(x[k]) for x in rows for k in ('loss', 'grad_norm_before_clip')),
            'Nonfinite training values')
    order = json.loads((run/'train_order_epoch_01.json').read_text())['image_ids']
    require(len(order) == len(set(order)) == 77, 'Invalid engineering train order')
    expected_path = ROOT/'reports/course_crop_focal/coverage_v1/expected_training_crops.csv'
    with expected_path.open() as stream:
        expected = {int(x['image_id']): bool(int(x['expected_crop']))
                    for x in csv.DictReader(stream) if x['epoch'] == '1'}
    seen, applied, calls, audit_hashes = [], 0, [], {}
    for rank in range(7):
        path = run/f'method_audit_rank{rank}.jsonl'
        audit_hashes[path.name] = file_sha(path)
        batches = [json.loads(x) for x in path.read_text().splitlines()]
        require(len(batches) == 3, 'Missing actual training forwards')
        require([len(x['images']) for x in batches] == ([6, 6, 2] if rank < 4 else [3, 3, 1]),
                'Wrong per-rank actual tail sizes')
        for batch in batches:
            require(batch['rank'] == rank and batch['method'] == method, 'Wrong actual intervention')
            n_calls = batch['focal_mask_loss_calls']
            require(n_calls > 0 if method == 'focal' else n_calls == 0, 'Wrong mask criterion calls')
            calls.append(n_calls)
            for item in batch['images']:
                cropped = item['crop']['applied']
                require(cropped == (expected[item['image_id']] if method == 'crop' else False),
                        'Actual crop differs from independent expected flag')
                require(item['input_hw'] == ([512, 512] if cropped else [1024, 1024]),
                        'Wrong actual image size')
                applied += cropped
                seen.append(item['image_id'])
    require(len(seen) == len(set(seen)) == 77 and set(seen) == set(order), 'Actual train coverage mismatch')
    sync = json.loads((run/'model_sync_epoch_01.json').read_text())
    require(sync['status'] == 'pass' and sync['world_size'] == 7, 'Model synchronization failed')
    selection = json.loads((run/'selection.json').read_text())
    require(selection['best_epoch'] == 1 and len(selection['epochs']) == 1, 'Wrong engineering selection')
    entry = selection['epochs'][0]
    require(entry['model_sha256'] == sync['model_sha256'], 'Selected model differs from synchronized model')
    val = Path(entry['output']).resolve()
    require(val.is_relative_to(run), 'Foreign validation output')
    evaluation = json.loads((val/'evaluation_run.json').read_text())
    require(evaluation['status'] == 'completed' and evaluation['scope'] == 'engineering_smoke'
            and evaluation['images'] == 14 and evaluation['split'] == 'val', 'Wrong evaluation scope')
    require(file_sha(val/'predictions.jsonl') == evaluation['predictions_sha256']
            and file_sha(val/'evaluation.json') == evaluation['evaluation_sha256'], 'Evaluation artifacts changed')
    prediction_rows = [json.loads(x) for x in (val/'predictions.jsonl').read_text().splitlines()]
    require(len(prediction_rows) == len({x['image_id'] for x in prediction_rows}) == 14,
            'Validation image coverage mismatch')
    reload = json.loads(reload_path.read_text())
    require(reload['status'] == 'passed' and not reload['test_used'] and len(reload['checks']) >= 2,
            'Missing actual prediction reload check')
    require(reload['reload']['checkpoint_sha256'] == state['best_checkpoint_sha256'] == file_sha(run/'best.pt')
            and reload['reload']['model_sha256'] == sync['model_sha256']
            and reload['reference_predictions_sha256'] == evaluation['predictions_sha256'],
            'Prediction reload checked another checkpoint or reference')
    return {'status': 'passed', 'method': method, 'run': str(run), 'run_sha256': file_sha(run/'run.json'),
        'protocol_sha256': file_sha(protocol_path), 'source_sha256': state['source_sha256'],
        'resolution': state['resolution'], 'world_size': 7, 'optimizer_steps': 3,
        'training_images': 77, 'validation_images': 14, 'actual_crop_images': applied,
        'actual_whole_images': 77-applied, 'focal_calls_per_forward': sorted(set(calls)),
        'model_sha256': sync['model_sha256'], 'rank_peak_allocated_bytes': state['rank_peak_allocated_bytes'],
        'actual_intervention_audit_sha256': audit_hashes, 'reload_check': str(reload_path.resolve()),
        'reload_check_sha256': file_sha(reload_path), 'checker_sha256': file_sha(__file__),
        'limitations': 'Three-step engineering validation only; not method accuracy evidence'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--reload-check', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.run, args.protocol, args.reload_check)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status', 'method', 'actual_crop_images', 'focal_calls_per_forward')}))


if __name__ == '__main__':
    main()
