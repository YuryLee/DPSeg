#!/usr/bin/env python3
"""Freeze a convergence extension from completed results and explicit user stops."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_seg.course_continuation_state import validated_parent_state
from rgbd_seg.course_optimized import file_sha, tensor_digest
from tools.run_course_dafusion import now, write_json
from tools.run_course_longer_extension import sources


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stopped_candidate(entry, input_identity, evidence):
    """Retain completed epochs when the user explicitly stops an unfinished arm."""
    for name in ('protocol', 'run', 'selection', 'stop'):
        binding = entry[name]
        require(file_sha(binding['path']) == binding['sha256'], f'Stopped-arm evidence changed: {name}')
        evidence.append({'path': str(Path(binding['path']).resolve()), 'sha256': binding['sha256']})
    p, run, selection, stop = [read(entry[k]['path']) for k in ('protocol', 'run', 'selection', 'stop')]
    require(stop['status'] == 'stopped_by_user' and stop['requested_by'] == 'user'
            and stop['arm'] == entry['arm'] and stop['process_group_absent']
            and stop['worker_processes_absent'] and bool(stop.get('user_objective')),
            'Unfinished arm requires a verified explicit user stop')
    binding = stop['supervisor_record']
    require(file_sha(binding['path']) == binding['sha256'], 'Stopped-arm supervisor evidence changed')
    evidence.append(binding)
    process = read(binding['path'])
    trials = [v for v in process['trials'] if v['arm'] == entry['arm']]
    require(len(trials) == 1 and trials[0]['pid'] == stop['pid']
            and trials[0]['status'] == 'failed' and trials[0]['exit_code'] == stop['supervisor_exit_code'] != 0
            and trials[0]['protocol_sha256'] == entry['protocol']['sha256']
            and process['status'] == 'failed' and not process['active_process_alive'],
            'Stopped-arm supervisor did not observe the requested termination')
    require(p['status'] == 'frozen' and p['arm'] == entry['arm']
            and run['scope'] == 'full_train_val' and not run['test_used']
            and run['protocol_sha256'] == entry['protocol']['sha256'] and not selection['test_used']
            and p['selected_resolution'] == 1024 and run['verified_inputs'] == p['input_sha256'] == input_identity,
            'Stopped-arm input or protocol identity differs')
    parent = run['parent_validation']
    require(selection['parent_candidate'] == parent, 'Stopped-arm parent candidate differs')
    expected = selection['epochs']
    require([v['epoch'] for v in expected] == stop['completed_audited_epochs']
            == list(range(parent['epoch'] + 1, parent['epoch'] + 1 + len(expected)))
            and len(entry['audits']) == len(expected), 'Stopped-arm completed epoch coverage differs')
    rows = [{'epoch': parent['epoch'], 'AP': parent['metrics']['mask_ap'],
             'APs': parent['metrics']['mask_ap_small']}]
    for binding, recorded in zip(entry['audits'], expected):
        require(file_sha(binding['path']) == binding['sha256'], 'Stopped-arm epoch audit changed')
        evidence.append(binding)
        audit = read(binding['path'])
        require(audit['status'] == 'passed' and audit['scope'] == 'completed_epoch_only'
                and audit['arm'] == entry['arm'] and audit['protocol_sha256'] == entry['protocol']['sha256']
                and not audit['test_used'] and len(audit['epochs']) == 1,
                'Stopped-arm epoch was not completely audited')
        row = audit['epochs'][0]
        require(row['epoch'] == recorded['epoch'] and row['training_images'] == 14432
                and row['validation_images'] == 2584 and row['updates'] == p['training']['updates_per_epoch']
                and row['AP'] == recorded['metrics']['mask_ap'] and row['APs'] == recorded['metrics']['mask_ap_small'],
                'Stopped-arm completed validation differs')
        rows.append(row)
    require(all(math.isfinite(v[k]) and 0 <= v[k] <= 1 for v in rows for k in ('AP', 'APs')),
            'Invalid stopped-arm metric')
    best = max(rows, key=lambda v: (v['APs'], v['AP'], -v['epoch']))
    require(selection['best_epoch'] == best['epoch'] and selection['best_key'] == [best['APs'], best['AP']],
            'Stopped-arm selected result differs')
    return {'arm': entry['arm'], 'protocol': p, 'run': run, 'best': best,
            'protocol_path': entry['protocol']['path'], 'stopped_by_user': True,
            'folder': str(Path(entry['run']['path']).resolve().parent)}


def checked_review(path):
    """Recompute the cross-arm winner from completed validation-only evidence.

Review schema: status/selection_split/test_used/reason/selected_arm and
completed_arms=[{arm, protocol:{path,sha256}, run:{path,sha256},
                 audit:{path,sha256}}]. An explicitly cancelled arm may instead
appear in stopped_arms with protocol/run/selection/stop bindings and a list of
completed-epoch audits. Its completed scores still participate in selection.
No review is created by this builder.
    """
    review = read(path)
    require(review['status'] == 'extension_required_after_convergence_review'
            and review['selection_split'] == 'val' and review['test_used'] is False
            and isinstance(review.get('reason'), str) and review['reason'].strip(),
            'A validation-only convergence review requiring extension is needed')
    entries = review['completed_arms']
    arms = [v['arm'] for v in entries]
    stopped = review.get('stopped_arms', [])
    all_arms = arms + [v['arm'] for v in stopped]
    require(entries and len(all_arms) == len(set(all_arms)) and {'lr5e5', 'lr2e5'}.issubset(all_arms),
            'Both original learning-rate arms must finish or have a verified explicit user stop')
    evidence = [{'path': str(Path(path).resolve()), 'sha256': file_sha(path)}]
    candidates = []
    input_identity = None
    for entry in entries:
        for name in ('protocol', 'run', 'audit'):
            binding = entry[name]
            require(file_sha(binding['path']) == binding['sha256'], f'Review evidence changed: {name}')
            evidence.append({'path': str(Path(binding['path']).resolve()), 'sha256': binding['sha256']})
        p, run, audit = [read(entry[k]['path']) for k in ('protocol', 'run', 'audit')]
        require(p['status'] == 'frozen' and p['arm'] == entry['arm'], 'Reviewed protocol identity differs')
        require(run['status'] == 'completed' and run['scope'] == 'full_train_val' and not run['test_used']
                and run['protocol_sha256'] == entry['protocol']['sha256'], 'Reviewed training is incomplete')
        require(p['selected_resolution'] == 1024 and run['verified_inputs'] == p['input_sha256'],
                'Reviewed input identity or resolution differs')
        if input_identity is None:
            input_identity = p['input_sha256']
        require(p['input_sha256'] == input_identity, 'Reviewed arms used different data or initialization')
        require(audit['status'] == 'passed' and audit['scope'] == 'completed_base_arm'
                and audit['arm'] == entry['arm'] and not audit['test_used']
                and audit['protocol_sha256'] == entry['protocol']['sha256'], 'Completed arm audit differs')
        start = p['continuation']['parent_epoch']
        require([v['epoch'] for v in audit['epochs']] == list(range(start + 1, start + p['training']['epochs'] + 1)),
                'Completed audit does not cover all declared epochs')
        require(all(v['training_images'] == 14432 and v['validation_images'] == 2584
                    and v['updates'] == p['training']['updates_per_epoch'] for v in audit['epochs']),
                'Reviewed audit uses incomplete training or validation coverage')
        parent = run['parent_validation']
        rows = [{'epoch': parent['epoch'], 'AP': parent['metrics']['mask_ap'],
                 'APs': parent['metrics']['mask_ap_small']}, *audit['epochs']]
        require(all(math.isfinite(v[k]) and 0 <= v[k] <= 1 for v in rows for k in ('AP', 'APs')),
                'Invalid reviewed metric')
        best = max(rows, key=lambda v: (v['APs'], v['AP'], -v['epoch']))
        require(run['best_epoch'] == best['epoch']
                and run['best_validation_key'] == [best['APs'], best['AP']], 'Reviewed winner differs')
        candidates.append({'arm': entry['arm'], 'protocol': p, 'run': run, 'best': best,
                           'protocol_path': entry['protocol']['path'],
                           'folder': str(Path(entry['run']['path']).resolve().parent)})
    candidates.extend(stopped_candidate(v, input_identity, evidence) for v in stopped)
    candidates.sort(key=lambda v: (-v['best']['APs'], -v['best']['AP'], v['best']['epoch'], v['arm']))
    selected = candidates[0]
    require(review['selected_arm'] == selected['arm'], 'Extension parent is not the full-validation winner')
    require(not selected.get('stopped_by_user'), 'Do not restart an arm explicitly stopped by the user')
    p = selected['protocol']
    require(selected['best']['epoch'] == p['continuation']['parent_epoch'] + p['training']['epochs'],
            'The selected best is not a final-epoch checkpoint with saved optimizer state')
    return selected, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, help='Default: continue from the parent terminal learning rate')
    parser.add_argument('--minimum-ratio', type=float, default=.1)
    parser.add_argument('--arm', default=None)
    parser.add_argument('--engineering-receipt', type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Extension protocol output already exists')
    require(args.epochs > 0 and 0 < args.minimum_ratio <= 1, 'Invalid extension schedule')
    selected, evidence = checked_review(args.review)
    import torch
    torch.set_num_threads(4)
    parent_path = Path(selected['protocol_path']).resolve()
    weights = Path(selected['folder']) / 'last.pt'
    parent = torch.load(weights, map_location='cpu', weights_only=False)
    counters = validated_parent_state(parent, selected['protocol'], file_sha(parent_path),
                                      selected['run']['model_metadata'], selected['best']['epoch'])
    require(tensor_digest(parent['model']) == counters['selection']['model_sha256'],
            'Parent model differs from its validated checkpoint')
    start_lr = max(group['lr'] for group in parent['optimizer']['param_groups'])
    peak_lr = start_lr if args.lr is None else args.lr
    require(math.isfinite(start_lr) and math.isfinite(peak_lr) and 0 < start_lr <= peak_lr,
            'Extension peak must be positive and at least the parent terminal learning rate')
    p = copy.deepcopy(selected['protocol'])
    p.pop('engineering_validation', None)
    p.update(schema='course_longer_extension_v1', status='frozen', created_at_utc=now(), frozen_at_utc=now(),
             source_sha256=sources(), selected_resolution=1024, extension_evidence=evidence,
             arm=args.arm or f"extension_{selected['arm']}_e{counters['parent_epoch']:02d}")
    require(re.fullmatch(r'[a-z0-9_]+', p['arm']) is not None, 'Invalid extension arm name')
    for name, digest in p['source_sha256'].items():
        require(file_sha(ROOT/name) == digest, f'Runtime source changed: {name}')
    p['continuation'] = {
        'parent_epoch': counters['parent_epoch'], 'parent_optimizer_steps': counters['parent_optimizer_steps'],
        'weights': {'path': str(weights), 'sha256': file_sha(weights)},
        'protocol': {'path': str(parent_path), 'sha256': file_sha(parent_path)},
        'optimizer': 'Restore AdamW moments and cumulative steps; reset only the declared LR schedule',
        'rng': 'Reseed ranks using seed+rank; this is not exact stochastic resumption',
        'data_order': f"Continue deterministic image order and hflip at epoch index {counters['parent_epoch']}",
        'initial_candidate': 'Retain the validated parent if extension epochs do not improve APs/AP'}
    train = p['training']
    train.update(epochs=args.epochs, lr=peak_lr, total_updates=train['updates_per_epoch'] * args.epochs,
                 total_image_exposures=14432 * args.epochs,
                 schedule={'start_lr': start_lr, 'warmup_steps': 100 if peak_lr > start_lr else 0,
                           'minimum_ratio': args.minimum_ratio, 'kind': 'Optional warmup, then cosine decay'})
    p['hardware']['reason'] = 'Retain the audited five-GPU base-training layout; other devices are excluded'
    p['limitations'] = ['Single seed; best means best among tested validation candidates',
                        'Rank RNG is reseeded while AdamW state and absolute data epoch are retained',
                        'No test or Kaggle score is used to select the extension parent or epoch']
    p['next_stages'] = ['Audit the completed extension and review validation convergence',
                        'Select among all tested base checkpoints using complete validation',
                        'Fit point heads to the selected base and compare complete systems',
                        'Lock the system before test evaluation and CSV export']
    p['material_passport']['verification_status'] = 'ENGINEERING_PENDING'
    if args.engineering_receipt:
        gate = read(args.engineering_receipt)
        require(gate['status'] == 'passed' and gate['source_sha256'] == p['source_sha256']
                and gate['parent_checkpoint_sha256'] == p['continuation']['weights']['sha256']
                and gate['parent_protocol_sha256'] == p['continuation']['protocol']['sha256'],
                'Exact-parent extension engineering receipt differs')
        require(gate.get('scope') == 'engineering_only_not_model_quality'
                and gate.get('training_contract') == p['training']
                and gate.get('hardware_contract') == p['hardware']
                and gate.get('input_sha256') == p['input_sha256'],
                'Engineering training, hardware or input settings differ')
        p['engineering_validation'] = {'path': str(args.engineering_receipt.resolve()),
                                        'sha256': file_sha(args.engineering_receipt)}
        p['material_passport']['verification_status'] = 'FORMAL_PENDING'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, p)
    print(json.dumps({'path': str(args.output.resolve()), 'sha256': file_sha(args.output),
                      'parent_epoch': counters['parent_epoch'], 'additional_epochs': args.epochs,
                      'start_lr': start_lr, 'peak_lr': peak_lr}))


if __name__ == '__main__':
    main()
