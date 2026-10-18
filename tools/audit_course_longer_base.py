#!/usr/bin/env python3
"""Bind continued-training scores to checkpoints, complete splits and updates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha, tensor_digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def checkpoint(path, protocol, digest):
    value = torch.load(path, map_location='cpu', weights_only=False)
    require(value['protocol_sha256'] == digest and value['source_sha256'] == protocol['source_sha256'],
            f'Checkpoint provenance differs: {path}')
    return value


def audit_epoch(folder, entry, protocol, protocol_sha, train_ids, val_ids):
    epoch, spec = entry['epoch'], protocol['training']
    world = len(protocol['hardware']['physical_gpus'])
    relative_epoch = epoch - protocol['continuation']['parent_epoch']
    count, updates, batch = len(train_ids), spec['updates_per_epoch'], spec['effective_batch_size']
    require(count == spec['images_per_epoch'] == entry['training_images'], 'Training size differs')
    require(updates == math.ceil(count / batch) and entry['global_step'] == relative_epoch * updates,
            'Epoch update count differs')
    order = read(folder / f'train_order_epoch_{epoch:02d}.json')
    require(order['epoch'] == epoch and len(order['image_ids']) == len(set(order['image_ids'])) == count
            and set(order['image_ids']) == set(train_ids), 'Training order coverage differs')
    rank_batches = order['rank_batches']
    require(len(rank_batches) == world and len({len(v) for v in rank_batches}) == 1,
            'Training rank batch coverage differs')
    recovered = []
    for micro in range(len(rank_batches[0])):
        for rank, local in enumerate(rank_batches):
            require(len(local[micro]) == spec['per_rank_batch_sizes'][rank], 'Local microbatch size differs')
            recovered.extend(train_ids[index] for index in local[micro])
    require(recovered == order['image_ids'], 'Rank batches do not partition the recorded training order')
    reference = None
    for rank in range(world):
        with (folder / f'train_rank{rank}.jsonl').open() as stream:
            rows = [row for line in stream if line.endswith('\n')
                    and (row := json.loads(line))['epoch'] == epoch]
        require([r['global_step'] for r in rows] == list(range((relative_epoch - 1) * updates + 1,
                                                             relative_epoch * updates + 1)),
                'Training update coverage differs')
        shared = []
        for step, row in enumerate(rows, 1):
            require(row['images_in_epoch'] == min(step * batch, count)
                    and row['global_batch_images'] == min(batch, count - (step - 1) * batch)
                    and row['total_global_step'] == protocol['continuation']['parent_optimizer_steps'] + row['global_step'],
                    'Training image/step counters differ')
            require(all(math.isfinite(row[key]) for key in ('loss', 'local_loss', 'grad_norm_before_clip', 'head_lr')),
                    'Non-finite training values')
            shared.append((row['loss'], row['images_in_epoch'], row['global_batch_images'], row['head_lr']))
        if reference is None:
            reference = shared
            mean = sum(r['loss'] * r['global_batch_images'] for r in rows) / count
            require(math.isclose(mean, entry['training_loss_mean'], rel_tol=1e-12), 'Epoch loss aggregation differs')
        else:
            require(shared == reference, 'Global training records disagree across ranks')
    sync = read(folder / f'model_sync_epoch_{epoch:02d}.json')
    require(sync == {'status': 'pass', 'world_size': world, 'model_sha256': entry['model_sha256']},
            'Model synchronization record differs')
    path = folder / f'epoch_{epoch:02d}.pt'
    values = checkpoint(path, protocol, protocol_sha)
    require(values['epoch'] == epoch and values['selection'] == entry
            and tensor_digest(values['model']) == entry['model_sha256'], 'Scored checkpoint tensor identity differs')
    del values
    val = Path(entry['output'])
    require(val.parent.resolve() == folder.resolve(), 'Validation directory is outside this arm')
    evaluation, receipt = read(val / 'evaluation.json'), read(val / 'evaluation_run.json')
    require(receipt['status'] == evaluation['status'] == 'completed'
            and receipt['scope'] == evaluation['scope'] == 'full_split'
            and receipt['split'] == evaluation['split'] == 'val'
            and receipt['images'] == evaluation['images'] == len(val_ids) == 2584
            and not receipt['ground_truth_in_inference'] and receipt['world_size'] == world,
            'Full validation status/scope differs')
    require(evaluation['metrics'] == entry['metrics'], 'Reported validation metrics differ')
    require(evaluation['iou_type'] == 'segm' and evaluation['max_detections'] == [1, 10, 100]
            and evaluation['area_ranges'][1] == [0, 1024]
            and evaluation['ground_truth_area'] == 'original annotation area preserved'
            and len(evaluation['iou_thresholds']) == 10
            and all(math.isclose(v, .5 + .05 * i) for i, v in enumerate(evaluation['iou_thresholds'])),
            'COCO APs evaluation semantics differ')
    for key in ('mask_ap', 'mask_ap_small'):
        require(math.isfinite(entry['metrics'][key]) and 0 <= entry['metrics'][key] <= 1, 'Invalid AP metric')
    for rank in range(world):
        shard = val / f'predictions_rank{rank}.jsonl'
        rank_receipt = read(val / f'rank{rank}.json')
        require(rank_receipt['status'] == 'completed' and rank_receipt['ordered_ids'] == val_ids[rank::world]
                and rank_receipt['prediction_sha256'] == file_sha(shard), 'Validation shard coverage/digest differs')
    merged = val / 'predictions.jsonl'
    require(file_sha(merged) == evaluation['predictions_sha256'] == receipt['predictions_sha256']
            and file_sha(val / 'evaluation.json') == receipt['evaluation_sha256'], 'Scored prediction binding differs')
    ids, instances = [], 0
    with merged.open() as stream:
        for line in stream:
            row = json.loads(line)
            ids.append(row['image_id'])
            predictions = row['predictions']
            require(len(predictions) <= 100 and all(v['category_id'] == 1 and math.isfinite(v['score'])
                    and 0 <= v['score'] <= 1 and v['segmentation']['size'] == [1024, 1024]
                    and isinstance(v['segmentation']['counts'], str) for v in predictions), 'Prediction schema differs')
            instances += len(predictions)
    require(ids == val_ids and instances == evaluation['predicted_instances'] == receipt['predicted_instances'],
            'Merged validation prediction coverage differs')
    return {'epoch': epoch, 'updates': updates, 'training_images': count, 'validation_images': len(ids),
            'AP': entry['metrics']['mask_ap'], 'APs': entry['metrics']['mask_ap_small'],
            'checkpoint': str(path.resolve()), 'checkpoint_sha256': file_sha(path),
            'model_tensor_sha256': entry['model_sha256'], 'predictions_sha256': file_sha(merged),
            'evaluation_sha256': receipt['evaluation_sha256']}


def audit(protocol_path, folder, control, only_epoch=None):
    p = read(protocol_path)
    digest = file_sha(protocol_path)
    require(p['status'] == 'frozen' and p['schema'] in ('course_longer_training_v1', 'course_longer_extension_v1'),
            'Invalid protocol')
    if p['schema'] == 'course_longer_extension_v1':
        require(bool(p.get('extension_evidence')), 'Missing extension review evidence')
        for item in p['extension_evidence']:
            require(file_sha(item['path']) == item['sha256'], 'Extension review evidence changed')
    for name, expected in p['source_sha256'].items():
        require(file_sha(ROOT / name) == expected, f'Frozen source changed: {name}')
    for item in (p['continuation']['weights'], p['continuation']['protocol']):
        require(file_sha(item['path']) == item['sha256'], 'Parent provenance differs')
    run, selection = read(folder / 'run.json'), read(folder / 'selection.json')
    require(run['protocol_sha256'] == digest and run['scope'] == 'full_train_val'
            and run['source_sha256'] == p['source_sha256'] and not run['test_used']
            and run['verified_inputs'] == p['input_sha256'] and not selection['test_used'], 'Run provenance differs')
    entries = selection['epochs']
    if only_epoch is None:
        require(run['status'] == 'completed', 'Full arm audit requires completed training')
        actual_inputs = {'archive': file_sha(p['archive']),
                         'pretrained': file_sha(Path(p['checkpoint']) / 'model.safetensors'),
                         'data_audit': file_sha(p['data_audit']['path'])}
        require(actual_inputs == p['input_sha256'], 'Experiment inputs changed after training')
        trial = [v for v in read(control / 'process.json')['trials'] if v['arm'] == p['arm']]
        require(len(trial) == 1 and trial[0]['status'] == 'completed' and trial[0]['exit_code'] == 0
                and trial[0]['protocol_sha256'] == digest, 'Supervisor did not observe a successful child exit')
        parent_epoch = p['continuation']['parent_epoch']
        require([v['epoch'] for v in entries] == list(range(parent_epoch + 1, parent_epoch + p['training']['epochs'] + 1)),
                'Not all declared epochs completed')
    else:
        entries = [v for v in entries if v['epoch'] == only_epoch]
        require(len(entries) == 1, 'Requested epoch has no completed selection entry')
    with CourseArchive(p['archive'], 'train') as data:
        train_ids = data.image_ids
    with CourseArchive(p['archive'], 'val') as data:
        val_ids = data.image_ids
    checked = [audit_epoch(folder, v, p, digest, train_ids, val_ids) for v in entries]
    if only_epoch is None:
        parent = torch.load(p['continuation']['weights']['path'], map_location='cpu', weights_only=False)
        parent_entry = parent['state']['val_results'][-1]
        require(parent_entry == selection['parent_candidate'] == run['parent_validation'], 'Parent candidate differs')
        winner = max([parent_entry, *entries], key=lambda r: (r['metrics']['mask_ap_small'], r['metrics']['mask_ap'], -r['epoch']))
        key = [winner['metrics']['mask_ap_small'], winner['metrics']['mask_ap']]
        require(selection['best_epoch'] == run['best_epoch'] == winner['epoch']
                and selection['best_key'] == run['best_validation_key'] == key, 'Best checkpoint selection differs')
        best = checkpoint(folder / 'best.pt', p, digest)
        require(best['selection'] == winner and tensor_digest(best['model']) == winner['model_sha256']
                and file_sha(folder / 'best.pt') == run['best_checkpoint_sha256'], 'Best checkpoint differs')
        del best
        last = checkpoint(folder / 'last.pt', p, digest)
        state, spec = last['state'], p['training']
        require(state['epoch'] == spec['epochs'] and state['global_step'] == spec['total_updates']
                and state['images_in_epoch'] == 0 and state['val_results'] == entries
                and tensor_digest(last['model']) == entries[-1]['model_sha256'], 'Last checkpoint state differs')
        require(run['optimizer_steps'] == spec['total_updates']
                and run['training_image_exposures'] == spec['total_image_exposures'], 'Final training counts differ')
        before, after = parent['optimizer']['state'], last['optimizer']['state']
        require(before.keys() == after.keys(), 'Optimizer parameter coverage differs')
        require(all(int(after[k]['step']) - int(before[k]['step']) == spec['total_updates'] for k in before),
                'AdamW moments were not continued for all declared updates')
        require(len(last['rank_rng']) == len(p['hardware']['physical_gpus']), 'Rank RNG states missing')
    return {'status': 'passed', 'scope': 'completed_epoch_only' if only_epoch is not None else 'completed_base_arm',
            'arm': p['arm'], 'at_utc': datetime.now(timezone.utc).isoformat(), 'epochs': checked,
            'auditor_sha256': file_sha(__file__),
            'protocol_sha256': digest, 'test_used': False,
            'material_passport': {'origin_skill': 'academic-research-suite/experiment-agent', 'mode': 'run',
                                 'verification_status': 'ACTUAL_ARTIFACTS_AUDITED'},
            'limitations': ['This does not complete point-head training or final-system evaluation',
                            'Training coverage binds the saved sampler partitions to consumed batch counts',
                            'Only completed epochs in this report have been audited']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--epoch', type=int, help='Audit only an immutable completed epoch, not the running arm')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.protocol, args.folder, args.control, args.epoch)
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'scope', 'arm', 'epochs')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
