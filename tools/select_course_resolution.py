#!/usr/bin/env python3
"""Audit completed paired runs before applying their frozen resolution gate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metric_key(metrics):
    values = tuple(metrics[k] for k in ('mask_ap_small', 'mask_ap'))
    require(all(type(v) in (float,int) and math.isfinite(v) and 0 <= v <= 1 for v in values),
            'Invalid AP selection metrics')
    return values


def select_resolution(high, low, gate):
    hs,ha = metric_key(high)
    ls,la = metric_key(low)
    gain,drop = hs-ls,la-ha
    accepted = gain >= gate['APsmall_gain_at_least'] and drop <= gate['AP_drop_at_most']
    return {'selected_resolution':1024 if accepted else 800,
            'adopt_1024':accepted, 'APsmall_gain':gain, 'AP_drop':drop,
            'gate':gate}


def audit_trial(directory, protocol, protocol_sha, resolution):
    run = read(directory/'run.json')
    train = protocol['training']
    require(run['status'] == 'completed' and run['scope'] == 'full_train_val', 'Trial is incomplete or smoke-only')
    require(run['resolution'] == resolution and run['protocol_sha256'] == protocol_sha,
            'Resolution/protocol identity differs')
    require(run['source_sha256'] == protocol['source_sha256'], 'Run sources differ from protocol')
    require(run['verified_inputs'] == protocol['input_sha256'], 'Run input identity differs')
    require(run['test_used'] is False and run['world_size'] == 7, 'Unexpected trial scope')
    require(run['optimizer_steps'] == train['total_updates'] and
            run['training_image_exposures'] == train['total_image_exposures'], 'Training budget incomplete')
    selection = read(directory/'selection.json')
    epochs = selection['epochs']
    require([row['epoch'] for row in epochs] == list(range(1,train['epochs']+1)), 'Missing or repeated validation epochs')
    for entry in epochs:
        epoch = entry['epoch']
        require(entry['training_images'] == 14432 and entry['global_step'] == epoch*438,
                'Epoch exposure/update count differs')
        validation = Path(entry['output']).resolve()
        require(validation.parent == directory.resolve(), 'Validation path outside this trial')
        receipt,score = read(validation/'evaluation_run.json'),read(validation/'evaluation.json')
        for record in (receipt,score):
            require(record['status'] == 'completed' and record['split'] == 'val'
                    and record['scope'] == 'full_split' and record['images'] == 2584,
                    'Full validation missing')
        require(score['metrics'] == entry['metrics'], 'Selection metrics differ from scored validation')
        metric_key(score['metrics'])
        require(sha(validation/'evaluation.json') == receipt['evaluation_sha256'], 'Evaluation hash changed')
        require(sha(validation/'predictions.jsonl') == receipt['predictions_sha256'] == score['predictions_sha256'],
                'Prediction hash changed')
        require(score['iou_type'] == 'segm' and score['max_detections'] == [1,10,100]
                and score['area_ranges'][1] == [0,1024]
                and len(score['iou_thresholds']) == 10
                and all(abs(v-(.5+i*.05)) < 1e-8 for i,v in enumerate(score['iou_thresholds'])),
                'Unexpected mask AP evaluation settings')
        sync = read(directory/f'model_sync_epoch_{epoch:02d}.json')
        require(sync['status'] == 'pass' and sync['world_size'] == 7
                and sync['model_sha256'] == entry['model_sha256'], 'Model sync receipt differs')
        order = read(directory/f'train_order_epoch_{epoch:02d}.json')
        require(len(order['image_ids']) == len(set(order['image_ids'])) == 14432, 'Incomplete image order')
        batches = order['rank_batches']
        require(len(batches) == 7 and all(len(b) == 438 for b in batches), 'Wrong rank batch count')
        indices = [i for step in range(438) for rank in batches for i in rank[step]]
        require(sorted(indices) == list(range(14432)), 'Training batch indices omit or duplicate images')
    best = max(epochs,key=lambda row: (*metric_key(row['metrics']),-row['epoch']))
    require(best['epoch'] == selection['best_epoch'] == run['best_epoch'], 'Best epoch selection differs')
    require(list(metric_key(best['metrics'])) == selection['best_key'] == run['best_validation_key'],
            'Best validation metrics differ')
    checkpoint_sha = sha(directory/'best.pt')
    require(checkpoint_sha == run['best_checkpoint_sha256'], 'Best checkpoint changed')
    return {'resolution':resolution,'best_epoch':best['epoch'],'metrics':best['metrics'],
            'best_checkpoint':str(directory/'best.pt'),'best_checkpoint_sha256':checkpoint_sha,
            'initial_model_sha256':run['initial_model_sha256'],
            'run_sha256':sha(directory/'run.json'),'selection_sha256':sha(directory/'selection.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--trials',type=Path,required=True)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require(not args.output.exists(),'Refusing to overwrite an existing resolution decision')
    protocol = read(args.protocol)
    require(protocol['status'] == 'frozen','Resolution protocol is not frozen')
    protocol_sha = sha(args.protocol)
    exit_receipt = read(args.control/'exit.json')
    require(exit_receipt['status'] == 'completed' and exit_receipt['exit_code'] == 0
            and exit_receipt['protocol_sha256'] == protocol_sha,'Supervisor has not completed successfully')
    require([t['resolution'] for t in exit_receipt['trials']] == [1024,800]
            and all(t['status'] == 'completed' and t['exit_code'] == 0 for t in exit_receipt['trials']),
            'Paired trials did not both complete successfully')
    for name,value in protocol['source_sha256'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT/name
        require(sha(path) == value, f'Frozen source changed: {name}')
    trials = {r:audit_trial(args.trials.resolve()/str(r),protocol,protocol_sha,r) for r in (1024,800)}
    require(trials[1024]['initial_model_sha256'] == trials[800]['initial_model_sha256'], 'Different initialization')
    for epoch in range(1,protocol['training']['epochs']+1):
        name = f'train_order_epoch_{epoch:02d}.json'
        require(sha(args.trials/'1024'/name) == sha(args.trials/'800'/name), 'Paired training image order differs')
    decision = select_resolution(trials[1024]['metrics'],trials[800]['metrics'],protocol['adoption_gate'])
    decision.update(status='completed',finished_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_path=str(args.protocol.resolve()),protocol_sha256=protocol_sha,
        trials=trials,test_used=False,scope='Select resolution for later independent crop and Focal arms',
        audit_source_sha256=sha(Path(__file__)),supervisor_exit_sha256=sha(args.control/'exit.json'),
        limitation='Single seed; this resolution choice does not establish improvement over historical best DA-Fusion')
    with args.output.open('x') as stream:
        json.dump(decision,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(decision,ensure_ascii=False))


if __name__ == '__main__':
    main()
