#!/usr/bin/env python3
"""Finish the predeclared five-stage experiment after the matched RGB training."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha
from rgbd_seg.course_progressive_ablation import STAGES
from rgbd_seg.course_tiled_probability import TILE_POLICY, PROBABILITY_POLICY


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def spec(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def make_protocol(gpus):
    from tools.run_course_progressive_rgb import sources
    source_hashes = sources()
    for path in ('artifacts/protocols/base_epoch15.json', 'artifacts/protocols/point_e15.json'):
        source_hashes.update(read(ROOT/path)['source_sha256'])
    for path in ('rgbd_seg/course_portable.py', 'rgbd_seg/course_progressive_ablation.py',
                 'rgbd_seg/course_tiled_point.py', 'rgbd_seg/course_tiled_probability.py',
                 'rgbd_seg/course_point_guard.py', 'rgbd_seg/course_depth_refinement.py',
                 'tools/run_course_progressive_ablation.py', 'tools/supervise_course_progressive_ablation.py',
                 'tools/check_course_prediction_reload.py', 'tools/audit_course_point_refinement.py'):
        source_hashes[path] = file_sha(ROOT/path)
    manifest = ROOT/'configs/course_e15_point.json'
    data = read(manifest)['dataset']
    archive = ROOT.parent/'downloads/buaa_image_segmentation/blender.zip'
    if archive.stat().st_size != data['archive_bytes'] or file_sha(archive) != data['archive_sha256']:
        raise ValueError('Dataset archive identity differs')
    return dict(schema='course_progressive_ablation_v1', status='frozen', created_at_utc=now(),
        stages=list(STAGES), gpus=gpus, worker_timeout_seconds=8*3600,
        archive=str(archive), archive_sha256=data['archive_sha256'],
        coco_checkpoint=str(ROOT/'pretrained/mask2former_swin_tiny_coco_instance'),
        inputs=dict(manifest=spec(manifest),
            base=spec(ROOT/'reports/course_longer_training/formal_v1/lr5e5/epoch_15.pt'),
            head=spec(ROOT/'reports/course_longer_training/point_e15_v1/formal/experiment/heads_epoch_06.pt'),
            coco_model=spec(ROOT/'pretrained/mask2former_swin_tiny_coco_instance/model.safetensors')),
        source_sha256=source_hashes, tile_policy=TILE_POLICY, probability_policy=PROBABILITY_POLICY,
        score_threshold=.01, mask_threshold=.5, max_detections=100,
        rgb_fixed_epoch=15, rgbd_fixed_epoch=15, depth_head_epoch=6,
        rgb_pending=True, parameter_search=False,
        test_policy='Lock all five validation metrics and exact weight hashes before test; no test tuning',
        limitations=[
            'One seed, fixed epoch15 base weights and prior validation-selected epoch6 point head.',
            'RGB and RGB-D match 1024 input, 15 epochs, global batch33 and two-stage AdamW schedule; '
            'historical RGB-D GPU/local microbatch layout differs, so gradients are not bitwise matched.',
            'Adding RGB-D also adds a second encoder and DS/DC fusion parameters.',
            'Later stages reuse frozen RGB-D weights; only the point head has additional training.',
            'Local test scores use archive annotations and differ from Kaggle public subset.'])


def execute(protocol, output, group, phase, state, state_path):
    command = [sys.executable, '-u', str(ROOT/'tools/run_course_progressive_ablation.py'),
        '--protocol', str(protocol), '--output', str(output), '--group', group, '--phase', phase]
    log = output/f'{group}_{phase}_supervisor.log'
    with log.open('x') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        state.update(phase=f'{group}_{phase}', child_pid=proc.pid, child_command=command, child_log=str(log))
        while proc.poll() is None:
            state.update(last_process_check_at_utc=now(), child_elapsed_seconds=time.monotonic()-started)
            write_json(state_path, state)
            # Each inference worker has its own frozen timeout and exit monitor.
            if time.monotonic()-started > 20*3600:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
                raise TimeoutError('Evaluation supervisor hard timeout')
            time.sleep(10)
        state.setdefault('children', []).append(dict(phase=f'{group}_{phase}', pid=proc.pid,
            exit_code=proc.returncode, log=str(log), elapsed_seconds=time.monotonic()-started))
        write_json(state_path, state)
        if proc.returncode:
            raise RuntimeError(f'{group} {phase} failed (exit {proc.returncode}); no automatic retry')


def audit_training(control):
    from rgbd_seg.course_optimized import tensor_digest
    from rgbd_seg.course_portable import verify_sources
    records = {}
    for phase, epochs, steps in (('initial', 5, 2190), ('continuation', 10, 4380)):
        run = read(control/phase/'run.json')
        process = read(control/f'{phase}_process.json')
        protocol = read(control/f'{phase}_protocol.json')
        if (run['status'] != 'completed' or process['exit_code'] != 0 or run['test_used']
                or run['optimizer_steps'] != steps or run['training_image_exposures'] != 14432*epochs
                or run['protocol_sha256'] != file_sha(control/f'{phase}_protocol.json')
                or run['source_sha256'] != protocol['source_sha256']):
            raise ValueError(f'Incomplete RGB training: {phase}')
        verify_sources(protocol['source_sha256'])
        epoch_records = []
        for epoch in range(1 if phase == 'initial' else 6, 6 if phase == 'initial' else 16):
            folder = control/phase/f'val_epoch_{epoch:02d}'
            evaluation = read(folder/'evaluation_run.json')
            if (evaluation['status'] != 'completed' or evaluation['images'] != 2584 or evaluation['depth_used']
                    or evaluation['predictions_sha256'] != file_sha(folder/'predictions.jsonl')):
                raise ValueError('RGB validation missing or changed')
            epoch_records.append(spec(folder/'evaluation_run.json'))
        records[phase] = dict(run=spec(control/phase/'run.json'), process=spec(control/f'{phase}_process.json'),
                              epochs=epoch_records, protocol=spec(control/f'{phase}_protocol.json'))
    weights = control/'continuation/epoch_15.pt'
    value = torch.load(weights, map_location='cpu', weights_only=True)
    tensor_sha = tensor_digest(value['model'])
    if value['epoch'] != 15 or value['selection']['model_sha256'] != tensor_sha:
        raise ValueError('Final RGB epoch15 tensor identity differs')
    return dict(status='passed', audited_at_utc=now(), epochs=15, training_images_per_epoch=14432,
        validation_images_per_epoch=2584, optimizer_steps=6570, records=records,
        weights=spec(weights), protocol=spec(control/'continuation_protocol.json'),
        tensor_sha256=tensor_sha, test_used=False)


def lock_validation(output, protocol, rgbd_protocol):
    metrics, inputs = {}, {}
    for group, path in (('rgbd', rgbd_protocol), ('rgb', protocol)):
        run_path = output/f'{group}_val_run.json'
        run = read(run_path)
        if run['status'] != 'completed' or run['protocol_sha256'] != file_sha(path):
            raise ValueError('Validation group is incomplete or has changed')
        metric_path = output/f'{group}_val/metrics.json'
        group_metrics = read(metric_path)
        if len(read(output/f'{group}_val/image_ids.json')) != 2584:
            raise ValueError('Incomplete validation coverage')
        metrics.update(group_metrics)
        inputs[f'{group}_run'] = spec(run_path)
        inputs[f'{group}_metrics'] = spec(metric_path)
        inputs[f'{group}_protocol'] = spec(path)
    if set(metrics) != set(STAGES): raise ValueError('Incomplete five-arm validation')
    for values in metrics.values():
        if any(not math.isfinite(values[k]) or not 0 <= values[k] <= 1 for k in ('mask_ap', 'mask_ap_small')):
            raise ValueError('Invalid full validation metric')
    write_json(output/'validation_lock.json', dict(status='locked', locked_at_utc=now(),
        stages=list(STAGES), protocol_sha256=file_sha(protocol), validation_inputs=inputs,
        metrics=metrics, test_used=False, parameters_changed_after_validation=False,
        selection='All five predeclared configurations; no test-based variant selection'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rgb-control', type=Path, required=True)
    parser.add_argument('--gpus', default='4,5,6')
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.output.resolve(); control = args.rgb_control.resolve()
    output.mkdir(parents=True, exist_ok=False)
    gpus = [int(n) for n in args.gpus.split(',')]
    if len(set(gpus)) != len(gpus) or not gpus: raise ValueError('Invalid GPU list')
    state_path = output/'supervisor.json'
    state = dict(status='running', pid=os.getpid(), started_at_utc=now(), phase='preparing')
    write_json(state_path, state)
    try:
        p = make_protocol(gpus)
        rgbd_protocol = output/'rgbd_protocol.json'; write_json(rgbd_protocol, p)
        execute(rgbd_protocol, output, 'rgbd', 'val', state, state_path)
        state.update(phase='waiting_for_rgb_training'); write_json(state_path, state)
        start_wait = time.monotonic()
        while True:
            training = read(control/'supervisor.json')
            if training['status'] == 'failed': raise RuntimeError('RGB training failed; inspect its supervisor receipt')
            if training['status'] == 'completed': break
            # A stale status alone cannot prove the training supervisor is alive.
            os.kill(training['supervisor_pid'], 0)
            if time.monotonic()-start_wait > 48*3600: raise TimeoutError('RGB training wait expired')
            state.update(last_training_process_check_at_utc=now(), training_phase=training['phase'])
            write_json(state_path, state); time.sleep(30)
        audit = audit_training(control); write_json(output/'rgb_training_audit.json', audit)
        final = copy.deepcopy(p)
        final.update(rgb_pending=False, finalized_at_utc=now(),
            rgb={k:audit[k] for k in ('weights', 'protocol', 'tensor_sha256')})
        for name in ('weights', 'protocol'):
            final['inputs']['rgb_'+name] = final['rgb'][name]
        final['inputs']['rgb_training_audit'] = spec(output/'rgb_training_audit.json')
        protocol = output/'protocol.json'; write_json(protocol, final)
        execute(protocol, output, 'rgb', 'val', state, state_path)
        lock_validation(output, protocol, rgbd_protocol)
        execute(protocol, output, 'all', 'test', state, state_path)
        state.update(status='completed', phase='five_stage_evaluation_complete', completed_at_utc=now(),
            comparison=spec(output/'comparison.json'))
    except BaseException as error:
        state.update(status='failed', failed_at_utc=now(), error_type=type(error).__name__, error=str(error))
        write_json(state_path, state)
        raise
    write_json(state_path, state)


if __name__ == '__main__':
    main()
