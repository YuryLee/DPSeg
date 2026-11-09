#!/usr/bin/env python3
"""Train the RGB-base point head and evaluate DPSeg without RGB-D fusion.

Reuse the original point-training loop unchanged: two GPUs, global batch 8,
six epochs and full-validation head selection. Depth is retained in the two
mask refiners, but never enters the RGB encoder or Mask2Former decoder.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SELF = 'tools/run_course_rgb_point_ablation.py'
DEFAULT_OUTPUT = ROOT/'reports/course_swinl_without_rgbd_v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def prepare(output):
    require(not output.exists(), 'Use a fresh run directory')
    require(shutil.disk_usage(output.parent).free >= 24*1024**3, 'Insufficient disk reserve')
    manifest_path = ROOT/'configs/dpseg_current.json'
    manifest = read(manifest_path)
    rgb_protocol_path = ROOT/'reports/course_swinl_rgb_v1/protocol.json'
    rgb_protocol = read(rgb_protocol_path)
    rgb_result = read(ROOT/'reports/course_swinl_rgb_v1/results.json')
    reference_system_path = ROOT/manifest['system_protocol']['path']
    reference_system = read(reference_system_path)
    sources = dict(manifest['source_sha256'])
    for key, value in rgb_protocol['source_sha256'].items():
        require(key not in sources or sources[key] == value, 'Original source contracts disagree')
        sources[key] = value
    for name in ('rgbd_seg/course_rgb_points.py', SELF):
        sources[name] = sha(ROOT/name)
    for name, checksum in sources.items():
        require(sha(ROOT/name) == checksum, 'Frozen source changed: '+name)
    require(rgb_result['epoch'] == 15 and sha(rgb_result['checkpoint']) == rgb_result['checkpoint_sha256'],
            'RGB base epoch/checkpoint changed')
    require(reference_system['hardware']['physical_gpus'] == ['0', '2'], 'Reference point GPU layout differs')
    from rgbd_seg.course_archive import CourseArchive
    ids = {}
    for split, expected in (('train', 14432), ('val', 2584), ('test', 2112)):
        with CourseArchive(rgb_protocol['archive'], split=split) as dataset:
            require(len(dataset) == expected, 'Unexpected split size')
            ids[split] = dataset.image_ids
    val_predictions = ROOT/'reports/course_swinl_rgb_v1/base15/val_epoch_15/predictions.jsonl'
    require(val_predictions.is_file(), 'Missing exact RGB validation reference')
    output.mkdir(parents=True)
    protocol = dict(schema='dpseg-without-rgbd-fusion-point-training-v1', status='frozen',
        created_at_utc=now(), scope='full_experiment', archive=rgb_protocol['archive'],
        checkpoint=rgb_protocol['checkpoint'], rgb_base_protocol=artifact(rgb_protocol_path),
        base_checkpoint=artifact(rgb_result['checkpoint']), base_checkpoint_sha256=rgb_result['checkpoint_sha256'],
        base_training_receipt=artifact(ROOT/'reports/course_swinl_rgb_v1/base15/run.json'),
        source_sha256=rgb_protocol['source_sha256'], system_source_sha256=sources,
        current_default_manifest=artifact(manifest_path), reference_system_protocol=artifact(reference_system_path),
        native_val_reference=artifact(val_predictions), image_ids=ids,
        hardware=dict(physical_gpus=['0', '2'], memory_fraction=.55, minimum_free_gib=48),
        training=dict(seed=17, epochs=6, batch_per_rank=4, global_batch=8, precision='float32',
            optimizer='AdamW', learning_rate=.0003, weight_decay=.01, clip_norm=1.,
            schedule='Unchanged train_points warmup and cosine schedule',
            implementation='tools.run_course_depth_backbone_system.train_points',
            paired_heads=True, inference_head='depth', training_images_per_epoch=14432,
            expected_steps_per_epoch=1804, expected_total_steps=10824),
        model=dict(backbone='Swin-Large', rgbd_feature_fusion=False, base_modality='RGB',
            fine_channels=448, base_epoch=15, base_frozen=True, depth_encoding='depth',
            depth_used_in_point_refinement=True, depth_used_in_probability_refinement=True,
            camera_intrinsics_used=False),
        inference=manifest['default'], evaluation=dict(
            selection='Full validation point-depth APsmall, then AP, then earlier epoch',
            validation_images=2584, test_images=2112, test_used_for_selection=False,
            selected_checkpoint_test_only=True, ground_truth_in_prediction=False),
        limits=dict(per_trial_seconds=48*3600, external_trial_seconds=48*3600+300,
                    disk_reserve_gib=20),
        completed_training_reused=True, new_base_training=False,
        data_archive_sha256=rgb_protocol['input_sha256']['archive'],
        definition='Remove dual-branch RGB-D feature fusion only; retain both depth mask refiners and current four-tile merging.')
    require(protocol['inference']['tile_size_px'] == 640 and protocol['inference']['strategy'] == 'global_guard_edge_consensus',
            'Current inference policy changed')
    write(output/'protocol.json', protocol)
    write(output/'status.json', dict(status='prepared', created_at_utc=now(), protocol_sha256=sha(output/'protocol.json')))
    print(str(output/'protocol.json'), flush=True)


def check_protocol(output):
    path = output/'protocol.json'
    protocol = read(path)
    for name, checksum in protocol['system_source_sha256'].items():
        require(sha(ROOT/name) == checksum, 'Frozen source changed: '+name)
    for key in ('base_checkpoint', 'rgb_base_protocol', 'base_training_receipt',
                'current_default_manifest', 'reference_system_protocol', 'native_val_reference'):
        require(sha(protocol[key]['path']) == protocol[key]['sha256'], 'Frozen artifact changed: '+key)
    protocol['_sha256'] = sha(path)
    return protocol


def worker(output, phase):
    import torch
    import torch.distributed as dist
    from torch import nn
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_depth_backbone_points import PairedPointHeads, predict_paired
    from rgbd_seg.course_depth_backbone_eval import evaluate
    from rgbd_seg.course_dpseg import predict_dpseg
    from rgbd_seg.course_rgb_points import load_rgb_base
    from tools.check_course_prediction_reload import compare_predictions
    from tools.run_course_depth_backbone_system import train_points

    protocol = check_protocol(output)
    rank, local_rank, world = (int(os.environ[name]) for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    require(os.environ['CUDA_VISIBLE_DEVICES'].split(',') == protocol['hardware']['physical_gpus'] and world == 2,
            'Training GPU layout changed')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(protocol['hardware']['memory_fraction'], device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(hours=2), device_id=device)
    folder = output/({'engineering': 'engineering', 'training': 'points6', 'evaluation': 'final_evaluation'}[phase])
    began = time.monotonic()

    def guard():
        if time.monotonic()-began > protocol['limits']['per_trial_seconds']:
            raise TimeoutError('Declared phase timeout exceeded')
        if shutil.disk_usage(output).free < protocol['limits']['disk_reserve_gib']*1024**3:
            raise OSError('Disk reserve below20GiB')

    run = dict(status='loading', phase=phase, started_at_utc=now(), protocol_sha256=protocol['_sha256'])
    try:
        if rank == 0:
            folder.mkdir(exist_ok=False)
            write(folder/'run.json', run)
        dist.barrier()
        adapter, receipt = load_rgb_base(protocol['rgb_base_protocol']['path'], protocol['base_checkpoint']['path'],
            device=device, expected_protocol_sha256=protocol['rgb_base_protocol']['sha256'],
            expected_weights_sha256=protocol['base_checkpoint_sha256'])
        torch.manual_seed(protocol['training']['seed'])
        heads = PairedPointHeads(fine_channels=448).to(device)
        write(folder/f'base_loading_rank{rank}.json', receipt)
        if phase == 'engineering':
            expected_ids = protocol['image_ids']['val'][:2*world][rank::world]
            references = {}
            with Path(protocol['native_val_reference']['path']).open() as stream:
                for line in stream:
                    item = json.loads(line)
                    if item['image_id'] in expected_ids:
                        references[item['image_id']] = item['predictions']
            require(set(references) == set(expected_ids), 'Engineering reference coverage differs')
            checks = []
            with CourseArchive(protocol['archive'], split='val') as dataset, torch.inference_mode():
                heads.eval()
                for image_id in expected_ids:
                    sample = load_sample(dataset, image_id, 'depth')
                    paired = predict_paired(adapter, heads, sample, head_names=('point_depth',))
                    native = [value.public() for value in paired['native']]
                    point = [value.public() for value in paired['point_depth']]
                    compare_predictions(references[image_id], native, score_tolerance=0.)
                    compare_predictions(native, point, score_tolerance=0.)
                    predictions, stats = predict_dpseg(adapter, heads.depth, sample)
                    require(stats['model_views'] == 5 and stats['geometry']['tile_size_px'] == 640
                            and stats['merge_strategy'] == 'global_guard_edge_consensus'
                            and stats['actual_model_input_size_verified'] and len(predictions) <= 100,
                            'Engineering current inference settings differ')
                    checks.append(dict(image_id=image_id, native_exact=True, zero_head_exact=True,
                                       instances=len(predictions), stats=stats))
                    del paired, native, point, predictions, sample
            write(folder/f'native_gate_rank{rank}.json', checks)
            dist.barrier()
            # The unchanged original loop exercises real point sampling, DDP,
            # optimizer updates, validation merge and COCO scoring on a subset.
            train_points(protocol, adapter, heads, folder, rank, world, guard, True)
        elif phase == 'training':
            gate = read(output/'engineering_gate.json')
            require(gate['status'] == 'passed' and gate['protocol_sha256'] == protocol['_sha256'],
                    'Matching engineering gate is required before full training')
            train_points(protocol, adapter, heads, folder, rank, world, guard, False)
        else:
            selection = read(output/'points6/selection.json')
            require(selection['status'] == 'locked' and not selection['test_used']
                    and selection['epochs'] == 6 and selection['global_steps'] == 10824
                    and selection['training_images'] == 14432 and selection['base_unchanged']
                    and selection['base_gradients_absent'] and selection['head_replicas_equal'],
                    'Full training and validation selection have not completed')
            checkpoint = selection['selected']['checkpoint']
            require(sha(checkpoint) == selection['selected_head_sha256'], 'Selected head changed')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            require(saved['protocol_sha256'] == protocol['_sha256']
                    and saved['base_checkpoint_sha256'] == protocol['base_checkpoint_sha256']
                    and saved['source_sha256'] == protocol['system_source_sha256']
                    and saved['epoch'] == selection['selected']['epoch']
                    and saved['fine_channels'] == 448, 'Point head provenance differs')
            heads.load_state_dict(saved['heads'], strict=True)
            heads.eval().requires_grad_(False)
            del saved

            class FullPredictor(nn.Module):
                def __init__(self, base, head):
                    super().__init__()
                    self.adapter, self.head = base, head
                    self.depth_encoding = 'depth'

                @property
                def device(self):
                    return self.adapter.device

                @torch.inference_mode()
                def predict_sample(self, sample, **kwargs):
                    return predict_dpseg(self.adapter, self.head, sample)[0]

            predictor = FullPredictor(adapter, heads.depth).eval()
            result = evaluate(predictor, protocol['archive'], folder/'test', rank=rank,
                              world_size=world, guard=guard, split='test')
            if rank == 0:
                write(folder/'model_identity.json', dict(variant='without_rgbd_fusion', base_modality='RGB',
                    rgbd_feature_fusion=False, depth_used_in_refinement=True, camera_intrinsics_used=False,
                    selected_epoch=selection['selected']['epoch'], base=protocol['base_checkpoint'],
                    head=artifact(checkpoint), model_input_size=[1024, 1024], inference=protocol['inference'],
                    metrics=result['metrics']))
        dist.barrier()
        if rank == 0:
            run.update(status='completed', finished_at_utc=now(), elapsed_seconds=time.monotonic()-began)
            write(folder/'run.json', run)
    except BaseException as error:
        if folder.exists():
            write(folder/f'failure_rank{rank}.json', dict(error=str(error), type=type(error).__name__, at_utc=now()))
        raise
    finally:
        dist.destroy_process_group()


def process(output, phase):
    protocol = check_protocol(output)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(protocol['hardware']['physical_gpus']),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    command = [str(ROOT/'.venv/bin/python'), '-u', '-m', 'torch.distributed.run', '--standalone',
               '--nnodes=1', '--nproc_per_node=2', str(ROOT/SELF), 'worker', '--output', str(output), '--phase', phase]
    with (output/f'{phase}.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        state = dict(status=phase, started_at_utc=now(), controller_pid=os.getpid(),
                     process_pid=child.pid, command=command, protocol_sha256=protocol['_sha256'])
        write(output/'status.json', state)
        while child.poll() is None:
            if time.monotonic()-started > protocol['limits']['external_trial_seconds']:
                import signal
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=60)
                raise TimeoutError('Declared external phase timeout exceeded')
            state.update(last_checked_at_utc=now(), elapsed_seconds=time.monotonic()-started,
                         log_bytes=(output/f'{phase}.log').stat().st_size)
            write(output/'status.json', state)
            time.sleep(10)
        write(output/f'{phase}_exit.json', dict(exit_code=child.returncode, finished_at_utc=now(),
                                              elapsed_seconds=time.monotonic()-started, pid=child.pid))
        require(child.returncode == 0, f'{phase} failed, inspect {phase}.log')


def finish(output):
    from tools.audit_course_point_refinement import compare_csv_predictions
    protocol = check_protocol(output)
    evaluation = output/'final_evaluation/test'
    result = read(evaluation/'evaluation.json')
    validation = read(evaluation/'submission_validation.json')
    evaluation_receipt = read(evaluation/'evaluation_run.json')
    selection = read(output/'points6/selection.json')
    ids = protocol['image_ids']['test']
    require(result['status'] == 'completed' and result['images'] == len(ids) == 2112,
            'Incomplete test evaluation')
    predictions_sha = sha(evaluation/'predictions.jsonl')
    require(result['predictions_sha256'] == predictions_sha
            and evaluation_receipt['status'] == 'completed'
            and evaluation_receipt['split'] == 'test' and evaluation_receipt['images'] == 2112
            and evaluation_receipt['predictions_sha256'] == predictions_sha
            and evaluation_receipt['evaluation_sha256'] == sha(evaluation/'evaluation.json')
            and evaluation_receipt['predicted_instances'] == result['predicted_instances']
                == validation['predictions'], 'Scored predictions or evaluation receipt changed')
    source_csv = evaluation/'submission.csv'
    require(validation['sha256'] == sha(source_csv) and validation['rows'] == 2112
            and validation['rle_native_decode_verified'] and validation['rle_runs_verified']
            and validation['id_order_verified'] and validation['image_size'] == [1024, 1024]
            and validation['category_ids'] == [1] and validation['max_detections'] == 100,
            'CSV validation incomplete')
    for phase in ('engineering', 'training', 'evaluation'):
        require(read(output/f'{phase}_exit.json')['exit_code'] == 0, 'Phase did not exit successfully: '+phase)
    count = compare_csv_predictions(source_csv, evaluation/'predictions.jsonl', ids)
    require(count == 2112, 'CSV differs from scored predictions')
    (output/'submissions').mkdir(exist_ok=False)
    target = output/'submissions/dpseg_swinl_without_rgbd_fusion_guarded640.csv'
    os.link(source_csv, target)
    result_record = dict(status='completed', variant='without_rgbd_fusion', images=2112,
        metrics=result['metrics'], evaluation=str(evaluation/'evaluation.json'),
        evaluation_sha256=sha(evaluation/'evaluation.json'), predictions=str(evaluation/'predictions.jsonl'),
        predictions_sha256=sha(evaluation/'predictions.jsonl'), csv=str(target), csv_sha256=sha(target),
        csv_rows_identical_to_scored_predictions=count, predicted_instances=validation['predictions'],
        protocol=artifact(output/'protocol.json'), head_selection=artifact(output/'points6/selection.json'),
        selected_epoch=selection['selected']['epoch'], base=protocol['base_checkpoint'],
        head=artifact(selection['selected']['checkpoint']), rgbd_feature_fusion=False,
        depth_used_in_point_refinement=True, depth_used_in_probability_refinement=True,
        camera_intrinsics_used=False, test_used_for_selection=False, inference=protocol['inference'],
        finished_at_utc=now())
    write(output/'results.json', result_record)


def controller(output):
    protocol = check_protocol(output)
    try:
        query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
        available = {row.split(',')[0].strip(): int(row.split(',')[1]) for row in query.splitlines()}
        require(all(available[g] >= protocol['hardware']['minimum_free_gib']*1024
                    for g in protocol['hardware']['physical_gpus']), 'Reserved training GPUs are not sufficiently free')
        process(output, 'engineering')
        smoke = read(output/'engineering/selection.json')
        require(smoke['epochs'] == 1 and smoke['global_steps'] == 2
                and smoke['base_unchanged'] and smoke['base_gradients_absent'] and smoke['head_replicas_equal'],
                'Distributed training smoke failed')
        checks = [item for rank in range(2) for item in read(output/f'engineering/native_gate_rank{rank}.json')]
        require(len(checks) == 4 and all(item['native_exact'] and item['zero_head_exact'] for item in checks),
                'RGB reference or zero-head gate failed')
        write(output/'engineering_gate.json', dict(status='passed', protocol_sha256=protocol['_sha256'],
            native_checks=checks, training_smoke=artifact(output/'engineering/selection.json'),
            ground_truth_used_only_for_training_and_scoring=True, passed_at_utc=now()))
        process(output, 'training')
        process(output, 'evaluation')
        finish(output)
        write(output/'status.json', dict(status='completed', exit_code=0, finished_at_utc=now(),
                                        results=str(output/'results.json')))
    except BaseException as error:
        write(output/'status.json', dict(status='failed', error=str(error), type=type(error).__name__, at_utc=now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'launch', 'controller', 'worker', 'finish'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--phase', choices=('engineering', 'training', 'evaluation'))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == 'prepare':
        prepare(output)
    elif args.command == 'launch':
        require(read(output/'status.json')['status'] == 'prepared', 'Only launch a newly prepared run')
        with (output/'controller.log').open('xb') as log:
            child = subprocess.Popen([str(ROOT/'.venv/bin/python'), '-u', str(ROOT/SELF), 'controller',
                                      '--output', str(output)], cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(output/'launch.json', dict(pid=child.pid, launched_at_utc=now(), detached=True))
        print(json.dumps(read(output/'launch.json')), flush=True)
    elif args.command == 'controller':
        controller(output)
    elif args.command == 'worker':
        require(args.phase is not None, 'Worker phase is required')
        worker(output, args.phase)
    else:
        finish(output)


if __name__ == '__main__':
    main()
