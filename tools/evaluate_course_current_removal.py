#!/usr/bin/env python3
"""Evaluate two independent removals from the adopted guarded640 DPSeg."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_course_overlap import read, write, digest, require, now, reference_rows, SEMANTICS, gpu_preflight
from tools.evaluate_course_swinl_removal import configure_and_load
from rgbd_seg.course_current_removal import VARIANTS, COMPONENTS

SOURCES = ('rgbd_seg/course_current_removal.py', 'tools/evaluate_course_current_removal.py',
           'tools/evaluate_course_overlap.py', 'tools/audit_course_point_refinement.py',
           'rgbd_seg/course_dafusion_eval.py')


def check_dependencies(plan):
    for name, checksum in plan['new_source_sha256'].items():
        require(digest(ROOT/name) == checksum, f'Frozen source changed: {name}')
    manifest = plan['current_default_manifest']
    require(digest(manifest['path']) == manifest['sha256'] and read(manifest['path']) == manifest['content'],
            'Current default manifest changed')
    for name, checksum in manifest['content']['source_sha256'].items():
        require(digest(ROOT/name) == checksum, f'Default source changed: {name}')
    from rgbd_seg.course_dpseg import OVERLAP_CONFIGS, DEFAULT_MERGE
    from rgbd_seg.course_guarded_merge import POLICY
    require(plan['variants'] == list(VARIANTS) and plan['components'] == COMPONENTS,
            'Removal definitions changed')
    require(plan['merge_strategy'] == DEFAULT_MERGE == 'global_guard_edge_consensus'
            and plan['geometry'] == json.loads(json.dumps(OVERLAP_CONFIGS['overlap40']))
            and plan['guarded_merge_policy'] == POLICY, 'Frozen inference policy changed')
    require(plan['gpus'] == ['5', '6', '7'], 'Unexpected physical GPU assignment')
    require(manifest['content']['default']['strategy'] == DEFAULT_MERGE
            and manifest['content']['default']['overlap'] == .4, 'Wrong current default')


def check_outputs(values, stats, image_id, plan):
    from rgbd_seg.course_submission import _Spec, _predictions
    require(tuple(values) == VARIANTS, 'Removal variant coverage changed')
    spec = _Spec([image_id], (1,), (1024, 1024), 100)
    require(json.loads(json.dumps(stats['geometry'])) == plan['geometry']
            and stats['model_views'] == 5 and stats['actual_model_input_size_verified']
            and stats['model_input_sizes'] == [[1024, 1024]]*5
            and stats['merge_strategy'] == plan['merge_strategy'], 'Actual view geometry changed')
    identities, scores = stats['selected_identity'], stats['selected_scores']
    require(len(identities) == len(set(tuple(item) for item in identities)) == len(scores)
            == stats['merging']['selected'] <= 100 and stats['point_identity_unchanged']
            and stats['removal_scores_and_order_unchanged'], 'Selected candidate identity changed')
    for predictions in values.values():
        normalized, _ = _predictions(predictions, spec, decode=False)
        require(normalized == predictions and [(p['category_id'], p['score']) for p in predictions]
                == [(1, value) for value in scores] and all(value >= .01 for value in scores),
                'Removal changed categories/scores/order or produced invalid RLE')


def engineering(output):
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_current_removal import predict_current_removals
    from rgbd_seg.course_dpseg import predict_dpseg
    from rgbd_seg.course_swinl_removal import predict_removals
    from tools.check_course_prediction_reload import compare_predictions
    plan = read(output/'protocol.json')
    check_dependencies(plan)
    adapter, heads = configure_and_load(plan)
    checks = []
    with CourseArchive(plan['archive'], split='val') as data, torch.inference_mode():
        require(all(image_id in data.image_ids for image_id in plan['engineering_val_ids']), 'Non-val gate ID')
        for image_id in plan['engineering_val_ids']:
            sample = load_sample(data, image_id, 'depth')
            values, stats, full, without_tiles, native = predict_current_removals(adapter, heads.depth, sample)
            check_outputs(values, stats, image_id, plan)
            independent_full, _ = predict_dpseg(adapter, heads.depth, sample, overlap=.4)
            independent_old, _ = predict_removals(adapter, heads.depth, sample)
            independent_native = adapter.predict_sample(sample, score_threshold=.01, max_detections=100)
            errors = dict(full=compare_predictions(independent_full, full, score_tolerance=0.),
                without_tiles=compare_predictions(independent_old['without_tiles'], without_tiles, score_tolerance=0.),
                native=compare_predictions(independent_native, native, score_tolerance=0.))
            checks.append(dict(image_id=image_id, full_without_tiles_native_exact=True,
                score_errors=errors, stats=stats, instances={name: len(values[name]) for name in VARIANTS}))
            print(json.dumps(checks[-1], allow_nan=False), flush=True)
    check_dependencies(plan)
    write(output/'engineering_gate.json', dict(status='passed', split='val', checks=checks,
        protocol_sha256=digest(output/'protocol.json'), ground_truth_used=False,
        parameter_changes=False, gpu=torch.cuda.get_device_name(0)))


def worker(output, rank):
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_current_removal import predict_current_removals
    from tools.check_course_prediction_reload import compare_predictions
    plan = read(output/'protocol.json')
    gate = read(output/'engineering_gate.json')
    require(gate['status'] == 'passed' and gate['protocol_sha256'] == digest(output/'protocol.json'), 'Invalid gate')
    check_dependencies(plan)
    adapter, heads = configure_and_load(plan)
    ids = plan['image_ids'][rank::len(plan['gpus'])]
    references = {name: reference_rows(plan['references'][name], ids)
                  for name in ('full', 'without_tiles', 'native')}
    began, counts = time.monotonic(), dict.fromkeys(VARIANTS, 0)
    with ExitStack() as stack, torch.inference_mode():
        data = stack.enter_context(CourseArchive(plan['archive'], split='test'))
        require(data.image_ids == plan['image_ids'], 'Official test coverage changed')
        streams = {name: stack.enter_context((output/f'{name}_rank{rank}.jsonl').open('x')) for name in VARIANTS}
        audit = stack.enter_context((output/f'checks_rank{rank}.jsonl').open('x'))
        for index, image_id in enumerate(ids):
            sample = load_sample(data, image_id, 'depth')
            require(set(sample) == {'image', 'depth_m', 'depth_valid'}, 'Unexpected inference input')
            values, stats, full, without_tiles, native = predict_current_removals(adapter, heads.depth, sample)
            check_outputs(values, stats, image_id, plan)
            for name, predictions in dict(full=full, without_tiles=without_tiles, native=native).items():
                compare_predictions(references[name].pop(image_id), predictions, score_tolerance=0.)
            for name in VARIANTS:
                streams[name].write(json.dumps(dict(image_id=image_id, predictions=values[name]), allow_nan=False)+'\n')
                counts[name] += len(values[name])
            audit.write(json.dumps(dict(image_id=image_id, full_without_tiles_native_exact=True,
                instances={name: len(values[name]) for name in VARIANTS}, stats=stats), allow_nan=False)+'\n')
            for stream in [*streams.values(), audit]:
                stream.flush()
            if index == 0 or (index+1) % 10 == 0 or index+1 == len(ids):
                progress = dict(status='inferring', rank=rank, images=index+1, expected=len(ids),
                                elapsed_seconds=time.monotonic()-began)
                write(output/f'progress_rank{rank}.json', progress)
                print(json.dumps(progress), flush=True)
            del sample, values, stats, full, without_tiles, native
    require(all(not rows for rows in references.values()), 'Uncompared reference images remain')
    check_dependencies(plan)
    write(output/f'rank{rank}.json', dict(status='completed', image_ids=ids, images=len(ids),
        gpu=torch.cuda.get_device_name(0), instances=counts, full_without_tiles_native_exact=True,
        ground_truth_in_prediction=False,
        artifacts={name: digest(output/f'{name}_rank{rank}.jsonl') for name in VARIANTS},
        checks_sha256=digest(output/f'checks_rank{rank}.jsonl')))


def score(output, variant):
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone_eval import merge_shards
    from rgbd_seg.course_dafusion_eval import score_predictions, validate_csv_fast
    from rgbd_seg.course_submission import SubmissionWriter
    from tools.audit_course_point_refinement import compare_csv_predictions
    plan = read(output/'protocol.json')
    check_dependencies(plan)
    ids, folder = plan['image_ids'], output/variant
    folder.mkdir()
    shards = [output/f'{variant}_rank{rank}.jsonl' for rank in range(len(plan['gpus']))]
    total = 0
    for rank, shard in enumerate(shards):
        receipt = read(output/f'rank{rank}.json')
        rank_ids = ids[rank::len(plan['gpus'])]
        require(receipt['status'] == 'completed' and receipt['image_ids'] == rank_ids
                and receipt['images'] == len(rank_ids) and receipt['full_without_tiles_native_exact']
                and not receipt['ground_truth_in_prediction'], 'Invalid rank receipt')
        require(receipt['artifacts'][variant] == digest(shard), 'Prediction shard changed')
        check_path = output/f'checks_rank{rank}.jsonl'
        require(receipt['checks_sha256'] == digest(check_path), 'Per-image checks changed')
        with check_path.open() as stream:
            checks = [json.loads(line) for line in stream]
        with shard.open() as stream:
            observed = [json.loads(line) for line in stream]
        require([row['image_id'] for row in checks] == [row['image_id'] for row in observed] == rank_ids
                and all(row['full_without_tiles_native_exact'] for row in checks), 'Incomplete reference verification')
        count = sum(len(row['predictions']) for row in observed)
        require(count == receipt['instances'][variant], 'Shard count differs')
        for row, audit in zip(observed, checks):
            require(len(row['predictions']) == audit['instances'][variant]
                    and [(p['category_id'],p['score']) for p in row['predictions']]
                    == [(1, value) for value in audit['stats']['selected_scores']], 'Selected candidates changed')
        total += count
    predictions = folder/'predictions.jsonl'
    count = merge_shards(shards, ids, predictions)
    require(count == total == plan['references']['full']['predicted_instances'], 'Merged instance count differs from full')
    with CourseArchive(plan['archive'], split='test') as data:
        require(data.image_ids == ids, 'Scoring archive IDs differ')
        evaluation = score_predictions(data, predictions, folder)
    reference = plan['references']['full']
    require(digest(reference['evaluation']) == reference['evaluation_sha256'], 'Full evaluation changed')
    baseline = read(reference['evaluation'])
    require({k:evaluation[k] for k in SEMANTICS} == {k:baseline[k] for k in SEMANTICS}, 'Evaluation settings differ')
    csv_path = output/'submissions'/f'dpseg_swinl_depth_no_k_guarded640_{variant}.csv'
    with SubmissionWriter(csv_path, ids) as writer, predictions.open() as stream:
        for line in stream:
            row = json.loads(line)
            writer.write(row['image_id'], row['predictions'])
    validation = validate_csv_fast(csv_path, ids)
    binding = compare_csv_predictions(csv_path, predictions, ids)
    require(binding == 2112 and validation['predictions'] == evaluation['predicted_instances'] == count,
            'CSV/RLE binding or instance count differs')
    write(folder/'csv_validation.json', validation)
    result = dict(status='completed', variant=variant, images=len(ids), predicted_instances=count,
        metrics=evaluation['metrics'], evaluation=str(folder/'evaluation.json'),
        evaluation_sha256=digest(folder/'evaluation.json'), predictions=str(predictions),
        predictions_sha256=digest(predictions), csv=str(csv_path), csv_sha256=digest(csv_path),
        csv_rows_identical_to_scored_predictions=binding)
    write(folder/'result.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)


def run_children(output, phase, specifications, timeout):
    children, logs = [], []
    began, timed_out = time.monotonic(), False
    try:
        for name, arguments, gpu in specifications:
            log = (output/f'{name}.log').open('x')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--output', str(output), *arguments],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            children.append((name, child))
        write(output/'status.json', dict(status=phase, started_at_utc=now(), controller_pid=os.getpid(),
            hard_timeout_seconds=timeout, processes=[dict(name=name,pid=child.pid) for name,child in children]))
        reported = set()
        while any(child.poll() is None for _, child in children):
            if time.monotonic()-began > timeout:
                timed_out = True
                print(f'{phase}: declared timeout; terminating only this phase children.', flush=True)
                raise TimeoutError(f'{phase} exceeded {timeout}s')
            for name, child in children:
                if child.poll() not in (None,0) and name not in reported:
                    reported.add(name)
                    print(f'{name} failed; remaining workers continue; no automatic retry.', flush=True)
            time.sleep(3)
        require(all(child.returncode == 0 for _,child in children), f'{phase} did not finish successfully')
    finally:
        if timed_out:
            for _,child in children:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
        write(output/f'{phase}_exits.json', {name:child.poll() for name,child in children})
        for log in logs:
            log.close()


def prepare(output):
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_dpseg import OVERLAP_CONFIGS
    from rgbd_seg.course_guarded_merge import POLICY
    manifest_path = ROOT/'configs/dpseg_current.json'
    manifest = read(manifest_path)
    ref_run = (ROOT/manifest['protocol']['path']).parent
    old, result, summary = (read(ref_run/name) for name in ('protocol.json','results.json','summary.json'))
    require(digest(ref_run/'protocol.json') == manifest['protocol']['sha256'] == result['protocol_sha256']
            and result['status'] == 'completed' and summary['status'] == 'verified', 'Unverified current reference')
    full = result['variants'][manifest['default']['strategy']]
    references = dict(full=full, without_tiles=old['without_tiles_reference'], native=old['native_reference'])
    for row in references.values():
        for key in ('evaluation','predictions'):
            require(digest(row[key]) == row[key+'_sha256'], 'Reference artifact changed')
    with CourseArchive(old['archive'], split='test') as data:
        require(data.image_ids == old['image_ids'] and len(data) == 2112, 'Require full official test set')
    gpus = ['5','6','7']
    capacity = gpu_preflight(gpus)
    disk = shutil.disk_usage(output.parent)
    require(disk.free >= 22*1024**3, 'Require 20 GiB reserve plus 2 GiB prediction budget')
    plan = {key:old[key] for key in ('archive','image_ids','weights','system_protocol','engineering_val_ids',
        'input_keys','base_epoch','head_epoch','precision','tf32','score_threshold','max_detections','camera_intrinsics_used')}
    source_names = list(dict.fromkeys([*manifest['source_sha256'], *SOURCES]))
    plan.update(created_at_utc=now(), gpus=gpus, gpu_capacity=capacity, variants=list(VARIANTS), components=COMPONENTS,
        authorized_request='补齐当前640×640/global_guard_edge_consensus完整方法的独立去点细化、去概率图细化两项消融',
        references=references, reference_protocol=dict(path=str(ref_run/'protocol.json'),sha256=digest(ref_run/'protocol.json')),
        reference_summary=dict(path=str(ref_run/'summary.json'),sha256=digest(ref_run/'summary.json')),
        current_default_manifest=dict(path=str(manifest_path),sha256=digest(manifest_path),content=manifest),
        new_source_sha256={name:digest(ROOT/name) for name in source_names},
        geometry=json.loads(json.dumps(OVERLAP_CONFIGS['overlap40'])), guarded_merge_policy=POLICY,
        merge_strategy='global_guard_edge_consensus', training=False, ground_truth_in_prediction=False,
        test_used_for_selection=False, default_changed=False, speed_benchmark=False,
        test_analysis='Exploratory fixed-protocol leave-one-module-out comparison on the previously examined '
            'complete local test set; no test-based retuning or validation-selected optimum.',
        comparison_scope='Identical selected candidate identities, original scores/order, five 1024-input views '
            'and checkpoints; independently remove only the learned point head or probability-depth filter.',
        formal_comparison='Exact full, native S1 and whole-image S3 reproduction on three engineering val '
            'images and every test image; reference outputs reused for final comparison.',
        storage=dict(free_bytes_before=disk.free, reserve_bytes=20*1024**3, save_features=False,
                     save_soft_probability=False, delete_existing_artifacts=False),
        material_passport=dict(source='ARS experiment-agent/code_runner', verification_status='PLANNED',
            scope='current guarded640 independent mask-refinement removals'))
    check_dependencies(plan)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',plan)
    write(output/'status.json',dict(status='prepared',at_utc=now()))


def run(output):
    plan = read(output/'protocol.json')
    require(read(output/'status.json')['status'] == 'prepared', 'Run already started')
    check_dependencies(plan)
    gpu_preflight(plan['gpus'])
    try:
        run_children(output,'engineering',[('engineering',['--engineering'],plan['gpus'][-1])],3600)
        gpu_preflight(plan['gpus'])
        run_children(output,'inferring',[(f'worker{rank}',['--worker',str(rank)],gpu) for rank,gpu in enumerate(plan['gpus'])],21600)
        run_children(output,'scoring',[(f'score_{name}',['--score',name],'') for name in VARIANTS],7200)
        check_dependencies(plan)
        results={name:read(output/name/'result.json') for name in VARIANTS}
        write(output/'results.json',dict(status='completed',finished_at_utc=now(),split='test',images=2112,
            protocol_sha256=digest(output/'protocol.json'),variants=results,references=plan['references'],
            training=False,test_used_for_selection=False,default_changed=False))
        write(output/'status.json',dict(status='completed',exit_code=0,finished_at_utc=now()))
    except BaseException as error:
        write(output/'status.json',dict(status='failed',error=str(error),at_utc=now()))
        raise
    with (output/'summary.log').open('x') as log:
        process=subprocess.run([sys.executable,'-u',str(ROOT/'tools/summarize_course_current_removal.py'),
            '--run',str(output),'--docs',str(ROOT/'docs/results')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    write(output/'summary_exit.json',dict(exit_code=process.returncode,at_utc=now()))
    require(process.returncode==0,'Completed inference/scoring but summary failed; inspect summary.log')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--run-prepared',action='store_true')
    parser.add_argument('--engineering',action='store_true')
    parser.add_argument('--worker',type=int)
    parser.add_argument('--score',choices=VARIANTS)
    args=parser.parse_args(); output=args.output.resolve()
    if args.engineering: engineering(output)
    elif args.worker is not None: worker(output,args.worker)
    elif args.score: score(output,args.score)
    elif args.run_prepared: run(output)
    else:
        prepare(output)
        if not args.prepare_only: run(output)


if __name__=='__main__':
    main()
