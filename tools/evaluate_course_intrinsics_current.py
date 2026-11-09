#!/usr/bin/env python3
"""Complete-system XYZ-vs-normalized-depth comparison using frozen DPSeg rules.

Only this entry point is new. Base, point, crop/K, probability, and current
candidate-merging implementations are reused without modifying frozen sources.
"""
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
SELF = 'tools/evaluate_course_intrinsics_current.py'
from tools.evaluate_course_overlap import read, write, digest, require, now, reference_rows, SEMANTICS, gpu_preflight
from tools.evaluate_course_swinl_removal import configure_and_load


def check_dependencies(plan):
    for name, checksum in plan['new_source_sha256'].items():
        require(digest(ROOT/name) == checksum, 'Frozen source changed: '+name)
    require(digest(plan['current_default_manifest']['path']) == plan['current_default_manifest']['sha256'], 'Current default changed')
    for row in plan['references'].values():
        for field in ('predictions', 'evaluation'):
            require(digest(row[field]) == row[field+'_sha256'], 'Reference artifact changed')
    require(plan['base_epoch'] == 15 and plan['head_epoch'] == 6 and not plan['training'], 'Fixed budget differs')


def predict(adapter, head, sample, *, legacy=False):
    import torch
    from rgbd_seg.course_depth_backbone_system import crop_sample
    from rgbd_seg.course_dpseg import DEFAULT_MERGE, OVERLAP_CONFIGS
    from rgbd_seg.course_guarded_merge import identity, select_candidates
    from rgbd_seg.course_swinl_removal import refine_candidates
    from rgbd_seg.course_tile_size import extract_view
    from rgbd_seg.course_tiled_probability import merge_candidates, refined_public
    expected = {'image', 'depth_m', 'depth_valid'} | ({'camera_K'} if adapter.depth_encoding == 'xyz' else set())
    require(set(sample) == expected and tuple(sample['image'].shape[-2:]) == (1024, 1024), 'Invalid inference fields')
    geometry = OVERLAP_CONFIGS['overlap40']
    cache, candidates, native = {}, [], None
    for view_id, bounds in enumerate(geometry['bounds']):
        view = sample if view_id == 0 else crop_sample(sample, bounds)
        if 'camera_K' in sample:
            left, top, _, _ = bounds
            affine = sample['camera_K'].new_tensor([[1, 0, -left], [0, 1, -top], [0, 0, 1]])
            require(torch.equal(view['camera_K'], affine @ sample['camera_K']), 'Crop K transform differs')
        current, cache[view_id] = extract_view(adapter, view, bounds, (1024, 1024), view_id)
        if view_id == 0:
            native = [c.public() for c in current]
        candidates.extend(current)
    selected, merging = select_candidates(candidates, DEFAULT_MERGE)
    refined = refine_candidates(adapter, head, selected, cache, (1024, 1024))
    require([identity(c) for c in refined] == [identity(c) for c in selected], 'Refinement changed identities')
    values, changes = refined_public(refined, sample, 'probability', adapter.device)
    expected_scores = [(1, c.score) for c in selected]
    require([(p['category_id'], p['score']) for p in values] == expected_scores, 'Refinement changed scores/order')
    old_values = None
    if legacy:
        old_selected, _ = merge_candidates(candidates)
        old_points = refine_candidates(adapter, head, old_selected, cache, (1024, 1024))
        old_values, _ = refined_public(old_points, sample, 'probability', adapter.device)
    return values, native, old_values, dict(geometry=geometry, merge_strategy=DEFAULT_MERGE,
        merging=merging, probability_changes=changes, model_views=5,
        selected_identity=[list(identity(c)) for c in selected], selected_scores=[c.score for c in selected],
        selected_count=len(selected), camera_intrinsics_used=adapter.depth_encoding == 'xyz',
        crop_intrinsics_verified=adapter.depth_encoding == 'xyz', no_ground_truth_in_prediction=True)


def engineering(output, arm, gpu):
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_depth_backbone_system import tiled_pair
    from rgbd_seg.course_dpseg import predict_dpseg
    from tools.check_course_prediction_reload import compare_predictions
    plan = read(output/'protocol.json'); check_dependencies(plan)
    load_plan = plan if arm == 'xyz' else plan['depth_load_plan']
    adapter, heads = configure_and_load(load_plan)
    checks = []
    ids = plan['engineering_val_ids']
    refs = {name:reference_rows(plan['references'][name], ids) for name in ('xyz_val_native', 'xyz_val_full')} if arm == 'xyz' else {}
    with CourseArchive(plan['archive'], split='val') as data, torch.inference_mode():
        for image_id in ids:
            sample = load_sample(data, image_id, arm if arm == 'xyz' else 'depth')
            values, native, old, stats = predict(adapter, heads.depth, sample, legacy=arm == 'xyz')
            if arm == 'xyz':
                compare_predictions(refs['xyz_val_native'][image_id], native, score_tolerance=0.)
                compare_predictions(refs['xyz_val_full'][image_id], old, score_tolerance=0.)
                original = tiled_pair(adapter, heads.depth, sample)[1]
                compare_predictions(original, old, score_tolerance=0.)
                compare_predictions(adapter.predict_sample(sample), native, score_tolerance=0.)
            else:
                original, original_stats = predict_dpseg(adapter, heads.depth, sample)
                compare_predictions(original, values, score_tolerance=0.)
                require(original_stats['merging'] == stats['merging'], 'Current merge statistics differ')
            checks.append(dict(image_id=image_id, split='val', exact_masks_scores_order=True, instances=len(values)))
            print(json.dumps(checks[-1]), flush=True)
    if arm == 'depth':
        test_ids = [plan['image_ids'][0], plan['image_ids'][-1]]
        refs = reference_rows(plan['references']['depth_current_full'], test_ids)
        with CourseArchive(plan['archive'], split='test') as data, torch.inference_mode():
            for image_id in test_ids:
                sample = load_sample(data, image_id, 'depth')
                values, _, _, _ = predict(adapter, heads.depth, sample)
                compare_predictions(refs[image_id], values, score_tolerance=0.)
                checks.append(dict(image_id=image_id, split='test', exact_masks_scores_order=True, current_saved_reference=True))
    check_dependencies(plan)
    write(output/f'engineering_{arm}_gpu{gpu}.json', dict(status='passed', checks=checks,
        protocol_sha256=digest(output/'protocol.json'), gpu=torch.cuda.get_device_name(0), no_ground_truth_in_prediction=True))


def worker(output, rank):
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_submission import _Spec, _predictions
    from tools.check_course_prediction_reload import compare_predictions
    plan = read(output/'protocol.json'); check_dependencies(plan)
    for name in plan['engineering_receipts']:
        gate = read(output/name)
        require(gate['status'] == 'passed' and gate['protocol_sha256'] == digest(output/'protocol.json'), 'Missing engineering gate')
    adapter, heads = configure_and_load(plan)
    ids = plan['image_ids'][rank::len(plan['gpus'])]
    native_refs = reference_rows(plan['references']['xyz_test_native'], ids)
    started, total = time.monotonic(), 0
    with ExitStack() as stack, torch.inference_mode():
        data = stack.enter_context(CourseArchive(plan['archive'], split='test'))
        require(data.image_ids == plan['image_ids'], 'Test IDs/order differs')
        stream = stack.enter_context((output/f'xyz_rank{rank}.jsonl').open('x'))
        audit = stack.enter_context((output/f'checks_rank{rank}.jsonl').open('x'))
        for index, image_id in enumerate(ids):
            sample = load_sample(data, image_id, 'xyz')
            values, native, _, stats = predict(adapter, heads.depth, sample)
            compare_predictions(native_refs.pop(image_id), native, score_tolerance=0.)
            normalized, _ = _predictions(values, _Spec([image_id], (1,), (1024, 1024), 100), decode=False)
            require(normalized == values, 'Prediction schema changed')
            stream.write(json.dumps(dict(image_id=image_id, predictions=values), allow_nan=False)+'\n')
            audit.write(json.dumps(dict(image_id=image_id, native_reference_exact=True, stats=stats), allow_nan=False)+'\n')
            stream.flush(); audit.flush(); total += len(values)
            if index == 0 or (index+1) % 10 == 0 or index+1 == len(ids):
                progress = dict(status='inferring', rank=rank, images=index+1, expected=len(ids), elapsed_seconds=time.monotonic()-started)
                write(output/f'progress_rank{rank}.json', progress); print(json.dumps(progress), flush=True)
            del sample, values, native, stats, normalized
    require(not native_refs, 'Native reference coverage incomplete'); check_dependencies(plan)
    write(output/f'rank{rank}.json', dict(status='completed', image_ids=ids, images=len(ids), instances=total,
        predictions_sha256=digest(output/f'xyz_rank{rank}.jsonl'), checks_sha256=digest(output/f'checks_rank{rank}.jsonl'),
        native_reference_exact=True, no_ground_truth_in_prediction=True, gpu=torch.cuda.get_device_name(0)))


def score(output):
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone_eval import merge_shards
    from rgbd_seg.course_dafusion_eval import score_predictions, validate_csv_fast
    from rgbd_seg.course_submission import SubmissionWriter
    from tools.audit_course_point_refinement import compare_csv_predictions
    plan = read(output/'protocol.json'); check_dependencies(plan)
    ids, total, shards = plan['image_ids'], 0, []
    folder = output/'xyz_current_full'; folder.mkdir()
    for rank in range(len(plan['gpus'])):
        receipt = read(output/f'rank{rank}.json'); shard = output/f'xyz_rank{rank}.jsonl'; audit = output/f'checks_rank{rank}.jsonl'
        rank_ids = ids[rank::len(plan['gpus'])]
        require(receipt['status'] == 'completed' and receipt['image_ids'] == rank_ids and receipt['no_ground_truth_in_prediction'], 'Worker incomplete')
        require(digest(shard) == receipt['predictions_sha256'] and digest(audit) == receipt['checks_sha256'], 'Worker artifact changed')
        rows = [json.loads(line) for line in shard.open()]; checks = [json.loads(line) for line in audit.open()]
        require([row['image_id'] for row in rows] == [row['image_id'] for row in checks] == rank_ids, 'ID coverage differs')
        for row, check in zip(rows, checks):
            require(check['native_reference_exact'] and check['stats']['crop_intrinsics_verified'], 'Inference verification absent')
            require([(p['category_id'], p['score']) for p in row['predictions']] == [(1,s) for s in check['stats']['selected_scores']], 'Candidates changed')
        count = sum(len(row['predictions']) for row in rows)
        require(count == receipt['instances'], 'Instance count differs'); total += count; shards.append(shard)
    predictions = folder/'predictions.jsonl'
    require(merge_shards(shards, ids, predictions) == total, 'Merge count differs')
    with CourseArchive(plan['archive'], split='test') as data:
        evaluation = score_predictions(data, predictions, folder)
    reference = read(plan['references']['depth_current_full']['evaluation'])
    require({k:evaluation[k] for k in SEMANTICS} == {k:reference[k] for k in SEMANTICS}, 'COCO evaluation semantics differ')
    csv_path = output/'submissions/dpseg_swinl_xyz_with_k_guarded640_full.csv'; csv_path.parent.mkdir()
    with SubmissionWriter(csv_path, ids) as writer, predictions.open() as stream:
        for line in stream:
            row = json.loads(line); writer.write(row['image_id'], row['predictions'])
    validation = validate_csv_fast(csv_path, ids); binding = compare_csv_predictions(csv_path, predictions, ids)
    require(binding == 2112 and validation['predictions'] == total == evaluation['predicted_instances'], 'CSV/prediction binding differs')
    write(folder/'csv_validation.json', validation)
    result = dict(status='completed', variant='xyz_current_full', images=2112, predicted_instances=total,
        metrics=evaluation['metrics'], evaluation=str(folder/'evaluation.json'), evaluation_sha256=digest(folder/'evaluation.json'),
        predictions=str(predictions), predictions_sha256=digest(predictions), csv=str(csv_path), csv_sha256=digest(csv_path),
        csv_rows_identical_to_scored_predictions=binding)
    write(folder/'result.json', result)
    write(output/'results.json', dict(status='completed', finished_at_utc=now(), split='test', images=2112,
        protocol_sha256=digest(output/'protocol.json'), variants=dict(xyz_current_full=result),
        depth_current_full=plan['references']['depth_current_full'], training=False, default_changed=False,
        scope='Camera intrinsics AND depth representation comparison with corresponding independently trained bases/heads; not a pure K toggle.'))
    print(json.dumps(result), flush=True)


def prepare(output, gpus):
    require(not output.exists(), 'Use a new output directory')
    require(shutil.disk_usage(output.parent).free >= 10*1024**3, 'Need 10 GiB free')
    prior = read(ROOT/'reports/course_swinl_current_removal_v1/protocol.json')
    xyz = ROOT/'reports/course_depth_backbone_v1_gpu02/swinl_xyz'
    system = xyz/'protocol_system.json'
    plan = {k:prior[k] for k in ('archive','image_ids','engineering_val_ids','current_default_manifest','base_epoch','head_epoch','precision','tf32','score_threshold','max_detections')}
    sources = dict(prior['new_source_sha256']); sources[SELF] = digest(ROOT/SELF)
    for name in ('rgbd_seg/course_dpseg.py','tools/evaluate_course_overlap.py','tools/check_course_prediction_reload.py','tools/audit_course_point_refinement.py'):
        sources[name] = digest(ROOT/name)
    refs = {}
    for name, path in [('xyz_val_native',xyz/'final_evaluation/val_native'),('xyz_val_full',xyz/'final_evaluation/val_full'),('xyz_test_native',xyz/'final_evaluation/test_native')]:
        refs[name] = {field:str(path/(field+'.jsonl' if field == 'predictions' else field+'.json')) for field in ('predictions','evaluation')}
        refs[name].update({field+'_sha256':digest(refs[name][field]) for field in ('predictions','evaluation')})
    refs['depth_current_full'] = read(ROOT/'reports/course_swinl_guarded_merge640_v1/global_guard_edge_consensus/result.json')
    plan.update(schema='dpseg-intrinsics-complete-system-v1', created_at_utc=now(), training=False,
        authorized_request='完整配置比较有无相机内参；缺失结果补跑评测', gpus=gpus,
        gpu_capacity=gpu_preflight(gpus), system_protocol=dict(path=str(system),sha256=digest(system)),
        weights={name:dict(path=str(path),sha256=digest(path)) for name,path in [('base',xyz/'base15/final.pt'),('heads',xyz/'points6/heads_epoch_06.pt')]},
        new_source_sha256=sources, references=refs, depth_load_plan=prior,
        base_selection='fixed epoch15',head_selection='corresponding full-validation selected epoch6',
        engineering_receipts=[f'engineering_xyz_gpu{gpus[0]}.json',f'engineering_xyz_gpu{gpus[-1]}.json',f'engineering_depth_gpu{gpus[0]}.json'],
        merge_strategy='global_guard_edge_consensus',tile_size_px=640,tiles=4,overlap=.4,
        point_refinement=True,probability_refinement=True,parameter_search=False,test_used_for_selection=False,
        test_analysis='Previously inspected local test split; fixed descriptive comparison, no new test tuning or significance claims.',
        comparison='XYZ+validity using K versus normalized Z+validity without K; each uses its matching trained base/head.',
        material_passport=dict(origin_skill='academic-research-suite',origin_mode='run',verification_status='PLANNED'))
    check_dependencies(plan)
    output.mkdir(parents=True); write(output/'protocol.json',plan); write(output/'status.json',dict(status='prepared',at_utc=now()))


def run_phase(output, phase, specifications, timeout):
    children, logs, started = [], [], time.monotonic()
    try:
        for name, args, gpu in specifications:
            log = (output/(name+'.log')).open('x'); logs.append(log)
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            child = subprocess.Popen([sys.executable,'-u',str(ROOT/SELF),'--output',str(output),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            children.append((name,child))
        write(output/'status.json',dict(status=phase,started_at_utc=now(),controller_pid=os.getpid(),hard_timeout_seconds=timeout,processes=[dict(name=n,pid=c.pid) for n,c in children]))
        while any(c.poll() is None for _,c in children):
            if time.monotonic()-started > timeout:
                for _,child in children:
                    if child.poll() is None: child.terminate()
                raise TimeoutError(phase+' hard timeout')
            time.sleep(3)
        require(all(c.returncode == 0 for _,c in children),phase+' failed; inspect logs')
    finally:
        write(output/(phase+'_exits.json'),{n:c.poll() for n,c in children})
        for log in logs: log.close()


def run(output):
    plan = read(output/'protocol.json'); require(read(output/'status.json')['status'] == 'prepared','Run already started')
    try:
        run_phase(output,'engineering_xyz',[(f'engineering_xyz_gpu{gpu}',['--engineering','xyz','--gpu',gpu],gpu) for gpu in (plan['gpus'][0],plan['gpus'][-1])],3600)
        run_phase(output,'engineering_depth',[(f'engineering_depth_gpu{plan["gpus"][0]}',['--engineering','depth','--gpu',plan['gpus'][0]],plan['gpus'][0])],3600)
        gpu_preflight(plan['gpus'])
        run_phase(output,'inferring',[(f'worker{rank}',['--worker',str(rank)],gpu) for rank,gpu in enumerate(plan['gpus'])],21600)
        run_phase(output,'scoring',[('score_xyz',['--score'],'')],7200)
        check_dependencies(plan); write(output/'status.json',dict(status='completed',finished_at_utc=now(),exit_code=0))
    except BaseException as error:
        write(output/'status.json',dict(status='failed',error=str(error),at_utc=now())); raise


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--gpus',default='0,2,5,6,7'); p.add_argument('--engineering',choices=['xyz','depth']); p.add_argument('--gpu')
    p.add_argument('--worker',type=int); p.add_argument('--score',action='store_true'); p.add_argument('--prepare-only',action='store_true'); p.add_argument('--run-prepared',action='store_true')
    a=p.parse_args(); output=a.output.resolve()
    if a.engineering: engineering(output,a.engineering,a.gpu)
    elif a.worker is not None: worker(output,a.worker)
    elif a.score: score(output)
    elif a.run_prepared: run(output)
    else:
        prepare(output,a.gpus.split(','))
        if not a.prepare_only: run(output)


if __name__ == '__main__': main()
