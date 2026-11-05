#!/usr/bin/env python3
"""Run three single-module removals from the frozen full Swin-Large DPSeg."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_course_swinl_ablation import read, digest, write

VARIANTS = ('full', 'without_point', 'without_probability', 'without_tiles')
NEW_SOURCES = ('rgbd_seg/course_swinl_removal.py', 'tools/evaluate_course_swinl_removal.py',
               'tools/evaluate_course_swinl_ablation.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def configure_and_load(plan):
    import torch
    from rgbd_seg.course_depth_backbone_points import PairedPointHeads
    from tools.run_course_depth_backbone_system import load_base, sources
    for name, checksum in plan['new_source_sha256'].items():
        if digest(ROOT/name) != checksum:
            raise ValueError(f'Frozen experiment code changed: {name}')
    system = read(plan['system_protocol']['path'])
    if digest(plan['system_protocol']['path']) != plan['system_protocol']['sha256']:
        raise ValueError('Original model protocol changed')
    if system['system_source_sha256'] != sources():
        raise ValueError('Original inference/model sources changed')
    for item in plan['weights'].values():
        if digest(item['path']) != item['sha256']:
            raise ValueError('Fixed inference weights changed')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    capacity = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 22*1024**3/capacity), 0)
    adapter = load_base(system, Path(plan['weights']['base']['path']), torch.device('cuda:0'))
    heads = PairedPointHeads(fine_channels=448)
    saved = torch.load(plan['weights']['heads']['path'], map_location='cpu', weights_only=True)
    if saved['epoch'] != 6 or saved['base_checkpoint_sha256'] != plan['weights']['base']['sha256']:
        raise ValueError('Expected the selected epoch6 head attached to the fixed epoch15 base')
    heads.load_state_dict(saved['heads'], strict=True)
    heads.eval().requires_grad_(False).to(adapter.device)
    return adapter, heads


def engineering(output):
    """Compare all four paths with separate original functions on fixed val IDs."""
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_depth_backbone_points import predict_paired
    from rgbd_seg.course_depth_backbone_system import tiled_pair, crop_sample
    from rgbd_seg.course_tiled_probability import view_bounds, infer_view, merge_candidates, refined_public
    from rgbd_seg.course_swinl_removal import predict_removals
    from tools.check_course_prediction_reload import compare_predictions
    plan = read(output/'protocol.json')
    adapter, heads = configure_and_load(plan)
    rows = []
    with CourseArchive(plan['archive'], split='val') as data, torch.inference_mode():
        for image_id in plan['engineering_val_ids']:
            sample = load_sample(data, image_id, 'depth')
            values, stats = predict_removals(adapter, heads.depth, sample)
            original_raw, original_full, merging = tiled_pair(adapter, heads.depth, sample)
            paired = predict_paired(adapter, heads, sample, head_names=('point_depth',))
            whole, _ = refined_public(paired['point_depth'], sample, 'probability', adapter.device)
            candidates = []
            for view_id, bounds in enumerate(view_bounds(1024, 1024)):
                view = sample if view_id == 0 else crop_sample(sample, bounds)
                candidates.extend(infer_view(adapter, view, bounds, (1024, 1024), view_id))
            selected, independent_merging = merge_candidates(candidates)
            no_point, _ = refined_public(selected, sample, 'probability', adapter.device)
            expected = dict(full=original_full, without_probability=original_raw,
                            without_point=no_point, without_tiles=whole)
            errors = {key: compare_predictions(expected[key], values[key], score_tolerance=0.)
                      for key in VARIANTS}
            if stats['merging'] != merging or merging != independent_merging:
                raise ValueError('Shared-view candidate merge differs from original paths')
            rows.append(dict(image_id=image_id, exact_masks_and_scores=True,
                score_errors=errors, instances={key: len(values[key]) for key in VARIANTS}))
            print(json.dumps(rows[-1]), flush=True)
    write(output/'engineering_gate.json', dict(status='passed', split='val', checks=rows,
        ground_truth_used=False, parameter_changes=False,
        protocol_sha256=digest(output/'protocol.json'), gpu=torch.cuda.get_device_name(0)))


def worker(output, rank):
    import torch
    from contextlib import ExitStack
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_swinl_removal import predict_removals
    from tools.check_course_prediction_reload import compare_predictions
    plan = read(output/'protocol.json')
    gate = read(output/'engineering_gate.json')
    if gate['status'] != 'passed' or gate['protocol_sha256'] != digest(output/'protocol.json'):
        raise ValueError('Engineering comparison has not passed for this protocol')
    adapter, heads = configure_and_load(plan)
    ids = plan['image_ids'][rank::len(plan['gpus'])]
    references = {}
    for variant, reference in plan['references'].items():
        if digest(reference['predictions']) != reference['predictions_sha256']:
            raise ValueError('Original reference predictions changed')
        selected_ids = set(ids)
        records = {}
        with Path(reference['predictions']).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['image_id'] in selected_ids:
                    records[row['image_id']] = row['predictions']
        if set(records) != selected_ids:
            raise ValueError('Reference image coverage differs')
        references[variant] = records
    began = time.monotonic()
    total_instances = {key: 0 for key in VARIANTS}
    with ExitStack() as stack, torch.inference_mode():
        data = stack.enter_context(CourseArchive(plan['archive'], split='test'))
        if data.image_ids != plan['image_ids']:
            raise ValueError('Test image IDs/order changed')
        streams = {key: stack.enter_context((output/f'{key}_rank{rank}.jsonl').open('x')) for key in VARIANTS}
        audit = stack.enter_context((output/f'checks_rank{rank}.jsonl').open('x'))
        for index, image_id in enumerate(ids):
            sample = load_sample(data, image_id, 'depth')
            values, stats = predict_removals(adapter, heads.depth, sample)
            # Require exact scores as well as masks; no relaxed accuracy tolerance.
            for variant, original in references.items():
                compare_predictions(original.pop(image_id), values[variant], score_tolerance=0.)
            for variant, predictions in values.items():
                streams[variant].write(json.dumps(dict(image_id=image_id, predictions=predictions), allow_nan=False)+'\n')
                total_instances[variant] += len(predictions)
            audit.write(json.dumps(dict(image_id=image_id, full_and_without_tiles_exact=True,
                unchanged_tiled_candidate_scores=True, **stats), allow_nan=False)+'\n')
            if index == 0 or (index+1) % 10 == 0 or index+1 == len(ids):
                for stream in [*streams.values(), audit]:
                    stream.flush()
                progress = dict(status='inferring', rank=rank, images=index+1, expected=len(ids),
                                elapsed_seconds=time.monotonic()-began)
                write(output/f'progress_rank{rank}.json', progress)
                print(json.dumps(progress), flush=True)
            del sample, values, stats
    write(output/f'rank{rank}.json', dict(status='completed', images=len(ids), image_ids=ids,
        gpu=torch.cuda.get_device_name(0), instances=total_instances,
        full_and_without_tiles_exact=True, ground_truth_in_prediction=False,
        artifacts={key: digest(output/f'{key}_rank{rank}.jsonl') for key in VARIANTS},
        checks_sha256=digest(output/f'checks_rank{rank}.jsonl')))


def score(output, variant):
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone_eval import merge_shards
    from rgbd_seg.course_dafusion_eval import score_predictions, validate_csv_fast
    from rgbd_seg.course_submission import SubmissionWriter
    from tools.audit_course_point_refinement import compare_csv_predictions
    plan = read(output/'protocol.json')
    ids = plan['image_ids']
    folder = output/variant
    folder.mkdir()
    shards = [output/f'{variant}_rank{rank}.jsonl' for rank in range(len(plan['gpus']))]
    for rank, shard in enumerate(shards):
        receipt = read(output/f'rank{rank}.json')
        if receipt['status'] != 'completed' or receipt['artifacts'][variant] != digest(shard):
            raise ValueError('Incomplete or changed shard')
    predictions = folder/'predictions.jsonl'
    count = merge_shards(shards, ids, predictions)
    with CourseArchive(plan['archive'], split='test') as data:
        evaluation = score_predictions(data, predictions, folder)
    if variant in plan['references']:
        reference = plan['references'][variant]
        if digest(reference['evaluation']) != reference['evaluation_sha256']:
            raise ValueError('Reference evaluation changed')
        if evaluation['metrics'] != read(reference['evaluation'])['metrics']:
            raise ValueError('Reproduced predictions did not reproduce every COCO metric exactly')
    csv_path = output/'submissions'/f'dpseg_swinl_depth_no_k_removal_{variant}.csv'
    with SubmissionWriter(csv_path, ids) as writer, predictions.open() as stream:
        for line in stream:
            row = json.loads(line)
            writer.write(row['image_id'], row['predictions'])
    receipt = validate_csv_fast(csv_path, ids)
    binding = compare_csv_predictions(csv_path, predictions, ids)
    write(folder/'csv_validation.json', receipt)
    result = dict(status='completed', variant=variant, images=len(ids), predicted_instances=count,
        metrics=evaluation['metrics'], evaluation=str(folder/'evaluation.json'),
        evaluation_sha256=digest(folder/'evaluation.json'), predictions=str(predictions),
        predictions_sha256=digest(predictions), csv=str(csv_path), csv_sha256=digest(csv_path),
        csv_rows_identical_to_scored_predictions=binding)
    write(folder/'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def run_children(output, phase, specifications, timeout):
    children, logs = [], []
    began = time.monotonic()
    try:
        for name, arguments, gpu in specifications:
            log = (output/f'{name}.log').open('x')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--output', str(output), *arguments]
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append((name, child))
        write(output/'status.json', dict(status=phase, started_at_utc=now(), hard_timeout_seconds=timeout,
            processes=[dict(name=name, pid=child.pid) for name, child in children]))
        while any(child.poll() is None for _, child in children):
            if time.monotonic()-began > timeout:
                for _, child in children:
                    if child.poll() is None:
                        child.terminate()
                        child.wait(timeout=30)
                raise TimeoutError(f'{phase} exceeded declared {timeout}s limit')
            time.sleep(3)
        exits = {name: child.returncode for name, child in children}
        write(output/f'{phase}_exits.json', exits)
        if any(exits.values()):
            raise RuntimeError(f'{phase} has unsuccessful processes: {exits}')
    finally:
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', type=Path, default=ROOT/'reports/course_swinl_depth_no_k_v1/swinl_depth_no_k')
    parser.add_argument('--archive', type=Path, default=ROOT.parent/'downloads/buaa_image_segmentation/blender.zip')
    parser.add_argument('--without-tiles-reference', type=Path, default=ROOT/'reports/course_swinl_report_ablation_v1/S3')
    parser.add_argument('--gpus', default='3,4,5,6,7')
    parser.add_argument('--engineering', action='store_true')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--score', choices=VARIANTS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.engineering:
        engineering(output)
        return
    if args.worker is not None:
        worker(output, args.worker)
        return
    if args.score:
        score(output, args.score)
        return
    from rgbd_seg.course_archive import CourseArchive
    run = args.run.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with CourseArchive(args.archive, split='test') as data:
        ids = data.image_ids
    with CourseArchive(args.archive, split='val') as data:
        engineering_ids = [data.image_ids[i] for i in (0, len(data)//2, len(data)-1)]
    if len(ids) != 2112 or len(set(ids)) != 2112:
        raise ValueError('Expected all 2112 official test images')
    weights = dict(base=run/'base15/final.pt', heads=run/'points6/heads_epoch_06.pt')
    references = {}
    for name, folder in [('full', run/'final_evaluation/test_full'), ('without_tiles', args.without_tiles_reference.resolve())]:
        evaluation, predictions = folder/'evaluation.json', folder/'predictions.jsonl'
        receipt = read(evaluation)
        if receipt['status'] != 'completed' or receipt['images'] != 2112 or receipt['predictions_sha256'] != digest(predictions):
            raise ValueError('Original full-test reference is not complete or changed')
        references[name] = dict(evaluation=str(evaluation), evaluation_sha256=digest(evaluation),
            predictions=str(predictions), predictions_sha256=digest(predictions))
    plan = dict(material_passport={'source':'academic-research-suite / experiment-agent run',
        'scope':'fixed-weight single-module removal, full local test', 'verification_status':'PLANNED'},
        authorized_request='分别去掉模块，先测试点细化、概率图细化、切片；RGB-D 融合最后再做',
        created_at_utc=now(), archive=str(args.archive.resolve()), image_ids=ids,
        engineering_val_ids=engineering_ids, gpus=args.gpus.split(','),
        weights={name: dict(path=str(path), sha256=digest(path)) for name, path in weights.items()},
        system_protocol=dict(path=str(run/'protocol_system.json'), sha256=digest(run/'protocol_system.json')),
        new_source_sha256={name: digest(ROOT/name) for name in NEW_SOURCES},
        references=references, training=False, test_used_for_selection=False,
        input_keys=['image', 'depth_m', 'depth_valid'], camera_intrinsics_used=False,
        base_epoch=15, head_epoch=6, precision='float32', tf32=False,
        score_threshold=.01, max_detections=100,
        variants={
            'full':dict(rgbd=True, point=True, probability=True, tiles=True),
            'without_point':dict(rgbd=True, point=False, probability=True, tiles=True),
            'without_probability':dict(rgbd=True, point=True, probability=False, tiles=True),
            'without_tiles':dict(rgbd=True, point=True, probability=True, tiles=False)},
        deferred=['without_rgbd_fusion'], ground_truth_in_prediction=False,
        engineering_comparison='3 fixed validation images; four original paths; exact masks and scores',
        formal_comparison='Every test image: full == original S4 and without_tiles == original S3, exactly',
        metric_comparison='Fresh COCO scoring for all four predictions; full/without_tiles metrics exactly match originals')
    write(output/'protocol.json', plan)
    try:
        run_children(output, 'engineering', [('engineering', ['--engineering'], plan['gpus'][-1])], 1800)
        specifications = [(f'worker{rank}', ['--worker', str(rank)], gpu) for rank, gpu in enumerate(plan['gpus'])]
        run_children(output, 'inferring', specifications, 10800)
        run_children(output, 'scoring', [(f'score_{variant}', ['--score', variant], '') for variant in VARIANTS], 3600)
        results = {variant: read(output/variant/'result.json') for variant in VARIANTS}
        write(output/'results.json', dict(status='completed', finished_at_utc=now(), split='test', images=2112,
            protocol_sha256=digest(output/'protocol.json'), variants=results, test_used_for_selection=False,
            training=False, deferred=['without_rgbd_fusion']))
        write(output/'status.json', dict(status='completed', exit_code=0, finished_at_utc=now()))
        print(json.dumps({name: row['metrics'] for name, row in results.items()}), flush=True)
    except BaseException as error:
        write(output/'status.json', dict(status='failed', error=str(error), at_utc=now()))
        raise


if __name__ == '__main__':
    main()
