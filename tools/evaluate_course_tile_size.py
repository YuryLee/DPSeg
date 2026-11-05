#!/usr/bin/env python3
"""Compare 512/640/768 tiles on the full local test set with frozen DPSeg."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
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
from tools.evaluate_course_swinl_removal import configure_and_load, NEW_SOURCES

SIZES = (512, 768)
SOURCES = (*NEW_SOURCES, 'rgbd_seg/course_tile_size.py', 'tools/evaluate_course_tile_size.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def reference_rows(reference, ids):
    if digest(reference['predictions']) != reference['predictions_sha256']:
        raise ValueError('Reference prediction file changed')
    wanted = set(ids)
    rows = {}
    with Path(reference['predictions']).open() as stream:
        for line in stream:
            item = json.loads(line)
            if item['image_id'] in wanted:
                if item['image_id'] in rows:
                    raise ValueError('Duplicate reference image')
                rows[item['image_id']] = item['predictions']
    if set(rows) != wanted:
        raise ValueError('Reference image coverage differs')
    return rows


def engineering(output):
    import torch
    from unittest.mock import patch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_depth_backbone_system import tiled_pair
    from rgbd_seg.course_tiled_probability import view_bounds
    from rgbd_seg.course_tile_size import axis_starts, tile_bounds, predict_sizes, geometry
    from tools.check_course_prediction_reload import compare_predictions
    plan = read(output/'protocol.json')
    if tile_bounds(1024, 1024, 640) != view_bounds(1024, 1024):
        raise ValueError('640 grid must reproduce the historical four-corner windows')
    for size, expected in [(512, [0, 410, 512]), (640, [0, 384]), (768, [0, 256])]:
        if axis_starts(1024, size, .2) != expected:
            raise ValueError('Grid differs from SAHI stride/end-alignment rule')
        cover = torch.zeros((1024, 1024), dtype=torch.int16)
        for left, top, right, bottom in tile_bounds(1024, 1024, size)[1:]:
            cover[top:bottom, left:right] += 1
        if not (cover > 0).all():
            raise ValueError('Tile grid leaves uncovered pixels')
    adapter, heads = configure_and_load(plan)
    checks = []
    with CourseArchive(plan['archive'], split='val') as data, torch.inference_mode():
        for image_id in plan['engineering_val_ids']:
            sample = load_sample(data, image_id, 'depth')
            predictions, stats, _ = predict_sizes(adapter, heads.depth, sample, (512, 640, 768))
            errors = {}
            for size in (512, 640, 768):
                # Independent legacy inference, changing only its window list.
                # The context manager restores the original function each time.
                with patch('rgbd_seg.course_depth_backbone_system.view_bounds',
                           lambda h, w, s=size: tile_bounds(h, w, s, .2)):
                    _, original, merging = tiled_pair(adapter, heads.depth, sample)
                errors[str(size)] = compare_predictions(original, predictions[str(size)], score_tolerance=0.)
                if merging != stats[str(size)]['merging']:
                    raise ValueError('Candidate merge differs from legacy implementation')
            checks.append(dict(image_id=image_id, exact_masks_and_scores=True, score_errors=errors))
            print(json.dumps(checks[-1]), flush=True)
    write(output/'engineering_gate.json', dict(status='passed', protocol_sha256=digest(output/'protocol.json'),
        checks=checks, geometry={str(s): geometry(s) for s in (512, 640, 768)},
        split='val', ground_truth_used=False, parameters_changed=False))


def worker(output, rank):
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_tile_size import predict_sizes
    from tools.check_course_prediction_reload import compare_predictions
    plan = read(output/'protocol.json')
    gate = read(output/'engineering_gate.json')
    if gate['status'] != 'passed' or gate['protocol_sha256'] != digest(output/'protocol.json'):
        raise ValueError('Engineering gate has not passed')
    adapter, heads = configure_and_load(plan)
    ids = plan['image_ids'][rank::len(plan['gpus'])]
    native_reference = reference_rows(plan['native_reference'], ids)
    checks_640 = reference_rows(plan['reference_640'], [ids[0], ids[-1]])
    began = time.monotonic()
    counts = {str(size): 0 for size in SIZES}
    with ExitStack() as stack, torch.inference_mode():
        data = stack.enter_context(CourseArchive(plan['archive'], split='test'))
        if data.image_ids != plan['image_ids']:
            raise ValueError('Official test IDs/order changed')
        streams = {str(s): stack.enter_context((output/f'tile{s}_rank{rank}.jsonl').open('x')) for s in SIZES}
        audit = stack.enter_context((output/f'checks_rank{rank}.jsonl').open('x'))
        for index, image_id in enumerate(ids):
            sample = load_sample(data, image_id, 'depth')
            sizes = (*SIZES, 640) if image_id in checks_640 else SIZES
            values, stats, native = predict_sizes(adapter, heads.depth, sample, sizes, plan['nominal_overlap_ratio'])
            compare_predictions(native_reference.pop(image_id), native, score_tolerance=0.)
            if '640' in values:
                compare_predictions(checks_640.pop(image_id), values.pop('640'), score_tolerance=0.)
            for size in SIZES:
                key = str(size)
                streams[key].write(json.dumps(dict(image_id=image_id, predictions=values[key]), allow_nan=False)+'\n')
                counts[key] += len(values[key])
            audit.write(json.dumps(dict(image_id=image_id, native_exact=True,
                reference_640_checked=index in (0, len(ids)-1), stats=stats), allow_nan=False)+'\n')
            if index == 0 or (index+1) % 10 == 0 or index+1 == len(ids):
                for stream in [*streams.values(), audit]:
                    stream.flush()
                progress = dict(status='inferring', rank=rank, images=index+1,
                    expected=len(ids), elapsed_seconds=time.monotonic()-began)
                write(output/f'progress_rank{rank}.json', progress)
                print(json.dumps(progress), flush=True)
            del sample, values, stats, native
    write(output/f'rank{rank}.json', dict(status='completed', image_ids=ids, images=len(ids),
        gpu=torch.cuda.get_device_name(0), instances=counts, native_exact=True,
        reference_640_exact_images=[ids[0], ids[-1]], ground_truth_in_prediction=False,
        artifacts={str(s): digest(output/f'tile{s}_rank{rank}.jsonl') for s in SIZES},
        checks_sha256=digest(output/f'checks_rank{rank}.jsonl')))


def score(output, size):
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone_eval import merge_shards
    from rgbd_seg.course_dafusion_eval import score_predictions, validate_csv_fast
    from rgbd_seg.course_submission import SubmissionWriter
    from tools.audit_course_point_refinement import compare_csv_predictions
    plan = read(output/'protocol.json')
    ids = plan['image_ids']
    folder = output/f'tile{size}'
    folder.mkdir()
    shards = [output/f'tile{size}_rank{rank}.jsonl' for rank in range(len(plan['gpus']))]
    for rank, shard in enumerate(shards):
        receipt = read(output/f'rank{rank}.json')
        if receipt['status'] != 'completed' or receipt['artifacts'][str(size)] != digest(shard):
            raise ValueError('Incomplete or changed shard')
    predictions = folder/'predictions.jsonl'
    count = merge_shards(shards, ids, predictions)
    with CourseArchive(plan['archive'], split='test') as data:
        evaluation = score_predictions(data, predictions, folder)
    csv_path = output/'submissions'/f'dpseg_swinl_depth_no_k_tile{size}_overlap20.csv'
    with SubmissionWriter(csv_path, ids) as writer, predictions.open() as stream:
        for line in stream:
            row = json.loads(line)
            writer.write(row['image_id'], row['predictions'])
    receipt = validate_csv_fast(csv_path, ids)
    binding = compare_csv_predictions(csv_path, predictions, ids)
    write(folder/'csv_validation.json', receipt)
    result = dict(status='completed', tile_size_px=size, images=len(ids), predicted_instances=count,
        metrics=evaluation['metrics'], evaluation=str(folder/'evaluation.json'),
        evaluation_sha256=digest(folder/'evaluation.json'), predictions=str(predictions),
        predictions_sha256=digest(predictions), csv=str(csv_path), csv_sha256=digest(csv_path),
        csv_rows_identical_to_scored_predictions=binding, recomputed=True)
    write(folder/'result.json', result)
    print(json.dumps(result), flush=True)


def run_children(output, phase, specifications, timeout):
    children, logs = [], []
    began = time.monotonic()
    try:
        for name, arguments, gpu in specifications:
            log = (output/f'{name}.log').open('x')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                '--output', str(output), *arguments], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append((name, child))
        write(output/'status.json', dict(status=phase, started_at_utc=now(), hard_timeout_seconds=timeout,
            processes=[dict(name=name, pid=child.pid) for name, child in children]))
        while any(child.poll() is None for _, child in children):
            if time.monotonic()-began > timeout:
                raise TimeoutError(f'{phase} exceeded declared {timeout}s limit')
            time.sleep(3)
        exits = {name: child.returncode for name, child in children}
        write(output/f'{phase}_exits.json', exits)
        if any(exits.values()):
            raise RuntimeError(f'{phase} failed: {exits}')
    finally:
        for _, child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, default=ROOT/'reports/course_swinl_removal_v1')
    parser.add_argument('--gpus', default='0,2,3,4,5,6,7')
    parser.add_argument('--engineering', action='store_true')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--score', type=int, choices=SIZES)
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
    from rgbd_seg.course_tile_size import geometry
    reference_run = args.reference_run.resolve()
    original = read(reference_run/'protocol.json')
    completed = read(reference_run/'results.json')
    if completed['status'] != 'completed' or completed['protocol_sha256'] != digest(reference_run/'protocol.json'):
        raise ValueError('Require a completed and unchanged full-model reference run')
    reference = completed['variants']['full']
    for kind in ('evaluation', 'predictions', 'csv'):
        if digest(reference[kind]) != reference[f'{kind}_sha256']:
            raise ValueError('Reference artifact changed')
    with CourseArchive(original['archive'], split='test') as data:
        if data.image_ids != original['image_ids'] or len(data) != 2112:
            raise ValueError('Require all 2112 test images')
    run = Path(original['system_protocol']['path']).parent
    native = read(run/'final_evaluation/test_native/evaluation.json')
    native_reference = dict(predictions=str(run/'final_evaluation/test_native/predictions.jsonl'),
                            predictions_sha256=native['predictions_sha256'])
    gpus = args.gpus.split(',')
    if len(set(gpus)) != len(gpus) or not all(gpu.isdecimal() for gpu in gpus):
        raise ValueError('GPU IDs must be distinct nonnegative integers')
    plan = {key: original[key] for key in ('archive', 'image_ids', 'weights', 'system_protocol',
        'engineering_val_ids', 'input_keys', 'base_epoch', 'head_epoch', 'precision', 'tf32',
        'score_threshold', 'max_detections', 'camera_intrinsics_used')}
    plan.update(material_passport=dict(source='academic-research-suite / experiment-agent run',
        verification_status='PLANNED', scope='fixed-weight tile-size sensitivity'),
        created_at_utc=now(), gpus=gpus, new_source_sha256={name: digest(ROOT/name) for name in SOURCES},
        reference_640=reference, native_reference=native_reference,
        training=False, ground_truth_in_prediction=False, test_used_for_selection=False,
        test_analysis='Post-hoc sensitivity comparison on the already examined local test set; no parameter selection',
        sizes=[512, 640, 768], recomputed_sizes=list(SIZES), nominal_overlap_ratio=.2,
        geometry={str(s): geometry(s) for s in (512, 640, 768)},
        geometry_reference='https://github.com/obss/sahi/blob/main/sahi/slicing.py',
        scope_note='SAHI-style window geometry only; retain DPSeg mask NMS, RGB-D inputs and both refiners',
        reference_640_run_protocol_sha256=digest(reference_run/'protocol.json'),
        speed_benchmark=False, deferred=['without_rgbd_fusion'])
    output.mkdir(parents=True, exist_ok=False)
    write(output/'protocol.json', plan)
    try:
        run_children(output, 'engineering', [('engineering', ['--engineering'], gpus[-1])], 1800)
        run_children(output, 'inferring', [(f'worker{rank}', ['--worker', str(rank)], gpu)
                                         for rank, gpu in enumerate(gpus)], 14400)
        run_children(output, 'scoring', [(f'score{size}', ['--score', str(size)], '') for size in SIZES], 3600)
        results = {str(s): read(output/f'tile{s}/result.json') for s in SIZES}
        results['640'] = dict(reference, tile_size_px=640, recomputed=False,
            reuse_reason='Exactly identical grid and complete frozen pipeline; independently verified on val and test sentinels')
        write(output/'results.json', dict(status='completed', finished_at_utc=now(), split='test', images=2112,
            protocol_sha256=digest(output/'protocol.json'), sizes=results, training=False,
            test_used_for_selection=False, deferred=['without_rgbd_fusion']))
        write(output/'status.json', dict(status='completed', exit_code=0, finished_at_utc=now()))
        print(json.dumps({key: result['metrics'] for key, result in results.items()}), flush=True)
    except BaseException as error:
        write(output/'status.json', dict(status='failed', error=str(error), at_utc=now()))
        raise


if __name__ == '__main__':
    main()
