#!/usr/bin/env python3
"""Evaluate protected whole-image merging with frozen 640-pixel DPSeg views."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_course_swinl_ablation import read, digest, write
from tools.evaluate_course_swinl_removal import configure_and_load, NEW_SOURCES

SOURCES = (*NEW_SOURCES, 'rgbd_seg/course_tile_size.py', 'rgbd_seg/course_guarded_merge.py',
           'tools/check_course_prediction_reload.py', 'tools/evaluate_course_guarded_merge.py')


def variants():
    from rgbd_seg.course_guarded_merge import VARIANTS
    return tuple(VARIANTS)

SEMANTICS = ('iou_type', 'max_detections', 'iou_thresholds', 'area_ranges',
             'ground_truth_area', 'prediction_area', 'gt_instances', 'gt_small_including_crowd')


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_dependencies(plan):
    """Freeze rules and all experiment sources, without optional SAHI dependencies."""
    for name, checksum in plan['new_source_sha256'].items():
        require(digest(ROOT/name) == checksum, f'Frozen experiment source changed: {name}')
    from rgbd_seg.course_guarded_merge import POLICY, VARIANT_DESCRIPTIONS
    require(list(variants()) == plan['variants'], 'Guarded-merge variant order changed')
    require(json.loads(json.dumps(POLICY)) == plan['guarded_merge_policy'], 'Frozen merge policy changed')
    require(json.loads(json.dumps(VARIANT_DESCRIPTIONS)) == plan['variant_descriptions'],
            'Frozen variant descriptions changed')


def reference_rows(reference, ids):
    require(digest(reference['predictions']) == reference['predictions_sha256'],
            'Reference prediction file changed')
    wanted, rows = set(ids), {}
    with Path(reference['predictions']).open() as stream:
        for line in stream:
            row = json.loads(line)
            image_id = row['image_id']
            if image_id in wanted:
                require(image_id not in rows, 'Duplicate reference image')
                rows[image_id] = row['predictions']
    require(set(rows) == wanted, 'Reference prediction coverage differs')
    return rows


def check_outputs(values, image_id):
    """Validate all public masks before any COCO native evaluation calls."""
    from rgbd_seg.course_submission import _Spec, _predictions
    require(tuple(values) == variants(), 'Missing, extra or reordered prediction variants')
    spec = _Spec([image_id], (1,), (1024, 1024), 100)
    for variant in variants():
        normalized, _ = _predictions(values[variant], spec, decode=False)
        require(normalized == values[variant], 'Prediction normalization changed a public output')
        require(all(prediction['score'] >= .01 for prediction in normalized),
                'Unexpected candidate below fixed score threshold')


def engineering(output):
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_depth_backbone_system import tiled_pair
    from rgbd_seg.course_guarded_merge import check_synthetic, predict_mergers
    from tools.check_course_prediction_reload import compare_predictions

    plan = read(output/'protocol.json')
    check_dependencies(plan)
    synthetic = check_synthetic()
    require(isinstance(synthetic, dict) and synthetic.get('status') == 'passed',
            'Guarded-merge synthetic checks did not pass')
    # The synthetic checks must not change the isolated runtime or frozen code.
    check_dependencies(plan)
    adapter, heads = configure_and_load(plan)
    checks = []
    with CourseArchive(plan['archive'], split='val') as data, torch.inference_mode():
        require(all(image_id in data.image_ids for image_id in plan['engineering_val_ids']),
                'Engineering image is outside validation split')
        for image_id in plan['engineering_val_ids']:
            sample = load_sample(data, image_id, 'depth')
            values, stats, native, legacy = predict_mergers(adapter, heads.depth, sample, verify_independent=True)
            check_outputs(values, image_id)
            _, independent_full, _ = tiled_pair(adapter, heads.depth, sample)
            independent_native = adapter.predict_sample(sample, score_threshold=.01, max_detections=100)
            errors = dict(legacy_full=compare_predictions(independent_full, legacy, score_tolerance=0.),
                          native=compare_predictions(independent_native, native, score_tolerance=0.))
            checks.append(dict(image_id=image_id, native_and_legacy_exact=True, score_errors=errors,
                instances={variant: len(values[variant]) for variant in variants()}, stats=stats))
            print(json.dumps(checks[-1], allow_nan=False), flush=True)
    write(output/'engineering_gate.json', dict(status='passed', split='val', checks=checks,
        synthetic=synthetic,
        protocol_sha256=digest(output/'protocol.json'), ground_truth_used=False,
        model_or_threshold_changes=False, gpu=torch.cuda.get_device_name(0)))


def worker(output, rank):
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_guarded_merge import predict_mergers
    from tools.check_course_prediction_reload import compare_predictions

    plan = read(output/'protocol.json')
    gate = read(output/'engineering_gate.json')
    require(gate['status'] == 'passed' and gate['protocol_sha256'] == digest(output/'protocol.json'), 'Engineering gate invalid')
    check_dependencies(plan)
    adapter, heads = configure_and_load(plan)
    ids = plan['image_ids'][rank::len(plan['gpus'])]
    require(bool(ids), 'Empty inference shard')
    native_reference = reference_rows(plan['native_reference'], ids)
    full_reference = reference_rows(plan['full_reference'], ids)
    began = time.monotonic()
    counts = dict.fromkeys(variants(), 0)
    with ExitStack() as stack, torch.inference_mode():
        data = stack.enter_context(CourseArchive(plan['archive'], split='test'))
        require(data.image_ids == plan['image_ids'], 'Official test IDs/order changed')
        streams = {variant: stack.enter_context((output/f'{variant}_rank{rank}.jsonl').open('x'))
                   for variant in variants()}
        audit = stack.enter_context((output/f'checks_rank{rank}.jsonl').open('x'))
        for index, image_id in enumerate(ids):
            sample = load_sample(data, image_id, 'depth')
            require(set(sample) == {'image', 'depth_m', 'depth_valid'}, 'Unexpected inference inputs')
            values, stats, native, legacy = predict_mergers(adapter, heads.depth, sample)
            check_outputs(values, image_id)
            compare_predictions(native_reference.pop(image_id), native, score_tolerance=0.)
            compare_predictions(full_reference.pop(image_id), legacy, score_tolerance=0.)
            for variant in variants():
                predictions = values[variant]
                streams[variant].write(json.dumps(dict(image_id=image_id, predictions=predictions),
                                                 allow_nan=False)+'\n')
                counts[variant] += len(predictions)
            audit.write(json.dumps(dict(image_id=image_id, native_exact=True, legacy_full_exact=True,
                instances={variant: len(values[variant]) for variant in variants()}, stats=stats),
                allow_nan=False)+'\n')
            if index == 0 or (index+1) % 10 == 0 or index+1 == len(ids):
                for stream in [*streams.values(), audit]:
                    stream.flush()
                progress = dict(status='inferring', rank=rank, images=index+1, expected=len(ids),
                                elapsed_seconds=time.monotonic()-began)
                write(output/f'progress_rank{rank}.json', progress)
                print(json.dumps(progress), flush=True)
            del sample, values, stats, native, legacy
    require(not native_reference and not full_reference, 'Uncompared reference images remain')
    check_dependencies(plan)
    write(output/f'rank{rank}.json', dict(status='completed', image_ids=ids, images=len(ids),
        gpu=torch.cuda.get_device_name(0), instances=counts, native_exact=True, legacy_full_exact=True,
        ground_truth_in_prediction=False,
        artifacts={variant: digest(output/f'{variant}_rank{rank}.jsonl') for variant in variants()},
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
                and receipt['images'] == len(rank_ids) and receipt['native_exact']
                and receipt['legacy_full_exact'] and not receipt['ground_truth_in_prediction'], 'Invalid rank receipt')
        require(receipt['artifacts'][variant] == digest(shard), 'Prediction shard changed')
        check_path = output/f'checks_rank{rank}.jsonl'
        require(receipt['checks_sha256'] == digest(check_path), 'Per-image checks changed')
        with check_path.open() as stream:
            checks = [json.loads(line) for line in stream]
        require([row['image_id'] for row in checks] == rank_ids
                and all(row['native_exact'] and row['legacy_full_exact'] for row in checks),
                'Reference comparison coverage incomplete')
        with shard.open() as stream:
            observed = [json.loads(line) for line in stream]
        require([row['image_id'] for row in observed] == rank_ids, 'Shard ID/order differs')
        count = sum(len(row['predictions']) for row in observed)
        require(count == receipt['instances'][variant]
                and all(len(row['predictions']) == check['instances'][variant]
                        for row, check in zip(observed, checks)), 'Shard instance count differs')
        total += count
    predictions = folder/'predictions.jsonl'
    count = merge_shards(shards, ids, predictions)
    require(count == total, 'Merged instance count differs')
    with CourseArchive(plan['archive'], split='test') as data:
        require(data.image_ids == ids, 'Evaluation archive IDs changed')
        evaluation = score_predictions(data, predictions, folder)
    reference = plan['full_reference']
    require(digest(reference['evaluation']) == reference['evaluation_sha256'], 'Reference evaluation changed')
    original = read(reference['evaluation'])
    require({key: evaluation[key] for key in SEMANTICS} == {key: original[key] for key in SEMANTICS},
            'COCO settings or ground-truth statistics differ from the full-model reference')
    csv_path = output/'submissions'/f'dpseg_swinl_depth_no_k_guarded640_{variant}.csv'
    with SubmissionWriter(csv_path, ids) as writer, predictions.open() as stream:
        for line in stream:
            row = json.loads(line)
            writer.write(row['image_id'], row['predictions'])
    validation = validate_csv_fast(csv_path, ids)
    binding = compare_csv_predictions(csv_path, predictions, ids)
    require(binding == 2112 and validation['predictions'] == evaluation['predicted_instances'] == count,
            'CSV binding or instance count differs')
    write(folder/'csv_validation.json', validation)
    result = dict(status='completed', variant=variant, images=len(ids), predicted_instances=count,
        metrics=evaluation['metrics'], evaluation=str(folder/'evaluation.json'),
        evaluation_sha256=digest(folder/'evaluation.json'), predictions=str(predictions),
        predictions_sha256=digest(predictions), csv=str(csv_path), csv_sha256=digest(csv_path),
        csv_rows_identical_to_scored_predictions=binding)
    write(folder/'result.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)


def gpu_preflight(gpus):
    command = ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
               '--format=csv,noheader,nounits']
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    devices = {}
    for line in result.stdout.splitlines():
        index, name, total, used, free, utilization = [cell.strip() for cell in line.split(',')]
        devices[index] = dict(name=name, total_mib=int(total), used_mib=int(used),
                              free_mib=int(free), utilization_percent=int(utilization))
    for gpu in gpus:
        require(gpu in devices and devices[gpu]['free_mib'] >= 24*1024,
                f'GPU {gpu} needs at least 24 GiB free; no existing process will be interrupted')
    return {gpu: devices[gpu] for gpu in gpus}


def run_children(output, phase, specifications, timeout):
    children, logs = [], []
    began = time.monotonic()
    timed_out = False
    try:
        for name, arguments, gpu in specifications:
            log = (output/f'{name}.log').open('x')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                '--output', str(output), *arguments], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT)
            children.append((name, child))
        write(output/'status.json', dict(status=phase, started_at_utc=now(), hard_timeout_seconds=timeout,
            processes=[dict(name=name, pid=child.pid) for name, child in children]))
        reported = set()
        while any(child.poll() is None for _, child in children):
            if time.monotonic()-began > timeout:
                timed_out = True
                print(f'{phase}: declared hard timeout reached; terminating only this phase children.', flush=True)
                raise TimeoutError(f'{phase} exceeded declared {timeout}s limit')
            for name, child in children:
                if child.poll() not in (None, 0) and name not in reported:
                    reported.add(name)
                    print(f'{phase}: {name} failed; other workers continue, no automatic retry.', flush=True)
            time.sleep(3)
        require(all(child.returncode == 0 for _, child in children), f'{phase} did not finish successfully')
    finally:
        if timed_out:
            for _, child in children:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
        write(output/f'{phase}_exits.json', {name: child.poll() for name, child in children})
        for log in logs:
            log.close()


def baseline_rescore_receipt(path, reference):
    from tools.summarize_course_swinl_removal import check_sources
    receipt = read(path)
    require(receipt['status'] == 'passed' and receipt['all_metrics_exact']
            and receipt['images'] == 2112 and receipt['metrics'] == reference['metrics']
            and receipt['reference_predictions_sha256'] == reference['predictions_sha256']
            and receipt['checker_sha256'] == digest(ROOT/'tools/check_course_sahi_reference.py')
            and digest(receipt['evaluation']) == receipt['evaluation_sha256'], 'Invalid existing baseline rescore')
    rescored = read(receipt['evaluation'])
    require(rescored['metrics'] == reference['metrics']
            and rescored['predictions_sha256'] == reference['predictions_sha256'], 'Rescored baseline differs')
    check_sources(rescored['source_sha256'])
    return dict(path=str(path), sha256=digest(path), reused=True, receipt=receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, default=ROOT/'reports/course_swinl_removal_v1')
    parser.add_argument('--baseline-rescore', type=Path,
                        default=ROOT/'reports/course_sahi_merge_v1/reference_rescore/verification.json')
    parser.add_argument('--gpus', default='1,3,4,5,6,7')
    parser.add_argument('--engineering', action='store_true')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--score')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.engineering:
        engineering(output)
        return
    if args.worker is not None:
        worker(output, args.worker)
        return
    if args.score:
        require(args.score in variants(), 'Unknown variant')
        score(output, args.score)
        return
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_guarded_merge import POLICY, VARIANT_DESCRIPTIONS
    from rgbd_seg.course_tiled_probability import TILE_POLICY, PROBABILITY_POLICY, view_bounds
    require(len(variants()) == 5 and len(set(variants())) == 5, 'Expected five fixed rule variants')
    reference_run = args.reference_run.resolve()
    original, completed = read(reference_run/'protocol.json'), read(reference_run/'results.json')
    require(completed['status'] == 'completed'
            and completed['protocol_sha256'] == digest(reference_run/'protocol.json'),
            'Require a completed and unchanged full-model reference run')
    full_reference = completed['variants']['full']
    without_tiles_reference = completed['variants']['without_tiles']
    for reference in (full_reference, without_tiles_reference):
        for kind in ('evaluation', 'predictions', 'csv'):
            require(digest(reference[kind]) == reference[f'{kind}_sha256'], 'Reference result changed')
    baseline_rescore = baseline_rescore_receipt(args.baseline_rescore.resolve(), full_reference)
    system_run = Path(original['system_protocol']['path']).parent
    native_path = system_run/'final_evaluation/test_native/evaluation.json'
    native = read(native_path)
    native_predictions = native_path.parent/'predictions.jsonl'
    require(native['status'] == 'completed' and native['split'] == 'test' and native['images'] == 2112
            and native['predictions_sha256'] == digest(native_predictions), 'Incomplete or changed native reference')
    native_reference = dict(evaluation=str(native_path), evaluation_sha256=digest(native_path),
                            predictions=str(native_predictions), predictions_sha256=native['predictions_sha256'])
    with CourseArchive(original['archive'], split='test') as data:
        require(data.image_ids == original['image_ids'] and len(data) == 2112
                and len(set(data.image_ids)) == 2112, 'Require all 2112 test images')
    gpus = args.gpus.split(',')
    require(len(set(gpus)) == len(gpus) and all(gpu.isdecimal() for gpu in gpus),
            'GPU IDs must be distinct nonnegative integers')
    require(len(gpus) <= 2112, 'Each inference process must receive at least one image')
    gpu_capacity = gpu_preflight(gpus)
    storage = shutil.disk_usage(output.parent)
    require(storage.free >= 4*1024**3, 'Need 4 GiB free for JSONL, CSV, audit and evaluation artifacts')
    plan = {key: original[key] for key in ('archive', 'image_ids', 'weights', 'system_protocol',
        'engineering_val_ids', 'input_keys', 'base_epoch', 'head_epoch', 'precision', 'tf32',
        'score_threshold', 'max_detections', 'camera_intrinsics_used')}
    plan.update(material_passport=dict(source='academic-research-suite / experiment-agent run',
        verification_status='PLANNED', scope='fixed-weight 640-pixel guarded candidate-selection comparison'),
        authorized_request='在640×640切片上尝试保护可靠整图候选、补充小目标、边缘完整性与跨视图一致性',
        created_at_utc=now(), gpus=gpus, variants=list(variants()), gpu_capacity=gpu_capacity,
        new_source_sha256={name: digest(ROOT/name) for name in SOURCES},
        full_reference=full_reference, without_tiles_reference=without_tiles_reference,
        native_reference=native_reference, baseline_rescore=baseline_rescore,
        reference_run_protocol_sha256=digest(reference_run/'protocol.json'),
        training=False, ground_truth_in_prediction=False, test_used_for_selection=False,
        test_analysis='Exploratory fixed-rule comparison on all 2112 already examined local test images; '
            'no retuning from test results, no validation-selected optimum, no default replacement',
        tile_size_px=640, view_bounds=view_bounds(1024, 1024),
        tile_policy={**TILE_POLICY, 'geometry': 'aligned RGB/depth/valid crop; no camera K or intrinsics used'},
        probability_policy=PROBABILITY_POLICY,
        guarded_merge_policy=json.loads(json.dumps(POLICY)),
        variant_descriptions=json.loads(json.dumps(VARIANT_DESCRIPTIONS)),
        comparison_scope='Only candidate selection/priority and cross-view conflicts change; masks, original scores, '
            'source-view refiners, score threshold and final top100 remain fixed',
        formal_comparison='Every test image: native == original S1 and independently refined legacy full == original S4 exactly',
        storage=dict(free_bytes_before=storage.free, minimum_free_bytes=4*1024**3,
            save_features=False, save_soft_probability=False, delete_existing_artifacts=False),
        default_changed=False, speed_benchmark=False, deferred=['without_rgbd_fusion'])
    check_dependencies(plan)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'protocol.json', plan)
    try:
        run_children(output, 'engineering', [('engineering', ['--engineering'], gpus[-1])], 1800)
        gpu_preflight(gpus)
        run_children(output, 'inferring', [(f'worker{rank}', ['--worker', str(rank)], gpu)
                                         for rank, gpu in enumerate(gpus)], 21600)
        run_children(output, 'scoring', [(f'score_{variant}', ['--score', variant], '')
                                       for variant in variants()], 7200)
        results = {variant: read(output/variant/'result.json') for variant in variants()}
        check_dependencies(plan)
        write(output/'results.json', dict(status='completed', finished_at_utc=now(), split='test', images=2112,
            protocol_sha256=digest(output/'protocol.json'), variants=results, full_reference=full_reference,
            without_tiles_reference=without_tiles_reference, baseline_rescore=baseline_rescore,
            training=False, test_used_for_selection=False, default_changed=False,
            deferred=['without_rgbd_fusion']))
        write(output/'status.json', dict(status='completed', exit_code=0, finished_at_utc=now()))
        print(json.dumps({key: row['metrics'] for key, row in results.items()}, allow_nan=False), flush=True)
    except BaseException as error:
        write(output/'status.json', dict(status='failed', error=str(error), at_utc=now()))
        raise


if __name__ == '__main__':
    main()
