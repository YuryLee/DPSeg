#!/usr/bin/env python3
"""Evaluate fixed four-tile overlap/size configurations with the current DPSeg."""
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
           'tools/check_course_prediction_reload.py', 'tools/evaluate_course_guarded_merge.py',
           'rgbd_seg/course_dpseg.py', 'tools/evaluate_course_overlap.py')


def variants():
    from rgbd_seg.course_dpseg import NEW_OVERLAP_VARIANTS
    return tuple(NEW_OVERLAP_VARIANTS)

SEMANTICS = ('iou_type', 'max_detections', 'iou_thresholds', 'area_ranges',
             'ground_truth_area', 'prediction_area', 'gt_instances', 'gt_small_including_crowd')


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_dependencies(plan):
    for name, checksum in plan['new_source_sha256'].items():
        require(digest(ROOT/name) == checksum, f'Frozen source changed: {name}')
    from rgbd_seg.course_dpseg import OVERLAP_CONFIGS
    from rgbd_seg.course_guarded_merge import POLICY
    require(list(variants()) == plan['new_variants'], 'New overlap variant order changed')
    require(json.loads(json.dumps(OVERLAP_CONFIGS)) == plan['overlap_configs'], 'Window configurations changed')
    require(json.loads(json.dumps(POLICY)) == plan['guarded_merge_policy'], 'Frozen merge policy changed')
    manifest = plan['current_default_manifest']
    require(digest(manifest['path']) == manifest['sha256'], 'Current default manifest changed')
    require(read(manifest['path']) == manifest['content'], 'Current default content changed')
    for name, checksum in manifest['content']['source_sha256'].items():
        require(digest(ROOT/name) == checksum, f'Current default source changed: {name}')
    require(manifest['content']['default']['strategy'] == 'global_guard_edge_consensus'
            and manifest['content']['default']['overlap'] == .4, 'Unexpected promoted current default')
    protocol = manifest['content']['protocol']
    require(digest(ROOT/protocol['path']) == protocol['sha256'] == plan['reference_run_protocol_sha256'],
            'Current default was promoted from a different reference run')


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



def check_window_configs(configs):
    import numpy as np
    expected = {'overlap00': (0., 512), 'overlap20': (.2, 569), 'overlap40': (.4, 640),
                'overlap60': (.6, 731), 'overlap80': (.8, 853)}
    require(list(configs) == list(expected), 'Unexpected overlap configuration order')
    for name, (target, size) in expected.items():
        item = configs[name]
        starts = [0, 1024-size]
        boxes = [[0, 0, 1024, 1024], [0, 0, size, size], [1024-size, 0, 1024, size],
                 [0, 1024-size, size, 1024], [1024-size, 1024-size, 1024, 1024]]
        require(item['tile_size_px'] == size and item['target_overlap_ratio'] == target
                and item['actual_overlap_px'] == 2*size-1024
                and abs(item['actual_overlap_ratio']-(2-1024/size)) < 1e-12
                and item['axis_starts_px'] == starts and item['bounds'] == boxes
                and item['tiles'] == 4 and item['views_including_whole'] == 5
                and item['model_input_size'] == [1024, 1024]
                and abs(item['magnification']-1024/size) < 1e-12,
                f'Window geometry/scale mismatch for {name}')
        covered = np.zeros((1024, 1024), dtype=bool)
        for left, top, right, bottom in boxes[1:]:
            covered[top:bottom, left:right] = True
        require(bool(covered.all()) and item['coverage_fraction'] == float(covered.mean()) == 1.,
                f'Four tiles do not fully cover the original image for {name}')
    return dict(status='passed', variants=list(expected), tiles=4, views_including_whole=5,
                full_coverage=True, model_input_size=[1024, 1024], geometry_and_scale_independently_verified=True)


def check_runtime_stats(stats, values, plan, include_reference):
    arms = stats
    expected = set(variants()) | ({'overlap40'} if include_reference else set())
    require(set(arms) == expected, 'Runtime overlap arm coverage differs')
    for name, item in arms.items():
        require(json.loads(json.dumps(item['geometry'])) == plan['overlap_configs'][name] and item['model_views'] == 5,
                'Runtime window geometry differs')
        require(item['actual_model_input_size_verified'] is True
                and item['model_input_sizes'] == [[1024, 1024]]*5, 'Actual forward input size differs')
        if name in values:
            require(item['merging']['selected'] == len(values[name]), 'Selected/output count differs')
        require(0 <= item['merging']['candidates'] <= 500, 'Unexpected candidate count')


def engineering(output):
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_guarded_merge import predict_mergers
    from rgbd_seg.course_dpseg import check_synthetic, predict_overlaps
    from tools.check_course_prediction_reload import compare_predictions

    plan = read(output/'protocol.json')
    check_dependencies(plan)
    geometry_check = check_window_configs(plan['overlap_configs'])
    synthetic = check_synthetic()
    require(isinstance(synthetic, dict) and synthetic.get('status') == 'passed', 'Window synthetic checks failed')
    adapter, heads = configure_and_load(plan)
    checks = []
    with CourseArchive(plan['archive'], split='val') as data, torch.inference_mode():
        require(all(image_id in data.image_ids for image_id in plan['engineering_val_ids']), 'Non-val engineering image')
        for image_id in plan['engineering_val_ids']:
            sample = load_sample(data, image_id, 'depth')
            values, stats, native, reference40 = predict_overlaps(adapter, heads.depth, sample,
                include_reference=True, verify_independent=True)
            check_outputs(values, image_id)
            check_runtime_stats(stats, values, plan, True)
            independent_values, _, _, _ = predict_mergers(adapter, heads.depth, sample)
            independent_native = adapter.predict_sample(sample, score_threshold=.01, max_detections=100)
            errors = dict(reference40=compare_predictions(independent_values['global_guard_edge_consensus'],
                                                         reference40, score_tolerance=0.),
                          native=compare_predictions(independent_native, native, score_tolerance=0.))
            checks.append(dict(image_id=image_id, native_and_reference40_exact=True, score_errors=errors,
                instances={variant: len(values[variant]) for variant in variants()}, stats=stats))
            print(json.dumps(checks[-1], allow_nan=False), flush=True)
    check_dependencies(plan)
    write(output/'engineering_gate.json', dict(status='passed', split='val', checks=checks, synthetic=synthetic, geometry_check=geometry_check,
        protocol_sha256=digest(output/'protocol.json'), ground_truth_used=False,
        model_or_threshold_changes=False, gpu=torch.cuda.get_device_name(0)))


def worker(output, rank):
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_dpseg import predict_overlaps
    from tools.check_course_prediction_reload import compare_predictions

    plan = read(output/'protocol.json')
    gate = read(output/'engineering_gate.json')
    require(gate['status'] == 'passed' and gate['protocol_sha256'] == digest(output/'protocol.json'), 'Invalid gate')
    check_dependencies(plan)
    adapter, heads = configure_and_load(plan)
    ids = plan['image_ids'][rank::len(plan['gpus'])]
    require(bool(ids), 'Empty inference shard')
    reference_ids = list(dict.fromkeys((ids[0], ids[-1])))
    native_reference = reference_rows(plan['native_reference'], ids)
    full_reference = reference_rows(plan['full_reference'], reference_ids)
    began, counts = time.monotonic(), dict.fromkeys(variants(), 0)
    with ExitStack() as stack, torch.inference_mode():
        data = stack.enter_context(CourseArchive(plan['archive'], split='test'))
        require(data.image_ids == plan['image_ids'], 'Test IDs/order changed')
        streams = {variant: stack.enter_context((output/f'{variant}_rank{rank}.jsonl').open('x'))
                   for variant in variants()}
        audit = stack.enter_context((output/f'checks_rank{rank}.jsonl').open('x'))
        for index, image_id in enumerate(ids):
            sample = load_sample(data, image_id, 'depth')
            require(set(sample) == {'image', 'depth_m', 'depth_valid'}, 'Unexpected inference input')
            checked_reference = image_id in reference_ids
            values, stats, native, reference40 = predict_overlaps(adapter, heads.depth, sample,
                                                                include_reference=checked_reference)
            check_outputs(values, image_id)
            check_runtime_stats(stats, values, plan, checked_reference)
            compare_predictions(native_reference.pop(image_id), native, score_tolerance=0.)
            if checked_reference:
                compare_predictions(full_reference.pop(image_id), reference40, score_tolerance=0.)
            else:
                require(reference40 is None, 'Unexpected unrequested reference inference')
            for variant in variants():
                predictions = values[variant]
                streams[variant].write(json.dumps(dict(image_id=image_id, predictions=predictions), allow_nan=False)+'\n')
                counts[variant] += len(predictions)
            audit.write(json.dumps(dict(image_id=image_id, native_exact=True,
                reference40_checked=checked_reference, reference40_exact=True if checked_reference else None,
                instances={variant: len(values[variant]) for variant in variants()}, stats=stats), allow_nan=False)+'\n')
            if index == 0 or (index+1) % 10 == 0 or index+1 == len(ids):
                for stream in [*streams.values(), audit]:
                    stream.flush()
                progress = dict(status='inferring', rank=rank, images=index+1, expected=len(ids),
                                elapsed_seconds=time.monotonic()-began)
                write(output/f'progress_rank{rank}.json', progress)
                print(json.dumps(progress), flush=True)
            del sample, values, stats, native, reference40
    require(not native_reference and not full_reference, 'Uncompared reference images remain')
    check_dependencies(plan)
    write(output/f'rank{rank}.json', dict(status='completed', image_ids=ids, images=len(ids),
        gpu=torch.cuda.get_device_name(0), instances=counts, native_exact=True,
        reference40_exact=True, reference40_image_ids=reference_ids, ground_truth_in_prediction=False,
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
                and receipt['reference40_exact']
                and receipt['reference40_image_ids'] == list(dict.fromkeys((rank_ids[0], rank_ids[-1])))
                and not receipt['ground_truth_in_prediction'], 'Invalid rank receipt')
        require(receipt['artifacts'][variant] == digest(shard), 'Prediction shard changed')
        check_path = output/f'checks_rank{rank}.jsonl'
        require(receipt['checks_sha256'] == digest(check_path), 'Per-image checks changed')
        with check_path.open() as stream:
            checks = [json.loads(line) for line in stream]
        require([row['image_id'] for row in checks] == rank_ids
                and all(row['native_exact'] for row in checks)
                and [row['image_id'] for row in checks if row['reference40_checked']] == receipt['reference40_image_ids']
                and all(row['reference40_exact'] is True if row['reference40_checked'] else row['reference40_exact'] is None
                        for row in checks),
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
    csv_path = output/'submissions'/f'dpseg_swinl_depth_no_k_{variant}.csv'
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



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, default=ROOT/'reports/course_swinl_guarded_merge640_v1')
    parser.add_argument('--current-default-manifest', type=Path)
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
        require(args.score in variants(), 'Unknown new overlap variant')
        score(output, args.score)
        return
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_dpseg import OVERLAP_CONFIGS
    from rgbd_seg.course_guarded_merge import POLICY
    from rgbd_seg.course_tiled_probability import TILE_POLICY, PROBABILITY_POLICY
    require(len(variants()) == len(set(variants())) == 4, 'Expected four new overlap variants')
    require(args.current_default_manifest is not None, 'Provide the promoted current-default manifest')
    manifest_path = args.current_default_manifest.resolve()
    manifest = dict(path=str(manifest_path), sha256=digest(manifest_path), content=read(manifest_path))
    reference_run = args.reference_run.resolve()
    original, completed = read(reference_run/'protocol.json'), read(reference_run/'results.json')
    require(completed['status'] == 'completed' and read(reference_run/'status.json')['exit_code'] == 0
            and completed['protocol_sha256'] == digest(reference_run/'protocol.json'),
            'Require the completed guarded-merge reference run')
    summary = read(reference_run/'summary.json')
    require(summary['status'] == 'verified' and summary['protocol']['sha256'] == completed['protocol_sha256'],
            'Reference run has not passed its full result/CSV audit')
    full_reference = completed['variants']['global_guard_edge_consensus']
    for kind in ('evaluation', 'predictions', 'csv'):
        require(digest(full_reference[kind]) == full_reference[f'{kind}_sha256'], 'Promoted full-model reference changed')
    native_reference = original['native_reference']
    for kind in ('evaluation', 'predictions'):
        require(digest(native_reference[kind]) == native_reference[f'{kind}_sha256'], 'Native reference changed')
    with CourseArchive(original['archive'], split='test') as data:
        require(data.image_ids == original['image_ids'] and len(data) == len(set(data.image_ids)) == 2112,
                'Require all 2112 official test images')
    gpus = args.gpus.split(',')
    require(len(set(gpus)) == len(gpus) and all(gpu.isdecimal() for gpu in gpus), 'Invalid GPU list')
    gpu_capacity = gpu_preflight(gpus)
    storage = shutil.disk_usage(output.parent)
    require(storage.free >= 4*1024**3, 'Need 4 GiB free for prediction/audit artifacts')
    plan = {key: original[key] for key in ('archive', 'image_ids', 'weights', 'system_protocol',
        'engineering_val_ids', 'input_keys', 'base_epoch', 'head_epoch', 'precision', 'tf32',
        'score_threshold', 'max_detections', 'camera_intrinsics_used')}
    plan.update(material_passport=dict(source='academic-research-suite / experiment-agent run',
        verification_status='PLANNED', scope='fixed four-tile full-coverage overlap/size comparison'),
        authorized_request='固定四角2×2切片和完整覆盖，比较0/20/40/60/80%重叠率；40%整图保护＋边缘＋一致性为当前完整版',
        created_at_utc=now(), gpus=gpus, new_variants=list(variants()), all_variants=list(OVERLAP_CONFIGS),
        gpu_capacity=gpu_capacity, new_source_sha256={name: digest(ROOT/name) for name in SOURCES},
        full_reference=full_reference, native_reference=native_reference,
        reference_variant='overlap40', reference_run_protocol_sha256=digest(reference_run/'protocol.json'),
        reference_run_summary=dict(path=str(reference_run/'summary.json'), sha256=digest(reference_run/'summary.json')),
        current_default_manifest=manifest, default_at_start='overlap40', default_after_run='overlap40',
        training=False, ground_truth_in_prediction=False, test_used_for_selection=False,
        test_analysis='Exploratory fixed-protocol comparison on all 2112 already examined local test images; '
            'no retuning from test results and no validation-selected optimum',
        overlap_configs=json.loads(json.dumps(OVERLAP_CONFIGS)),
        tile_policy={**TILE_POLICY, 'tile_size_px': 'per overlap_configs',
            'overlap_at_1024_px': 'per overlap_configs',
            'representative': 'global_guard_edge_consensus selection; unchanged score used for final ordering',
            'geometry': 'four corner-anchored 2x2 tiles, full coverage, aligned RGB/depth/valid; no camera K'},
        probability_policy=PROBABILITY_POLICY, guarded_merge_policy=json.loads(json.dumps(POLICY)),
        guarded_merge_policy_scope='select_candidates thresholds only; the historical POLICY.tile_size_px=640 '
            'is not consulted by select_candidates, active geometry is in overlap_configs',
        merge_variant='global_guard_edge_consensus', views_per_configuration=5,
        comparison_scope='Tile count, full coverage, 1024 model input, weights and merge/refinement rules are fixed; '
            'crop size, actual overlap and effective object scale necessarily change together; not pure overlap causality',
        formal_comparison='Native S1 exactly reproduced for all test images; current 640 full model exactly reproduced '
            'on three engineering val images and the first/last test image in each rank; full 2112 reference reused',
        storage=dict(free_bytes_before=storage.free, minimum_free_bytes=4*1024**3,
            save_features=False, save_soft_probability=False, delete_existing_artifacts=False),
        default_changed_by_experiment=False, speed_benchmark=False)
    check_dependencies(plan)
    check_window_configs(plan['overlap_configs'])
    output.mkdir(parents=True, exist_ok=False)
    write(output/'protocol.json', plan)
    try:
        run_children(output, 'engineering', [('engineering', ['--engineering'], gpus[-1])], 3600)
        gpu_preflight(gpus)
        run_children(output, 'inferring', [(f'worker{rank}', ['--worker', str(rank)], gpu)
                                         for rank, gpu in enumerate(gpus)], 21600)
        run_children(output, 'scoring', [(f'score_{variant}', ['--score', variant], '')
                                       for variant in variants()], 7200)
        results = {variant: read(output/variant/'result.json') for variant in variants()}
        check_dependencies(plan)
        write(output/'results.json', dict(status='completed', finished_at_utc=now(), split='test', images=2112,
            protocol_sha256=digest(output/'protocol.json'), variants=results, full_reference=full_reference,
            training=False, test_used_for_selection=False, default_changed_by_experiment=False,
            default_after_run='overlap40', reference_variant='overlap40'))
        write(output/'status.json', dict(status='completed', exit_code=0, finished_at_utc=now()))
        print(json.dumps({key: row['metrics'] for key, row in results.items()}, allow_nan=False), flush=True)
    except BaseException as error:
        write(output/'status.json', dict(status='failed', error=str(error), at_utc=now()))
        raise


if __name__ == '__main__':
    main()
