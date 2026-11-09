#!/usr/bin/env python3
"""Evaluate the trained Large plain point head in the adopted guarded640 system.

Frozen sources are imported unchanged. No-point and depth-point outputs are
reproduced for every test image before evaluating the new plain-point arm.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SELF = 'tools/evaluate_course_plain_point.py'
from tools.evaluate_course_overlap import read, write, digest, require, now, reference_rows, SEMANTICS, gpu_preflight
from tools.evaluate_course_swinl_removal import configure_and_load


def check_dependencies(plan):
    manifest = plan['current_default_manifest']
    require(digest(manifest['path']) == manifest['sha256'], 'Current manifest changed')
    for name, checksum in plan['new_source_sha256'].items():
        require(digest(ROOT/name) == checksum, 'Frozen source changed: '+name)
    for reference in plan['references'].values():
        for field in ('predictions', 'evaluation'):
            require(digest(reference[field]) == reference[field+'_sha256'], 'Reference changed: '+field)
    require(plan['head_epoch'] == 6 and plan['base_epoch'] == 15 and not plan['training'], 'Fixed budget changed')


def predict(adapter, heads, sample):
    import torch
    from rgbd_seg.course_depth_backbone_system import crop_sample
    from rgbd_seg.course_dpseg import DEFAULT_MERGE, OVERLAP_CONFIGS, _check_sample
    from rgbd_seg.course_guarded_merge import identity, select_candidates
    from rgbd_seg.course_swinl_removal import refine_candidates
    from rgbd_seg.course_tile_size import extract_view
    from rgbd_seg.course_tiled_probability import refined_public
    _check_sample(adapter, sample)
    require(heads.plain.depth_channels == 0 and heads.depth.depth_channels == 9, 'Wrong head input')
    geometry = OVERLAP_CONFIGS['overlap40']
    cache, candidates = {}, []
    for view_id, bounds in enumerate(geometry['bounds']):
        view = sample if view_id == 0 else crop_sample(sample, bounds)
        values, cache[view_id] = extract_view(adapter, view, bounds, (1024, 1024), view_id)
        candidates.extend(values)
    selected, merging = select_candidates(candidates, DEFAULT_MERGE)
    expected = [(1, c.score) for c in selected]
    results, changes = {}, {}
    results['without_point'], changes['without_point'] = refined_public(selected, sample, 'probability', adapter.device)
    for name, head in (('plain_point', heads.plain), ('depth_point', heads.depth)):
        refined = refine_candidates(adapter, head, selected, cache, (1024, 1024))
        require([identity(c) for c in refined] == [identity(c) for c in selected], 'Candidate identities changed')
        results[name], changes[name] = refined_public(refined, sample, 'probability', adapter.device)
    for values in results.values():
        require([(p['category_id'], p['score']) for p in values] == expected, 'Head changed count, class, score or order')
    require(DEFAULT_MERGE == 'global_guard_edge_consensus', 'Merge policy changed')
    return results, dict(selected_identity=[list(identity(c)) for c in selected],
        selected_scores=[c.score for c in selected], selected_count=len(selected),
        candidates=len(candidates), merging=merging, probability_changes=changes,
        geometry=geometry, model_views=5, original_scores_and_order_unchanged=True)


def engineering(output, gpu):
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_dpseg import predict_dpseg
    from rgbd_seg.course_current_removal import predict_current_removals
    from tools.check_course_prediction_reload import compare_predictions
    plan = read(output/'protocol.json')
    check_dependencies(plan)
    adapter, heads = configure_and_load(plan)
    checks = []
    with CourseArchive(plan['archive'], split='val') as data, torch.inference_mode():
        for image_id in plan['engineering_val_ids']:
            require(image_id in data.image_ids, 'Engineering image is not val')
            sample = load_sample(data, image_id, 'depth')
            values, stats = predict(adapter, heads, sample)
            plain, _ = predict_dpseg(adapter, heads.plain, sample)
            removals, _, full, _, _ = predict_current_removals(adapter, heads.depth, sample)
            for name, expected in dict(plain_point=plain, depth_point=full, without_point=removals['without_point']).items():
                compare_predictions(expected, values[name], score_tolerance=0.)
            checks.append(dict(image_id=image_id, independent_arms_exact=True, instances=stats['selected_count']))
            print(json.dumps(checks[-1]), flush=True)
    check_dependencies(plan)
    write(output/f'engineering_gpu{gpu}.json', dict(status='passed', split='val', checks=checks,
        protocol_sha256=digest(output/'protocol.json'), gpu=torch.cuda.get_device_name(0),
        no_ground_truth_in_prediction=True))


def worker(output, rank):
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_submission import _Spec, _predictions
    from tools.check_course_prediction_reload import compare_predictions
    plan = read(output/'protocol.json')
    check_dependencies(plan)
    for gpu in plan['engineering_gpus']:
        gate = read(output/f'engineering_gpu{gpu}.json')
        require(gate['status'] == 'passed' and gate['protocol_sha256'] == digest(output/'protocol.json'), 'Invalid engineering gate')
    adapter, heads = configure_and_load(plan)
    ids = plan['image_ids'][rank::len(plan['gpus'])]
    references = {name: reference_rows(row, ids) for name, row in plan['references'].items()}
    started, total = time.monotonic(), 0
    with ExitStack() as stack, torch.inference_mode():
        data = stack.enter_context(CourseArchive(plan['archive'], split='test'))
        require(data.image_ids == plan['image_ids'], 'Test coverage/order changed')
        stream = stack.enter_context((output/f'plain_point_rank{rank}.jsonl').open('x'))
        audit = stack.enter_context((output/f'checks_rank{rank}.jsonl').open('x'))
        for index, image_id in enumerate(ids):
            sample = load_sample(data, image_id, 'depth')
            values, stats = predict(adapter, heads, sample)
            for name, reference in references.items():
                compare_predictions(reference.pop(image_id), values[name], score_tolerance=0.)
            predictions = values['plain_point']
            normalized, _ = _predictions(predictions, _Spec([image_id], (1,), (1024, 1024), 100), decode=False)
            require(normalized == predictions and len(predictions) == stats['selected_count'], 'Invalid prediction schema')
            stream.write(json.dumps(dict(image_id=image_id, predictions=predictions), allow_nan=False)+'\n')
            audit.write(json.dumps(dict(image_id=image_id, references_exact=True, stats=stats), allow_nan=False)+'\n')
            stream.flush(); audit.flush()
            total += len(predictions)
            if index == 0 or (index+1) % 10 == 0 or index+1 == len(ids):
                progress = dict(status='inferring', rank=rank, images=index+1, expected=len(ids),
                    elapsed_seconds=time.monotonic()-started, references_exact=True)
                write(output/f'progress_rank{rank}.json', progress)
                print(json.dumps(progress), flush=True)
            del sample, values, stats, predictions, normalized
    require(all(not row for row in references.values()), 'Reference coverage incomplete')
    check_dependencies(plan)
    write(output/f'rank{rank}.json', dict(status='completed', image_ids=ids, images=len(ids),
        instances=total, references_exact=True, no_ground_truth_in_prediction=True,
        predictions_sha256=digest(output/f'plain_point_rank{rank}.jsonl'),
        checks_sha256=digest(output/f'checks_rank{rank}.jsonl'), gpu=torch.cuda.get_device_name(0)))


def score(output):
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone_eval import merge_shards
    from rgbd_seg.course_dafusion_eval import score_predictions, validate_csv_fast
    from rgbd_seg.course_submission import SubmissionWriter
    from tools.audit_course_point_refinement import compare_csv_predictions
    plan = read(output/'protocol.json')
    check_dependencies(plan)
    ids, folder, total, shards = plan['image_ids'], output/'plain_point', 0, []
    folder.mkdir()
    for rank in range(len(plan['gpus'])):
        receipt = read(output/f'rank{rank}.json')
        shard = output/f'plain_point_rank{rank}.jsonl'
        check_path = output/f'checks_rank{rank}.jsonl'
        rank_ids = ids[rank::len(plan['gpus'])]
        require(receipt['status'] == 'completed' and receipt['image_ids'] == rank_ids
            and receipt['references_exact'] and receipt['no_ground_truth_in_prediction'], 'Worker did not complete checks')
        require(digest(shard) == receipt['predictions_sha256'] and digest(check_path) == receipt['checks_sha256'], 'Worker artifact changed')
        with shard.open() as stream, check_path.open() as audit:
            rows, checks = [json.loads(line) for line in stream], [json.loads(line) for line in audit]
        require([row['image_id'] for row in rows] == [row['image_id'] for row in checks] == rank_ids, 'Shard ID mismatch')
        for row, check in zip(rows, checks):
            require(check['references_exact'] and len(row['predictions']) == check['stats']['selected_count']
                and [(p['category_id'], p['score']) for p in row['predictions']]
                == [(1, s) for s in check['stats']['selected_scores']], 'Selected candidate check failed')
        count = sum(len(row['predictions']) for row in rows)
        require(count == receipt['instances'], 'Instance count differs')
        total += count
        shards.append(shard)
    predictions = folder/'predictions.jsonl'
    require(merge_shards(shards, ids, predictions) == total == plan['references']['depth_point']['predicted_instances'], 'Merged count differs')
    with CourseArchive(plan['archive'], split='test') as data:
        evaluation = score_predictions(data, predictions, folder)
    for reference in plan['references'].values():
        old = read(reference['evaluation'])
        require({k:evaluation[k] for k in SEMANTICS} == {k:old[k] for k in SEMANTICS}, 'Evaluation semantics differ')
    csv_path = output/'submissions/dpseg_swinl_depth_no_k_guarded640_plain_point_e06.csv'
    csv_path.parent.mkdir()
    with SubmissionWriter(csv_path, ids) as writer, predictions.open() as stream:
        for line in stream:
            row = json.loads(line)
            writer.write(row['image_id'], row['predictions'])
    validation = validate_csv_fast(csv_path, ids)
    binding = compare_csv_predictions(csv_path, predictions, ids)
    require(binding == 2112 and validation['predictions'] == total == evaluation['predicted_instances'], 'CSV binding differs')
    write(folder/'csv_validation.json', validation)
    result = dict(status='completed', variant='plain_point', images=2112, predicted_instances=total,
        metrics=evaluation['metrics'], evaluation=str(folder/'evaluation.json'),
        evaluation_sha256=digest(folder/'evaluation.json'), predictions=str(predictions),
        predictions_sha256=digest(predictions), csv=str(csv_path), csv_sha256=digest(csv_path),
        csv_rows_identical_to_scored_predictions=binding)
    write(folder/'result.json', result)
    write(output/'results.json', dict(status='completed', finished_at_utc=now(), split='test', images=2112,
        protocol_sha256=digest(output/'protocol.json'), variants=dict(plain_point=result), references=plan['references'],
        training=False, test_used_for_selection=False, default_changed=False))
    publish(output)


def publish(output):
    result = read(output/'results.json')
    rows = [('无点头', result['references']['without_point']),
            ('普通点头', result['variants']['plain_point']),
            ('深度点头', result['references']['depth_point'])]
    docs = ROOT/'docs/results'
    stem = 'course_point_heads_swinl_current'
    fields = ('配置', 'AP_s', 'AR_s', 'AP', 'CSV路径', 'CSV_SHA256')
    with (docs/(stem+'.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(fields)
        for name, row in rows:
            metrics = row['metrics']
            writer.writerow((name, metrics['mask_ap_small'], metrics['mask_ar_small'], metrics['mask_ap'], row['csv'], row['csv_sha256']))
    lines = ['# Swin-Large 当前完整流程的点头类型消融', '',
        '统一采用Swin-Large＋无内参深度、第15轮冻结基座、1024×1024输入、整图＋四个640×640切片（40%重叠）、整图优先＋截断处理＋跨视图一致性合并，以及深度感知概率图细化。', '',
        '以下为完整本地测试集2,112张图的COCO掩码评测，分数范围0–1，与Kaggle分数分别报告。', '',
        '| 点头配置 | APₛ ↑ | ARₛ ↑ | AP ↑ |', '|---|---:|---:|---:|']
    for name, row in rows:
        metrics = row['metrics']
        lines.append(f"| {name} | {metrics['mask_ap_small']:.6f} | {metrics['mask_ar_small']:.6f} | {metrics['mask_ap']:.6f} |")
    lines += ['', '普通与深度点头均来自同一成对训练的第6轮检查点。两者共享冻结的448维RGB-D特征、粗掩码logit、MLP隐藏宽度256及三层结构，深度点头额外接收9维显式局部深度线索。普通点头也使用RGB-D融合特征，不是纯RGB模型或原版PointRend的直接复现。', '',
        '三组使用相同候选身份、类别、原始置信度、排序和最多100实例限制。各点头使用相同的选点算法与设置；后续轮次的具体不确定点位置可随各自掩码变化。', '',
        '本次只新增普通点头的完整测试推理与评测；无点头及深度点头逐图重新推理并与既有预测逐掩码、逐分数、逐次序精确核对，表中两行沿用已核验的原指标。普通点头统一固定第6轮，未在测试集选轮，也不宣称其为普通点头的验证最优轮次。', '',
        '该测试集此前已被查看，本表为固定协议、单次训练种子的描述性消融；没有新增训练、参数搜索、统计显著性或Kaggle结果。共享前向用于节省实验计算，运行总耗时不能作为各方法独立FPS。', '',
        '## 文件与核验', '', f'- 本轮协议与结果：`{output}`。',
        '- 普通点头CSV全量压缩RLE解码，且逐行绑定COCO评分所用JSONL；原图尺寸1024×1024、category_id=1、每图最多100实例。']
    for name, row in rows:
        lines.append(f"- {name} CSV：`{row['csv']}`。")
    plan = read(output/'protocol.json')
    lines += ['', f"来源：ARS experiment-agent/run；协议SHA-256：`{digest(output/'protocol.json')}`；状态：COMPLETED。"]
    (docs/(stem+'.md')).write_text('\n'.join(lines)+'\n')
    write(docs/(stem+'.json'), dict(**result, material_passport=dict(origin_skill='academic-research-suite', origin_mode='run',
        verification_status='FULL_TEST_AND_CSV_BOUND', protocol=str(output/'protocol.json')), plan=plan))
    print(json.dumps(dict(status='published', metrics=result['variants']['plain_point']['metrics'], report=str(docs/(stem+'.md')))), flush=True)


def prepare(output, gpus):
    require(not output.exists(), 'Use a fresh output directory')
    require(shutil.disk_usage(output.parent).free > 22*1024**3, 'Insufficient disk reserve')
    prior = read(ROOT/'reports/course_swinl_current_removal_v1/protocol.json')
    sources = dict(prior['new_source_sha256'])
    sources[SELF] = digest(ROOT/SELF)
    plan = {key:prior[key] for key in ('archive', 'image_ids', 'weights', 'system_protocol', 'engineering_val_ids',
        'current_default_manifest', 'base_epoch', 'head_epoch', 'precision', 'tf32', 'score_threshold', 'max_detections')}
    no_point = read(ROOT/'reports/course_swinl_current_removal_v1/without_point/result.json')
    depth_point = read(ROOT/'reports/course_swinl_guarded_merge640_v1/global_guard_edge_consensus/result.json')
    plan.update(schema='dpseg-large-plain-point-evaluation-v1', created_at_utc=now(),
        authorized_request='Large普通点头的完整评测重新评测一下', training=False,
        gpus=gpus, engineering_gpus=[gpus[0], gpus[-1]], new_source_sha256=sources,
        references=dict(without_point=no_point, depth_point=depth_point),
        ordinary_head_selection='Fixed existing paired epoch6 for equal-budget comparison; no test selection',
        comparison='Same selected view/query candidates and original scores; only point head differs. Probability-depth filter retained.',
        test_analysis='Previously examined complete local test split; exploratory fixed-protocol descriptive ablation.',
        gpu_capacity=gpu_preflight(gpus), test_used_for_selection=False, default_changed=False,
        material_passport=dict(origin_skill='academic-research-suite', origin_mode='run', verification_status='PLANNED'))
    require(len(plan['image_ids']) == 2112 and len(set(plan['image_ids'])) == 2112, 'Wrong full test IDs')
    check_dependencies(plan)
    output.mkdir(parents=True)
    write(output/'protocol.json', plan)
    write(output/'status.json', dict(status='prepared', at_utc=now()))


def run_phase(output, phase, specifications, timeout):
    children, logs, started = [], [], time.monotonic()
    try:
        for name, args, gpu in specifications:
            log = (output/(name+'.log')).open('x'); logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            child = subprocess.Popen([sys.executable, '-u', str(ROOT/SELF), '--output', str(output), *args],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            children.append((name, child))
        write(output/'status.json', dict(status=phase, started_at_utc=now(), controller_pid=os.getpid(),
            processes=[dict(name=name, pid=child.pid) for name, child in children], hard_timeout_seconds=timeout))
        while any(child.poll() is None for _, child in children):
            if time.monotonic()-started > timeout:
                print('Declared hard timeout reached; terminating only this phase children.', flush=True)
                for _, child in children:
                    if child.poll() is None: child.terminate()
                for _, child in children:
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                raise TimeoutError(phase)
            time.sleep(3)
        require(all(child.returncode == 0 for _, child in children), phase+' failed; inspect logs. No automatic retry.')
    finally:
        write(output/(phase+'_exits.json'), {name:child.poll() for name, child in children})
        for log in logs: log.close()


def run(output):
    plan = read(output/'protocol.json')
    require(read(output/'status.json')['status'] == 'prepared', 'Run already started')
    try:
        run_phase(output, 'engineering', [(f'engineering_gpu{gpu}', ['--engineering', gpu], gpu)
            for gpu in plan['engineering_gpus']], 3600)
        gpu_preflight(plan['gpus'])
        run_phase(output, 'inferring', [(f'worker{rank}', ['--worker', str(rank)], gpu)
            for rank, gpu in enumerate(plan['gpus'])], 21600)
        run_phase(output, 'scoring', [('score_plain_point', ['--score'], '')], 7200)
        check_dependencies(plan)
        write(output/'status.json', dict(status='completed', finished_at_utc=now(), exit_code=0))
    except BaseException as error:
        write(output/'status.json', dict(status='failed', error=str(error), at_utc=now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', default='0,2,5,6,7')
    parser.add_argument('--engineering')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    args = parser.parse_args(); output = args.output.resolve()
    if args.engineering is not None: engineering(output, args.engineering)
    elif args.worker is not None: worker(output, args.worker)
    elif args.score: score(output)
    elif args.run_prepared: run(output)
    else:
        prepare(output, args.gpus.split(','))
        if not args.prepare_only: run(output)


if __name__ == '__main__':
    main()
