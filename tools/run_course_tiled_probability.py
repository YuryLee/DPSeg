#!/usr/bin/env python3
"""Run a frozen validation-only comparison on disjoint available GPUs."""
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
import torch
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_depth_data import load_inference_sample
from rgbd_seg.course_distributed_eval import merge_shards, write_json
from rgbd_seg.course_dafusion_eval import score_predictions
from rgbd_seg.course_optimized import file_sha, load_for_inference
from rgbd_seg.course_tiled_probability import (
    PROBABILITY_POLICY, TILE_POLICY, VARIANTS, predict_variants, refine_probability_masks,
)
from tools.check_course_prediction_reload import compare_predictions


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path):
    p = read(path)
    require(p['status'] == 'frozen' and p['scope'] in ('engineering_smoke', 'full_validation'), 'Invalid scope')
    require(p['tile_policy'] == TILE_POLICY and p['probability_policy'] == PROBABILITY_POLICY, 'Policy changed')
    require(p['variants'] == list(VARIANTS) and p['gpus'] == [2, 3, 5, 6, 7], 'Unexpected arms/devices')
    for name, digest in p['source_sha256'].items():
        require(file_sha(name) == digest, f'Frozen source changed: {name}')
    for key in ('weights', 'training_protocol', 'baseline_predictions', 'binary_depth_predictions'):
        require(file_sha(p[key]['path']) == p[key]['sha256'], f'Frozen input changed: {key}')
    if p['scope'] == 'full_validation':
        engineering = p['engineering']
        require(file_sha(engineering['path']) == engineering['sha256'], 'Engineering evidence changed')
        run = read(engineering['path'])
        require(run['status'] == 'completed' and run['scope'] == 'engineering_smoke'
                and run['images'] == 10 and run['source_sha256'] == p['source_sha256'], 'Engineering checks missing')
    return p


def reference_records(path, ids):
    wanted, values = set(ids), {}
    with Path(path).open() as stream:
        for line in stream:
            item = json.loads(line)
            if item['image_id'] in wanted:
                values[item['image_id']] = item['predictions']
    require(set(values) == wanted, 'Missing reference images')
    return values


def worker(p, output, rank):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(p['gpus'][rank])
            and torch.cuda.device_count() == 1, 'Unexpected GPU mapping')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter, receipt = load_for_inference(p['weights']['path'], p['training_protocol']['path'], device='cuda:0')
    require(receipt['method'] == 'native' and receipt['resolution'] == 1024 and receipt['epoch'] == 5, 'Model differs')
    old_rank = p['training_gpus'].index(p['gpus'][rank])
    reference_dir = Path(p['baseline_predictions']['path']).parent
    check_ids = read(reference_dir/f'rank{old_rank}.json')['ordered_ids'][:2]
    references = reference_records(p['baseline_predictions']['path'], check_ids)
    depth_references = reference_records(p['binary_depth_predictions']['path'], check_ids)
    rank_ids = read(output/'image_ids.json')[rank::len(p['gpus'])]
    # Compare CPU/GPU arithmetic at a physical depth discontinuity with missing depth.
    generator = torch.Generator().manual_seed(17)
    soft = torch.rand((3, 33, 35), generator=generator)
    raw = soft > .5
    z = torch.ones(33, 35)*.5
    z[:, 17:] += .01
    valid = torch.ones_like(z, dtype=torch.bool)
    valid[3:7, 4:8] = False
    z[~valid] = float('nan')
    cpu, stats = refine_probability_masks(soft, raw, z, valid)
    gpu, gpu_stats = refine_probability_masks(soft.cuda(), raw.cuda(), z.cuda(), valid.cuda())
    require(torch.equal(cpu, gpu.cpu()) and stats == gpu_stats, 'CPU/GPU probability refinement mismatch')
    totals = {name: 0 for name in VARIANTS}
    timing = {'native_seconds': 0., 'tile_extra_seconds': 0.,
              'refinement_seconds': {name: 0. for name in VARIANTS if 'depth' in name}}
    checks = []
    began = time.monotonic()
    with CourseArchive(p['archive'], 'val') as data, ExitStack() as stack, torch.inference_mode():
        streams = {name: stack.enter_context((output/name/f'predictions_rank{rank}.jsonl').open('x'))
                   for name in VARIANTS}
        by_id = {item['id']: item for item in data.images}
        first = load_inference_sample(data, check_ids[0])
        for _ in range(3):
            adapter.predict_sample(first, score_threshold=.01, max_detections=100)
        # Bounded reload checks use the original GPU class and official rank IDs.
        for image_id in check_ids:
            sample = load_inference_sample(data, image_id)
            actual = adapter.predict_sample(sample, score_threshold=.01, max_detections=100)
            error = compare_predictions(references[image_id], actual)
            checks.append({'image_id': image_id, 'reload_max_score_difference': error})
        for position, image_id in enumerate(rank_ids):
            sample = load_inference_sample(data, image_id)
            expected = adapter.predict_sample(sample) if position == 0 else None
            def native_check(values):
                if expected is not None:
                    compare_predictions(expected, values)
                if image_id in references:
                    compare_predictions(references[image_id], values)
            outputs, diagnostics = predict_variants(adapter, sample, native_check=native_check)
            if image_id in depth_references:
                compare_predictions(depth_references[image_id], outputs['native_binary_depth'])
            for name in VARIANTS:
                values = outputs[name]
                require(len(values) <= 100, 'Too many instances')
                base = outputs['tiled' if name.startswith('tiled') else 'native']
                if 'depth' in name:
                    require([(v['category_id'], v['score']) for v in values] ==
                            [(v['category_id'], v['score']) for v in base], 'Refinement changed score/order/count')
                totals[name] += len(values)
                streams[name].write(json.dumps({'image_id': image_id, 'file_name': by_id[image_id]['file_name'],
                    'predictions': values}, allow_nan=False)+'\n')
            timing['native_seconds'] += diagnostics['native_seconds']
            timing['tile_extra_seconds'] += diagnostics['tile_extra_seconds']
            for name, value in diagnostics['refinement_seconds'].items():
                timing['refinement_seconds'][name] += value
            if p['scope'] == 'engineering_smoke' or position == 0:
                write_json(output/f'diagnostics_rank{rank}_image{image_id}.json', diagnostics)
            if position == 0 or (position+1) % 10 == 0 or position+1 == len(rank_ids):
                for stream in streams.values():
                    stream.flush()
                write_json(output/f'progress_rank{rank}.json', {'phase': 'inferring', 'rank': rank,
                    'physical_gpu': p['gpus'][rank], 'images': position+1, 'expected': len(rank_ids),
                    'elapsed_seconds': time.monotonic()-began, 'instances': totals})
    write_json(output/f'rank{rank}.json', {'status': 'completed', 'rank': rank, 'pid': os.getpid(),
        'physical_gpu': p['gpus'][rank], 'ordered_ids': rank_ids, 'images': len(rank_ids), 'instances': totals,
        'reload_checks': checks, 'cpu_gpu_probability_check_passed': True, 'reload': receipt,
        'timing': timing, 'elapsed_seconds': time.monotonic()-began,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'predictions_sha256': {name: file_sha(output/name/f'predictions_rank{rank}.jsonl') for name in VARIANTS}})


def score(p, output, name):
    ids = read(output/'image_ids.json')
    paths = [output/name/f'predictions_rank{rank}.jsonl' for rank in range(len(p['gpus']))]
    for rank, path in enumerate(paths):
        receipt = read(output/f'rank{rank}.json')
        require(receipt['status'] == 'completed' and receipt['ordered_ids'] == ids[rank::len(p['gpus'])]
                and receipt['predictions_sha256'][name] == file_sha(path), 'Incomplete or changed prediction shard')
    count = merge_shards(paths, ids, output/name/'predictions.jsonl')
    with CourseArchive(p['archive'], 'val') as data:
        result = score_predictions(data, output/name/'predictions.jsonl', output/name,
                                   image_ids=ids if p['scope'] == 'engineering_smoke' else None)
    require(result['images'] == len(ids) and result['predicted_instances'] == count, 'Scoring coverage differs')
    write_json(output/name/'scored.json', {'status': 'completed', 'images': len(ids),
        'evaluation_sha256': file_sha(output/name/'evaluation.json'), 'scope': p['scope']})


def summarize(p, output):
    results = {name: read(output/name/'evaluation.json')['metrics'] for name in VARIANTS}
    baseline = results['native_binary_depth']
    comparisons = {}
    for name in ('native_probability_depth', 'tiled_binary_depth', 'tiled_probability_depth'):
        gain = results[name]['mask_ap_small']-baseline['mask_ap_small']
        drop = baseline['mask_ap']-results[name]['mask_ap']
        comparisons[name] = {'APsmall_gain': gain, 'AP_drop': drop,
                             'eligible': gain >= .005 and drop <= .005}
    eligible = [name for name, result in comparisons.items() if result['eligible']]
    selected = max(eligible, key=lambda name: (results[name]['mask_ap_small'], results[name]['mask_ap'])) if eligible else 'native_binary_depth'
    report = {'status': 'completed', 'scope': p['scope'], 'images': len(read(output/'image_ids.json')),
        'material_passport': {'origin_skill': 'academic-research-suite/experiment-agent', 'mode': 'run',
                              'verification_status': 'VERIFIED_PAIRED_OUTPUTS'},
        'metrics': results, 'comparisons_to_binary_depth': comparisons, 'selected_variant': selected,
        'selection_is_formal': p['scope'] == 'full_validation', 'test_accessed': False,
        'historical_best_APsmall': .5256361837733781,
        'baseline_APsmall_difference_from_historical': baseline['mask_ap_small']-.5256361837733781,
        'protocol_sha256': file_sha(p['_path']),
        'limits': ['One checkpoint and one fixed configuration per intervention; no significance claim',
                   'Crop-view scores are unchanged and may differ in calibration from whole-image scores',
                   'Tile inference includes representative selection and top100; those can remove instances']}
    write_json(output/'comparison.json', report)
    names = {'native': '整图原始预测', 'native_binary_depth': '整图＋现有二值深度细化',
             'native_probability_depth': '整图＋概率图深度细化', 'tiled': '整图＋切片',
             'tiled_binary_depth': '整图＋切片＋二值深度细化', 'tiled_probability_depth': '整图＋切片＋概率图深度细化'}
    lines = ['# 切片推理与概率图深度细化对照', '', f"范围：{p['scope']}，{report['images']} 张验证图像。", '',
             '| 配置 | Mask APₛ | Mask AP | ARₛ | 相对现有深度版 APₛ |', '|---|---:|---:|---:|---:|']
    for name in VARIANTS:
        m = results[name]
        lines.append(f"| {names[name]} | {m['mask_ap_small']:.6f} | {m['mask_ap']:.6f} | {m['mask_ar_small']:.6f} | {m['mask_ap_small']-baseline['mask_ap_small']:+.6f} |")
    lines += ['', f"按预定门槛选择：**{names[selected]}**。工程样本的选择仅为流程检查，不用于正式结论。", '',
              '改进门槛：相对同轮二值深度版，APₛ 至少提高 0.005，整体 AP 下降不超过 0.005。', '',
              '概率版替换二值深度后处理：保留原始置信度，使用相同 5×5 邻域、空间 σ=1.5 像素、深度 σ=1 mm，一次加权后再二值化。',
              '切片版保留整图预测，增加四个 640×640 窗口；同步裁剪 RGB/深度并平移相机内参；按跨视图 mask IoU>0.5 去重后保留前 100 个。', '',
              '模型权重、训练配置和历史结果均保持冻结。无测试集标签参与，无 Kaggle 公榜调参。']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    return report


def parent(p, protocol_path, output):
    require(not output.exists(), 'Output must be new')
    output.mkdir(parents=True)
    for name in VARIANTS:
        (output/name).mkdir()
    with CourseArchive(p['archive'], 'val') as data:
        require(len(data) == 2584, 'Validation split differs')
        ids = data.image_ids
    if p['scope'] == 'engineering_smoke':
        folder = Path(p['baseline_predictions']['path']).parent
        chosen = set()
        for gpu in p['gpus']:
            chosen.update(read(folder/f"rank{p['training_gpus'].index(gpu)}.json")['ordered_ids'][:2])
        ids = [image_id for image_id in ids if image_id in chosen]
        require(len(ids) == 10, 'Engineering subset coverage differs')
    write_json(output/'image_ids.json', ids)
    run = {'status': 'running', 'scope': p['scope'], 'images': len(ids), 'pid': os.getpid(),
           'started_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': file_sha(protocol_path),
           'source_sha256': p['source_sha256'], 'gpus': p['gpus'], 'test_accessed': False, 'workers': []}
    write_json(output/'run.json', run)
    processes = []
    with ExitStack() as stack:
        def launch(command, label, gpu=None):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            log = stack.enter_context((output/f'{label}.log').open('x'))
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(child)
            return child
        def wait(children, deadline):
            start = time.monotonic()
            while any(child.poll() is None for child in children):
                if time.monotonic()-start > deadline:
                    run['timeout_notice'] = 'Predeclared hard timeout; terminating only this experiment process group'
                    write_json(output/'run.json', run)
                    for child in children:
                        if child.poll() is None:
                            child.terminate()
                    for child in children:
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                    raise TimeoutError('Experiment hard timeout')
                time.sleep(1)
            require(all(child.returncode == 0 for child in children), 'Experiment process failed; inspect preserved logs')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--protocol', str(protocol_path), '--output', str(output)]
        try:
            workers = []
            for rank, gpu in enumerate(p['gpus']):
                child = launch(command+['--rank', str(rank)], f'worker{rank}', gpu)
                workers.append(child)
                run['workers'].append({'rank': rank, 'physical_gpu': gpu, 'pid': child.pid})
            write_json(output/'run.json', run)
            wait(workers, p['worker_timeout_seconds'])
            run['phase'] = 'scoring'
            run['worker_exit_codes'] = [child.returncode for child in workers]
            write_json(output/'run.json', run)
            scorers = [launch(command+['--score', name], f'score_{name}') for name in VARIANTS]
            wait(scorers, p['scoring_timeout_seconds'])
            summary = summarize(p, output)
            run.update(status='completed', selected_variant=summary['selected_variant'], exit_code=0)
        except BaseException as error:
            run.update(status='failed', error_type=type(error).__name__, error=str(error), exit_code=1)
            raise
        finally:
            run['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            write_json(output/'run.json', run)
            write_json(output/'exit.json', {'status': run['status'], 'exit_code': run.get('exit_code', 1),
                                           'process_exit_codes': [child.poll() for child in processes]})
    print(json.dumps({'status': run['status'], 'output': str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--score', choices=VARIANTS)
    args = parser.parse_args()
    p = checked(args.protocol)
    p['_path'] = str(args.protocol.resolve())
    if args.rank is not None:
        worker(p, args.output.resolve(), args.rank)
    elif args.score:
        score(p, args.output.resolve(), args.score)
    else:
        parent(p, args.protocol.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
