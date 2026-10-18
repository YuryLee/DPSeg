#!/usr/bin/env python3
"""One fixed full-test report, reusing predictions and frozen validation choices.

The user's explicit request to compare all configurations on test authorizes
scoring previously rejected validation candidates for this report. Historical
protocols and their adoption decisions remain unchanged. No parameter selection
or training uses test annotations; only the standalone COCO scorer receives GT.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import csv
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
from rgbd_seg.course_finetune_eval import validate_csv_fast
from rgbd_seg.course_optimized import file_sha, load_for_inference
from rgbd_seg.course_submission import SubmissionWriter
from rgbd_seg.course_tiled_probability import VARIANTS, TILE_POLICY, PROBABILITY_POLICY, predict_variants
from tools.check_course_prediction_reload import compare_predictions

OUT = ROOT/'reports/course_all_test_comparison'
LAYOUT = [2, 3, 5, 6, 7, 2, 3, 5, 6, 7]
NEW = ('native_probability_depth', 'tiled', 'tiled_binary_depth', 'tiled_probability_depth')


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_sha(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inventory():
    def item(key, label, *, evaluation=None, predictions=None, transform=None, new=None, public=None):
        row = {'key': key, 'label': label, 'kaggle_public_APs': public}
        if evaluation:
            row['existing_evaluation'] = binding(ROOT/evaluation)
        if predictions:
            row['predictions'] = binding(ROOT/predictions)
        if transform:
            row['transform'] = transform
        if new:
            row['new_variant'] = new
        return row
    export = 'reports/kaggle_batch_20261016/exports_v1'
    return [
        item('pretrained_maskrcnn', 'Mask R-CNN：公开权重直接推理', evaluation='reports/course_pretrained/maskrcnn/evaluation.json'),
        item('pretrained_mask2former', 'Mask2Former：公开权重直接推理', evaluation='reports/course_pretrained/mask2former/evaluation.json'),
        item('finetuned_maskrcnn', 'Mask R-CNN：课程微调', evaluation='reports/course_finetune/maskrcnn/test/evaluation.json'),
        item('finetuned_mask2former', 'Mask2Former：课程微调', evaluation='reports/course_finetune/mask2former/test/evaluation.json', public=.14189),
        item('historical_dafusion', 'DA-Fusion 800：历史单卡', evaluation='reports/course_dafusion/formal_v1/test/evaluation.json', public=.28063),
        item('sampling', 'DA-Fusion 800＋小目标点采样', evaluation='reports/course_small_target/formal_v1/test/evaluation.json'),
        item('cleanup_original', 'DA-Fusion 800＋最大连通域', predictions='reports/course_dafusion/formal_v1/test/predictions.jsonl', transform='largest_component'),
        item('cleanup_sampling', 'DA-Fusion 800＋点采样＋最大连通域', predictions='reports/course_small_target/formal_v1/test/predictions.jsonl', transform='largest_component'),
        item('native_800', 'DA-Fusion 800：多卡配对基线', predictions=f'{export}/native_800/native_800/predictions.jsonl', public=.25015),
        item('native_1024', 'DA-Fusion 1024：多卡基线', predictions=f'{export}/native_1024/native_1024/predictions.jsonl', public=.35025),
        item('crop_1024', '1024＋局部裁剪训练', predictions=f'{export}/crop_1024/crop_1024/predictions.jsonl', public=.33079),
        item('focal_1024', '1024＋Focal（γ=2）', predictions=f'{export}/focal_1024/focal_1024/predictions.jsonl', public=.33763),
        item('spatial_1024', '1024＋普通空间平滑', predictions=f'{export}/native_1024/spatial_1024/predictions.jsonl'),
        item('depth_1024', '1024＋二值深度细化', predictions='reports/course_small_target_final/test_v1/predictions.jsonl', public=.40457),
        item('native_probability_depth', '1024＋概率图深度细化', new='native_probability_depth'),
        item('tiled', '1024＋切片推理', new='tiled'),
        item('tiled_binary_depth', '1024＋切片＋二值深度细化', new='tiled_binary_depth'),
        item('tiled_probability_depth', '1024＋切片＋概率深度细化', new='tiled_probability_depth'),
    ]


def prepare():
    require(not OUT.exists(), 'Comparison output must be new')
    OUT.mkdir()
    old = read(ROOT/'reports/course_tiled_probability/protocol_full_v1.json')
    sources = dict(old['source_sha256'])
    sources[str(Path(__file__).resolve())] = file_sha(__file__)
    sources[str(ROOT/'tools/run_course_component_cleanup.py')] = file_sha(ROOT/'tools/run_course_component_cleanup.py')
    protocol = {'schema': 'course_all_test_comparison_v1', 'status': 'frozen',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'material_passport': {'origin_skill': 'academic-research-suite/experiment-agent', 'mode': 'run/analyze'},
        'authorization': 'User explicitly requested all completed experiments compared using test-set AP and APs',
        'archive': old['archive'], 'weights': old['weights'], 'training_protocol': old['training_protocol'],
        'validation_selection': binding(ROOT/'reports/course_tiled_probability/decision.json'),
        'test_annotation_audit': binding(ROOT/'reports/course_pretrained/dataset_audit.json'),
        'worker_layout': LAYOUT, 'tile_policy': TILE_POLICY, 'probability_policy': PROBABILITY_POLICY,
        'source_sha256': sources, 'inventory': inventory(), 'images': 2112,
        'parameter_policy': 'All model choices and inference parameters frozen from validation; one descriptive test comparison',
        'worker_timeout_seconds': 10800, 'scoring_timeout_seconds': 3600, 'upload_to_kaggle': False}
    require(read(protocol['test_annotation_audit']['path'])['actual_test_gt_present'], 'Actual test annotation evidence missing')
    write_json(OUT/'protocol.json', protocol)
    return protocol


def checked(path):
    p = read(path)
    require(p['status'] == 'frozen' and p['worker_layout'] == LAYOUT, 'Protocol differs')
    require(p['tile_policy'] == TILE_POLICY and p['probability_policy'] == PROBABILITY_POLICY, 'Inference policy differs')
    for name, digest in p['source_sha256'].items():
        require(file_sha(name) == digest, f'Frozen source changed: {name}')
    for key in ('weights', 'training_protocol', 'validation_selection', 'test_annotation_audit'):
        require(file_sha(p[key]['path']) == p[key]['sha256'], f'Bound input changed: {key}')
    return p


def model_worker(p, rank, engineering=False):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(LAYOUT[rank]) and torch.cuda.device_count() == 1, 'GPU mapping differs')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, reload = load_for_inference(p['weights']['path'], p['training_protocol']['path'], device='cuda:0')
    folder = OUT/('engineering' if engineering else 'export')
    reference = read(folder/f'reference_rank{rank}.json')
    ids = reference['image_ids']
    refs = reference['records']
    began = time.monotonic()
    totals = {name: 0 for name in VARIANTS}
    with CourseArchive(p['archive'], 'val' if engineering else 'test') as data, ExitStack() as stack:
        streams = {} if engineering else {name: stack.enter_context((folder/name/f'predictions_rank{rank}.jsonl').open('x')) for name in VARIANTS}
        by_id = {record['id']: record for record in data.images}
        first = load_inference_sample(data, ids[0])
        for _ in range(2):
            model.predict_sample(first)
        for position, image_id in enumerate(ids):
            sample = load_inference_sample(data, image_id)
            expected = refs[str(image_id)]
            values, diagnostics = predict_variants(model, sample,
                native_check=lambda actual: compare_predictions(expected['native'], actual))
            compare_predictions(expected['native_binary_depth'], values['native_binary_depth'])
            if engineering:
                for name in VARIANTS:
                    compare_predictions(expected[name], values[name])
            else:
                for name in VARIANTS:
                    require(len(values[name]) <= 100, 'Instance cap exceeded')
                    base = values['tiled' if name.startswith('tiled') else 'native']
                    if 'depth' in name:
                        require([x['score'] for x in base] == [x['score'] for x in values[name]], 'Refinement changed score/count/order')
                    streams[name].write(json.dumps({'image_id': image_id, 'file_name': by_id[image_id]['file_name'], 'predictions': values[name]}, allow_nan=False)+'\n')
                    totals[name] += len(values[name])
            if position == 0 or (position+1) % 10 == 0 or position+1 == len(ids):
                for stream in streams.values():
                    stream.flush()
                write_json(folder/f'progress_rank{rank}.json', {'rank': rank, 'physical_gpu': LAYOUT[rank],
                    'images': position+1, 'expected': len(ids), 'elapsed_seconds': time.monotonic()-began})
    write_json(folder/f'rank{rank}.json', {'status': 'completed', 'rank': rank, 'physical_gpu': LAYOUT[rank],
        'images': len(ids), 'ordered_ids': ids, 'instances': totals, 'engineering': engineering,
        'native_and_binary_depth_exact_checks': len(ids), 'all_six_exact_checks': len(ids) if engineering else 0,
        'reload': reload, 'elapsed_seconds': time.monotonic()-began,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'prediction_sha256': {} if engineering else {name: file_sha(folder/name/f'predictions_rank{rank}.jsonl') for name in VARIANTS}})


def score_one(p, key):
    row = next(row for row in p['inventory'] if row['key'] == key)
    folder = OUT/'scores'/key
    folder.mkdir(parents=True, exist_ok=False)
    if row.get('new_variant'):
        source = OUT/'export'/row['new_variant']/'predictions.jsonl'
    else:
        source = Path(row['predictions']['path'])
        require(file_sha(source) == row['predictions']['sha256'], 'Saved prediction input changed')
    original_source = binding(source)
    if row.get('transform'):
        # This import intentionally lives in the CPU-only role: the historical
        # helper disables CUDA globally. Its old adoption gate is not edited.
        from tools.run_course_component_cleanup import transform
        destination = folder/'predictions.jsonl'
        count = 0
        with source.open() as stream, destination.open('x') as saved, ThreadPoolExecutor(max_workers=4) as pool:
            import itertools
            while lines := list(itertools.islice(stream, 16)):
                for result, stats in pool.map(transform, [json.loads(line) for line in lines]):
                    saved.write(json.dumps(result, allow_nan=False)+'\n')
                    count += 1
            require(count == 2112, 'Incomplete cleanup source')
        source = destination
    with CourseArchive(p['archive'], 'test') as dataset:
        require(len(dataset) == 2112, 'Full test split differs')
        result = score_predictions(dataset, source, folder)
        require(result['images'] == 2112 and result['gt_instances'] == 75462, 'Wrong test evaluation scope')
        if row.get('new_variant'):
            filename = {'native_probability_depth': 'rgbd_dafusion_swin_t_1024_e05_probdepth.csv',
                        'tiled': 'rgbd_dafusion_swin_t_1024_e05_tile640.csv',
                        'tiled_binary_depth': 'rgbd_dafusion_swin_t_1024_e05_tile640_binarydepth.csv',
                        'tiled_probability_depth': 'rgbd_dafusion_swin_t_1024_e05_tile640_probdepth.csv'}[key]
            csv_path = OUT/'submissions'/filename
            with source.open() as stream, SubmissionWriter(csv_path, dataset.image_ids) as writer:
                for line in stream:
                    item = json.loads(line)
                    writer.write(item['image_id'], item['predictions'])
            validation = validate_csv_fast(csv_path, dataset.image_ids)
            write_json(folder/'submission_validation.json', validation)
    receipt = {'status': 'completed', 'source': original_source, 'scored_predictions': binding(source),
               'evaluation': binding(folder/'evaluation.json'), 'images': 2112, 'key': key}
    if row.get('new_variant'):
        receipt['submission'] = binding(csv_path)
    write_json(folder/'receipt.json', receipt)


def make_references(p, engineering=False):
    folder = OUT/('engineering' if engineering else 'export')
    folder.mkdir()
    split = 'val' if engineering else 'test'
    with CourseArchive(p['archive'], split) as data:
        ids = data.image_ids[:10] if engineering else data.image_ids
    sources = ({name: ROOT/'reports/course_tiled_probability/full_v1'/name/'predictions.jsonl' for name in VARIANTS}
               if engineering else {'native': Path(next(r for r in p['inventory'] if r['key'] == 'native_1024')['predictions']['path']),
                                    'native_binary_depth': Path(next(r for r in p['inventory'] if r['key'] == 'depth_1024')['predictions']['path'])})
    by_id = {image_id: {} for image_id in ids}
    for name, path in sources.items():
        with path.open() as stream:
            for line in stream:
                item = json.loads(line)
                if item['image_id'] in by_id:
                    by_id[item['image_id']][name] = item['predictions']
    require(all(len(row) == len(sources) for row in by_id.values()), 'Reference coverage differs')
    for rank in range(len(LAYOUT)):
        rank_ids = ids[rank::len(LAYOUT)]
        write_json(folder/f'reference_rank{rank}.json', {'image_ids': rank_ids,
                   'records': {str(image_id): by_id[image_id] for image_id in rank_ids}})
    write_json(folder/'image_ids.json', ids)
    if not engineering:
        for name in VARIANTS:
            (folder/name).mkdir()


def summarize(p):
    rows = []
    public = {row['configuration']: row for row in csv.DictReader((ROOT/'reports/kaggle_batch_20261016/results.csv').open())}
    for item in p['inventory']:
        path = Path(item['existing_evaluation']['path']) if item.get('existing_evaluation') else OUT/'scores'/item['key']/'evaluation.json'
        if item.get('existing_evaluation'):
            require(file_sha(path) == item['existing_evaluation']['sha256'], 'Historical evaluation changed')
        else:
            receipt = read(OUT/'scores'/item['key']/'receipt.json')
            require(receipt['status'] == 'completed' and receipt['evaluation']['sha256'] == file_sha(path), 'Scoring receipt differs')
        value = read(path)
        require(value['split'] == 'test' and value['images'] == 2112 and value['gt_instances'] == 75462, 'Incomparable test scope')
        m = value['metrics']
        score = item.get('kaggle_public_APs')
        if item['key'] in public and public[item['key']]['kaggle_public_score']:
            score = float(public[item['key']]['kaggle_public_score'])
        row = {'configuration': item['key'], 'method': item['label'], 'test_mask_AP': m['mask_ap'],
               'test_mask_APs': m['mask_ap_small'], 'kaggle_public_APs': score,
               'images': 2112, 'evaluation_path': str(path), 'evaluation_sha256': file_sha(path)}
        if item.get('new_variant'):
            row['submission_csv'] = read(OUT/'scores'/item['key']/'receipt.json')['submission']['path']
        rows.append(row)
    selected = read(p['validation_selection']['path'])['selected_variant']
    report = {'status': 'completed', 'scope': 'Full 2112-image course test comparison', 'scale': '0–1',
              'material_passport': {'origin_skill': 'academic-research-suite/experiment-agent', 'verification_status': 'VERIFIED_ARTIFACT_BINDINGS'},
              'rows': rows, 'validation_selected_variant': selected,
              'highest_observed_test_APs': max(rows, key=lambda r: r['test_mask_APs'])['configuration'],
              'test_labels_used': 'Standalone COCO scoring only; parameters and validation selection unchanged',
              'actual_test_gt_present': True, 'protocol_sha256': file_sha(OUT/'protocol.json')}
    write_json(OUT/'comparison.json', report)
    fields = ['configuration', 'method', 'test_mask_AP', 'test_mask_APs', 'kaggle_public_APs', 'images', 'evaluation_path', 'evaluation_sha256', 'submission_csv']
    with (OUT/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 所有课程实验：完整测试集 AP 与 APₛ 对比', '',
        '统一范围：下载包中全部 2,112 张 test 图片、75,462 个真实实例，其中 9,447 个 small。指标为 COCO Mask AP@[.50:.95] 与 Mask APₛ@[.50:.95]，分数均为 0–1。', '',
        '| 方法 | 测试 Mask AP | 测试 Mask APₛ | 已知 Kaggle 公榜 APₛ |', '|---|---:|---:|---:|']
    for row in rows:
        official = '—' if row['kaggle_public_APs'] is None else f"{row['kaggle_public_APs']:.5f}"
        lines.append(f"| {row['method']} | {row['test_mask_AP']:.6f} | {row['test_mask_APs']:.6f} | {official} |")
    lines += ['', '## 口径说明', '',
        '- 北航下载的 blender.zip 实际有 annotations/test/*.json.gz 实例标注，见此前数据审计。比赛说明中的“测试不公开标注”与该下载包内容不一致；上一条回答仅依照比赛说明，未反映这个已核实的实际情况。',
        '- 本地测试列使用完整 2,112 张。Kaggle 公榜约使用 30% 的测试图片，官方实现也尚未取得，不能把两个分数当作同一范围。公榜未公布整体 Mask AP，因此该列只列 APₛ。',
        '- 表中微调模型均采用此前验证集选定的权重。小目标点采样版选第 4 轮，其余训练配置选第 5 轮。历史单卡 batch8、9020 次更新；多卡 batch33、2190 次更新，同为 5 轮但训练预算不同。',
        '- 最大连通域两组此前验证失败，因此当时没有执行测试；此次按用户“所有实验用测试集对比”的明确要求补齐评分，原协议及采用决定保留。',
        '- 最新四组方案参数沿用完整验证集实验，进行一次固定测试比较；没有按测试成绩修改参数、重新训练或更改原验证集选择。',
        '- 历史 Mask2Former 与单卡 DA-Fusion 的公榜分数来自用户此前提供的记录；本轮六组公榜分数取自本地已核实提交台账。新四组 CSV 已生成并校验，尚未上传。', '',
        '## 文件', '', '[对比数据 CSV](comparison.csv) · [原始数值和来源](comparison.json) · [对比图 PNG](comparison.png) · [对比图 PDF](comparison.pdf)', '',
        'GraspNet 和 T-LESS 是早期不同数据集的探索，详见原报告；这些分数不能与课程测试集作同表性能排名。']
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
    # Publication/export chart: use the installed CJK font and a writable cache.
    os.environ['MPLCONFIGDIR'] = '/tmp/course-test-comparison-mpl'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np
    font = FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    fig, ax = plt.subplots(figsize=(14, 12), layout='constrained')
    y = np.arange(len(rows))
    ax.barh(y-.18, [r['test_mask_AP'] for r in rows], height=.33, color='#356990', label='Mask AP')
    ax.barh(y+.18, [r['test_mask_APs'] for r in rows], height=.33, color='#dc8c35', label='Mask APs')
    for i, row in enumerate(rows):
        ax.text(row['test_mask_AP']+.009, i-.18, f"{row['test_mask_AP']:.4f}", va='center', fontsize=8)
        ax.text(row['test_mask_APs']+.009, i+.18, f"{row['test_mask_APs']:.4f}", va='center', fontsize=8)
    ax.set_yticks(y, [r['method'] for r in rows], fontproperties=font, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, .1))
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    ax.set_title('课程完整测试集：全部配置的 AP 与 APs', fontproperties=font, fontsize=17, pad=16)
    ax.set_xlabel('COCO Mask AP@[0.50:0.95] · 0–1', fontsize=11)
    ax.legend(loc='lower right')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.savefig(OUT/'comparison.png', dpi=200)
    fig.savefig(OUT/'comparison.pdf')
    plt.close(fig)
    return report


def parent(p):
    state = {'status': 'running', 'phase': 'engineering', 'pid': os.getpid(),
             'started_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': file_sha(OUT/'protocol.json')}
    write_json(OUT/'run.json', state)
    (OUT/'scores').mkdir()
    (OUT/'submissions').mkdir()
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--protocol', str(OUT/'protocol.json')]
    children = []
    with ExitStack() as stack:
        def spawn(arguments, label, gpu=None):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu), OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            log = stack.enter_context((OUT/f'{label}.log').open('x'))
            child = subprocess.Popen(command+arguments, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
            return child
        def wait(group, timeout):
            began = time.monotonic()
            while any(child.poll() is None for child in group):
                if time.monotonic()-began > timeout:
                    for child in group:
                        if child.poll() is None:
                            child.terminate()
                    raise TimeoutError('Predeclared hard deadline exceeded')
                time.sleep(1)
            require(all(child.returncode == 0 for child in group), 'A comparison process failed; inspect saved logs')
        try:
            make_references(p, engineering=True)
            engineers = [spawn(['--gpu-worker', str(rank), '--engineering'], f'engineering_rank{rank}', gpu) for rank, gpu in enumerate(LAYOUT)]
            wait(engineers, 600)
            write_json(OUT/'engineering.json', {'status': 'passed', 'images': 10, 'all_six_exact_checks': 60,
                'two_processes_per_gpu': True, 'exit_codes': [child.returncode for child in engineers]})
            state['phase'] = 'test_inference_and_saved_prediction_scoring'
            write_json(OUT/'run.json', state)
            scorers = [spawn(['--score', row['key']], f"score_{row['key']}") for row in p['inventory'] if row.get('predictions')]
            make_references(p, engineering=False)
            workers = [spawn(['--gpu-worker', str(rank)], f'worker_rank{rank}', gpu) for rank, gpu in enumerate(LAYOUT)]
            state['workers'] = [{'rank': rank, 'gpu': gpu, 'pid': workers[rank].pid} for rank, gpu in enumerate(LAYOUT)]
            write_json(OUT/'run.json', state)
            wait(workers, p['worker_timeout_seconds'])
            ids = read(OUT/'export/image_ids.json')
            for name in VARIANTS:
                paths = [OUT/'export'/name/f'predictions_rank{rank}.jsonl' for rank in range(len(LAYOUT))]
                for rank, path in enumerate(paths):
                    receipt = read(OUT/'export'/f'rank{rank}.json')
                    require(receipt['ordered_ids'] == ids[rank::len(LAYOUT)] and receipt['prediction_sha256'][name] == file_sha(path), 'Export shard changed')
                merge_shards(paths, ids, OUT/'export'/name/'predictions.jsonl')
            state['phase'] = 'final_test_scoring_and_csv_validation'
            write_json(OUT/'run.json', state)
            scorers.extend(spawn(['--score', name], f'score_{name}') for name in NEW)
            wait(scorers, p['scoring_timeout_seconds'])
            report = summarize(p)
            state.update(status='completed', phase='completed', images=2112, configurations=len(report['rows']), exit_code=0)
        except BaseException as error:
            state.update(status='failed', error_type=type(error).__name__, error=str(error), exit_code=1)
            raise
        finally:
            state['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            write_json(OUT/'run.json', state)
            write_json(OUT/'exit.json', {'status': state['status'], 'exit_code': state.get('exit_code', 1),
                                        'process_exit_codes': [child.poll() for child in children]})
    print(json.dumps({'status': state['status'], 'report': str(OUT/'comparison.md')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--gpu-worker', type=int)
    parser.add_argument('--engineering', action='store_true')
    parser.add_argument('--score')
    args = parser.parse_args()
    if args.protocol is None:
        p = prepare()
    else:
        p = checked(args.protocol)
    if args.gpu_worker is not None:
        model_worker(p, args.gpu_worker, args.engineering)
    elif args.score:
        score_one(p, args.score)
    else:
        parent(p)


if __name__ == '__main__':
    main()
