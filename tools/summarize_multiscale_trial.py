#!/usr/bin/env python3
"""Audit all 30 frozen inference-policy results; select only on development metrics."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METHODS = ('full', 'highres', 'crop_fuse')
COCO = ('mask_ap', 'mask_ap50', 'mask_ap75', 'mask_ap_small', 'mask_ap_medium', 'mask_ap_large',
        'mask_ar_1', 'mask_ar_10', 'mask_ar_100', 'mask_ar_small', 'mask_ar_medium', 'mask_ar_large')
EXPECTED = {('graspnet', split, seed, mode) for split in ('val', 'test')
            for seed, mode in ((17, 'E2'), (23, 'E2'), (41, 'E2'), (17, 'E0'))}
EXPECTED |= {('tless', 'test', 17, mode) for mode in ('E0', 'E2')}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def describe(values):
    require(bool(values) and all(math.isfinite(v) for v in values), 'Expected nonempty finite values')
    return {'n': len(values), 'mean': statistics.mean(values),
            'sample_std_ddof1': statistics.stdev(values) if len(values) > 1 else None,
            'values': values}


def gate_development(rows):
    indexed = {(r['dataset'], r['split'], r['seed'], r['mode'], r['method']): r for r in rows}
    decisions = {}
    for method in METHODS[1:]:
        paired = []
        for seed in (17, 23, 41):
            base = indexed[('graspnet', 'val', seed, 'E2', 'full')]
            trial = indexed[('graspnet', 'val', seed, 'E2', method)]
            paired.append({'seed': seed, 'full_ap_small': base['metrics']['mask_ap_small'],
                           'trial_ap_small': trial['metrics']['mask_ap_small'],
                           'delta_ap_small_pp': trial['metrics']['mask_ap_small'] - base['metrics']['mask_ap_small'],
                           'delta_ap_pp': trial['metrics']['mask_ap'] - base['metrics']['mask_ap'],
                           'latency_ms': trial['timing']['mean_ms']})
        small = describe([p['delta_ap_small_pp'] for p in paired])
        overall = describe([p['delta_ap_pp'] for p in paired])
        conditions = {'mean_delta_ap_small_ge_2pp': small['mean'] >= 2,
                      'positive_small_seeds_at_least_2': sum(p['delta_ap_small_pp'] > 0 for p in paired) >= 2,
                      'mean_delta_ap_ge_minus1pp': overall['mean'] >= -1}
        decisions[method] = {'paired': paired, 'delta_ap_small_pp': small, 'delta_ap_pp': overall,
                             'mean_ap_small': statistics.mean(p['trial_ap_small'] for p in paired),
                             'mean_latency_ms': statistics.mean(p['latency_ms'] for p in paired),
                             'conditions': conditions, 'passed': all(conditions.values())}
    eligible = [method for method in METHODS[1:] if decisions[method]['passed']]
    ranked = sorted(eligible, key=lambda method: (-decisions[method]['mean_ap_small'], decisions[method]['mean_latency_ms']))
    return {'selection_data': 'GraspNet development only, E2 seeds17/23/41; no test/TLESS input to ranking',
            'thresholds': {'mean_delta_ap_small_pp_min': 2, 'positive_seeds_min': 2, 'mean_delta_ap_pp_min': -1},
            'methods': decisions, 'eligible': ranked, 'preferred_optional_mode': ranked[0] if ranked else None,
            'demo_default': 'full', 'interpretation': 'Predeclared engineering gate, not statistical significance. Equal APsmall means are broken by lower mean development latency.'}


def audit(experiment, tolerance):
    from rgbd_seg.data import dataset_fingerprint, RGBDDataset
    from PIL import Image
    experiment = Path(experiment).resolve()
    matrix_path = experiment / 'matrix.json'
    matrix = read(matrix_path)
    require(matrix.get('status') == 'completed', 'Matrix is not completed; refuse partial result selection')
    keys = [(r['dataset'], r['split'], r['seed'], r['mode']) for r in matrix['entries']]
    require(len(keys) == 10 and set(keys) == EXPECTED, 'Matrix must contain exactly the ten prespecified combinations')
    require(len(set(keys)) == len(keys), 'Duplicate matrix combination')
    rows, regressions = [], []
    sources, datasets, checkpoints, devices, protocol, policy = {}, {}, {}, None, None, None
    training_sha = None
    sample_contracts = {}
    for entry, identity in zip(matrix['entries'], keys):
        require(entry.get('status') == 'completed' and entry.get('exit_code') == 0, f'Incomplete/failed job: {identity}')
        directory = Path(entry['output']).resolve()
        require(directory.is_relative_to(experiment), 'Matrix output escapes experiment root')
        summary = read(directory / 'summary.json')
        metadata = read(directory / 'metadata.json')
        require(summary.get('status') == 'completed' and summary['metadata'] == metadata, f'Incomplete or mismatched summary: {directory}')
        require(set(summary['settings']) == set(METHODS), f'Missing or extra inference setting: {directory}')
        dataset_name, split, seed, mode = identity
        require(metadata['model']['seed'] == seed and metadata['model']['mode'] == mode and metadata['split'] == split, f'Matrix/model identity mismatch: {identity}')
        require(metadata['classes'] == ['object'], 'Class contract changed')
        require({'rgbd_seg/model.py', 'rgbd_seg/data.py', 'rgbd_seg/evaluation.py', 'rgbd_seg/multiscale_inference.py', 'tools/evaluate_multiscale.py'}.issubset(metadata['source_sha256']), 'Missing critical source SHA')
        require(metadata['training_steps'] == 800, 'Unexpected training step count')
        require(metadata['input_contract']['gt_used_for_crops'] is False and metadata['input_contract']['mask_threshold'] == 0.5, 'GT-free inference or mask threshold changed')
        checkpoint = Path(metadata['checkpoint'])
        checkpoint_sha = checkpoints.setdefault(str(checkpoint), sha(checkpoint))
        require(metadata['checkpoint_sha256'] == checkpoint_sha, f'Checkpoint bytes differ: {checkpoint}')
        run_path = Path(metadata['run_path'])
        require(sha(run_path) == metadata['run_sha256'], f'Training receipt changed: {run_path}')
        run = read(run_path)
        require(run['checkpoint_sha256'] == checkpoint_sha and run['config'] == metadata['model'] and run['exit_status'] == 'completed', 'Training metadata mismatch')
        require(run['dataset_sha256'] == metadata['training_dataset_sha256'], 'Training dataset identity mismatch')
        if training_sha is None:
            training_sha = metadata['training_dataset_sha256']
        require(metadata['training_dataset_sha256'] == training_sha, 'Different training datasets across checkpoints')
        data_root = Path(metadata['dataset_root'])
        if dataset_name not in datasets:
            datasets[dataset_name] = {'dataset_sha256': dataset_fingerprint(data_root), 'manifest_sha256': sha(data_root / 'manifest.json'), 'root': str(data_root)}
        require(metadata['dataset_sha256'] == datasets[dataset_name]['dataset_sha256'] and metadata['manifest_sha256'] == datasets[dataset_name]['manifest_sha256'], 'Evaluation data fingerprint mismatch')
        contract_key = (dataset_name, split)
        if contract_key not in sample_contracts:
            sample_contracts[contract_key] = []
            for sample_record in RGBDDataset(data_root, split).samples:
                with Image.open(sample_record['rgb_path']) as rgb:
                    width, height = rgb.size
                sample_contracts[contract_key].append({'sample_id': sample_record['id'], 'image_id': sample_record['image_id'], 'height': height, 'width': width})
        sample_contract = sample_contracts[contract_key]
        if dataset_name == 'graspnet':
            require(metadata['dataset_sha256'] == training_sha, 'GraspNet train/eval manifest identity mismatch')
        require(metadata['protocol_sha256'] == matrix['protocol_sha256'] == sha(metadata['protocol_document']), 'Protocol SHA differs from frozen matrix/current document')
        if protocol is None:
            protocol = metadata['protocol_sha256']
            policy = metadata['inference_protocol']
            devices = metadata['device']
        require(metadata['protocol_sha256'] == protocol and metadata['inference_protocol'] == policy, 'Inference policies differ across jobs')
        require(metadata['device'] == devices, 'Timing GPU/threads differ across jobs')
        for name, digest in metadata['source_sha256'].items():
            require(sha(ROOT / name) == digest, f'Current source differs: {name}')
            require(name not in sources or sources[name] == digest, f'Source differed between jobs: {name}')
            sources[name] = digest
        setting_rows = {}
        counts = None
        for method in METHODS:
            path = directory / method / 'evaluation.json'
            result = read(path)
            require(result.get('status') == 'completed' and result['method'] == method, f'Incomplete setting: {path}')
            require(all(result.get(k) == v for k, v in metadata.items()), f'Setting metadata differs: {path}')
            require(sha(path) == summary['settings'][method]['evaluation_sha256'], f'Evaluation SHA differs from job summary: {path}')
            require(result['metrics'] == summary['settings'][method]['metrics'] and result['timing'] == summary['settings'][method]['timing'], 'Summary metric/timing mismatch')
            require(set(result['metrics']) == set(COCO), 'Expected all twelve COCO metrics')
            require(all(v is None or isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 100 for v in result['metrics'].values()), 'Nonfinite/out-of-range metric')
            require(result['metrics']['mask_ap_small'] is not None, 'This matrix requires nonempty original small GT')
            require(result['timing']['warmup'] == 3 and result['timing']['mean_ms'] > 0 and math.isfinite(result['timing']['mean_ms']), 'Invalid timing measurement')
            predpath = directory / method / 'predictions.jsonl'
            require(sha(predpath) == result['predictions_sha256'], f'Prediction RLE bytes changed: {predpath}')
            with predpath.open() as stream:
                predicted_ids = []
                count = 0
                for index, line in enumerate(stream):
                    require(index < len(sample_contract), 'Too many prediction rows')
                    predicted = json.loads(line)
                    contract = sample_contract[index]
                    require(all(predicted.get(k) == v for k, v in contract.items()), f'Prediction ID/image_id/original dimensions mismatch: {predpath}, row {index}')
                    require(all(rle.get('size') == [contract['height'], contract['width']] for rle in predicted['masks_rle']), f'Prediction RLE dimensions differ from original image: {predpath}, row {index}')
                    require(len(predicted['masks_rle']) == len(predicted['scores']) == len(predicted['labels']) == len(predicted['boxes']), 'Prediction candidate vector lengths differ')
                    predicted_ids.append(predicted['sample_id'])
                    count += 1
            expected_ids = [frame['sample_id'] for frame in result['original_area_counts']['per_frame']]
            require(predicted_ids == expected_ids and count == metadata['samples'] == len(sample_contract), 'Prediction/sample order mismatch')
            if counts is None:
                counts = result['original_area_counts']
            require(counts == result['original_area_counts'], 'Original GT areas differ across settings')
            extra = None
            if dataset_name == 'graspnet':
                extra = read(directory / method / 'extra.json')['metrics_macro']
            else:
                require(not (directory / method / 'extra.json').exists(), 'TLESS UOIS metrics prohibited')
            row = {'dataset': dataset_name, 'split': split, 'seed': seed, 'mode': mode, 'method': method,
                   'metrics': result['metrics'], 'extra_macro': extra, 'timing': result['timing'],
                   'original_area_counts': result['original_area_counts']['totals'], 'samples': metadata['samples'],
                   'parameters': metadata['parameters'], 'checkpoint_bytes': metadata['checkpoint_bytes'],
                   'checkpoint_sha256': checkpoint_sha, 'dataset_sha256': metadata['dataset_sha256'],
                   'prediction_identity_dimensions_verified': True, 'evaluation_file': str(path), 'evaluation_sha256': sha(path), 'predictions_sha256': result['predictions_sha256']}
            setting_rows[method] = row
        base = setting_rows['full']
        for method, row in setting_rows.items():
            row['latency_ratio_vs_full'] = row['timing']['mean_ms'] / base['timing']['mean_ms']
            row['delta_vs_full_pp'] = {metric: None if row['metrics'][metric] is None or base['metrics'][metric] is None else row['metrics'][metric] - base['metrics'][metric] for metric in COCO}
            rows.append(row)
        previous_path = ROOT / 'reports/v3/tless' / mode / 'evaluation.json' if dataset_name == 'tless' else checkpoint.parent / ('evaluation.json' if split == 'val' else 'evaluation_test.json')
        previous = read(previous_path)
        require(previous['checkpoint_sha256'] == checkpoint_sha and previous['dataset_sha256'] == metadata['dataset_sha256'] and previous['split'] == split, 'Historical reference identity mismatch')
        require(set(previous['metrics']) == set(COCO), 'Historical reference lacks twelve metrics')
        comparison = {}
        for metric in COCO:
            old, new = previous['metrics'][metric], base['metrics'][metric]
            require((old is None) == (new is None), f'Historical unavailable-bin mismatch: {metric}')
            delta = None if old is None else new - old
            comparison[metric] = {'historical': old, 'current': new, 'delta_pp': delta}
            require(delta is None or abs(delta) <= tolerance, f'Historical full regression exceeds {tolerance}pp: {identity}, {metric}, delta={delta}')
        regressions.append({'dataset': dataset_name, 'split': split, 'seed': seed, 'mode': mode,
                            'reference': str(previous_path), 'reference_sha256': sha(previous_path), 'metrics': comparison,
                            'max_abs_delta_pp': max(abs(v['delta_pp']) for v in comparison.values() if v['delta_pp'] is not None)})
    require(len(rows) == 30, 'Expected exactly thirty complete rows')
    grouped = {}
    for split in ('val', 'test'):
        grouped[split] = {}
        for method in METHODS:
            group = [r for r in rows if r['dataset'] == 'graspnet' and r['split'] == split and r['mode'] == 'E2' and r['method'] == method]
            group.sort(key=lambda r: r['seed'])
            grouped[split][method] = {'seeds': [r['seed'] for r in group],
                'metrics': {m: describe([r['metrics'][m] for r in group]) for m in COCO if all(r['metrics'][m] is not None for r in group)},
                'paired_delta_vs_full_pp': {m: describe([r['delta_vs_full_pp'][m] for r in group]) for m in COCO if all(r['delta_vs_full_pp'][m] is not None for r in group)}}
    return {'status': 'complete_and_verified', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'matrix_sha256': sha(matrix_path), 'protocol_sha256': protocol, 'source_sha256': sources,
            'summarizer_sha256': sha(__file__), 'device': devices, 'inference_protocol': policy,
            'datasets': datasets, 'rows': rows, 'e2_three_seed_descriptive': grouped,
            'full_historical_regression': {'absolute_tolerance_pp': tolerance, 'all_within_tolerance': True, 'comparisons': regressions,
                'interpretation': 'Every actual difference is retained. Numerical agreement within tolerance is not exact bitwise reproducibility or proof of a hardware-only cause.'},
            'development_gate': gate_development(rows),
            'limits': ['All development/test/TLESS data were previously observed; exploratory reuse, not new blind confirmation.',
                       'Training: 64 images from two scenes; development: four small GT; GraspNet test: one small GT; TLESS: 52 small GT with related views.',
                       'Three training seeds describe initialization/training variation; views and seeds are not independent new scene samples. No significance test.',
                       'Fixed-weight inference-policy comparison; cannot establish a causal depth benefit or isolate crop scale from context/filtering/merging.']}


def markdown(result):
    gate = result['development_gate']
    out = ['# 固定权重推理策略实验汇总', '',
           f"全部 **30/30** 组结果完整，checkpoint、数据、协议及评测源码校验通过。开发门槛推荐的可选模式：**{gate['preferred_optional_mode'] or '无'}**；Demo 默认仍为 full。", '',
           '## 开发集门槛（只使用 E2 三种子 val）', '',
           '| 设置 | ΔAPsmall 均值±样本SD（百分点） | ΔAP 均值±样本SD（百分点） | APsmall正增益种子 | 通过 |',
           '|---|---:|---:|---:|---|']
    for method, item in gate['methods'].items():
        small, ap = item['delta_ap_small_pp'], item['delta_ap_pp']
        positives = sum(p['delta_ap_small_pp'] > 0 for p in item['paired'])
        out.append(f"| {method} | {small['mean']:+.3f} ± {small['sample_std_ddof1']:.3f} | {ap['mean']:+.3f} ± {ap['sample_std_ddof1']:.3f} | {positives}/3 | {'是' if item['passed'] else '否'} |")
    out += ['', '门槛：APsmall平均增加≥2个百分点、至少2/3种子为正、总体AP平均下降≤1个百分点。均值和SD以训练种子为单位，SD使用ddof=1；这是工程门槛，不是显著性判断。test与TLESS不参与选择。', '',
            '![开发集配对结果](development_ap_small.png)', '', '## 全部结果与同卡成本', '',
            'AP/AR为0–100尺度；时间为同GPU三次预热后的逐图均值，倍率以同checkpoint、同数据split的full为基准；显存单位GiB=2³⁰字节。', '',
            '| 数据/划分 | seed | 模型 | 设置 | AP | APsmall | ARsmall | ms/图 | 倍率 | 已分配/保留GiB | 小GT |',
            '|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['rows']:
        t, m = row['timing'], row['metrics']
        allocated = t['peak_cuda_allocated_bytes']
        reserved = t['peak_cuda_reserved_bytes']
        memory = f'{allocated/2**30:.3f}/{reserved/2**30:.3f}' if allocated is not None and reserved is not None else '—'
        out.append(f"| {row['dataset']}/{row['split']} | {row['seed']} | {row['mode']} | {row['method']} | {m['mask_ap']:.3f} | {m['mask_ap_small']:.3f} | {m['mask_ar_small']:.3f} | {t['mean_ms']:.2f} | {row['latency_ratio_vs_full']:.2f}× | {memory} | {row['original_area_counts']['small']} |")
    out += ['', '全部十二项COCO、逐设置来源SHA、三种子逐项均值/SD与配对差保存在 [summary.json](summary.json)。保留显存可能包含上一设置的缓存；已分配峰值在各设置预热后重置，含常驻模型。', '', '## full 与历史评测核对', '',
            f"容差为 {result['full_historical_regression']['absolute_tolerance_pp']} 个百分点；实际最大差异逐项保存在JSON，不能把容差内一致写成逐位完全复现。", '',
            '| 数据/划分 | seed | 模型 | 十二项最大绝对差（百分点） |', '|---|---:|---|---:|']
    for row in result['full_historical_regression']['comparisons']:
        out.append(f"| {row['dataset']}/{row['split']} | {row['seed']} | {row['mode']} | {row['max_abs_delta_pp']:.9f} |")
    out += ['', '## 解释范围', '',
            '- 训练仅64图、2场景；开发小GT仅4个、GraspNet测试仅1个。TLESS有52个小GT，但同物体视角相关。',
            '- 所有评测集都已查看过，本轮是探索性复用，没有新的独立确认集。TLESS不使用会压平重叠mask的额外指标。',
            '- highres/crop_fuse增加计算量；裁块还改变上下文与候选过滤/合并，不能把增益全部归因于像素数。',
            '- E0控制只含seed17，不能据此重写三种子深度收益结论。默认full保持不变，增强模式仅按开发门槛进入可选项。', '']
    return '\n'.join(out)


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    rows = result['rows']
    colors = ('#2b6cb0', '#c05621', '#2f855a')
    for seed, color in zip((17, 23, 41), colors):
        values = [next(r['metrics']['mask_ap_small'] for r in rows if (r['dataset'], r['split'], r['mode'], r['seed'], r['method']) == ('graspnet', 'val', 'E2', seed, method)) for method in METHODS]
        axes[0].plot(range(3), values, 'o-', color=color, label=f'seed {seed}')
        axes[1].plot([0, 1], [values[1] - values[0], values[2] - values[0]], 'o-', color=color)
    axes[0].set_xticks(range(3), METHODS)
    axes[0].set_ylabel('Mask AP small (0–100)')
    axes[0].set_title('E2 development: paired seeds')
    axes[0].legend(frameon=False)
    axes[1].set_xticks([0, 1], METHODS[1:])
    axes[1].set_ylabel('Change vs full (percentage points)')
    axes[1].axhline(0, color='gray', linewidth=.8)
    axes[1].axhline(2, color='black', linestyle=':', label='Mean-gain gate: +2 pp')
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title('Descriptive only; four small GT')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'development_ap_small.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default=str(ROOT / 'reports/v4/experiment'))
    parser.add_argument('--output', default=str(ROOT / 'reports/v4/trial_summary'))
    parser.add_argument('--regression-tolerance-pp', type=float, default=0.05)
    args = parser.parse_args()
    require(math.isfinite(args.regression_tolerance_pp) and args.regression_tolerance_pp >= 0, 'Tolerance must be finite and nonnegative')
    output = Path(args.output)
    require(not output.exists() or output.is_dir() and not any(output.iterdir()), 'Refusing to overwrite nonempty output directory')
    result = audit(args.experiment, args.regression_tolerance_pp)
    result['command'] = sys.argv
    output.mkdir(parents=True, exist_ok=True)
    plot(result, output)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (output / 'summary.md').write_text(markdown(result))
    print(json.dumps({'output': str(output.resolve()), 'rows': len(result['rows']), 'development_gate': result['development_gate']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
