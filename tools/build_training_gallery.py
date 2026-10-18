#!/usr/bin/env python3
"""Build a self-contained v5 gallery from completed RLEs; never run a model.

Freeze six cases using GT only before building: all four GraspNet development
frames, its one small-containing test frame, and the T-LESS test frame containing
the smallest positive small GT (sample ID breaks ties). No score/gain selection.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from rgbd_seg.data import RGBDDataset, dataset_fingerprint
from tools.build_multiscale_gallery import (best_same_class, crop_bounds, decode_record,
                                            overlay, read_json, sha256, PALETTE)
from tools.evaluate_multiscale import ensure_empty

VARIANTS = ('baseline', 'bce2', 'bace', 'freqfusion')
SEEDS = (17, 23, 41)
RULE = ('All four GraspNet val frames; every GraspNet test frame with positive original GT area <=1024 '
        '(expected one frame); one T-LESS test frame with globally minimum positive small GT area, '
        'ties resolved by ascending sample ID. No predictions or metric gains are consulted.')


def json_write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def datasets(graspnet, tless):
    result = {('graspnet', split): RGBDDataset(graspnet, split) for split in ('val', 'test')}
    result['tless', 'test'] = RGBDDataset(tless, 'test')
    for (name, _), dataset in result.items():
        expected = 'public_graspnet_subset' if name == 'graspnet' else 'public_tless_subset'
        if dataset.manifest['provenance'] != expected or dataset.classes != ['object']:
            raise ValueError('This gallery requires the registered public object-class datasets')
    return result


def select_cases(data):
    selected, external_candidates = [], []
    for (name, split), dataset in data.items():
        for index, record in enumerate(dataset.samples):
            _, target = dataset[index]
            areas = target['masks'].flatten(1).sum(1).tolist()
            small = [i for i, area in enumerate(areas) if 0 < area <= 1024]
            case = {'dataset': name, 'split': split, 'sample_id': record['id'],
                    'image_id': record['image_id'], 'index': index, 'small_gt_indices': small,
                    'gt_areas_px': areas, 'scene': record.get('source_scene'),
                    'frame': record.get('source_frame')}
            if name == 'graspnet' and (split == 'val' or small):
                selected.append(case)
            elif name == 'tless' and small:
                external_candidates.append((min(areas[i] for i in small), record['id'], case))
    if sum(c['split'] == 'val' for c in selected) != 4 or len(selected) != 5:
        raise ValueError('Expected four GraspNet development cases and exactly one small-containing test case')
    if not external_candidates:
        raise ValueError('T-LESS has no small GT for the registered selection rule')
    selected.append(min(external_candidates, key=lambda item: item[:2])[2])
    return selected


def selection_receipt(data):
    roots = {name: dataset.root for (name, _), dataset in data.items()}
    return {'format': 'v5_gallery_selection_v1', 'selection_rule': RULE,
            'selection_inputs': 'Only manifest records and original GT masks; no model outputs or scores',
            'datasets': {name: {'path': str(root), 'manifest_sha256': sha256(root / 'manifest.json'),
                                'dataset_sha256': dataset_fingerprint(root)} for name, root in roots.items()},
            'cases': select_cases(data)}


def read_evaluation(root, name, seed, variant, split, dataset, fingerprint):
    folder = Path(root) / name / f'seed{seed}' / variant / split
    path = folder / 'evaluation.json'
    value = read_json(path)
    required = {'exit_status': 'completed', 'phase': 'formal', 'method': 'full', 'variant': variant,
                'training_seed': seed, 'training_steps': 800, 'split': split,
                'samples': len(dataset), 'classes': dataset.classes,
                'dataset_sha256': fingerprint, 'manifest_sha256': sha256(dataset.root / 'manifest.json'),
                'provenance': dataset.manifest['provenance'], 'evaluation_depth_scale': dataset.depth_scale,
                'mask_threshold': .5, 'score_threshold': .01}
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f'Unexpected evaluation identity {path}: {key}')
    if value['model'].get('mode') != 'E2' or value['model'].get('paper_variant') != variant:
        raise ValueError('Expected E2 with the declared v5 training variant')
    prediction_path = folder / 'predictions.jsonl'
    if sha256(prediction_path) != value['prediction_sha256']:
        raise ValueError(f'Prediction checksum mismatch: {prediction_path}')
    records = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    if [(r['sample_id'], r['image_id']) for r in records] != [(r['id'], r['image_id']) for r in dataset.samples]:
        raise ValueError('Predictions do not cover the complete split in original order')
    receipt = {'dataset': name, 'seed': seed, 'variant': variant, 'split': split,
               'evaluation_path': str(path.resolve()), 'evaluation_sha256': sha256(path),
               'predictions_path': str(prediction_path.resolve()), 'predictions_sha256': value['prediction_sha256'],
               'checkpoint_sha256': value['checkpoint_sha256'], 'protocol_sha256': value['protocol']['sha256'],
               'source_sha256': value['source_sha256']}
    return value, {r['sample_id']: r for r in records}, receipt


def describe(values):
    if any(value is None for value in values):
        return {'mean': None, 'sample_sd': None, 'values': values}
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values) if len(values) > 1 else None,
            'values': values}


def development_table(results):
    rows, gates = [], {}
    for variant in VARIANTS:
        values = [results['graspnet', seed, variant, 'val'] for seed in SEEDS]
        rows.append({'variant': variant,
                     **{metric: describe([v['metrics'][metric] for v in values])
                        for metric in ('mask_ap', 'mask_ap_small', 'mask_ar_small')},
                     'latency_ms': describe([v['timing']['mean_ms'] for v in values])})
        if variant != 'baseline':
            pairs = [{'seed': seed,
                      'ap_small_delta_pp': results['graspnet', seed, variant, 'val']['metrics']['mask_ap_small']
                         - results['graspnet', seed, 'baseline', 'val']['metrics']['mask_ap_small'],
                      'ap_delta_pp': results['graspnet', seed, variant, 'val']['metrics']['mask_ap']
                         - results['graspnet', seed, 'baseline', 'val']['metrics']['mask_ap']} for seed in SEEDS]
            small = statistics.mean(p['ap_small_delta_pp'] for p in pairs)
            overall = statistics.mean(p['ap_delta_pp'] for p in pairs)
            positives = sum(p['ap_small_delta_pp'] > 0 for p in pairs)
            gates[variant] = {'paired': pairs, 'mean_ap_small_delta_pp': small, 'mean_ap_delta_pp': overall,
                              'positive_seeds': positives, 'passed': small >= 2 and positives >= 2 and overall >= -1}
    return {'rows': rows, 'gates': gates, 'seeds': list(SEEDS),
            'bace_minus_bce2_ap_small_pp': describe([
                results['graspnet', seed, 'bace', 'val']['metrics']['mask_ap_small'] -
                results['graspnet', seed, 'bce2', 'val']['metrics']['mask_ap_small'] for seed in SEEDS])}


def encode_image(image, full=False):
    buffer = io.BytesIO()
    if full:
        image = image.copy()
        image.thumbnail((960, 960), Image.Resampling.LANCZOS)
        image.save(buffer, format='JPEG', quality=85, optimize=True)
        mime = 'jpeg'
    else:
        image.save(buffer, format='PNG', optimize=True)
        mime = 'png'
    return f'data:image/{mime};base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def build(args):
    data = datasets(args.graspnet_data, args.tless_data)
    actual_selection = selection_receipt(data)
    selection = read_json(args.selection)
    if selection != actual_selection:
        raise ValueError('Frozen GT selection or original data changed; do not silently choose new cases')
    # The completed 28-run summary is mandatory; no provisional/example metrics.
    summary = read_json(args.summary)
    summary_sha = sha256(args.summary)
    results, predictions, receipts = {}, {}, []
    for name, seeds, splits in (('graspnet', SEEDS, ('val', 'test')), ('tless', (17,), ('test',))):
        fingerprint = selection['datasets'][name]['dataset_sha256']
        for seed in seeds:
            for variant in VARIANTS:
                for split in splits:
                    key = name, seed, variant, split
                    value, rows, receipt = read_evaluation(args.evaluations, *key, data[name, split], fingerprint)
                    results[key] = value
                    if seed == 17:
                        predictions[key] = rows
                    receipts.append(receipt)
    if len({r['protocol_sha256'] for r in receipts}) != 1 or len({json.dumps(r['source_sha256'], sort_keys=True) for r in receipts}) != 1:
        raise ValueError('The 28 evaluations must share a frozen protocol and implementation')
    for seed in SEEDS:
        for variant in VARIANTS:
            if len({r['checkpoint_sha256'] for r in receipts if r['seed'] == seed and r['variant'] == variant}) != 1:
                raise ValueError('Dataset/split evaluations must use the same checkpoint for each seed/variant')
    development = development_table(results)
    validate_summary(summary, receipts, development)
    output = ensure_empty(args.output)
    cases, images = [], {}
    for ordinal, selected in enumerate(selection['cases']):
        name, split, sid = selected['dataset'], selected['split'], selected['sample_id']
        dataset = data[name, split]
        _, target = dataset[selected['index']]
        source = dataset.samples[selected['index']]
        with Image.open(source['rgb_path']) as source_image:
            image = source_image.convert('RGB')
        width, height = image.size
        gt_masks = target['masks'].numpy().astype(bool)
        labels = target['labels'].tolist()
        decoded = {variant: decode_record(predictions[name, 17, variant, split][sid], height, width, dataset.classes)
                   for variant in VARIANTS}
        prefix = f'case_{ordinal:02d}'
        def save(picture, tag, full=False):
            key = f'{prefix}_{tag}'
            images[key] = encode_image(picture, full)
            return key
        case = {**selected, 'key': prefix, 'width': width, 'height': height,
                'instances': len(gt_masks), 'small_instances': len(selected['small_gt_indices']),
                'full_images': {'rgb': save(image, 'rgb', True), 'gt': save(overlay(image, gt_masks), 'gt', True)},
                'candidate_counts': {}, 'small_cases': []}
        for variant in VARIANTS:
            case['full_images'][variant] = save(overlay(image, decoded[variant]['masks']), variant, True)
            case['candidate_counts'][variant] = len(decoded[variant]['scores'])
        for ordinal_small, gt_index in enumerate(selected['small_gt_indices']):
            gt = gt_masks[gt_index]
            bounds = crop_bounds(gt)
            left, top, right, bottom = bounds
            crop = image.crop(tuple(bounds))
            detail_prefix = f'small_{ordinal_small:02d}'
            detail = {'gt_index': gt_index, 'original_area': int(gt.sum()), 'crop_xyxy': bounds,
                      'images': {'rgb': save(crop, detail_prefix + '_rgb'),
                                 'gt': save(overlay(crop, [gt[top:bottom, left:right]], PALETTE[0]), detail_prefix + '_gt')},
                      'matches': {}, 'iou_delta_vs_baseline': {}}
            for variant in VARIANTS:
                match = best_same_class(gt, labels[gt_index], decoded[variant])
                local_masks = [] if match is None else [decoded[variant]['masks'][match.pop('local_index'), top:bottom, left:right]]
                detail['matches'][variant] = match
                detail['images'][variant] = save(overlay(crop, local_masks, PALETTE[1]), detail_prefix + '_' + variant)
            baseline = detail['matches']['baseline']
            baseline_iou = baseline['iou'] if baseline else 0.
            detail['iou_delta_vs_baseline'] = {variant: (detail['matches'][variant]['iou'] if detail['matches'][variant] else 0.) - baseline_iou
                                                for variant in VARIANTS[1:]}
            case['small_cases'].append(detail)
        cases.append(case)
    for receipt in receipts:
        if sha256(receipt['evaluation_path']) != receipt['evaluation_sha256'] or sha256(receipt['predictions_path']) != receipt['predictions_sha256']:
            raise RuntimeError('Frozen evaluation changed during gallery generation')
    if selection_receipt(data) != selection:
        raise RuntimeError('Original data changed during gallery generation')
    if sha256(args.summary) != summary_sha:
        raise RuntimeError('Trial summary changed during gallery generation')
    manifest = {'format': 'v5_training_gallery_v1', 'status': 'completed',
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'seed_for_images': 17,
                'selection': selection, 'selection_sha256': sha256(args.selection),
                'summary_path': str(Path(args.summary).resolve()), 'summary_sha256': summary_sha,
                'development': development, 'evaluation_receipts': receipts, 'cases': cases,
                'display_score_threshold': .01, 'mask_threshold': .5,
                'image_encoding': 'Full views: JPEG quality85, longest side<=960, presentation only. Small crops: original-pixel lossless PNG. All IoUs use full original-resolution masks.',
                'diagnostic': 'GT-assisted per-GT maximum same-class IoU among all score>=0.01 candidates; independent matching, predictions may repeat across GT. Not deployment recall or COCO AP.',
                'default_live_demo': 'Unchanged original model; this page does not switch online inference',
                'source_sha256': {name: sha256(ROOT / name) for name in ('tools/build_training_gallery.py', 'tools/build_multiscale_gallery.py', 'rgbd_seg/data.py')}}
    standalone = {**manifest, 'embedded_images': images}
    embedded = json.dumps(standalone, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    html = HTML.replace('__DATA__', embedded)
    size = len(html.encode('utf-8'))
    if size >= 30 * 1024 * 1024:
        raise ValueError(f'Self-contained gallery exceeds the fixed 30MiB budget: {size} bytes; no cases silently dropped')
    json_write(output / 'gallery.json', manifest)
    (output / 'index.html').write_text(html, encoding='utf-8')
    result = {'status': 'completed', 'cases': len(cases), 'small_cases': sum(len(c['small_cases']) for c in cases),
              'html_bytes': size, 'html_sha256': sha256(output / 'index.html'), 'output': str(output.resolve())}
    json_write(output / 'build_receipt.json', result)
    print(json.dumps(result, ensure_ascii=False))
    return result


def validate_summary(summary, receipts, development):
    """Require the independently verified full matrix and identical gate decisions."""
    if summary.get('status') != 'complete_and_verified':
        raise ValueError('The complete v5 trial summary is required')
    if summary.get('counts') != {'completed_training': 12, 'expected_training': 12,
                                'completed_evaluations': 28, 'expected_evaluations': 28}:
        raise ValueError('Expected the full twelve-training/twenty-eight-evaluation matrix')
    rows = summary.get('evaluations', [])
    expected = {(r['dataset'], r['seed'], r['variant'], r['split']): r for r in receipts}
    if len(rows) != len(expected):
        raise ValueError('Trial summary must cover all 28 evaluations')
    seen = set()
    for row in rows:
        key = row['dataset'], row['seed'], row['variant'], row['split']
        if key not in expected or key in seen or row['evaluation_sha256'] != expected[key]['evaluation_sha256']:
            raise ValueError('Trial summary and frozen evaluation files disagree')
        seen.add(key)
    if summary['protocol']['sha256'] != receipts[0]['protocol_sha256']:
        raise ValueError('Summary and evaluation protocol hashes differ')
    gate = summary['development_gate']
    if gate.get('status') != 'complete':
        raise ValueError('Cannot display a provisional development gate')
    for variant, decision in development['gates'].items():
        if gate['choices'][variant]['passed'] != decision['passed']:
            raise ValueError('Independent gallery/summary gate decisions differ')
        expected_mean = next(row['mask_ap_small']['mean'] for row in development['rows'] if row['variant'] == variant)
        if not np.isclose(gate['choices'][variant]['mean_val_ap_small'], expected_mean, rtol=0, atol=1e-10):
            raise ValueError('Summary and evaluation development means differ')


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark"><title>四种训练方法 · 固定案例对照</title>
<style>
:root{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#101621;color:#e8eef7;--zoom:1;--line:#354157;--muted:#b1bdcf;--accent:#95e0c7}*{box-sizing:border-box}body{margin:0}main{max-width:1850px;margin:auto;padding:32px}h1{font-size:30px;line-height:1.25;margin:10px 0}h2{font-size:19px;margin:25px 0 12px}h3{font-size:15px}.eyebrow{font-size:11px;letter-spacing:2px;color:var(--accent)}p{font-size:13px;line-height:1.75;color:var(--muted);max-width:1150px}p,footer,td,th,.caption,.detail-title{overflow-wrap:anywhere}.notice{background:#242a22;border:1px solid #637155;padding:15px 18px;border-radius:9px;color:#e3e8cf;max-width:none}.scope{border-left:3px solid #d8ac72;padding-left:14px}.methods{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.method{background:#182232;padding:14px;border:1px solid var(--line);border-radius:8px}.method p{margin-bottom:0;font-size:12px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px}table{width:100%;border-collapse:collapse;min-width:720px;font-size:12px}th,td{text-align:left;padding:12px;border-bottom:1px solid var(--line)}th{color:var(--muted);background:#182232}.controls{display:flex;gap:22px;align-items:center;flex-wrap:wrap;padding:16px 0}label{font-size:12px}button,select{font:inherit;color:inherit;background:#1b293d;border:1px solid #4b5c75;padding:9px;border-radius:6px}select{max-width:100%}button{cursor:pointer}input{accent-color:var(--accent)}button:focus-visible,select:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid var(--accent);outline-offset:3px}.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px}.card{background:#172233;border:1px solid var(--line);border-radius:8px;overflow:hidden;min-width:0}.card-title{padding:10px;font-size:12px;font-weight:700}.image-window{overflow:auto;max-height:65vh;background:#0b111b}.image-window img{display:block;width:calc(100% * var(--zoom));max-width:none;height:auto}.caption{padding:10px;font-size:11px;line-height:1.6;color:var(--muted);min-height:54px}.detail{margin:22px 0}.detail-title{font-size:13px;font-weight:650;margin-bottom:10px}.bad{color:#ffafa0}.good{color:var(--accent)}a{color:#b9d4ff}footer{margin-top:30px;padding-top:18px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}details{margin-top:18px}summary{cursor:pointer;font-size:13px}#integrity{max-width:none;font-size:11px}.pill{display:inline-block;border:1px solid var(--line);border-radius:20px;padding:4px 9px;font-size:11px;color:var(--accent)}@media(max-width:1250px){.grid{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:760px){main{padding:18px}.methods{grid-template-columns:repeat(2,minmax(0,1fr))}h1{font-size:25px}.controls{display:block}.controls label,.controls button{display:block;margin:12px 0}.controls label select{display:block;margin-top:7px;width:100%}}@media(max-width:480px){.grid,.methods{grid-template-columns:1fr}.image-window{max-height:60vh}}
</style></head><body><main>
<div class="eyebrow">TRAINING METHODS · FIXED CASES · SEED 17</div><h1>同一训练预算，四种方法</h1><span class="pill">静态实验记录 · 不运行模型</span>
<p>原模型、掩码损失加权、BACE 辅助损失与 FreqFusion 特征融合。图片固定展示 seed17；结论依据三个随机种子的完整开发结果，不能只看一张图。</p><p class="notice" id="verdict"></p>
<div class="methods"><section class="method"><strong>baseline · 原训练方法</strong><p>E2 RGB-D 模型，原始检测和 mask BCE 损失；本轮从同一 COCO 权重重新训练。</p></section><section class="method"><strong>BCE2 · 损失权重对照</strong><p>mask BCE 额外加一次，检验单纯提高掩码损失权重的效果。</p></section><section class="method"><strong>BACE · 辅助掩码损失</strong><p>原 BCE 加固定 scale=4 的辅助项，仅训练时使用 GT。推理仍使用原 mask head。</p></section><section class="method"><strong>FreqFusion · 特征融合</strong><p>在 RGB FPN 加入低通、高通及偏移采样模块；沿用预训练主干和原深度路径。</p></section></div>
<h2>三随机种子的开发结果</h2><p>seed17 / 23 / 41；每组同为 800 步。表中为均值 ± 样本标准差；AP、APₛ、ARₛ 单位为百分比，差值为百分点。种子重复不等于三个独立数据集。耗时为同卡 batch1、float32、4 线程、3 次预热后的完整评测均值，包含传输与掩码回传，不含读图和绘图。</p><div class="table-wrap"><table><thead><tr><th>方法</th><th>整体 AP</th><th>小目标 APₛ</th><th>小目标 ARₛ</th><th>对基线 ΔAPₛ</th><th>提高的种子</th><th>开发门槛</th><th>评测耗时 / ms</th></tr></thead><tbody id="metrics"></tbody></table></div><p id="bace-control"></p>
<p class="scope">本轮均为研究性对照；已有在线 Demo 继续使用原模型，不自动切换候选。任务为 object 单类实例分割，不是 88 种物体身份分类。GraspNet 开发集只有 4 个 small GT，测试集只有 1 个；T-LESS 是单独的跨数据集压力检查。测试结果不参与挑选方法，不把这三个数据范围合并成一个分数。</p>
<h2>固定案例与小目标诊断</h2><p>完整保留 GraspNet 全部 4 张开发图、唯一含 small 的测试图，再按最小非零 small GT 面积选 1 张 T-LESS 图（同面积按样本 ID 排序）。先按 GT 冻结选择，再读取预测；变差案例也保留。T-LESS 重叠真值以独立掩码保留。</p>
<div class="controls"><label>数据范围 <select id="scope"><option value="all">全部 6 个固定案例</option><option value="graspnet/val">GraspNet · 开发集</option><option value="graspnet/test">GraspNet · 测试集</option><option value="tless/test">T-LESS · 压力检查</option></select></label><label>选择案例 <select id="case" aria-label="选择固定案例"></select></label><label>图像放大 <input id="zoom" aria-label="图像放大" type="range" min="1" max="4" step=".25" value="1"> <output id="zoom-value">1.00×</output></label><button id="reset" type="button">复位视图</button></div><p id="case-info" aria-live="polite"></p>
<h3>完整视野 · 所有候选</h3><div id="full" class="grid"></div><p>RGB、GT、baseline、BCE2、BACE、FreqFusion。候选阈值固定为 score ≥ 0.01，与在线 Demo 默认 0.5 不同；掩码阈值为 0.5。概览图仅为节省页面体积而缩小，评测和下方 IoU 始终按原图像素计算。颜色区分实例，不表示身份类别。</p>
<h3>全部 small GT · 原像素局部图</h3><p>small 为原始可见掩码面积 0 &lt; area ≤ 1024。每个 GT 独立寻找同类最大 IoU 预测，局部窗口为 GT 框外扩 32 原图像素。这是 GT 辅助的事后诊断，不是部署检测；同一预测可被多个 GT 选中，不能把局部图的匹配率当成 COCO AP 或检出率。</p><div id="details"></div>
<footer><a id="download" href="#">下载案例、诊断与完整来源记录</a><details><summary>查看来源与校验值</summary><p id="integrity"></p></details><p>所有图片已嵌入此 HTML，可离线查看。此页不调用在线模型，也不会改变 Demo 的推理方式。</p></footer>
<script id="gallery-data" type="application/json">__DATA__</script><script>
'use strict';const $=id=>document.getElementById(id),data=JSON.parse($('gallery-data').textContent);const names={baseline:'baseline',bce2:'BCE2',bace:'BACE',freqfusion:'FreqFusion'},columns=[['rgb','RGB'],['gt','实例真值'],...Object.entries(names)];
function number(v){return v===null?'—':v.toFixed(2)}function describe(v){return number(v.mean)+(v.sample_sd===null?'':' ± '+number(v.sample_sd))}function sign(v){return (v>=0?'+':'')+number(v)}
const passed=Object.entries(data.development.gates).filter(([k,v])=>v.passed).map(([k])=>names[k]);$('verdict').textContent=(passed.length?'达到预定开发门槛的候选：'+passed.join('、')+'。':'三种改法均未达到预定开发门槛。')+' 门槛：小目标 AP 平均增加至少 2 点、至少 2/3 种子提高、整体 AP 平均下降不超过 1 点。这是小样本工程判断，不是统计显著性结论；在线默认模型保持不变。';
for(const row of data.development.rows){const gate=data.development.gates[row.variant],tr=document.createElement('tr');for(const text of [names[row.variant],describe(row.mask_ap),describe(row.mask_ap_small),describe(row.mask_ar_small),gate?sign(gate.mean_ap_small_delta_pp):'基准',gate?gate.positive_seeds+'/3':'—',gate?(gate.passed?'达到':'未达到'):'基准',describe(row.latency_ms)]){const td=document.createElement('td');td.textContent=text;tr.append(td)}$('metrics').append(tr)}$('bace-control').textContent='BACE 相对 BCE2 的配对小目标 AP 差值：'+describe(data.development.bace_minus_bce2_ap_small_pp)+' 点。若 BACE 没有胜过 BCE2，不能把相对 baseline 的收益特指为边界算式有效。';
function card(key,imageKey,caption,tone=''){const article=document.createElement('article');article.className='card';const title=document.createElement('div');title.className='card-title';title.textContent=columns.find(c=>c[0]===key)[1];const frame=document.createElement('div');frame.className='image-window';const img=document.createElement('img'),uri=data.embedded_images[imageKey];if(typeof uri!=='string'||!/^data:image\/(png|jpeg);base64,[A-Za-z0-9+/=]+$/.test(uri))throw Error('无效内嵌图片');img.src=uri;img.alt=title.textContent+'：'+caption;frame.append(img);const text=document.createElement('div');text.className='caption '+tone;text.textContent=caption;article.append(title,frame,text);return article}
function label(c){return (c.dataset==='graspnet'?'GraspNet':'T-LESS')+' · '+(c.split==='val'?'开发':'测试')+' · 场景 '+c.scene+' / 帧 '+c.frame}
function render(){const item=data.cases.find(c=>c.key===$('case').value);$('full').replaceChildren();$('details').replaceChildren();if(!item){$('case-info').textContent='当前范围没有固定案例。';return}$('case-info').textContent=label(item)+' · '+item.sample_id+' · 原图 '+item.width+'×'+item.height+' · '+item.instances+' 个实例 / '+item.small_instances+' 个 small';for(const [key] of columns)$('full').append(card(key,item.full_images[key],key==='rgb'?'原始颜色图':key==='gt'?'全部原始实例真值':item.candidate_counts[key]+' 个候选 · score ≥ 0.01'));if(!item.small_cases.length){const p=document.createElement('p');p.textContent='该帧没有 small GT；作为固定开发案例仍完整展示。';$('details').append(p)}for(const [index,detail] of item.small_cases.entries()){const section=document.createElement('section');section.className='detail';const title=document.createElement('div');title.className='detail-title';title.textContent='small '+(index+1)+' · GT索引 '+detail.gt_index+' · 原始面积 '+detail.original_area+' px · 窗口 ['+detail.crop_xyxy.join(', ')+']';const grid=document.createElement('div');grid.className='grid';for(const [key] of columns){let text=key==='rgb'?'原像素局部，未再次推理':key==='gt'?'绿色：本实例真值':'',tone='';if(key in detail.matches){const match=detail.matches[key];text=match?'GT辅助 IoU '+match.iou.toFixed(3)+' · score '+match.score.toFixed(3)+(match.iou===0?' · 无重叠':''):'没有同类候选 · IoU 0';if(key!=='baseline'){const delta=detail.iou_delta_vs_baseline[key];text+=' · ΔIoU '+(delta>=0?'+':'')+delta.toFixed(3);tone=delta< -1e-9?'bad':delta>1e-9?'good':''}}grid.append(card(key,detail.images[key],text,tone))}section.append(title,grid);$('details').append(section)}}
function filter(){const scope=$('scope').value;$('case').replaceChildren();for(const item of data.cases.filter(c=>scope==='all'||c.dataset+'/'+c.split===scope)){const option=document.createElement('option');option.value=item.key;option.textContent=label(item)+' · '+item.small_instances+' 个small';$('case').append(option)}render()}function zoom(){document.documentElement.style.setProperty('--zoom',$('zoom').value);$('zoom-value').textContent=Number($('zoom').value).toFixed(2)+'×'}$('scope').addEventListener('change',filter);$('case').addEventListener('change',render);$('zoom').addEventListener('input',zoom);$('reset').addEventListener('click',()=>{$('zoom').value=1;zoom();document.querySelectorAll('.image-window').forEach(el=>{el.scrollTop=0;el.scrollLeft=0})});$('download').addEventListener('click',event=>{event.preventDefault();const {embedded_images,...metadata}=data;const url=URL.createObjectURL(new Blob([JSON.stringify(metadata,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='training-methods-gallery.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});$('integrity').textContent='固定选样 SHA256 '+data.selection_sha256+'；完整试验汇总 SHA256 '+data.summary_sha256+'；协议 SHA256 '+data.evaluation_receipts[0].protocol_sha256+'。每个评测、预测文件、checkpoint及源码的完整校验值在下载记录中。';filter();
</script></main></body></html>'''


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graspnet-data', type=Path, default=ROOT / 'data/graspnet_expanded_v2')
    parser.add_argument('--tless-data', type=Path, default=ROOT / 'data/tless_confirmation')
    parser.add_argument('--evaluations', type=Path, default=ROOT / 'reports/v5/evaluation')
    parser.add_argument('--summary', type=Path, default=ROOT / 'reports/v5/trial_summary.json')
    parser.add_argument('--selection', type=Path, default=ROOT / 'reports/v5/gallery_selection.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/v5/gallery')
    parser.add_argument('--freeze-selection', action='store_true', help='Only read original GT and write the new selection receipt; no evaluation files are read')
    args = parser.parse_args()
    if args.freeze_selection:
        selection = selection_receipt(datasets(args.graspnet_data, args.tless_data))
        args.selection.parent.mkdir(parents=True, exist_ok=True)
        json_write(args.selection, selection)
        print(json.dumps({'selection': str(args.selection), 'sha256': sha256(args.selection), 'cases': selection['cases']}, ensure_ascii=False))
    else:
        build(args)


if __name__ == '__main__':
    main()
