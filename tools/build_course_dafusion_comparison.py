#!/usr/bin/env python3
"""Audit a completed DA-Fusion run and render fixed saved predictions; never infer.

Writes comparison_summary.json, comparison_report.md and a self-contained gallery.
Incomplete/smoke runs are rejected before creating any output. Existing artifacts
are never overwritten; --check-existing repeats only browser verification.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_submission import _Spec, _predictions
from tools.build_course_pretrained_gallery import decode
from tools.summarize_course_finetune import require, read, sha256, check_hash, metrics, selected_epoch

INDICES = (0, 400, 800, 1200, 1600, 2000)
METRICS = ('mask_ap_small', 'mask_ap', 'mask_ap50', 'mask_ap75', 'mask_ar_small')
LABELS = {'mask_ap_small': 'APsmall', 'mask_ap': 'AP', 'mask_ap50': 'AP50',
          'mask_ap75': 'AP75', 'mask_ar_small': 'ARsmall'}
NAMES = {'maskrcnn': 'Mask R-CNN · RGB', 'mask2former': 'Mask2Former · RGB',
         'dafusion': 'DA-Fusion 课程适配 · RGB-D'}
COLORS = ((65, 218, 178), (255, 183, 90), (123, 158, 255),
          (237, 130, 194), (168, 143, 247), (83, 210, 231))


def sources(mapping):
    require(isinstance(mapping, dict) and mapping, 'Missing source hashes')
    for name, digest in mapping.items():
        relative = Path(name)
        path = ROOT / relative if len(relative.parts) > 1 else ROOT / 'rgbd_seg' / relative
        require(path.resolve().is_relative_to(ROOT), 'Source path outside project')
        check_hash(path, digest)


def full_evaluation(folder, split, count, *, depth):
    evaluation, receipt = read(folder / 'evaluation.json'), read(folder / 'evaluation_run.json')
    for document in (evaluation, receipt):
        require(document.get('status') == 'completed' and document.get('split') == split,
                f'Incomplete/wrong evaluation: {folder}')
        require(document.get('scope') == 'full_split' and document.get('images') == count
                and document.get('full_split_images') == count, 'Full official split required')
        require(document.get('depth_used') is depth, 'Evaluation modality receipt differs')
    require(evaluation['iou_type'] == 'segm' and evaluation['max_detections'] == [1, 10, 100]
            and evaluation['ground_truth_area'] == 'original annotation area preserved', 'COCO protocol differs')
    require(evaluation['source_sha256'] == receipt['source_sha256'], 'Evaluation source receipts disagree')
    sources(evaluation['source_sha256'])
    metrics(evaluation)
    require(all(evaluation['metrics'].get(k) is not None for k in METRICS), 'Required metrics unavailable')
    check_hash(folder / 'evaluation.json', receipt['evaluation_sha256'])
    require(evaluation['predictions_sha256'] == receipt['predictions_sha256'], 'Raw hashes disagree')
    return evaluation, receipt


def selected_predictions(folder, dataset, expected_digest):
    raw = folder / 'predictions.jsonl'
    check_hash(raw, expected_digest)
    saved, rows, candidates = {}, 0, 0
    spec = _Spec(dataset.image_ids, (1,), (1024, 1024), 100)
    with raw.open() as stream:
        for index, line in enumerate(stream):
            require(index < len(dataset), 'Extra prediction rows')
            row = json.loads(line)
            expected = dataset.images[index]
            require(type(row['image_id']) is int and row['image_id'] == expected['id']
                    and row['file_name'] == expected['file_name'], 'Prediction ID/file/order mismatch')
            require(isinstance(row['predictions'], list) and len(row['predictions']) <= 100,
                    'Invalid prediction count')
            if index in INDICES:
                # Python run-length bounds before native decode, for displayed masks only.
                row['predictions'], _ = _predictions(row['predictions'], spec, decode=False)
                require(all(p['score'] >= .01 for p in row['predictions']), 'Prediction threshold differs')
                saved[index] = row
            rows += 1
            candidates += len(row['predictions'])
    require(rows == len(dataset) and set(saved) == set(INDICES), 'Missing official prediction rows/cases')
    return saved, {'path': str(raw), 'sha256': expected_digest, 'rows': rows,
                   'predictions': candidates, 'mask_decode_repeated': 'Only six displayed cases'}


def submission_receipt(folder, receipt, count, predicted):
    path = folder / 'submission_validation.json'
    check_hash(path, receipt['submission_validation_sha256'])
    validation = read(path)
    require(validation['rows'] == count and validation['predictions'] == predicted
            and validation['image_size'] == [1024, 1024] and validation['category_ids'] == [1]
            and validation['max_detections'] == 100, 'Submission scope differs')
    require(all(validation.get(k) is True for k in
                ('id_order_verified', 'rle_runs_verified', 'rle_native_decode_verified')),
            'Submission validation incomplete')
    require(validation['sha256'] == receipt['submission_sha256'], 'Submission hashes disagree')
    check_hash(folder / 'submission.csv', validation['sha256'])
    sources(validation['source_sha256'])
    return {'csv_sha256': validation['sha256'], 'validation_sha256': sha256(path),
            'native_decode_repeated': False}


def audit_dafusion(folder, dataset, baseline):
    run = read(folder / 'run.json')
    require(run.get('status') == 'completed', 'DA-Fusion formal run has not completed; no comparison generated')
    require(run.get('model') == 'dafusion' and run.get('scope') == 'full_train_val_test'
            and run.get('depth_used') is True and run.get('test_used_for_checkpoint_selection') is False,
            'Only completed formal RGB-D train/val/test runs are accepted')
    args = run['arguments']
    require(args['epochs'] == 5 and args['smoke_steps'] == 0
            and run['training_images_seen'] == 5 * 14432, 'Full five-epoch training required')
    require(run['training_data']['images'] == 14432, 'Training image count differs')
    require(Path(args['archive']).resolve() == dataset.zip_path, 'Training/gallery archive paths differ')
    protocol_path = Path(args['protocol'])
    check_hash(protocol_path, run['protocol_sha256'])
    protocol = read(protocol_path)
    require(protocol['status'] == 'frozen' and protocol.get('frozen_at_utc'), 'Formal frozen protocol required')
    require(protocol['source_sha256'] == run['source_sha256'], 'Source differs from frozen protocol')
    sources(run['source_sha256'])
    for key in ('epochs', 'batch_size', 'accumulation', 'seed', 'lr', 'precision', 'workers', 'threads'):
        require(args[key] == protocol['training'][key], f'Training setting differs: {key}')
    require(run['verified_input_sha256']['archive_sha256']
            == protocol['archive_sha256_from_completed_download_receipt'], 'Recorded archive SHA differs')
    initialization = read(folder / 'initialization.json')
    require(initialization == run['model_metadata'], 'Initialization receipt differs')
    check_hash(Path(args['checkpoint']) / 'model.safetensors', initialization['pretrained_sha256'])
    require(initialization['pretrained_sha256'] == baseline['models']['mask2former']['initial_checkpoint_sha256'],
            'DA-Fusion and Mask2Former COCO initializations differ')
    selection = read(folder / 'selection.json')
    best = selected_epoch(selection['epochs'])
    require(selection['validation_scope'] == 'full_2584_images'
            and selection['test_used_for_selection'] is False
            and best == selection['best_epoch'] == run['best_epoch'], 'Best-epoch selection differs')
    effective = args['batch_size'] * args['accumulation']
    steps_per_epoch = math.ceil(14432 / effective)
    require(run['optimizer_steps'] == steps_per_epoch * 5, 'Optimizer coverage differs')
    validation = []
    for entry in selection['epochs']:
        epoch = entry['epoch']
        require(entry['training_images'] == 14432 and entry['global_step'] == epoch * steps_per_epoch,
                'Selection entry training scope differs')
        val_folder = Path(entry['output']).resolve()
        require(val_folder.parent == folder, 'Validation output outside formal run')
        evaluation, receipt = full_evaluation(val_folder, 'val', 2584, depth=True)
        require(evaluation['metrics'] == entry['metrics'], 'Selected val metrics differ')
        require(Path(receipt['archive']).resolve() == dataset.zip_path, 'Validation archive differs')
        validation.append({'epoch': epoch, 'metrics': evaluation['metrics'], 'path': str(val_folder),
                           'evaluation_sha256': sha256(val_folder / 'evaluation.json')})
        order_path = folder / f'train_order_epoch_{epoch:02d}.json'
        order = read(order_path)
        require(order['epoch'] == epoch and order['count'] == len(order['image_ids']) == 14432
                and len(set(order['image_ids'])) == 14432, 'Training order coverage differs')
        digest = hashlib.sha256(json.dumps(order['image_ids']).encode()).hexdigest()
        require(digest == order['ordered_image_ids_sha256']
                == baseline['models']['mask2former']['train_orders'][epoch - 1]['ordered_image_ids_sha256'],
                'DA-Fusion image order differs from the frozen RGB baseline')
    selected = selection['epochs'][best - 1]
    require(selection['best_key'] == [selected['metrics']['mask_ap_small'], selected['metrics']['mask_ap']],
            'Best selection key differs')
    checkpoint_sha = check_hash(folder / 'best.pt', run['best_checkpoint_sha256'])
    import torch
    checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True, mmap=True)
    require(checkpoint['epoch'] == best and checkpoint['selection'] == selected
            and checkpoint['model_name'] == 'dafusion' and checkpoint['source_sha256'] == run['source_sha256']
            and checkpoint['protocol_sha256'] == run['protocol_sha256']
            and checkpoint['model_metadata'] == initialization, 'Best checkpoint provenance differs')
    del checkpoint
    test_folder = Path(run['test_output']).resolve()
    require(test_folder.parent == folder, 'Test output outside formal run')
    check_hash(test_folder / 'evaluation_run.json', run['test_evaluation_run_sha256'])
    evaluation, receipt = full_evaluation(test_folder, 'test', 2112, depth=True)
    require(receipt['evaluated_checkpoint_sha256'] == checkpoint_sha and receipt['selected_epoch'] == best,
            'Test checkpoint differs from selected best')
    require(evaluation['metrics'] == run['test_metrics'], 'Run/test metrics differ')
    require(Path(receipt['archive']).resolve() == dataset.zip_path, 'Test archive differs')
    saved, raw = selected_predictions(test_folder, dataset, receipt['predictions_sha256'])
    require(raw['predictions'] == evaluation['predicted_instances'], 'Predicted counts differ')
    csv = submission_receipt(test_folder, receipt, 2112, raw['predictions'])
    return {'name': NAMES['dafusion'], 'run': str(folder), 'metrics': evaluation['metrics'],
            'best_epoch': best, 'best_checkpoint_sha256': checkpoint_sha, 'raw': raw,
            'test_path': str(test_folder), 'validation': validation, 'submission': csv,
            'gt_instances': evaluation['gt_instances'],
            'gt_area_groups_including_crowd': evaluation['gt_area_groups_including_crowd'],
            'protocol': {'path': str(protocol_path), 'sha256': run['protocol_sha256']},
            'run_sha256': sha256(folder / 'run.json'), 'selection_sha256': sha256(folder / 'selection.json'),
            'initialization_sha256': sha256(folder / 'initialization.json'),
            'evaluation_sha256': sha256(test_folder / 'evaluation.json'),
            'evaluation_run_sha256': sha256(test_folder / 'evaluation_run.json'),
            'model_metadata': initialization, 'training_arguments': args,
            'timing_descriptive_only': receipt.get('timing')}, saved


def png_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def mask_layer(mask, ordinal, size):
    color = COLORS[ordinal % len(COLORS)]
    alpha = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)
    bounds = alpha.getbbox()
    # Sparse cropped sprites avoid keeping hundreds of decoded 768² RGBA images
    # in the browser. A tiny mask lost only in the display downsample stays an
    # explicitly transparent sprite; it still counts in the saved candidates.
    if bounds is None:
        return {'mask': png_uri(Image.new('RGBA', (1, 1))), 'xywh': [0, 0, 1, 1]}
    left, top, right, bottom = bounds
    layer = Image.new('RGBA', (right-left, bottom-top), (*color, 0))
    layer.putalpha(alpha.crop(bounds))
    return {'mask': png_uri(layer), 'xywh': [left, top, right-left, bottom-top]}


def gallery_cases(dataset, annotations, saved, fixed_cases):
    cases = []
    for index, previous in zip(INDICES, fixed_cases):
        record = dataset.images[index]
        require(previous['index'] == index and previous['image_id'] == record['id']
                and previous['file_name'] == record['file_name'], 'Fixed cases differ from original gallery')
        image = dataset.load_rgb(index).convert('RGB')
        original_size = image.size
        # Presentation-only downsampling; metrics and source RLEs remain 1024².
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        picture = io.BytesIO()
        image.save(picture, format='WEBP', quality=88, method=6)
        case = {'key': f'case_{index:04d}', 'index': index, 'image_id': record['id'],
                'file_name': record['file_name'], 'original_size': list(original_size),
                'display_size': list(image.size), 'rgb': 'data:image/webp;base64,'
                    + base64.b64encode(picture.getvalue()).decode('ascii'), 'gt': [], 'predictions': {}}
        for ordinal, annotation in enumerate(annotations[record['id']]):
            if annotation['area'] > 0:
                case['gt'].append({'annotation_id': annotation['id'],
                    **mask_layer(decode(annotation['segmentation'], 1024, 1024), ordinal, image.size)})
        for name in ('mask2former', 'dafusion'):
            case['predictions'][name] = [{'score': item['score'], **mask_layer(
                decode(item['segmentation'], 1024, 1024), ordinal, image.size)}
                for ordinal, item in enumerate(saved[name][index]['predictions'])]
        cases.append(case)
    return cases


HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DA-Fusion · 课程实例分割对比</title><style>
*{box-sizing:border-box}body{margin:0;background:#111927;color:#e6edf7;font:15px/1.65 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1600px;margin:auto;padding:24px}h1{font-size:clamp(24px,4vw,38px);margin:10px 0}h2{font-size:20px}.muted{color:#a9bbce}.tag{color:#71dbc1;font-size:12px;letter-spacing:.12em}.notice{padding:14px 18px;border-left:3px solid #71dbc1;background:#1a2738;border-radius:8px}label{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.controls{display:flex;flex-wrap:wrap;gap:18px;align-items:center;padding:18px 0}select,button{background:#24354c;color:inherit;border:1px solid #506780;border-radius:7px;padding:8px;max-width:100%;font:inherit}input{accent-color:#71dbc1;max-width:100%}input[type=range]{width:190px}button{cursor:pointer}:focus-visible{outline:3px solid #71dbc1;outline-offset:3px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0;min-width:0;background:#1b293c;border:1px solid #34475d;border-radius:12px;overflow:hidden}figcaption{padding:12px;font-weight:600}canvas{display:block;width:100%;height:auto;background:#0b1019}.count{display:block;font-size:12px;color:#aec0d5;font-weight:400}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{text-align:right;padding:11px 14px;border-bottom:1px solid #33465c;white-space:nowrap}td:first-child,th:first-child{text-align:left}.case-info,footer{overflow-wrap:anywhere}footer{margin-top:22px;font-size:12px;color:#a9bbce}.busy{min-height:26px;color:#71dbc1}@media(max-width:900px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:520px){main{padding:14px}.grid{grid-template-columns:1fr}.controls{gap:12px}label{width:100%}select{width:100%}td,th{padding:8px}.notice{padding:12px}}
</style><main><div class="tag">SAVED PREDICTIONS · FIXED COURSE TEST CASES</div><h1>深度融合，实际改变了什么？</h1>
<p class="muted">Mask2Former 与 DA-Fusion 课程适配的已保存预测 · 六个预先固定案例</p>
<p class="notice">全量测试指标来自 2112 张图。下方示例固定为第 0、400、800、1200、1600、2000 张，不能代替总体统计。DA-Fusion 使用 RGB-D；两条 RGB 基线仅用 RGB。新模型增加骨干和参数，差异不能仅归因于深度。</p>
<div class="table-wrap"><table><thead><tr><th>模型</th><th>APsmall</th><th>AP</th><th>AP50</th><th>AP75</th><th>ARsmall</th></tr></thead><tbody id="metrics"></tbody></table></div>
<p class="muted">指标范围 0–1；全量评估保留 score ≥ 0.01。滑杆仅控制显示，不重新评分、不改变模型输出。</p>
<div class="controls"><label>固定案例 <select id="case" aria-label="选择固定案例"></select></label><label>展示分数 ≥ <output id="score-value">0.50</output><input id="score" type="range" min="0.01" max="1" step="0.01" value="0.5" aria-label="展示分数阈值"></label><button id="reset" type="button">恢复 0.50</button></div>
<p id="case-info" class="case-info muted"></p><p id="status" class="busy" role="status" aria-live="polite">正在载入真实图像…</p>
<section class="grid" aria-label="四列实例分割对比"><figure><figcaption>RGB<span class="count">同一幅输入</span></figcaption><canvas id="rgb" aria-label="RGB原图"></canvas></figure><figure><figcaption>GT<span id="gt-count" class="count"></span></figcaption><canvas id="gt" aria-label="GT独立实例掩码"></canvas></figure><figure><figcaption>Mask2Former · RGB<span id="mask2former-count" class="count"></span></figcaption><canvas id="mask2former" aria-label="Mask2Former掩码"></canvas></figure><figure><figcaption>DA-Fusion · RGB-D<span id="dafusion-count" class="count"></span></figcaption><canvas id="dafusion" aria-label="DA-Fusion掩码"></canvas></figure></section>
<p class="muted">独立掩码半透明叠加，保留重叠；颜色只区分单幅图内的预测实例，不表示跨模型一一匹配。页面缩略图用于浏览，评估始终使用原始 1024×1024 掩码。</p><h2>比较边界</h2><ul id="limitations"></ul><footer id="provenance"></footer></main>
<script id="data" type="application/json">__DATA__</script><script>
'use strict';const data=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id),cache=new Map();let generation=0;window.__galleryReady=false;
for(const row of data.metric_rows){const tr=document.createElement('tr');for(const value of [row.name,...data.metric_keys.map(k=>row.metrics[k].toFixed(4))]){const td=document.createElement('td');td.textContent=value;tr.append(td)}el('metrics').append(tr)}
for(const c of data.cases){const o=document.createElement('option');o.value=c.key;o.textContent=`第 ${c.index} 张 · ${c.image_id}`;el('case').append(o)}for(const text of data.limitations){const li=document.createElement('li');li.textContent=text;el('limitations').append(li)}
el('provenance').textContent=`结果审计 SHA256：${data.summary_sha256}。仅展示已保存结果；页面无需联网。`;
function load(uri){if(!cache.has(uri))cache.set(uri,new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error('图像载入失败'));im.src=uri}));return cache.get(uri)}
async function render(){const token=++generation;window.__galleryReady=false;const c=data.cases.find(x=>x.key===el('case').value),threshold=Number(el('score').value);el('score-value').value=threshold.toFixed(2);el('case-info').textContent=`${c.file_name} · 原图 ${c.original_size.join('×')}`;el('status').textContent='正在绘制…';try{const base=await load(c.rgb),groups={rgb:[],gt:c.gt,mask2former:c.predictions.mask2former.filter(x=>x.score>=threshold),dafusion:c.predictions.dafusion.filter(x=>x.score>=threshold)};const decoded={};for(const name of Object.keys(groups))decoded[name]=await Promise.all(groups[name].map(x=>load(x.mask)));if(token!==generation)return;for(const name of Object.keys(groups)){const canvas=el(name);canvas.width=c.display_size[0];canvas.height=c.display_size[1];const ctx=canvas.getContext('2d');ctx.drawImage(base,0,0);ctx.globalAlpha=.38;for(let i=0;i<decoded[name].length;i++)ctx.drawImage(decoded[name][i],...groups[name][i].xywh);ctx.globalAlpha=1;if(name!=='rgb')el(name+'-count').textContent=name==='gt'?`${groups[name].length} 个真实实例`:`显示 ${groups[name].length} / 保存 ${c.predictions[name].length} 个候选`}el('status').textContent='图像已就绪 · 滑杆只改变显示';window.__galleryCase=c.key;window.__galleryReady=true}catch(error){el('status').textContent=error.message;console.error(error)}}
el('case').addEventListener('change',render);el('score').addEventListener('input',render);el('reset').addEventListener('click',()=>{el('score').value='.5';render()});render();
</script></html>'''


def safe_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def report(summary):
    rows = ['# DA-Fusion 课程适配：完成后的比较', '',
            '全部数值来自完整测试集；0–1 范围，未重新训练或评分。', '',
            '| 模型 | APsmall | AP | AP50 | AP75 | ARsmall |', '|---|---:|---:|---:|---:|---:|']
    for name, model in summary['models'].items():
        rows.append('| ' + NAMES[name] + ' | ' + ' | '.join(f'{model["metrics"][k]:.6f}' for k in METRICS) + ' |')
    rows += ['', '## DA-Fusion 相对 RGB 模型的差值', '', '单位为百分点（差值 ×100），不代表统计显著性。', '',
             '| 比较对象 | APsmall | AP | AP50 | AP75 | ARsmall |', '|---|---:|---:|---:|---:|---:|']
    for baseline, values in summary['delta_percentage_points'].items():
        rows.append('| ' + NAMES[baseline] + ' | ' + ' | '.join(f'{values[k]:+.3f}' for k in METRICS) + ' |')
    rows += ['', f'DA-Fusion 最佳轮次：{summary["models"]["dafusion"]["best_epoch"]}，仅按完整验证集 APsmall、AP、较早轮次选择。',
             '', '## 固定案例', '', '[打开离线对比页](gallery/index.html)。固定索引 0/400/800/1200/1600/2000，未按效果挑选；默认展示 score≥0.5，全量评估为≥0.01。',
             '', '## 限制', ''] + ['- ' + text for text in summary['limitations']]
    rows += ['', '## 审计范围', '', '核对正式状态、训练顺序、五轮完整验证选择、最佳 checkpoint、测试 ID/覆盖、来源 SHA 与 CSV 校验收据。只解码六例展示掩码；未重复全部 CSV native decode，也未重算 AP。',
             '完整 ZIP 的 SHA 采用正式训练校验记录；本工具重新核验当前测试标注成员和六幅 RGB，并不重算约20GB整包 SHA。', '']
    return '\n'.join(rows)


def build(args):
    folder = args.run.resolve()
    # Fast refusal, even before archive/baseline loading or output creation.
    run = read(folder / 'run.json')
    require(run.get('status') == 'completed' and run.get('scope') == 'full_train_val_test',
            'DA-Fusion formal run is incomplete or smoke-only; no outputs created')
    output = args.output.resolve()
    targets = [output / 'comparison_summary.json', output / 'comparison_report.md', output / 'gallery']
    require(all(not p.exists() and not p.is_symlink() for p in targets), 'Refusing to overwrite comparison artifacts')
    baseline = read(args.baseline_summary)
    require(baseline.get('status') == 'complete_and_verified'
            and baseline.get('schema') == 'course_finetune_comparison_v1', 'Verified RGB comparison required')
    require(baseline['counts'] == {'train_per_epoch': 14432, 'epochs': 5, 'val': 2584, 'test': 2112},
            'RGB baseline study scope differs')
    fixed = read(args.baseline_gallery)
    require(fixed['status'] == 'completed' and fixed['fixed_indices_zero_based'] == list(INDICES)
            and len(fixed['cases']) == len(INDICES), 'Original fixed-case gallery required')
    require(fixed['comparison_summary_sha256'] == sha256(args.baseline_summary), 'Baseline gallery/summary binding differs')
    check_hash(Path(baseline['protocol']['path']), baseline['protocol']['sha256'])
    models, saved = {}, {}
    with CourseArchive(args.archive, 'test') as dataset:
        require(len(dataset) == 2112 and {c['id'] for c in dataset.categories} == {1}, 'Official component test split required')
        for name in ('maskrcnn', 'mask2former'):
            audited = baseline['models'][name]
            test, test_folder = audited['test'], Path(audited['test']['path'])
            check_hash(test_folder / 'evaluation.json', test['evaluation_sha256'])
            check_hash(test_folder / 'evaluation_run.json', test['evaluation_run_sha256'])
            evaluation, receipt = full_evaluation(test_folder, 'test', 2112, depth=False)
            require(evaluation['metrics'] == test['metrics'] and Path(receipt['archive']).resolve() == dataset.zip_path,
                    'RGB baseline evaluation/archive differs')
            require(receipt['evaluated_checkpoint_sha256'] == audited['best_checkpoint_sha256'], 'RGB checkpoint binding differs')
            selected, raw = selected_predictions(test_folder, dataset, test['raw']['sha256'])
            require(raw['predictions'] == evaluation['predicted_instances'], 'RGB prediction count differs')
            csv = submission_receipt(test_folder, receipt, 2112, raw['predictions'])
            models[name] = {'name': NAMES[name], 'metrics': test['metrics'], 'raw': raw,
                'best_epoch': audited['best_epoch'], 'best_checkpoint_sha256': audited['best_checkpoint_sha256'],
                'gt_instances': evaluation['gt_instances'], 'gt_area_groups_including_crowd': evaluation['gt_area_groups_including_crowd'],
                'evaluation_sha256': test['evaluation_sha256'], 'evaluation_run_sha256': test['evaluation_run_sha256'],
                'test_path': str(test_folder), 'submission': csv}
            if name == 'mask2former':
                saved[name] = selected
        models['dafusion'], saved['dafusion'] = audit_dafusion(folder, dataset, baseline)
        for name in ('maskrcnn', 'mask2former'):
            for key in ('gt_instances', 'gt_area_groups_including_crowd'):
                require(models[name][key] == models['dafusion'][key], 'GT populations differ across models')
        annotations = defaultdict(list)
        ground_truth = dataset.coco_ground_truth()
        require(len(ground_truth['annotations']) == models['dafusion']['gt_instances'], 'Current GT count differs')
        for annotation in ground_truth['annotations']:
            annotations[annotation['image_id']].append(annotation)
        annotation_digest, rgb_sha = hashlib.sha256(), {}
        with ZipFile(args.archive) as archive:
            for name in dataset.annotation_members:
                annotation_digest.update(name.encode() + b'\0' + archive.read(name))
            for index in INDICES:
                name = dataset.images[index]['file_name']
                rgb_sha[name] = hashlib.sha256(archive.read(name)).hexdigest()
        require(annotation_digest.hexdigest() == fixed['annotation_members_sha256']
                and rgb_sha == fixed['selected_rgb_sha256'], 'Current selected inputs differ from audited baseline gallery')
        cases = gallery_cases(dataset, annotations, saved, fixed['cases'])
    limitations = [
        '单种子、各五轮训练；不宣称充分收敛或统计显著。',
        'DA-Fusion 为公开 DS/DC 重实现的课程适配，使用双 Swin-Tiny、XYZ+valid；不是论文 HHA/Swin-L 结果复现。',
        'DA-Fusion 与 Mask2Former 保持相同原生损失，但新增深度、另一条骨干和融合参数；微批累积的归一化与随机性也可能影响比较，差值不能单独归因于深度。',
        'Mask R-CNN 与两种 Transformer 模型的架构和原生损失不同；其结果用于任务性能比较。',
        '此前已查看该测试集上的 RGB 基线成绩；本轮训练前固定协议，最佳轮次只由完整验证集选择。',
        '六个展示案例沿用既定索引，未按增益筛选；展示阈值与0.01评估阈值不同，滑杆不改变指标。',
        '本次没有统一重测三模型速度；历史耗时仅描述各自运行，不能据此断言加速。',
    ]
    summary = {'status': 'complete_and_verified', 'schema': 'course_dafusion_comparison_v1',
        'status_scope': 'Artifact consistency audit; not independent reproduction or repeated scoring',
        'material_passport': {'origin_skill': 'academic-research-suite/experiment-agent', 'mode': 'analyze',
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'verification_status': 'ANALYZED'},
        'counts': baseline['counts'], 'models': models, 'metric_scale': '0–1',
        'delta_percentage_points': {name: {k: 100 * (models['dafusion']['metrics'][k] - models[name]['metrics'][k])
            for k in METRICS} for name in ('maskrcnn', 'mask2former')}, 'limitations': limitations,
        'fixed_cases': [{k: case[k] for k in ('key', 'index', 'image_id', 'file_name')} for case in cases],
        'display_score_threshold_default': .5, 'evaluation_score_threshold': .01,
        'source_binding': {'generator_sha256': sha256(__file__), 'baseline_summary_path': str(args.baseline_summary.resolve()),
            'baseline_summary_sha256': sha256(args.baseline_summary), 'baseline_gallery_sha256': sha256(args.baseline_gallery),
            'test_annotation_members_sha256': annotation_digest.hexdigest(), 'selected_rgb_sha256': rgb_sha,
            'full_archive_rehashed': False}}
    text = json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    payload = {'cases': cases, 'metric_keys': METRICS,
               'metric_rows': [{'name': model['name'], 'metrics': model['metrics']} for model in models.values()],
               'limitations': limitations, 'summary_sha256': hashlib.sha256(text.encode()).hexdigest()}
    html = HTML.replace('__DATA__', safe_json(payload))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'gallery').mkdir()
    with (output / 'comparison_summary.json').open('x') as stream:
        stream.write(text)
    with (output / 'comparison_report.md').open('x') as stream:
        stream.write(report(summary))
    with (output / 'gallery/index.html').open('x') as stream:
        stream.write(html)
    receipt = {'status': 'completed', 'fixed_cases': summary['fixed_cases'], 'new_inference': False,
        'summary_sha256': sha256(output / 'comparison_summary.json'), 'html_sha256': sha256(output / 'gallery/index.html'),
        'html_bytes': len(html.encode()), 'generator_sha256': sha256(__file__), 'display_score_threshold_default': .5}
    (output / 'gallery/gallery.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    if args.browser_check:
        browser_check(output)
    print(json.dumps({'status': 'completed', 'output': str(output), 'html_bytes': len(html.encode()),
                      'browser_checked': args.browser_check}))


def browser_check(output):
    from playwright.sync_api import sync_playwright
    receipt = read(output / 'gallery/gallery.json')
    check_hash(output / 'gallery/index.html', receipt['html_sha256'])
    check_hash(output / 'comparison_summary.json', receipt['summary_sha256'])
    checks = output / 'gallery/browser'
    checks.mkdir(exist_ok=True)
    errors, external, states = [], [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1500, 'height': 1100}, offline=True)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda req: external.append(req.url) if req.url.startswith(('http:', 'https:')) else None)
        page.goto((output / 'gallery/index.html').as_uri())
        metric_text = page.locator('#metrics').inner_text()
        for case in receipt['fixed_cases']:
            page.locator('#case').select_option(case['key'])
            for threshold in (.01, .5, 1.):
                page.locator('#score').evaluate('(el,value)=>{el.value=value;el.dispatchEvent(new Event("input"))}', str(threshold))
                page.wait_for_function('window.__galleryReady===true')
                require(page.evaluate('window.__galleryCase') == case['key'], 'Case render differs')
                require(page.locator('#score-value').evaluate('el=>el.value') == f'{threshold:.2f}', 'Threshold control differs')
                require(page.locator('canvas').count() == 4, 'Missing comparison panels')
                require(page.locator('#metrics').inner_text() == metric_text, 'Display filter changed full-test metrics')
                counts = {name: page.locator('#' + name + '-count').inner_text() for name in ('mask2former', 'dafusion')}
                states.append({'case': case['key'], 'threshold': threshold, 'counts': counts})
        page.locator('#reset').click()
        page.wait_for_function('window.__galleryReady===true')
        page.screenshot(path=str(checks / 'desktop.png'), full_page=True)
        for width in (390, 320):
            page.set_viewport_size({'width': width, 'height': 1100})
            require(page.evaluate('document.documentElement.scrollWidth<=innerWidth'), 'Horizontal overflow')
            page.screenshot(path=str(checks / f'mobile-{width}.png'), full_page=True)
        require(not errors and not external, f'JS errors or external requests: {errors}, {external}')
        browser.close()
    (checks / 'browser_test.json').write_text(json.dumps({'status': 'passed', 'states': states,
        'viewport_widths': [1500, 390, 320], 'js_errors': errors, 'external_requests': external,
        'offline': True, 'new_inference': False}, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'reports/course_dafusion/formal_v1',
                        help='Completed formal DA-Fusion run directory; defaults to formal_v1')
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--baseline-summary', type=Path, default=ROOT / 'reports/course_finetune/comparison_summary.json')
    parser.add_argument('--baseline-gallery', type=Path, default=ROOT / 'reports/course_finetune/gallery/gallery.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/course_dafusion')
    parser.add_argument('--browser-check', action='store_true')
    parser.add_argument('--check-existing', action='store_true')
    args = parser.parse_args()
    if args.check_existing:
        browser_check(args.output.resolve())
    else:
        build(args)


if __name__ == '__main__':
    main()
