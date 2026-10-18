#!/usr/bin/env python3
"""Build a static, GT-assisted diagnostic gallery from frozen seed17 E2 RLEs.

No model loading or inference. Selection is every GraspNet validation frame plus
all test frames containing an original-area small GT; never select by gain.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from rgbd_seg.data import RGBDDataset, dataset_fingerprint

METHODS = ('full', 'highres', 'crop_fuse')
DISPLAY_THRESHOLD = .01
SMALL_AREA = 32 ** 2
CROP_PADDING = 32
PALETTE = [(65, 218, 178), (255, 183, 90), (123, 158, 255), (237, 130, 194), (168, 143, 247), (83, 210, 231)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_frozen(folder, dataset, fingerprint):
    folder = Path(folder).resolve()
    summary = read_json(folder / 'summary.json')
    if summary.get('status') != 'completed':
        raise ValueError(f'Evaluation is not completed: {folder}')
    expected_ids = [record['id'] for record in dataset.samples]
    methods, receipts = {}, {}
    for method in METHODS:
        evaluation_path = folder / method / 'evaluation.json'
        prediction_path = folder / method / 'predictions.jsonl'
        evaluation = read_json(evaluation_path)
        if sha256(evaluation_path) != summary['settings'][method]['evaluation_sha256']:
            raise ValueError(f'Evaluation SHA mismatch: {method}')
        required = {'status': 'completed', 'method': method, 'split': dataset.split,
                    'dataset_sha256': fingerprint, 'manifest_sha256': sha256(dataset.root / 'manifest.json'),
                    'provenance': 'public_graspnet_subset', 'training_seed': 17}
        for key, value in required.items():
            if evaluation.get(key) != value:
                raise ValueError(f'Frozen evaluation identity mismatch: {method}/{key}')
        if evaluation['model'].get('mode') != 'E2' or evaluation['classes'] != dataset.classes:
            raise ValueError('Gallery requires seed17 E2 with matching dataset classes')
        if sha256(prediction_path) != evaluation['predictions_sha256']:
            raise ValueError(f'Prediction SHA mismatch: {method}')
        records = [json.loads(line) for line in prediction_path.read_text().splitlines()]
        if [record['sample_id'] for record in records] != expected_ids:
            raise ValueError(f'Prediction order/coverage differs from complete {dataset.split} split: {method}')
        methods[method] = {record['sample_id']: record for record in records}
        receipts[method] = {'evaluation_path': str(evaluation_path), 'evaluation_sha256': sha256(evaluation_path),
                            'predictions_path': str(prediction_path), 'predictions_sha256': sha256(prediction_path),
                            'checkpoint_sha256': evaluation['checkpoint_sha256'],
                            'protocol_sha256': evaluation['protocol_sha256'],
                            'inference_source_sha256': evaluation['source_sha256']['rgbd_seg/multiscale_inference.py']}
    for key in ('checkpoint_sha256', 'protocol_sha256', 'inference_source_sha256'):
        if len({receipt[key] for receipt in receipts.values()}) != 1:
            raise ValueError(f'Methods do not share the same {key}')
    return methods, receipts



def development_summary(val_runs):
    """Read all three completed development runs; do not consult test metrics."""
    parent = Path(val_runs).resolve().parents[1]
    rows, receipts = [], []
    dataset_hashes, protocol_hashes = set(), set()
    for seed in (17, 23, 41):
        folder = parent / f'seed{seed}_E2' / 'val'
        summary = read_json(folder / 'summary.json')
        if summary.get('status') != 'completed':
            raise ValueError('Three-seed development matrix must be complete')
        row = {'seed': seed}
        for method in METHODS:
            path = folder / method / 'evaluation.json'
            result = read_json(path)
            digest = sha256(path)
            if digest != summary['settings'][method]['evaluation_sha256']:
                raise ValueError('Development result SHA differs from its completed summary')
            if (result.get('status'), result.get('split'), result.get('training_seed'), result['model'].get('mode'), result.get('method')) != ('completed', 'val', seed, 'E2', method):
                raise ValueError('Unexpected development run identity')
            row[method] = {key: result['metrics'][key] for key in ('mask_ap', 'mask_ap_small')}
            dataset_hashes.add(result['dataset_sha256'])
            protocol_hashes.add(result['protocol_sha256'])
            receipts.append({'path': str(path), 'sha256': digest})
        rows.append(row)
    if len(dataset_hashes) != 1 or len(protocol_hashes) != 1:
        raise ValueError('Development runs must share dataset and protocol')
    means = {method: float(np.mean([row[method]['mask_ap_small'] for row in rows])) for method in METHODS}
    gate = {}
    for method in ('highres', 'crop_fuse'):
        gains = [row[method]['mask_ap_small'] - row['full']['mask_ap_small'] for row in rows]
        ap_delta = float(np.mean([row[method]['mask_ap'] - row['full']['mask_ap'] for row in rows]))
        mean_gain, positives = float(np.mean(gains)), sum(delta > 0 for delta in gains)
        gate[method] = {'mean_ap_small_gain_pp': mean_gain, 'positive_seeds': positives,
                        'mean_ap_delta_pp': ap_delta,
                        'passed': mean_gain >= 2 and positives >= 2 and ap_delta >= -1}
    return {'seeds': [17, 23, 41], 'mean_ap_small': means, 'gate': gate,
            'gate_rule': 'mean APsmall gain >=2pp; positive in >=2/3 seeds; mean AP change >=-1pp',
            'online_method': 'full', 'rows': rows, 'receipts': receipts,
            'dataset_sha256': next(iter(dataset_hashes)), 'protocol_sha256': next(iter(protocol_hashes))}


def decode_record(record, height, width, classes):
    if (record['height'], record['width']) != (height, width):
        raise ValueError('Prediction is not at original image dimensions')
    n = len(record['scores'])
    if any(len(record[key]) != n for key in ('boxes', 'labels', 'masks_rle')):
        raise ValueError('Inconsistent prediction vectors')
    masks, scores, labels, original_indices = [], [], [], []
    for index, (score, label, rle) in enumerate(zip(record['scores'], record['labels'], record['masks_rle'])):
        if not np.isfinite(score) or not 0 <= score <= 1 or label not in range(1, len(classes) + 1):
            raise ValueError('Invalid score or class label')
        if rle['size'] != [height, width]:
            raise ValueError('RLE size differs from original image')
        if score < DISPLAY_THRESHOLD:
            continue
        mask = mask_utils.decode({'size': rle['size'], 'counts': rle['counts'].encode('ascii')}).astype(bool)
        if mask.shape != (height, width):
            raise ValueError('Decoded RLE has wrong dimensions')
        if not mask.any():
            continue
        masks.append(mask)
        scores.append(float(score))
        labels.append(int(label))
        original_indices.append(index)
    return {'masks': np.stack(masks) if masks else np.zeros((0, height, width), dtype=bool),
            'scores': scores, 'labels': labels, 'original_indices': original_indices}


def overlay(image, masks, color=None):
    canvas = np.asarray(image).copy().astype(np.float32)
    for index, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        rgb = np.asarray(color or PALETTE[index % len(PALETTE)], dtype=np.float32)
        canvas[mask] = .55 * canvas[mask] + .45 * rgb
        interior = mask.copy()
        interior[1:] &= mask[:-1]
        interior[:-1] &= mask[1:]
        interior[:, 1:] &= mask[:, :-1]
        interior[:, :-1] &= mask[:, 1:]
        canvas[mask & ~interior] = rgb
    return Image.fromarray(canvas.clip(0, 255).astype(np.uint8))


def best_same_class(gt, label, prediction):
    """Independent per-GT maximum IoU, not one-to-one detection matching."""
    best = None
    for index, (mask, score, predicted_label) in enumerate(zip(prediction['masks'], prediction['scores'], prediction['labels'])):
        if predicted_label != label:
            continue
        intersection = int(np.count_nonzero(gt & mask))
        union = int(np.count_nonzero(gt | mask))
        iou = intersection / union if union else 0.
        key = (iou, score, -index)
        if best is None or key > best[0]:
            best = (key, index)
    if best is None:
        return None
    key, index = best
    return {'local_index': index, 'prediction_index': prediction['original_indices'][index],
            'score': prediction['scores'][index], 'iou': key[0],
            'mask_area': int(prediction['masks'][index].sum()), 'label': int(label)}


def crop_bounds(mask, padding=CROP_PADDING):
    yy, xx = np.nonzero(mask)
    h, w = mask.shape
    return [max(0, int(xx.min()) - padding), max(0, int(yy.min()) - padding),
            min(w, int(xx.max()) + 1 + padding), min(h, int(yy.max()) + 1 + padding)]


def build(data_root, val_runs, test_runs, output):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Refusing to overwrite nonempty gallery: {output}')
    dataset = {split: RGBDDataset(data_root, split) for split in ('val', 'test')}
    if dataset['val'].manifest['provenance'] != 'public_graspnet_subset':
        raise ValueError('This gallery is scoped to the GraspNet experiment')
    fingerprint = dataset_fingerprint(data_root)
    development = development_summary(val_runs)
    if development['dataset_sha256'] != fingerprint:
        raise ValueError('Development matrix and gallery data differ')
    predictions, receipts = {}, {}
    for split, path in (('val', val_runs), ('test', test_runs)):
        predictions[split], receipts[split] = load_frozen(path, dataset[split], fingerprint)
    for key in ('checkpoint_sha256', 'protocol_sha256', 'inference_source_sha256'):
        if receipts['val']['full'][key] != receipts['test']['full'][key]:
            raise ValueError(f'Validation/test records differ in {key}')
    # Selection consults only original GT, before prediction inspection or rendering.
    selected = []
    for split in ('val', 'test'):
        for index, record in enumerate(dataset[split].samples):
            _, target = dataset[split][index]
            areas = target['masks'].flatten(1).sum(1).numpy()
            small_indices = np.where((areas > 0) & (areas <= SMALL_AREA))[0].tolist()
            if split == 'val' or small_indices:
                selected.append((split, index, small_indices))
    if sum(split == 'val' for split, _, _ in selected) != 4 or sum(split == 'test' for split, _, _ in selected) != 1:
        raise ValueError('Expected all four validation frames and exactly one test frame with small GT; no silent truncation')
    (output / 'assets').mkdir(parents=True)
    cases = []
    def save(image, name):
        path = output / 'assets' / (name + '.png')
        image.save(path)
        return 'assets/' + path.name
    for case_index, (split, index, small_indices) in enumerate(selected):
        sample, target = dataset[split][index]
        source = dataset[split].samples[index]
        image = Image.open(source['rgb_path']).convert('RGB')
        width, height = image.size
        masks = target['masks'].numpy().astype(bool)
        labels = target['labels'].tolist()
        pred = {method: decode_record(predictions[split][method][source['id']], height, width, dataset[split].classes)
                for method in METHODS}
        key = f'case_{case_index:03d}'
        case = {'key': key, 'sample_id': source['id'], 'split': split,
                'name': f"{'开发' if split == 'val' else '测试'} · 场景 {source.get('source_scene', '?')} · 帧 {source.get('source_frame', '?')}",
                'width': width, 'height': height, 'instances': len(masks), 'small_instances': len(small_indices),
                'selection': 'all validation frames' if split == 'val' else 'test frame contains original-area small GT',
                'full_images': {'rgb': save(image, key + '_rgb'), 'gt': save(overlay(image, masks), key + '_gt')},
                'displayed_candidates': {method: len(pred[method]['scores']) for method in METHODS}, 'small_cases': []}
        for method in METHODS:
            case['full_images'][method] = save(overlay(image, pred[method]['masks']), key + '_' + method)
        for number, gt_index in enumerate(small_indices):
            gt = masks[gt_index]
            bounds = crop_bounds(gt)
            left, top, right, bottom = bounds
            crop = image.crop(tuple(bounds))
            detail_key = f'{key}_small_{number:02d}'
            local_gt = gt[top:bottom, left:right]
            detail = {'gt_index': gt_index, 'class': dataset[split].classes[labels[gt_index] - 1],
                      'original_area': int(gt.sum()), 'crop_xyxy': bounds,
                      'images': {'rgb': save(crop, detail_key + '_rgb'),
                                 'gt': save(overlay(crop, [local_gt], PALETTE[0]), detail_key + '_gt')},
                      'matches': {}}
            for method in METHODS:
                match = best_same_class(gt, labels[gt_index], pred[method])
                if match is None:
                    local_masks = []
                else:
                    local_masks = [pred[method]['masks'][match['local_index'], top:bottom, left:right]]
                    match.pop('local_index')
                detail['matches'][method] = match
                detail['images'][method] = save(overlay(crop, local_masks, PALETTE[1]), detail_key + '_' + method)
            baseline = detail['matches']['full']
            baseline_iou = baseline['iou'] if baseline else 0.
            detail['iou_change_from_full'] = {method: (detail['matches'][method]['iou'] if detail['matches'][method] else 0.) - baseline_iou
                                              for method in ('highres', 'crop_fuse')}
            case['small_cases'].append(detail)
        cases.append(case)
        del sample, target, masks, pred
    if dataset_fingerprint(data_root) != fingerprint:
        raise RuntimeError('Dataset bytes changed while rendering')
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'completed',
                'dataset': str(data_root), 'dataset_sha256': fingerprint, 'seed': 17, 'model': 'E2',
                'development_summary': development,
                'display_threshold': DISPLAY_THRESHOLD, 'mask_threshold': .5, 'small_definition': '0 < original GT area <= 1024 pixels',
                'crop_padding_original_pixels': CROP_PADDING,
                'selection': 'Every validation frame plus every test frame with a small GT; no selection by prediction quality or gain',
                'diagnostic': 'Per-small-GT maximum mask IoU among same-class candidates with score>=0.01, computed on the full original image. Independent matching may reuse a prediction for multiple GTs. GT-assisted retrospective diagnosis, not deployment, detection recall or COCO AP.',
                'receipts': receipts, 'cases': cases,
                'source_sha256': sha256(Path(__file__).resolve())}
    (output / 'gallery.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    standalone = dict(manifest)
    standalone['embedded_images'] = {str(path.relative_to(output)): 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')
                                     for path in sorted((output / 'assets').glob('*.png'))}
    embedded = json.dumps(standalone, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    (output / 'index.html').write_text(HTML.replace('__GALLERY_DATA__', embedded), encoding='utf-8')
    pngs = sorted((output / 'assets').glob('*.png'))
    result = {'status': 'completed', 'output': str(output), 'cases': len(cases),
              'small_cases': sum(len(case['small_cases']) for case in cases), 'pngs': len(pngs),
              'diagnostic_iou_decreases': sum(delta < -1e-9 for case in cases for detail in case['small_cases']
                                             for delta in detail['iou_change_from_full'].values())}
    print(json.dumps(result, ensure_ascii=False))
    return result


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>冻结权重 · 三种推理策略对照</title><style>
:root{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#0d1420;color:#e4ecf5;--zoom:1}*{box-sizing:border-box}body{margin:0}main{max-width:1700px;margin:auto;padding:28px}h1{font-size:27px;margin:8px 0 14px}h2{font-size:18px;margin:26px 0 12px}p,footer,.detail-head{overflow-wrap:anywhere}p{line-height:1.7;color:#b2c0d1;font-size:13px}.eyebrow{color:#a7e9d1;font-size:11px;letter-spacing:2px}.notice{padding:14px 17px;border:1px solid #806a43;background:#28251e;color:#efdbb7;border-radius:9px}.controls{display:flex;gap:22px;align-items:center;flex-wrap:wrap;padding:18px 0}label{font-size:12px}select,button{font:inherit;color:inherit;background:#1a273a;border:1px solid #3e4b61;border-radius:6px;padding:8px}input{accent-color:#a7e9d1}select{max-width:100%}.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}.card{min-width:0;background:#152133;border:1px solid #344055;border-radius:8px;overflow:hidden}.title{font-size:12px;font-weight:650;padding:10px}.window{overflow:auto;max-height:60vh;background:#101824}.window img{display:block;width:calc(100% * var(--zoom));max-width:none;height:auto}.caption{font-size:11px;color:#b4c2d4;padding:10px;line-height:1.6;min-height:53px}.detail{margin:20px 0 30px}.detail-head{font-size:14px;font-weight:600;margin-bottom:10px}.bad{color:#ffaaa0}.good{color:#a7e9d1}a{color:#aacdff}footer{border-top:1px solid #334258;margin-top:28px;padding-top:16px;font-size:12px;color:#9fafc4}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #a7e9d1;outline-offset:3px}@media(max-width:1050px){.grid{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:650px){main{padding:16px}.grid{grid-template-columns:1fr}.controls{display:block}.controls>label{display:block;margin:12px 0}h1{font-size:23px}.window{max-height:65vh}}
</style></head><body><main>
<div class="eyebrow">FROZEN WEIGHTS · POST-EVALUATION DIAGNOSTICS</div><h1>同一权重，三种推理策略</h1>
<p>seed17 · E2 · 原图、整图放大、全图加四角裁块。这里读取已完成评测的掩码，不调用模型。</p>
<p id="verdict" class="notice"></p>
<p class="notice">固定展示全部 4 张开发图，以及按真值小目标面积选出的 1 张测试图；不按改进幅度挑图。候选显示阈值固定为 0.01，与在线 demo 的默认 0.5 不同。局部图使用真值寻找最大 IoU 同类预测，属于 GT 辅助诊断，不是部署时的检测或独立检出结果。变差与未重叠的情况同样保留。</p>
<div class="controls"><label>选择案例 <select id="case" aria-label="选择案例"></select></label><label>图像放大 <input id="zoom" type="range" min="1" max="4" step="0.25" value="1"><output id="zoom-value">1.00×</output></label><button id="reset">复位</button></div>
<p id="case-info" aria-live="polite"></p><h2>原图完整视野</h2><div id="full" class="grid"></div>
<p>五列依次为 RGB、GT 和三种策略的所有可见候选。实例颜色只用于区分掩码，不代表不同类别。</p>
<h2>全部 small 真值的原图局部放大</h2><p>small 按原始可见掩码面积 0 &lt; area ≤ 1024 定义；局部窗口为真值框外扩 32 原图像素。下方各策略只画该真值的最大 IoU 同类预测，IoU 在完整原图计算。匹配逐真值独立，同一预测可能用于多个真值；这张表不能代替一对一检测评估或 COCO AP。</p>
<div id="details"></div><footer><a href="#" id="download-data">下载案例、匹配信息与来源哈希</a><p id="integrity"></p>所有 PNG 已嵌入此 HTML；页面无需联网。</footer>
<script id="gallery-data" type="application/json">__GALLERY_DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('gallery-data').textContent), $=id=>document.getElementById(id);
const summary=data.development_summary,passed=Object.entries(summary.gate).filter(([key,value])=>value.passed).map(([key])=>key);$('verdict').textContent=(passed.length?'通过开发门槛的策略：'+passed.join('、')+'；此页仅记录实验。':'两种改法均未达到预定开发门槛，在线推理继续 full。')+' 三种子开发 APₛ 均值：full '+summary.mean_ap_small.full.toFixed(2)+' / highres '+summary.mean_ap_small.highres.toFixed(2)+' / crop_fuse '+summary.mean_ap_small.crop_fuse.toFixed(2)+'。同一开发集仅4个小目标，属于探索性结果。门槛为 APₛ 平均提高至少2点、至少2/3种子提高、整体AP平均下降不超过1点。下方图像固定展示seed17，未择优种子。';
const columns=[['rgb','RGB'],['gt','实例真值'],['full','full · 原图'],['highres','highres · 整图放大'],['crop_fuse','crop_fuse · 全图 + 四角']];
function card(key,path,caption,kind=''){const node=document.createElement('article');node.className='card';const title=document.createElement('div');title.className='title';title.textContent=columns.find(item=>item[0]===key)[1];const frame=document.createElement('div');frame.className='window';const img=document.createElement('img');if(!/^assets\/case_\d{3}(?:_small_\d{2})?_(?:rgb|gt|full|highres|crop_fuse)\.png$/.test(path))throw new Error('Invalid local image path');const embedded=data.embedded_images[path];if(typeof embedded!=='string'||!embedded.startsWith('data:image/png;base64,'))throw new Error('Missing embedded PNG');img.src=embedded;img.alt=title.textContent+'：'+caption;frame.append(img);const text=document.createElement('div');text.className='caption '+kind;text.textContent=caption;node.append(title,frame,text);return node;}
function render(){const item=data.cases.find(entry=>entry.key===$('case').value);$('full').replaceChildren();$('details').replaceChildren();$('case-info').textContent=item.name+' · '+item.sample_id+' · '+item.width+'×'+item.height+' · '+item.instances+' 个GT实例 · '+item.small_instances+' 个small';for(const [key]of columns){const caption=key==='rgb'?'原始颜色图':key==='gt'?'原始像素真值（展示参考）':item.displayed_candidates[key]+' 个候选 · score ≥ 0.01';$('full').append(card(key,item.full_images[key],caption));}if(!item.small_cases.length){const p=document.createElement('p');p.textContent='该帧没有原始面积 small 实例。完整图仍保留，未因缺少小目标而删去开发样例。';$('details').append(p);}for(const [index,detail] of item.small_cases.entries()){const section=document.createElement('section');section.className='detail';const title=document.createElement('div');title.className='detail-head';title.textContent='small '+(index+1)+' · GT索引 '+detail.gt_index+' · '+detail.class+' · '+detail.original_area+' px · 原图窗口 ['+detail.crop_xyxy.join(', ')+']';const grid=document.createElement('div');grid.className='grid';for(const [key]of columns){let text=key==='rgb'?'原图局部，无再次推理':key==='gt'?'绿色：该实例真值':'';let kind='';if(detail.matches[key]!==undefined){const match=detail.matches[key];text=match?'GT辅助最大IoU '+match.iou.toFixed(3)+' · score '+match.score.toFixed(3)+(match.iou===0?' · 无重叠':''):'无同类候选 · IoU 0';if(key!=='full'){const delta=detail.iou_change_from_full[key];text+=' · 对full ΔIoU '+(delta>=0?'+':'')+delta.toFixed(3);kind=delta<-1e-9?'bad':delta>1e-9?'good':'';}}grid.append(card(key,detail.images[key],text,kind));}section.append(title,grid);$('details').append(section);}}
for(const item of data.cases){const option=document.createElement('option');option.value=item.key;option.textContent=item.name+' · '+item.small_instances+' 个small';$('case').append(option);}function zoom(){document.documentElement.style.setProperty('--zoom',$('zoom').value);$('zoom-value').textContent=Number($('zoom').value).toFixed(2)+'×';}$('case').addEventListener('change',render);$('zoom').addEventListener('input',zoom);$('reset').addEventListener('click',()=>{$('zoom').value=1;zoom();document.querySelectorAll('.window').forEach(node=>{node.scrollTop=0;node.scrollLeft=0;});});$('download-data').addEventListener('click',event=>{event.preventDefault();const {embedded_images,...metadata}=data;const url=URL.createObjectURL(new Blob([JSON.stringify(metadata,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='multiscale-gallery.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});$('integrity').textContent='数据 SHA256 '+data.dataset_sha256+' · 冻结协议 SHA256 '+data.receipts.val.full.protocol_sha256;render();
</script></main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/graspnet_expanded_v2')
    parser.add_argument('--val-runs', type=Path, default=ROOT / 'reports/v4/experiment/graspnet/seed17_E2/val')
    parser.add_argument('--test-runs', type=Path, default=ROOT / 'reports/v4/experiment/graspnet/seed17_E2/test')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/v4/gallery')
    args = parser.parse_args()
    build(args.data, args.val_runs, args.test_runs, args.output)


if __name__ == '__main__':
    main()
