#!/usr/bin/env python3
"""Read saved predictions and create fixed-case course diagnostics; no inference."""
from __future__ import annotations
import argparse
import base64
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_submission import _Spec, _checked_runs
from tools.build_multiscale_gallery import overlay, crop_bounds, PALETTE
from tools.evaluate_multiscale import ensure_empty, sha256

INDICES = (0, 400, 800, 1200, 1600, 2000)
MODELS = ('maskrcnn', 'mask2former')
DISPLAY_THRESHOLD = .5


def read(path):
    return json.loads(Path(path).read_text())


def load_saved(folder, name, dataset, indices):
    folder = Path(folder)
    run, evaluation = read(folder / 'run.json'), read(folder / 'evaluation.json')
    if (run.get('status') != 'completed' or run.get('scope') != 'full_test'
            or run.get('model') != name or run.get('images_completed') != len(dataset)
            or run.get('training_performed') is not False or run.get('depth_used') is not False
            or run.get('score_threshold') != .01 or run.get('max_detections') != 100):
        raise ValueError(f'{name}: expected completed, full-test, unfine-tuned RGB inference')
    if Path(run['archive']).resolve() != dataset.zip_path or run['archive_bytes'] != dataset.zip_path.stat().st_size:
        raise ValueError('Run and gallery use different course archives')
    if evaluation.get('status') != 'completed' or evaluation.get('images') != len(dataset) or evaluation.get('split') != 'test':
        raise ValueError('Complete test evaluation is required')
    for metric in ('mask_ap', 'mask_ap50', 'mask_ap75', 'mask_ap_small'):
        value = evaluation['metrics'][metric]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('Metrics must be finite 0–1 values, or null for unavailable bins')
    prediction_path = folder / 'predictions.jsonl'
    if sha256(prediction_path) != run.get('predictions_sha256'):
        raise ValueError('Saved prediction SHA differs from completed run')
    selected, count = {}, 0
    with prediction_path.open() as stream:
        for index, line in enumerate(stream):
            if index >= len(dataset):
                raise ValueError('Extra prediction rows')
            row, expected = json.loads(line), dataset.images[index]
            if row['image_id'] != expected['id'] or row['file_name'] != expected['file_name']:
                raise ValueError('Prediction IDs, files or ordering differ from official test records')
            if len(row['predictions']) > 100 or len(row['original_category_ids']) != len(row['predictions']):
                raise ValueError('Prediction count/original category receipts differ')
            if index in indices:
                selected[index] = row
            count += 1
    if count != len(dataset):
        raise ValueError('Missing prediction rows')
    receipt = {'folder': str(folder.resolve()), 'run_sha256': sha256(folder / 'run.json'),
               'evaluation_sha256': sha256(folder / 'evaluation.json'), 'predictions_sha256': sha256(prediction_path),
               'model_metadata': run['model_metadata'], 'metrics': evaluation['metrics'],
               'evaluated_images': len(dataset), 'timing': run.get('timing'),
               'evaluation_scope': evaluation.get('interpretation'), 'display_threshold': DISPLAY_THRESHOLD}
    return selected, receipt


def decode(rle, height, width):
    if not isinstance(rle, dict) or rle.get('size') != [height, width]:
        raise ValueError('Masks must use original image dimensions')
    spec = _Spec(['diagnostic'], (1,), (height, width), 100)
    area = _checked_runs(rle.get('counts'), spec)
    mask = mask_utils.decode({'size': [height, width], 'counts': rle['counts'].encode('ascii')}).astype(bool)
    if mask.shape != (height, width) or int(mask.sum()) != area:
        raise ValueError('RLE decoding differs from checked original-area runs')
    return mask


def predictions_for_display(row, height, width):
    result = []
    for index, prediction in enumerate(row['predictions']):
        score = prediction['score']
        if prediction['category_id'] != 1 or isinstance(score, bool) or not isinstance(score, (float, int)) or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid component prediction/score')
        if score >= DISPLAY_THRESHOLD:
            result.append({'mask': decode(prediction['segmentation'], height, width), 'score': score,
                           'original_category_id': row['original_category_ids'][index], 'prediction_index': index})
    return result


def encode(image, full=False):
    buffer = io.BytesIO()
    if full:
        image = image.copy()
        image.thumbnail((960, 960), Image.Resampling.LANCZOS)
        image.save(buffer, format='JPEG', quality=87, optimize=True)
        mime = 'jpeg'
    else:
        image.save(buffer, format='PNG', optimize=True)
        mime = 'png'
    return f'data:image/{mime};base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def best_match(gt, predictions):
    values = []
    for prediction in predictions:
        mask = prediction['mask']
        iou = int((gt & mask).sum()) / int((gt | mask).sum())
        values.append((iou, prediction['score'], -prediction['prediction_index'], prediction))
    if not values:
        return None, []
    iou, score, _, prediction = max(values, key=lambda item: item[:3])
    return {'iou': iou, 'score': score, 'prediction_index': prediction['prediction_index'],
            'original_category_id': prediction['original_category_id']}, [prediction['mask']]


def render_case(dataset, index, annotations, saved, images):
    record = dataset.images[index]
    image = dataset.load_rgb(index)
    width, height = image.size
    key = f'case_{index:04d}'
    def store(picture, tag, full=False):
        image_key = f'{key}_{tag}'
        images[image_key] = encode(picture, full)
        return image_key
    # Original annotations and overlapping masks remain independent.
    positive = [annotation for annotation in annotations if annotation['area'] > 0]
    gt = [decode(annotation['segmentation'], height, width) for annotation in positive]
    pred = {name: predictions_for_display(saved[name][index], height, width) for name in MODELS}
    full = {'rgb': image, 'gt': overlay(image, gt)}
    for name in MODELS:
        full[name] = overlay(image, [p['mask'] for p in pred[name]])
    case = {'key': key, 'index': index, 'image_id': record['id'], 'file_name': record['file_name'],
            'width': width, 'height': height, 'gt_instances': len(annotations), 'positive_gt_instances': len(positive),
            'gt_crowd_instances': sum(bool(a.get('iscrowd')) for a in annotations),
            'raw_candidates': {name: len(saved[name][index]['predictions']) for name in MODELS},
            'display_candidates': {name: len(pred[name]) for name in MODELS},
            'full_images': {name: store(picture, name, True) for name, picture in full.items()}, 'small_detail': None}
    small = [(a['area'], a['id'], i, a) for i, a in enumerate(positive) if 0 < a['area'] < 1024]
    if small:
        _, _, gt_index, annotation = min(small, key=lambda value: value[:2])
        selected_gt = gt[gt_index]
        bounds = crop_bounds(selected_gt)
        left, top, right, bottom = bounds
        crop = image.crop(tuple(bounds))
        detail = {'annotation_id': annotation['id'], 'original_annotation_area': annotation['area'],
                  'decoded_area': int(selected_gt.sum()), 'iscrowd': annotation.get('iscrowd', 0),
                  'crop_xyxy': bounds, 'matches': {},
                  'images': {'rgb': store(crop, 'small_rgb'),
                             'gt': store(overlay(crop, [selected_gt[top:bottom, left:right]], PALETTE[0]), 'small_gt')}}
        for name in MODELS:
            match, masks = best_match(selected_gt, pred[name])
            detail['matches'][name] = match
            detail['images'][name] = store(overlay(crop, [m[top:bottom, left:right] for m in masks], PALETTE[1]), 'small_' + name)
        case['small_detail'] = detail
    return case, full


@lru_cache(maxsize=16)
def font(size, cjk=False):
    # DroidSansFallbackFull supplies Chinese but lacks Latin letters/digits.
    # Pillow does not implement browser/fontconfig fallback automatically.
    paths = (('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',) if cjk == 'symbols' else
             ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf') if cjk else
             ('/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError('Contact sheet requires local Latin and Chinese fonts')


@lru_cache(maxsize=2048)
def _font_for_character(size, character):
    for candidate in (font(size), font(size, cjk=True), font(size, cjk='symbols')):
        glyph, missing = candidate.getmask(character), candidate.getmask('\uffff')
        if character.isspace() or glyph.size != missing.size or bytes(glyph) != bytes(missing):
            return candidate
    raise ValueError(f'No contact-sheet font supports U+{ord(character):04X}')


def draw_label(draw, position, text, *, size, fill):
    """Render mixed Chinese/Latin runs on a shared baseline, with glyph checks."""
    runs = []
    for character in text:
        chosen = _font_for_character(size, character)
        if runs and runs[-1][0] is chosen:
            runs[-1][1] += character
        else:
            runs.append([chosen, character])
    x, y = position
    for chosen, run in runs:
        draw.text((x, y + size), run, font=chosen, fill=fill, anchor='ls')
        x += chosen.getlength(run)


def contact_sheet(cases, full_views):
    cell, gap, label = 320, 12, 48
    canvas = Image.new('RGB', (4 * cell + 5 * gap, len(cases) * (cell + label + gap) + 75), '#101824')
    draw = ImageDraw.Draw(canvas)
    draw_label(draw, (gap, 10), f'课程数据 · 预训练模型直接测试 · 固定 {len(cases)} 例', size=23, fill='#f0f4f8')
    draw_label(draw, (gap, 43), '展示 score ≥ 0.5；完整 AP 使用 score ≥ 0.01；RGB 输入，未微调，所有 COCO 类映射为 component', size=14, fill='#b5c5d8')
    columns = [('rgb', 'RGB'), ('gt', 'GT'), ('maskrcnn', 'Mask R-CNN'), ('mask2former', 'Mask2Former')]
    for row, (case, views) in enumerate(zip(cases, full_views)):
        top = 75 + row * (cell + label + gap)
        for col, (name, title) in enumerate(columns):
            left = gap + col * (cell + gap)
            if name in MODELS:
                title += f" · {case['display_candidates'][name]} 个"
            elif name == 'gt':
                title += f" · {case['gt_instances']} 个"
            draw_label(draw, (left, top), title, size=17, fill='#e8eff6')
            draw_label(draw, (left, top + 24), f"索引 {case['index']} · ID {case['image_id']}", size=13, fill='#acbdd1')
            thumb = views[name].copy()
            thumb.thumbnail((cell, cell), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (left, top + label))
    return canvas


def build(args, indices=INDICES):
    dataset = CourseArchive(args.archive, split='test')
    if max(indices) >= len(dataset) or len(set(indices)) != len(indices):
        raise ValueError('Fixed gallery indices must all exist and be unique; no substitution by result quality')
    truth = dataset.coco_ground_truth()
    by_id = defaultdict(list)
    for annotation in truth['annotations']:
        by_id[annotation['image_id']].append(annotation)
    saved, receipts = {}, {}
    for name in MODELS:
        saved[name], receipts[name] = load_saved(getattr(args, name), name, dataset, indices)
    deviation_path = Path(args.maskrcnn).parent / 'execution_deviation.json'
    execution_deviation = None
    if deviation_path.exists():
        deviation = read(deviation_path)
        execution_deviation = {'path': str(deviation_path.resolve()), 'sha256': sha256(deviation_path),
                               'record': {key: deviation[key] for key in
                                          ('recorded_at_utc', 'reason', 'original_protocol_sha256', 'deviation',
                                           'frozen_inference_and_evaluation_settings_changed',
                                           'timing_comparison_controlled', 'timing_interpretation') if key in deviation}}
    output = ensure_empty(args.output)
    image_data, cases, full_views = {}, [], []
    for index in indices:
        case, views = render_case(dataset, index, by_id[dataset.image_ids[index]], saved, image_data)
        cases.append(case)
        full_views.append(views)
    contact_sheet(cases, full_views).save(output / 'contact_sheet.png', optimize=True)
    annotation_digest = hashlib.sha256()
    rgb_hashes = {}
    with ZipFile(args.archive) as archive:
        for member in dataset.annotation_members:
            annotation_digest.update(member.encode() + b'\0' + archive.read(member))
        for case in cases:
            rgb_hashes[case['file_name']] = hashlib.sha256(archive.read(case['file_name'])).hexdigest()
    manifest = {'status': 'completed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'fixed_indices_zero_based': list(indices), 'selection': 'Predeclared test-order indices; never chosen by model score, IoU or AP',
                'archive': str(dataset.zip_path), 'archive_bytes': dataset.zip_path.stat().st_size,
                'annotation_members_sha256': annotation_digest.hexdigest(), 'selected_rgb_sha256': rgb_hashes,
                'archive_hash_scope': 'Annotation members and selected RGB bytes hashed; entire ZIP not rehashed here',
                'display_score_threshold': .5, 'evaluation_score_threshold': .01, 'metrics_scale': '0–1',
                'small_crop_selection': 'One smallest positive original-annotation area <1024 per fixed frame; tie by annotation ID; GT-assisted post-evaluation diagnosis only',
                'small_matching': 'Maximum full-original-image mask IoU among displayed component candidates (score>=0.5). Not deployment detection or a new evaluation.',
                'models': receipts, 'cases': cases, 'contact_sheet_sha256': sha256(output / 'contact_sheet.png'),
                'execution_deviation': execution_deviation,
                'timing_interpretation': 'Per-model timings are descriptive only, not a controlled speed comparison. Background CPU validation may affect preprocessing/postprocessing latency; do not claim a speed advantage.',
                'source_sha256': {name: sha256(ROOT / name) for name in ('tools/build_course_pretrained_gallery.py', 'rgbd_seg/course_archive.py', 'rgbd_seg/course_submission.py', 'tools/build_multiscale_gallery.py')}}
    for name in MODELS:
        folder = Path(getattr(args, name))
        for file, key in (('run.json', 'run_sha256'), ('evaluation.json', 'evaluation_sha256'), ('predictions.jsonl', 'predictions_sha256')):
            if sha256(folder / file) != receipts[name][key]:
                raise RuntimeError('Saved results changed during rendering')
    if execution_deviation and sha256(deviation_path) != execution_deviation['sha256']:
        raise RuntimeError('Execution-deviation receipt changed during rendering')
    (output / 'gallery.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    data = json.dumps({**manifest, 'embedded_images': image_data}, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    (output / 'index.html').write_text(HTML.replace('__DATA__', data), encoding='utf-8')
    result = {'status': 'completed', 'output': str(output.resolve()), 'cases': len(cases),
              'small_crops': sum(c['small_detail'] is not None for c in cases), 'html_bytes': (output / 'index.html').stat().st_size}
    print(json.dumps(result, ensure_ascii=False))
    return result


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>课程数据 · 预训练模型直接测试</title><style>
:root{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#101824;color:#e9eff6;--zoom:1}*{box-sizing:border-box}body{margin:0}main{max-width:1600px;margin:auto;padding:28px}h1{font-size:27px}h2{font-size:18px;margin-top:25px}p{font-size:13px;color:#b6c4d7;line-height:1.7;overflow-wrap:anywhere}.notice{background:#242b25;border:1px solid #536b59;padding:14px;border-radius:8px}.table-wrap{overflow:auto}table{width:100%;min-width:560px;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:12px;border-bottom:1px solid #364458}th{color:#a8dcca}.controls{display:flex;align-items:center;gap:22px;flex-wrap:wrap;padding:18px 0}select,button{font:inherit;color:inherit;background:#1d2d44;padding:8px;border:1px solid #47607d;border-radius:6px}select{max-width:100%}label{font-size:13px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.card{border:1px solid #364458;border-radius:8px;background:#192537;overflow:hidden;min-width:0}.title{padding:10px;font-size:13px;font-weight:650}.window{overflow:auto;max-height:70vh}.window img{display:block;width:calc(100% * var(--zoom));height:auto;max-width:none}.caption{padding:10px;font-size:12px;color:#b6c4d7;line-height:1.6;overflow-wrap:anywhere}footer{margin-top:25px;padding-top:18px;border-top:1px solid #364458;font-size:12px;overflow-wrap:anywhere}a{color:#b8d8ff}input{accent-color:#9ae0c8}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #9ae0c8;outline-offset:3px}@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:550px){main{padding:16px}.grid{grid-template-columns:1fr}.controls{display:block}.controls label,.controls button{display:block;margin:14px 0}select{width:100%}h1{font-size:23px}}
</style></head><body><main><p>FIXED PRETRAINED WEIGHTS · COURSE TEST SET</p><h1>课程数据：两种预训练模型直接测试</h1><p class="notice">两模型仅使用 RGB，直接加载 COCO 预训练权重，未在课程数据上微调、未使用深度。所有原 COCO 前景类别统一映射为 component（类别 1）。这是跨领域直接迁移检查；两模型骨干与模型结构不同，不能把成绩差异仅归因于实例分割方法。</p><h2>完整测试集指标 · 0–1</h2><p id="scope"></p><div class="table-wrap"><table><thead><tr><th>模型</th><th>AP</th><th>AP₅₀</th><th>AP₇₅</th><th>小目标 APₛ</th></tr></thead><tbody id="metrics"></tbody></table></div><p id="architectures"></p><p id="timing-note"></p><p>指标使用保存的全部 score ≥ 0.01 预测，最多 100 个实例；此页只展示 score ≥ 0.5 的掩码。显示较少的掩码不代表指标计算时丢弃了其他候选。COCO APₛ 使用官方面积范围，此页局部诊断按指定规则选 area &lt; 1024。</p><h2>固定案例 · 不按结果挑选</h2><p>按官方测试顺序固定选择零起始索引 0、400、800、1200、1600、2000。掩码独立半透明叠加，保留实例重叠；颜色仅区分实例，不表示类别或跨模型配对。</p><div class="controls"><label>案例 <select id="case"></select></label><label>放大 <input id="zoom" type="range" min="1" max="4" step=".25" value="1"> <output id="zoom-value">1.00×</output></label><button id="reset">复位</button></div><p id="case-info"></p><div id="full" class="grid"></div><h2>小目标局部 · GT 辅助诊断</h2><p>每帧仅按 GT 选原始面积最小且 0 &lt; area &lt; 1024 的实例，同面积按 annotation ID 排序。展示该 GT 与最大原图 IoU 的预测（score ≥ 0.5）。局部图和匹配在评测后生成，未参与推理，不构成检出率或新测试成绩。原始标注的 crowd 属性保留在导出记录中。</p><p id="small-info"></p><div id="small" class="grid"></div><footer><a href="#" id="download">下载案例与来源记录</a><p id="integrity"></p><p>全部图片嵌入 HTML，可直接离线打开；完整视野为展示用缩略图，小目标局部保留原始像素。</p></footer><script type="application/json" id="gallery-data">__DATA__</script><script>
'use strict';const $=id=>document.getElementById(id),data=JSON.parse($('gallery-data').textContent),columns=[['rgb','RGB'],['gt','GT'],['maskrcnn','Mask R-CNN'],['mask2former','Mask2Former']];for(const key of ['maskrcnn','mask2former']){const tr=document.createElement('tr');for(const value of [columns.find(c=>c[0]===key)[1],...['mask_ap','mask_ap50','mask_ap75','mask_ap_small'].map(m=>data.models[key].metrics[m]===null?'—':data.models[key].metrics[m].toFixed(4))]){const td=document.createElement('td');td.textContent=value;tr.append(td)}$('metrics').append(tr)}$('scope').textContent='完整测试 '+data.models.maskrcnn.evaluated_images+' 张；下方6张图仅用于固定样例展示，表中指标不随案例切换。';$('architectures').textContent=Object.entries(data.models).map(([name,m])=>name+'：'+(m.model_metadata.name||m.model_metadata.model_id||'详见来源记录')).join('；');
$('timing-note').textContent=data.execution_deviation?.record?.timing_comparison_controlled===false?'执行说明：两模型的 GPU 推理依次进行，但 Mask R-CNN 的 CPU 提交校验与后续 Mask2Former 推理存在时间重叠。后台 CPU 负载可能影响预处理和后处理耗时；导出记录中的各模型耗时仅作描述，不能据此断言速度优势。':'导出记录中的各模型耗时仅作描述性记录；本页不作受控速度对比或加速结论。';
function card(key,imageKey,caption){const article=document.createElement('article');article.className='card';const title=document.createElement('div');title.className='title';title.textContent=columns.find(c=>c[0]===key)[1];const frame=document.createElement('div');frame.className='window';const img=document.createElement('img'),uri=data.embedded_images[imageKey];if(typeof uri!=='string'||!/^data:image\/(jpeg|png);base64,/.test(uri))throw Error('Missing embedded image');img.src=uri;img.alt=title.textContent+' '+caption;frame.append(img);const text=document.createElement('div');text.className='caption';text.textContent=caption;article.append(title,frame,text);return article}function render(){const c=data.cases.find(c=>c.key===$('case').value);$('full').replaceChildren();$('small').replaceChildren();$('case-info').textContent='索引 '+c.index+' · image_id '+c.image_id+' · '+c.width+'×'+c.height+' · '+c.file_name;for(const [key]of columns){const text=key==='rgb'?'原始颜色图':key==='gt'?c.gt_instances+' 个真值实例（含 '+c.gt_crowd_instances+' 个 crowd）':c.display_candidates[key]+' 个显示候选 / '+c.raw_candidates[key]+' 个保存候选';$('full').append(card(key,c.full_images[key],text))}const d=c.small_detail;if(!d){$('small-info').textContent='该固定样例没有满足 0 < area < 1024 的 GT；不替换为其他图片。';return}$('small-info').textContent='GT '+d.annotation_id+' · 原始标注面积 '+d.original_annotation_area+' px · 解码面积 '+d.decoded_area+' px · crowd='+d.iscrowd+' · 原图窗口 ['+d.crop_xyxy.join(', ')+']';for(const [key]of columns){const m=d.matches[key];const text=key==='rgb'?'原像素局部，无再次推理':key==='gt'?'绿色：选定GT':m?'GT辅助 IoU '+m.iou.toFixed(3)+' · score '+m.score.toFixed(3)+(m.iou===0?' · 无重叠':''):'无 score ≥ 0.5 候选';$('small').append(card(key,d.images[key],text))}}for(const c of data.cases){const o=document.createElement('option');o.value=c.key;o.textContent='索引 '+c.index+' · image_id '+c.image_id;$('case').append(o)}function zoom(){document.documentElement.style.setProperty('--zoom',$('zoom').value);$('zoom-value').textContent=Number($('zoom').value).toFixed(2)+'×'}$('case').addEventListener('change',render);$('zoom').addEventListener('input',zoom);$('reset').addEventListener('click',()=>{$('zoom').value=1;zoom();document.querySelectorAll('.window').forEach(n=>{n.scrollLeft=0;n.scrollTop=0})});$('download').addEventListener('click',e=>{e.preventDefault();const {embedded_images,...metadata}=data,url=URL.createObjectURL(new Blob([JSON.stringify(metadata,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='course-pretrained-gallery.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});$('integrity').textContent='标注成员 SHA256 '+data.annotation_members_sha256+'；各模型的预测与评测校验值在下载记录中。';render();
</script></main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--maskrcnn', type=Path, default=ROOT / 'reports/course_pretrained/maskrcnn')
    parser.add_argument('--mask2former', type=Path, default=ROOT / 'reports/course_pretrained/mask2former')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/course_pretrained/gallery')
    build(parser.parse_args())


if __name__ == '__main__':
    main()
