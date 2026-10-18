#!/usr/bin/env python3
"""Read epoch-one validation outputs and render six predetermined cases.

No model loading, GPU, inference, checkpoint selection, or test-set access.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from rgbd_seg.course_archive import CourseArchive
from tools.build_course_pretrained_gallery import decode, draw_label
from tools.build_multiscale_gallery import overlay
from tools.build_course_dafusion_comparison import full_evaluation, selected_predictions
from tools.summarize_course_finetune import require, read, sha256, check_hash

INDICES = (0, 400, 800, 1200, 1600, 2000)
THRESHOLD = .5
TITLES = {'rgb': 'RGB', 'gt': 'GT', 'mask2former': 'Mask2Former · epoch 1', 'dafusion': 'DA-Fusion · epoch 1'}


def snapshot(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def load(folder, name, dataset):
    evaluation, receipt = full_evaluation(folder, 'val', 2584, depth=name == 'dafusion')
    require(Path(receipt['archive']).resolve() == dataset.zip_path, 'Validation/archive mismatch')
    selection, selection_sha = snapshot(folder.parent / 'selection.json')
    entries = [entry for entry in selection['epochs'] if entry['epoch'] == 1]
    require(len(entries) == 1 and entries[0]['metrics'] == evaluation['metrics']
            and Path(entries[0]['output']).resolve() == folder.resolve()
            and entries[0]['training_images'] == 14432, 'Epoch-one selection receipt mismatch')
    run, run_sha = snapshot(folder.parent / 'run.json')
    require(run['model'] == name and run['scope'] == 'full_train_val_test', 'Wrong formal model run')
    protocol = Path(run['arguments']['protocol'])
    check_hash(protocol, run['protocol_sha256'])
    link = {'folder': str(folder.resolve()), 'metrics': evaluation['metrics'],
            'evaluation_sha256': sha256(folder / 'evaluation.json'),
            'evaluation_run_sha256': sha256(folder / 'evaluation_run.json'),
            'predictions_sha256': receipt['predictions_sha256'], 'source_sha256': receipt['source_sha256'],
            'selection_snapshot_sha256': selection_sha, 'run_snapshot_sha256': run_sha,
            'run_status_at_read': run['status'], 'epoch_entry': entries[0],
            'protocol_path': str(protocol), 'protocol_sha256': run['protocol_sha256'],
            'gt_instances': evaluation['gt_instances'], 'gt_area_groups_including_crowd': evaluation['gt_area_groups_including_crowd'],
            'epoch_checkpoint_sha256': None,
            'checkpoint_note': 'Epoch validation bound to runner selection/epoch output; no persistent per-epoch checkpoint asserted'}
    if not (folder / 'predictions.jsonl').exists():
        require(name == 'mask2former', 'Required DA validation raw predictions are missing')
        link['raw_status'] = 'missing; no replacement predictions used'
        return None, link
    selected, raw = selected_predictions(folder, dataset, receipt['predictions_sha256'])
    require(raw['predictions'] == evaluation['predicted_instances'], 'Prediction counts differ')
    link.update(raw_status='present_and_sha_verified', raw=raw)
    return selected, link


def diagnostics(predictions, gt_masks, gt_rles):
    union = np.zeros((1024, 1024), dtype=bool)
    for mask in gt_masks:
        union |= mask
    shown = [(i, p) for i, p in enumerate(predictions) if p['score'] >= THRESHOLD]
    masks = [decode(p['segmentation'], 1024, 1024) for _, p in shown]
    rles = [p['segmentation'] for _, p in shown]
    # Purely post-inference GT-assisted diagnostics, never filtering candidates.
    iou = mask_utils.iou(rles, gt_rles, [0] * len(gt_rles)) if rles and gt_rles else np.zeros((len(rles), len(gt_rles)))
    details = []
    for j, ((index, p), mask) in enumerate(zip(shown, masks)):
        area = int(mask.sum())
        outside = int(np.count_nonzero(mask & ~union))
        details.append({'prediction_index': index, 'score': p['score'], 'area': area,
                        'frame_fraction': area / mask.size, 'outside_all_gt_pixels': outside,
                        'outside_all_gt_fraction': outside / area,
                        'max_same_class_gt_iou': float(iou[j].max()) if gt_rles else 0.})
    displayed_union = np.zeros_like(union)
    for mask in masks:
        displayed_union |= mask
    return masks, {'raw_candidates': len(predictions), 'display_candidates': len(shown),
                  'largest_display_masks': sorted(details, key=lambda d: (-d['area'], d['prediction_index']))[:5],
                  'largest_background_dominant_masks': sorted(
                      [d for d in details if d['outside_all_gt_fraction'] > .5],
                      key=lambda d: (-d['area'], d['prediction_index']))[:5],
                  'gt_union_pixels': int(union.sum()), 'display_union_pixels': int(displayed_union.sum()),
                  'display_union_outside_gt_pixels': int((displayed_union & ~union).sum()),
                  'union_precision': float((displayed_union & union).sum() / max(1, displayed_union.sum())),
                  'union_recall': float((displayed_union & union).sum() / max(1, union.sum())),
                  'gt_matched_iou_ge_0_5': int((iou.max(0) >= .5).sum()) if len(rles) and len(gt_rles) else 0}


def sheet(cases, views, columns, cell):
    gap, label, header = 10, 56, 84
    canvas = Image.new('RGB', (len(columns)*(cell+gap)+gap, header+len(cases)*(cell+label+gap)), '#101824')
    draw = ImageDraw.Draw(canvas)
    draw_label(draw, (12, 10), '第一轮验证集 · 固定案例预览 · 非最终测试', size=22, fill='#e8eff6')
    draw_label(draw, (12, 44), 'score ≥ 0.5；RGB / GT / M2 / DA；全部模型均为 epoch 1；颜色不表示跨模型匹配', size=14, fill='#afc5d8')
    for row, (case, view) in enumerate(zip(cases, views)):
        top = header + row*(cell+label+gap)
        for col, name in enumerate(columns):
            left = gap + col*(cell+gap)
            count = case['gt_instances'] if name == 'gt' else case['models'].get(name, {}).get('display_candidates')
            title = TITLES[name] + (f' · {count}' if count is not None else '')
            draw_label(draw, (left, top), title, size=16, fill='#e8eff6')
            draw_label(draw, (left, top+26), f"val[{case['index']}] · {case['image_id']}", size=13, fill='#afc5d8')
            canvas.paste(view[name].resize((cell, cell), Image.Resampling.LANCZOS), (left, top+label))
    return canvas


def build(args):
    require(not args.output.exists(), 'Preview output must be new; no overwrite')
    with CourseArchive(args.archive, 'val') as dataset:
        require(len(dataset) == 2584, 'Full validation split required')
        saved, receipts = {}, {}
        for name in ('mask2former', 'dafusion'):
            predictions, receipt = load(getattr(args, name), name, dataset)
            receipts[name] = receipt
            if predictions is not None:
                saved[name] = predictions
        require(receipts['mask2former']['gt_instances'] == receipts['dafusion']['gt_instances']
                and receipts['mask2former']['gt_area_groups_including_crowd'] == receipts['dafusion']['gt_area_groups_including_crowd'],
                'Validation GT populations differ')
        gt = dataset.coco_ground_truth()
        by_id = defaultdict(list)
        for annotation in gt['annotations']:
            by_id[annotation['image_id']].append(annotation)
        cases, views = [], []
        columns = ['rgb', 'gt'] + list(saved)
        annotation_sha, rgb_sha = hashlib.sha256(), {}
        with ZipFile(args.archive) as archive:
            for member in dataset.annotation_members:
                annotation_sha.update(member.encode()+b'\0'+archive.read(member))
            for index in INDICES:
                record = dataset.images[index]
                image = dataset.load_rgb(index)
                rgb_sha[record['file_name']] = hashlib.sha256(archive.read(record['file_name'])).hexdigest()
                annotations = [a for a in by_id[record['id']] if a['area'] > 0]
                gt_masks = [decode(a['segmentation'], 1024, 1024) for a in annotations]
                gt_rles = [a['segmentation'] for a in annotations]
                case = {'index': index, 'image_id': record['id'], 'file_name': record['file_name'],
                        'gt_instances': len(annotations), 'gt_small_instances': sum(a['area'] <= 1024 for a in annotations),
                        'models': {}}
                view = {'rgb': image, 'gt': overlay(image, gt_masks)}
                for name in saved:
                    masks, case['models'][name] = diagnostics(saved[name][index]['predictions'], gt_masks, gt_rles)
                    view[name] = overlay(image, masks)
                cases.append(case)
                views.append(view)
    result = {'status': 'completed_interim_visualization', 'scope': 'first_epoch_validation_only_not_final_test',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'epoch': 1, 'split': 'val',
              'fixed_indices_zero_based': list(INDICES), 'display_score_threshold': THRESHOLD,
              'evaluation_score_threshold': .01, 'selection': 'Predetermined validation indices; no selection by performance',
              'receipts': receipts, 'cases': cases, 'test_accessed': False, 'new_inference': False,
              'gt_assisted_diagnostics': 'Display-only union overlap and max same-class instance IoU; not AP, matching, or inference filtering',
              'annotation_members_sha256': annotation_sha.hexdigest(), 'selected_rgb_sha256': rgb_sha,
              'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                 (Path(__file__), ROOT/'tools/build_course_dafusion_comparison.py', ROOT/'tools/build_course_pretrained_gallery.py',
                  ROOT/'tools/build_multiscale_gallery.py', ROOT/'rgbd_seg/course_archive.py')}}
    args.output.mkdir(parents=True)
    files = []
    for case, view in zip(cases, views):
        path = args.output / f"case_{case['index']:04d}.png"
        sheet([case], [view], columns, 512).save(path, optimize=True)
        files.append(path)
    contact = args.output / 'contact_sheet.png'
    sheet(cases, views, columns, 300).save(contact, optimize=True)
    files.append(contact)
    result['image_sha256'] = {p.name: sha256(p) for p in files}
    (args.output / 'preview.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    lines = ['# 第一轮验证集预览', '', '**仅为 epoch 1 validation，不是最终 test，也不改变五轮训练计划。**', '',
             '- 固定验证索引：0、400、800、1200、1600、2000；与此前 test 样本不是同一集合。',
             '- 仅从保存的预测绘图；未推理、未使用 GPU、未按效果挑图。展示 score≥0.5；完整指标评估阈值为0.01。',
             '- RGB / GT / Mask2Former epoch1 / DA-Fusion epoch1；每个独立掩码半透明覆盖，保留重叠。',
             '- 单张图的颜色仅区分本图实例；不同模型颜色不表示实例对应。', '',
             '## 完整验证集的既有指标（0–1）', '', '| 模型 | APsmall | AP |', '|---|---:|---:|']
    for name in receipts:
        m = receipts[name]['metrics']
        lines.append(f"| {name} epoch1 | {m['mask_ap_small']:.6f} | {m['mask_ap']:.6f} |")
    lines += ['', '## 固定图像', '', '| 验证索引 | image ID | GT | M2显示 | DA显示 |', '|---:|---:|---:|---:|---:|']
    for case in cases:
        m = case['models']
        lines.append(f"| [{case['index']}](case_{case['index']:04d}.png) | {case['image_id']} | {case['gt_instances']} | {m.get('mask2former',{}).get('display_candidates','缺少原始预测')} | {m['dafusion']['display_candidates']} |")
    lines += ['', '![六个固定验证案例](contact_sheet.png)', '',
              '六例不能推断全验证集表现或纯深度因果。JSON中的GT辅助重叠统计仅用于检查显示错位/背景误检，不是新的评估或候选过滤。',
              '预测文件SHA、完整ID顺序、评估/来源收据及第一轮selection记录均已关联核验；训练中的run/selection按读取时刻保存SHA，不要求整个正式run已完成。',
              '源码没有保存每轮独立checkpoint的固定SHA，因此本预览不捏造该项；第一轮身份由runner的selection条目及val_epoch_01收据关联。', '']
    (args.output / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'status': result['status'], 'output': str(args.output),
                      'cases': [{'index': c['index'], 'image_id': c['image_id'], 'counts': {k:v['display_candidates'] for k,v in c['models'].items()}} for c in cases]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=Path('/data/yinli/workspace/downloads/buaa_image_segmentation/blender.zip'))
    p.add_argument('--dafusion', type=Path, default=ROOT/'reports/course_dafusion/formal_v1/val_epoch_01')
    p.add_argument('--mask2former', type=Path, default=ROOT/'reports/course_finetune/mask2former/val_epoch_01')
    p.add_argument('--output', type=Path, default=ROOT/'reports/course_dafusion/interim_preview_epoch01')
    build(p.parse_args())


if __name__ == '__main__':
    main()
