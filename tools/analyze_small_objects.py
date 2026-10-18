#!/usr/bin/env python3
"""Audit every small original-resolution GT instance; crops are display-only."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch
from rgbd_seg.data import RGBDDataset, dataset_fingerprint
from rgbd_seg.instance_metrics import _boundary, _near_boundary
from rgbd_seg.model import load_model


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def mask_iou_matrix(gt_masks, pred_masks, gt_labels, pred_labels):
    """Same-class IoU for full-image binary masks; empty pairs have IoU zero."""
    gt_masks, pred_masks = np.asarray(gt_masks, bool), np.asarray(pred_masks, bool)
    if gt_masks.ndim != 3 or pred_masks.ndim != 3 or gt_masks.shape[1:] != pred_masks.shape[1:]:
        raise ValueError('Full-resolution GT and prediction masks must have matching HxW')
    if len(gt_labels) != len(gt_masks) or len(pred_labels) != len(pred_masks):
        raise ValueError('Mask and class counts differ')
    ga, pa = gt_masks.sum((1, 2)), pred_masks.sum((1, 2))
    ious = np.zeros((len(gt_masks), len(pred_masks)), dtype=float)
    for index, mask in enumerate(gt_masks):
        intersection = pred_masks[:, mask].sum(1)
        union = ga[index] + pa - intersection
        ious[index] = np.divide(intersection, union, out=np.zeros(len(pa), dtype=float), where=union > 0)
        ious[index, np.asarray(pred_labels) != gt_labels[index]] = 0
    return ious


def assign_masks(ious, scores, score_threshold):
    """Sum-IoU Hungarian assignment across ALL GT objects, not only small ones."""
    keep = np.flatnonzero(np.asarray(scores) >= score_threshold)
    matches = {}
    if ious.shape[0] and len(keep):
        rows, cols = linear_sum_assignment(-ious[:, keep])
        for row, col in zip(rows, cols):
            prediction_index = int(keep[col])
            if ious[row, prediction_index] > 0:
                matches[int(row)] = prediction_index
    return matches


def pair_diagnostics(gt, prediction, tolerance):
    """Full-mask region and contour diagnostics; values 0–1 except pixel areas."""
    gt, prediction = np.asarray(gt, bool), np.asarray(prediction, bool)
    ga, pa = int(gt.sum()), int(prediction.sum())
    intersection = int((gt & prediction).sum())
    union = ga + pa - intersection
    gb, pb = _boundary(gt), _boundary(prediction)
    bp = float((pb & _near_boundary(gb, tolerance)).sum()/pb.sum()) if pb.any() else 0.0
    br = float((gb & _near_boundary(pb, tolerance)).sum()/gb.sum()) if gb.any() else 0.0
    return {'iou': intersection/union if union else None,
            'dice_f1': 2*intersection/(ga+pa) if ga+pa else None,
            'region_precision': intersection/pa if pa else 0.0,
            'region_recall': intersection/ga if ga else None,
            'pred_area_pixels': pa, 'signed_area_error_pixels': pa-ga,
            'boundary_precision': bp, 'boundary_recall': br,
            'boundary_f1': 2*bp*br/(bp+br) if bp+br else 0.0,
            'boundary_tolerance_pixels': tolerance}


def classify_error(best_iou, best_retained_iou, matched_iou):
    if matched_iou >= .75:
        return 'matched_iou_ge075'
    if matched_iou >= .5:
        return 'matched_iou_050_to_075'
    if best_retained_iou >= .5:
        return 'one_to_one_assignment_competition'
    if best_iou >= .5:
        return 'adequate_mask_below_display_score_threshold'
    if best_iou > 0:
        return 'insufficient_mask_overlap'
    return 'no_overlapping_same_class_candidate'


def inspect_prediction(target, prediction, raw_instance_ids, max_area=1024, score_threshold=.5):
    gt = target['masks'].cpu().numpy().astype(bool)
    pred = prediction['masks'].cpu().numpy()
    if pred.ndim == 4:
        pred = pred[:, 0]
    scores = prediction['scores'].cpu().numpy()
    if pred.ndim != 3 or len(pred) != len(scores) or not np.isfinite(pred).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid prediction masks/scores')
    pred = pred >= .5
    labels = prediction['labels'].cpu().numpy()
    if len(raw_instance_ids) != len(gt):
        raise ValueError('Raw instance IDs and target masks differ')
    ious = mask_iou_matrix(gt, pred, target['labels'].cpu().numpy(), labels)
    assignments = assign_masks(ious, scores, score_threshold)
    radius = math.ceil(.003*math.hypot(*gt.shape[1:]))
    rows = []
    for index, mask in enumerate(gt):
        area = int(mask.sum())
        if not 0 < area <= max_area:
            continue
        best = int(np.argmax(ious[index])) if len(pred) and ious[index].max() > 0 else None
        retained = np.flatnonzero(scores >= score_threshold)
        best_retained_iou = float(ious[index, retained].max()) if len(retained) else 0.0
        assigned = assignments.get(index)
        assigned_mask = pred[assigned] if assigned is not None else np.zeros_like(mask)
        details = pair_diagnostics(mask, assigned_mask, radius)
        y, x = np.nonzero(mask)
        best_iou = float(ious[index, best]) if best is not None else 0.0
        rows.append({'target_index': index, 'original_instance_id': int(raw_instance_ids[index]),
                     'class_index': int(target['labels'][index]), 'gt_area_pixels': area,
                     'gt_bbox_xyxy': [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)],
                     'best_candidate_index_any_score': best, 'best_candidate_iou_any_score': best_iou,
                     'best_candidate_score': float(scores[best]) if best is not None else None,
                     'best_retained_candidate_iou': best_retained_iou,
                     'assigned_candidate_index': assigned, 'assigned_score': float(scores[assigned]) if assigned is not None else None,
                     'assigned_pair': details,
                     'assigned_boundary_f1_at_1px': pair_diagnostics(mask, assigned_mask, 1)['boundary_f1'],
                     'error_category': classify_error(best_iou, best_retained_iou, details['iou'])})
    return rows, gt, pred


def save_crop(path, rgb, gt, pred, row):
    x1, y1, x2, y2 = row['gt_bbox_xyxy']
    h, w = gt.shape
    x1, y1, x2, y2 = max(0, x1-32), max(0, y1-32), min(w, x2+32), min(h, y2+32)
    crop = rgb[y1:y2, x1:x2]
    panels = [Image.fromarray(crop)]
    captions = ['Original RGB']
    for key, label in [('best_candidate_index_any_score', 'Best candidate (any score)'), ('assigned_candidate_index', 'Assigned at score >= 0.5')]:
        candidate = row[key]
        pm = pred[candidate] if candidate is not None else np.zeros_like(gt)
        overlay = rgb.astype(float).copy()
        overlay[gt] = .60*overlay[gt]+.40*np.array([0, 255, 80])
        overlay[pm] = .60*overlay[pm]+.40*np.array([255, 0, 190])
        overlay[_boundary(gt)] = [0, 255, 80]
        overlay[_boundary(pm)] = [255, 0, 190]
        panels.append(Image.fromarray(overlay[y1:y2, x1:x2].astype(np.uint8)))
        captions.append(label)
    scale = 3
    panel_width = max(210, panels[0].width*scale)
    panel_height = panels[0].height*scale
    canvas = Image.new('RGB', (3*panel_width, panel_height+70), (22, 28, 36))
    draw = ImageDraw.Draw(canvas)
    for i, (panel, caption) in enumerate(zip(panels, captions)):
        panel = panel.resize((panel.width*scale, panel.height*scale), Image.Resampling.NEAREST)
        canvas.paste(panel, (i*panel_width, 30))
        draw.text((i*panel_width+4, 6), caption, fill='white')
    draw.text((4, panel_height+36), f"GT area {row['gt_area_pixels']}px | assigned IoU {row['assigned_pair']['iou']:.3f} | green GT / magenta prediction", fill='white')
    canvas.save(path)
    return [x1, y1, x2, y2]



def render_gallery(result):
    names = {
        'matched_iou_ge075': '匹配良好：IoU≥0.75',
        'matched_iou_050_to_075': '已匹配，掩码仍有偏差：IoU在0.5至0.75之间',
        'one_to_one_assignment_competition': '一对一匹配竞争',
        'adequate_mask_below_display_score_threshold': '存在IoU≥0.5的候选，但分数低于显示阈值',
        'insufficient_mask_overlap': '最佳候选与GT重合不足',
        'no_overlapping_same_class_candidate': '没有与GT重合的同类候选',
    }
    cards = []
    for tag, entry in result['models'].items():
        for row in entry['instances']:
            score = row['best_candidate_score']
            score_text = f'{score:.3f}' if score is not None else '无候选'
            label = f"{tag} · {'验证' if row['split']=='val' else '测试'} · {row['sample_id']} · 原始ID {row['original_instance_id']}"
            detail = f"GT面积 {row['gt_area_pixels']} 像素；最佳候选 IoU {row['best_candidate_iou_any_score']:.3f}，分数 {score_text}；显示阈值0.5下一对一匹配 IoU {row['assigned_pair']['iou']:.3f}，Dice {row['assigned_pair']['dice_f1']:.3f}。"
            filename = html.escape(row['crop'], quote=True)
            cards.append(f'<section><h2>{html.escape(label)}</h2><p>{html.escape(names[row["error_category"]])}</p><p>{html.escape(detail)}</p><a href="{filename}" target="_blank" rel="noopener"><img src="{filename}" alt="小目标GT和预测掩码对照；点击放大"></a></section>')
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>全部原始小目标诊断</title><style>body{background:#161c24;color:#eef2f7;font:15px system-ui;margin:24px;line-height:1.55}img{max-width:100%;height:auto}h2{font-size:16px;overflow-wrap:anywhere}section{border-top:1px solid #546070;padding:12px 0}a{color:#8acbff}</style><h1>全部原始小目标诊断</h1><p>包含原始掩码面积1–1024像素的全部标注。绿色为GT，品红为预测；依次为原图、最佳候选、显示阈值0.5下一对一分配的候选。点击图片可放大。先做全图推理与匹配，再裁剪展示。</p><p>低分候选没有显示，不等于在所有置信度阈值下完全漏检。这份逐例诊断不是AP；同一物体的不同视角相关，不能当作独立重复。</p>'+''.join(cards)+'</html>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/graspnet_expanded_v2')
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--splits', nargs='+', choices=('val', 'test'), default=['val', 'test'])
    parser.add_argument('--output', required=True, help='Directory for JSON, all small-object crops, and HTML index')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {'dataset_sha256': dataset_fingerprint(args.data), 'protocol': {
        'selection': 'Every GT object with original mask area >0 and <=1024 pixels. Only frames containing such GT are inferred, on the full original image; no result-based case selection.',
        'area_boundary_note': 'COCOeval small area includes 1024 exactly; the project audit uses <1024. Objects exactly1024 are explicitly counted here.',
        'matching': 'Same-class sum-IoU Hungarian one-to-one matching against ALL GT in the full image, using predictions with score>=0.5. Overlapping predicted masks are kept. Positive-IoU assignments only.',
        'best_candidate': 'Independent maximum same-class IoU among all model-emitted candidates, including scores below0.5; diagnostic only, may reuse a candidate across GT.',
        'mask_threshold': .5, 'score_threshold': .5,
        'region_dice_f1': '2*intersection/(GT area+prediction area), computed on original-resolution full masks; unmatched nonempty GT has Dice0.',
        'boundary': 'DAVIS-style pixel-transition contour; matched pair F1 at ceil(0.003*original-image diagonal), plus1pixel sensitivity diagnostic. Boundary metrics are full-image, not measured on display crops.',
        'limitations': 'Not COCO AP matching or a published benchmark summary. A one-pixel/relative tolerance can be large relative to tiny objects. Views of an object are correlated; do not treat crops as independent replicates. No causal error attribution.',
        'crop_legend': 'RGB; GT green + best candidate magenta; GT green + assigned candidate magenta.32pixel GT-box margin,3x nearest display enlargement; contours and metrics computed before cropping.'}, 'splits': {}, 'models': {}}
    prepared = {}
    for split in args.splits:
        ds = RGBDDataset(args.data, split)
        selected = []
        gt_count, small_count, exact1024 = 0, 0, 0
        for index in range(len(ds)):
            sample, target = ds[index]
            areas = target['area']
            gt_count += len(areas)
            small_count += int(((areas > 0) & (areas <= 1024)).sum())
            exact1024 += int((areas == 1024).sum())
            if ((areas > 0) & (areas <= 1024)).any():
                raw = np.asarray(Image.open(ds.samples[index]['instance_path']))
                ids = np.unique(raw); ids = ids[ids != 0]
                selected.append((index, sample, target, ids))
        result['splits'][split] = {'total_images': len(ds), 'gt_instances': gt_count, 'small_instances': small_count,
                                   'area_exactly1024_instances': exact1024, 'inferred_images': len(selected),
                                   'inferred_sample_ids': [ds.samples[x[0]]['id'] for x in selected]}
        prepared[split] = (ds, selected)
    seen = set()
    for path in args.checkpoints:
        model, checkpoint = load_model(path, args.device)
        tag = f"{checkpoint['config']['mode']}_seed{checkpoint['config']['seed']}"
        if tag in seen:
            raise ValueError(f'Duplicate mode/seed identifier: {tag}')
        seen.add(tag)
        model_result = {'checkpoint': str(Path(path).resolve()), 'checkpoint_sha256': sha256(path), 'config': checkpoint['config'], 'instances': []}
        with torch.inference_mode():
            for split, (ds, selected) in prepared.items():
                if checkpoint['classes'] != ds.classes:
                    raise ValueError('Checkpoint/dataset classes differ')
                for index, sample, target, ids in selected:
                    prediction = model([{key: value.to(args.device) for key, value in sample.items()}])[0]
                    prediction = {key: value.cpu() for key, value in prediction.items()}
                    rows, gt, pred = inspect_prediction(target, prediction, ids)
                    rgb = (sample['image'].permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
                    for row in rows:
                        row.update({'split': split, 'sample_id': ds.samples[index]['id'], 'original_image_hw': list(rgb.shape[:2])})
                        if ds.manifest['provenance'] == 'public_graspnet_subset':
                            row['graspnet_object_index'] = row['original_instance_id']-1
                        filename = f"{tag}_{split}_{row['sample_id']}_id{row['original_instance_id']}.png"
                        row['display_crop_xyxy'] = save_crop(output/filename, rgb, gt[row['target_index']], pred, row)
                        row['crop'] = filename
                        model_result['instances'].append(row)
        result['models'][tag] = model_result
        del model, checkpoint
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    (output/'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    (output/'index.html').write_text(render_gallery(result))
    print(json.dumps({'output': str(output.resolve()), 'splits': result['splits'], 'models': list(result['models'])}, indent=2))


if __name__ == '__main__':
    main()
