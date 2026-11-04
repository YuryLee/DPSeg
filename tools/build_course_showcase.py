#!/usr/bin/env python3
"""Build an offline GT/S0-S4 case gallery from saved, scored predictions."""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import io
import json
from functools import lru_cache
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as coco_mask
from scipy.ndimage import binary_erosion
from rgbd_seg.course_archive import CourseArchive
from tools.select_course_showcase import ARCHIVE, LABELS, paths, sha

PAIRS = {'small_baseline': ('S0','S4'), 'small_tiled': ('S2','S3'),
         'success_small': ('S0','S4'), 'success_component': ('S0','S4'),
         'point_boundary': ('S1','S2'), 'probability_boundary': ('S3','S4'),
         'regression': ('S3','S4'), 'unresolved': ('S0','S4'), 'false_positive': ('S0','S1')}
TITLES = {'small_baseline':'小目标漏检：完整方案找回',
          'success_small':'成功案例：小目标轮廓完整',
          'success_component':'成功案例：器件主体分割准确',
          'small_tiled':'小目标漏检：切片补充细节',
          'point_boundary':'边界偏差：深度点头修正',
          'probability_boundary':'边界偏差：概率图深度细化',
          'regression':'局限案例：细化后重合度下降',
          'unresolved':'局限案例：仍未找回的小目标',
          'false_positive':'背景误检：融合后减少'}
GROUPS = {'small_baseline':'漏检', 'small_tiled':'漏检', 'point_boundary':'边界',
          'success_small':'成功', 'success_component':'成功',
          'probability_boundary':'边界', 'regression':'局限', 'unresolved':'局限',
          'false_positive':'误检'}
PALETTE = [(20,184,166),(139,92,246),(245,158,11),(236,72,153),
           (59,130,246),(132,204,22),(249,115,22),(6,182,212)]
FONT = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
LATIN_FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


@lru_cache(maxsize=24)
def font_for(size, latin):
    return ImageFont.truetype(LATIN_FONT if latin else FONT, size)


def text(draw, xy, value, size=17, fill='#34445a'):
    """Use Latin and CJK fonts explicitly; Droid's CJK fallback lacks Latin."""
    x,y = xy
    for ch in str(value):
        font = font_for(size, ord(ch) < 0x2e80)
        draw.text((x,y), ch, font=font, fill=fill)
        x += font.getlength(ch)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def bounds(box, minimum=112):
    x,y,w,h = box
    side = min(1024, max(minimum, int(max(w,h)*1.7)+24))
    left = max(0, min(1024-side, round(x+w/2-side/2)))
    top = max(0, min(1024-side, round(y+h/2-side/2)))
    return [left,top,left+side,top+side]


def decode(rle):
    return coco_mask.decode(rle).astype(bool)


def png(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode('ascii')


def overlay(rgb, masks, colors, *, outlines=False):
    result = np.array(rgb, dtype=np.float32).copy()
    for mask, color in zip(masks, colors):
        if not outlines:
            result[mask] = result[mask]*.52 + np.array(color)*.48
        edge = mask & ~binary_erosion(mask)
        result[edge] = color
    return Image.fromarray(np.uint8(np.clip(result,0,255)))


def load_sources(selection):
    for stage, spec in selection['source_predictions'].items():
        if sha(spec['path']) != spec['sha256']:
            raise ValueError(f'{stage} predictions changed')


def load_records(ids, split='val'):
    records = {}
    for stage,path in paths(split).items():
        records[stage] = {}
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['image_id'] in ids:
                    records[stage][row['image_id']] = row['predictions']
        if set(records[stage]) != set(ids):
            raise ValueError('Missing selected image')
    return records


def enrich(candidate, annotations, records, threshold):
    row = dict(candidate)
    image_id = row['image_id']
    gt = annotations[image_id]
    row['before'], row['after'] = PAIRS[row['kind']]
    row['group'] = GROUPS[row['kind']]
    row['title'] = TITLES[row['kind']]
    row['bounds'] = bounds(row['bbox'])
    if row['kind'] == 'false_positive':
        ref = decode(records['S0'][image_id][row['prediction_index']]['segmentation'])
        gt_union = decode(coco_mask.merge([a['segmentation'] for a in gt])) if gt else np.zeros_like(ref)
        row['gt_coverage'] = float((ref & gt_union).sum()/ref.sum())
        row['remaining_coverage'] = {}
        for stage in ('S1','S4'):
            values = [v['segmentation'] for v in records[stage][image_id] if v['score'] >= threshold]
            union = decode(coco_mask.merge(values)) if values else np.zeros_like(ref)
            row['remaining_coverage'][stage] = float((ref & union).sum()/ref.sum())
        if row['gt_coverage'] > .05 or row['remaining_coverage']['S4'] > .1:
            return None
        if row['remaining_coverage']['S1'] > .1:
            row['after'] = 'S4'
            row['title'] = '背景误检：完整方案减少'
    return row


def target_and_predictions(row, gt, values, threshold):
    if row['kind'] == 'false_positive':
        reference = values['S0'][row['prediction_index']]['segmentation']
        target = decode(coco_mask.merge([a['segmentation'] for a in gt])) if gt else np.zeros((1024,1024),bool)
    else:
        reference = next(a['segmentation'] for a in gt if a['id'] == row['annotation_id'])
        target = decode(reference)
    selected, ious = {}, {}
    for stage, predictions in values.items():
        valid = [i for i,v in enumerate(predictions) if v['score'] >= threshold]
        overlap = (coco_mask.iou([predictions[i]['segmentation'] for i in valid], [reference], [0])[:,0]
                   if valid else np.empty(0))
        winner = valid[int(overlap.argmax())] if len(overlap) and overlap.max() > 0 else None
        selected[stage] = winner
        ious[stage] = float(overlap.max()) if len(overlap) else 0.
    return target, reference, selected, ious


def comparison(row, target, reference, values, selected, stage, threshold):
    """Compare one instance, or all predictions inside the fixed FP region."""
    if row['kind'] == 'false_positive':
        rles = [p['segmentation'] for p in values[stage] if p['score'] >= threshold]
        pred = decode(coco_mask.merge(rles)) if rles else np.zeros_like(target)
        region = decode(reference)
        return target & region, pred & region
    pred = (decode(values[stage][selected[stage]]['segmentation'])
            if selected[stage] is not None else np.zeros_like(target))
    return target, pred


def export_assets(folder, data, row, rgb, colored, gt_all, gt, values, threshold):
    """Export original bytes and lossless independent masks (overlaps retained)."""
    folder.mkdir(parents=True)
    metadata = data.images[row['image_index']]
    with ZipFile(ARCHIVE) as archive:
        (folder/'rgb.png').write_bytes(archive.read(metadata['file_name']))
        (folder/'depth_raw.png').write_bytes(archive.read(metadata['depth_file_name']))
    Image.fromarray(colored).save(folder/'depth_preview.png')
    write(folder/'image_metadata.json',metadata)
    write(folder/'gt_coco.json',dict(images=[metadata],annotations=gt_all,categories=data.categories))
    (folder/'gt_masks').mkdir()
    for ann in gt_all:
        Image.fromarray(decode(ann['segmentation']).astype(np.uint8)*255).save(
            folder/'gt_masks'/f"ann_{ann['id']}.png")
    masks = [decode(a['segmentation']) for a in gt]
    colors = [PALETTE[i%len(PALETTE)] for i in range(len(gt))]
    overlay(rgb,masks,colors).save(folder/'gt_overlay.png')
    color_map = np.zeros((1024,1024,3),dtype=np.uint8)
    for mask,color in zip(masks,colors):
        color_map[mask] = color
    Image.fromarray(color_map).save(folder/'gt_instances_color.png')
    for stage,preds in values.items():
        stage_dir = folder/'predictions'/stage
        (stage_dir/'masks').mkdir(parents=True)
        write(stage_dir/'predictions.json',dict(image_id=row['image_id'],predictions=preds))
        matrix = coco_mask.iou([p['segmentation'] for p in preds],
                              [a['segmentation'] for a in gt],[0]*len(gt))
        visible_masks,visible_colors=[],[]
        for j,pred in enumerate(preds):
            m = decode(pred['segmentation'])
            Image.fromarray(m.astype(np.uint8)*255).save(stage_dir/'masks'/f'pred_{j:03d}.png')
            if pred['score'] >= threshold:
                color = PALETTE[int(matrix[j].argmax())%len(PALETTE)] if len(gt) and matrix[j].max()>=.1 else (245,158,11)
                visible_masks.append(m);visible_colors.append(color)
        overlay(rgb,visible_masks,visible_colors).save(stage_dir/'overlay.png')
        color_map[:] = 0
        for m,color in zip(visible_masks,visible_colors):
            color_map[m] = color
        Image.fromarray(color_map).save(stage_dir/'instances_color.png')
    target,ref,selected,ious=target_and_predictions(row,gt,values,threshold)
    (folder/'focus').mkdir()
    Image.fromarray(decode(ref).astype(np.uint8)*255).save(folder/'focus'/'reference_mask.png')
    write(folder/'focus'/'selection.json',dict(row,selected_prediction_indices=selected,mask_ious=ious))
    for stage,idx in selected.items():
        m=decode(values[stage][idx]['segmentation']) if idx is not None else np.zeros((1024,1024),bool)
        Image.fromarray(m.astype(np.uint8)*255).save(folder/'focus'/f'{stage}_best_overlap_mask.png')


def preview_row(row, data, gt, values, threshold, width=256):
    rgb = data.load_rgb(row['image_index'])
    target, reference, selected, _ = target_and_predictions(row,gt,values,threshold)
    l,t,r,b = row['bounds']
    crop = np.array(rgb)[t:b,l:r]
    tiles = [Image.fromarray(crop), overlay(crop,[target[t:b,l:r]],[(20,184,166)])]
    for stage in (row['before'],row['after']):
        truth,pred = comparison(row,target,reference,values,selected,stage,threshold)
        truth,pred = truth[t:b,l:r],pred[t:b,l:r]
        colors = [(20,184,166),(244,63,94),(245,158,11)]
        tiles.append(overlay(crop,[truth & pred,truth & ~pred,pred & ~truth],colors))
    canvas = Image.new('RGB',(4*width, width+72),'#f5f7fa')
    d = ImageDraw.Draw(canvas)
    detail = (f"area {row['gt_area']} | IoU {row['ious'][row['before']]:.3f} → {row['ious'][row['after']]:.3f}"
              if row['kind'] != 'false_positive'
              else f"score {row['score']:.3f} | GT cover {row['gt_coverage']:.3f} | remaining {row['remaining_coverage'][row['after']]:.3f}")
    text(d,(8,3),f"{row['image_id']} | {row.get('annotation_id',row.get('prediction_index'))} | {detail}",15)
    for i,(tile,label) in enumerate(zip(tiles,['RGB','GT',row['before']+' 误差',row['after']+' 误差'])):
        text(d,(i*width+8,27),label,15)
        canvas.paste(tile.resize((width,width),Image.Resampling.NEAREST),(i*width,60))
    return canvas


def shortlist(selection, output):
    output.mkdir(parents=True,exist_ok=False)
    load_sources(selection)
    considered = []
    for kind, candidates in selection['candidates'].items():
        seen = set()
        for row in candidates:
            if row['image_id'] in seen:
                continue
            considered.append(row)
            seen.add(row['image_id'])
            if len(seen) >= (55 if kind == 'false_positive' else 10):
                break
    records = load_records({r['image_id'] for r in considered},selection['split'])
    with CourseArchive(ARCHIVE,selection['split']) as data:
        annotations = defaultdict(list)
        for ann in data.coco_ground_truth()['annotations']:
            if ann['area'] > 0 and not ann.get('iscrowd',0):
                annotations[ann['image_id']].append(ann)
        accepted = defaultdict(list)
        for candidate in considered:
            row = enrich(candidate, annotations, records, selection['display_threshold'])
            if row is not None:
                accepted[row['kind']].append(row)
        for kind, rows in accepted.items():
            rows[:] = rows[:8]
            tiles = [preview_row(row,data,annotations[row['image_id']],
                     {s:values[row['image_id']] for s,values in records.items()},
                     selection['display_threshold']) for row in rows]
            sheet = Image.new('RGB',(1024,328*len(tiles)),'white')
            for i,tile in enumerate(tiles):
                sheet.paste(tile,(0,i*328))
            sheet.save(output / f'{kind}.png')
        write(output / 'shortlist.json',dict(threshold=selection['display_threshold'],cases=accepted))
    print(json.dumps({k:len(v) for k,v in accepted.items()}))


def build(selection, choices, output):
    output.mkdir(parents=True,exist_ok=False)
    (output/'figures').mkdir()
    load_sources(selection)
    records = load_records({r['image_id'] for r in choices},selection['split'])
    payload = dict(split=selection['split'],threshold=selection['display_threshold'],
                   labels={f'S{i}':v for i,v in enumerate(LABELS)},cases=[])
    with CourseArchive(ARCHIVE,selection['split']) as data:
        annotations, all_annotations = defaultdict(list), defaultdict(list)
        for ann in data.coco_ground_truth()['annotations']:
            all_annotations[ann['image_id']].append(ann)
            if ann['area'] > 0 and not ann.get('iscrowd',0):
                annotations[ann['image_id']].append(ann)
        for n,row in enumerate(choices):
            row = dict(row)
            image_id = row['image_id']
            gt = annotations[image_id]
            values = {stage:table[image_id] for stage,table in records.items()}
            target, reference, selected, ious = target_and_predictions(row,gt,values,payload['threshold'])
            rgb = data.load_rgb(row['image_index'])
            depth,valid = data.load_depth(row['image_index'])
            from matplotlib import colormaps
            low,high = np.percentile(depth[valid],[2,98])
            scaled = np.clip((depth-low)/max(high-low,1e-8),0,1)
            colored = np.uint8(colormaps['viridis'](scaled)[...,:3]*255)
            colored[~valid] = 0
            case = dict(row,id=f'case-{n+1:02d}',rgb=png(rgb),
                        depth=png(Image.fromarray(colored)),ground_truth=gt,
                        reference_rle=reference, predictions={})
            for stage,preds in values.items():
                matrix = (coco_mask.iou([v['segmentation'] for v in preds],
                                      [a['segmentation'] for a in gt],[0]*len(gt))
                          if preds and gt else np.zeros((len(preds),len(gt))))
                focus = (coco_mask.iou([v['segmentation'] for v in preds],[reference],[0])[:,0]
                         if preds else [])
                case['predictions'][stage] = []
                for j,pred in enumerate(preds):
                    matched = int(matrix[j].argmax()) if len(gt) and matrix[j].max() >= .1 else None
                    case['predictions'][stage].append(dict(pred,prediction_index=j,
                        matched_gt_index=matched, focus_iou=float(focus[j]),
                        best_gt_iou=float(matrix[j].max()) if len(gt) else 0.))
            case['default_selected_indices'] = selected
            payload['cases'].append(case)
            export_assets(output/'cases'/f"{case['id']}_{image_id}",data,row,rgb,colored,
                          all_annotations[image_id],gt,values,payload['threshold'])
            # Large, lossless figure for slides: RGB + GT + all five stages.
            l,t,r,b = row['bounds'];crop=np.array(rgb)[t:b,l:r]
            width=320
            fig=Image.new('RGB',(7*width, width+185),'white')
            draw=ImageDraw.Draw(fig)
            text(draw,(12,8),row['title']+f"  ·  image_id {image_id}",22)
            images=[Image.fromarray(crop),overlay(crop,[target[t:b,l:r]],[(20,184,166)])]
            for stage in payload['labels']:
                truth,pred=comparison(row,target,reference,values,selected,stage,payload['threshold'])
                truth,pred=truth[t:b,l:r],pred[t:b,l:r]
                images.append(overlay(crop,[truth & pred,truth & ~pred,pred & ~truth],
                                      [(20,184,166),(244,63,94),(245,158,11)]))
            for j,(tile,label) in enumerate(zip(images,['RGB','GT',*payload['labels'].values()])):
                text(draw,(j*width+10,43),label,17)
                fig.paste(tile.resize((width,width),Image.Resampling.NEAREST),(j*width,78))
                if j>=2 and row['kind']!='false_positive':
                    text(draw,(j*width+10,402),f"IoU {ious[f'S{j-2}']:.3f}",17)
            for k,(label,color) in enumerate(zip(['与 GT 重合','GT 中有、预测缺失','预测多出的像素'],
                                              [(20,184,166),(244,63,94),(245,158,11)])):
                x=12+k*300
                draw.rectangle((x,444,x+15,459),fill=color)
                text(draw,(x+24,440),label,18)
            note = '仅原误检区域参与误差着色' if row['kind']=='false_positive' else f"GT 可见面积 {row['gt_area']} px²；局部最佳 Mask IoU，不是 AP"
            text(draw,(12,474),f"本地 test 归档 · 展示置信度 ≥ {payload['threshold']:.2f} · {note}",17)
            fig.save(output/'figures'/f"{case['id']}_{row['kind']}.png")
    raw=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    safe=raw.replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    template=(ROOT/'web/showcase.html').read_text()
    (output/'index.html').write_text(template.replace('__SHOWCASE_DATA__',safe))
    write(output/'cases.json',payload)
    write(output/'provenance.json',dict(status='completed',split=selection['split'],cases=len(choices),
        images=[r['image_id'] for r in choices], selection='post-hoc qualitative cases, manually inspected',
        prediction_sources=selection['source_predictions'], gt_source=str(ARCHIVE),
        source_sha256={p:sha(ROOT/p) for p in ['tools/build_course_showcase.py','tools/select_course_showcase.py','web/showcase.html']},
        gt_and_predictions_unmodified=True, inference_performed=False,
        default_display_threshold=payload['threshold'],
        masks='Original COCO compressed RLE; crop and color affect display only',
        images_are_synthetic_course_renders=True))
    print(json.dumps({'completed':str(output/'index.html'),'cases':len(choices)}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--choices',type=Path)
    args=parser.parse_args()
    selection=json.loads(args.selection.read_text())
    if args.choices:
        build(selection,json.loads(args.choices.read_text()),args.output)
    else:
        shortlist(selection,args.output)
