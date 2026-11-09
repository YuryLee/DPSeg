#!/usr/bin/env python3
"""Select traceable qualitative cases from the frozen Swin-Large test outputs.

This is post-hoc figure selection, never a model or threshold tuning procedure.
"""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as cm
from rgbd_seg.course_archive import CourseArchive
from tools.build_course_showcase import text, bounds

ARCHIVE = Path('/data/yinli/workspace/downloads/buaa_image_segmentation/blender.zip')
RUN = ROOT/'reports/course_swinl_depth_no_k_v1/swinl_depth_no_k'
OUT = ROOT/'outputs/report_figures_swinl_v1'
SOURCES = {'M1': RUN/'final_evaluation/test_native/predictions.jsonl',
           'M4': RUN/'final_evaluation/test_full/predictions.jsonl'}
THRESHOLD = .5

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def load_records(path, ids=None):
    result = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if ids is None or row['image_id'] in ids:
                result[row['image_id']] = row['predictions']
    return result

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = CourseArchive(ARCHIVE, split='test')
    gt = data.coco_ground_truth()
    anns = defaultdict(list)
    for ann in gt['annotations']:
        anns[ann['image_id']].append(ann)
    values = {stage: load_records(path) for stage, path in SOURCES.items()}
    pools = defaultdict(list)
    for index, image_id in enumerate(data.image_ids):
        truth = [a for a in anns[image_id] if a['area'] > 0 and not a.get('iscrowd', 0)]
        rles = [a['segmentation'] for a in truth]
        metrics = {}
        for stage in SOURCES:
            pred = values[stage][image_id]
            visible = [i for i, p in enumerate(pred) if p['score'] >= THRESHOLD]
            ious = cm.iou([pred[i]['segmentation'] for i in visible], rles, [0]*len(truth))
            if not visible:
                ious = np.zeros((0, len(truth)))
            metrics[stage] = (visible, ious)
        best = {s: m.max(0) if len(m) else np.zeros(len(truth))
                for s, (_, m) in metrics.items()}
        for j, a in enumerate(truth):
            if a['area'] < 70:
                continue
            before, after = float(best['M1'][j]), float(best['M4'][j])
            row = dict(image_id=image_id, image_index=index, annotation_id=a['id'],
                       area=a['area'], bbox=a['bbox'], bounds=bounds(a['bbox'], minimum=128),
                       iou_M1=before, iou_M4=after,
                       prediction_indices={s: int(v[int(m[:, j].argmax())])
                         if len(m) and m[:, j].max() > 0 else None
                         for s, (v, m) in metrics.items()})
            choices = []
            if a['area'] < 1024 and before < .1 and after > .85:
                choices.append(('recovered', after-before))
            if a['area'] < 1024 and .5 < before < .85 and after > .96:
                choices.append(('boundary', after-before))
            if 200 < a['area'] < 1024 and before > .97 and after > .99:
                choices.append(('good_small', after))
            if 2500 < a['area'] < 8000 and before > .98 and after > .99:
                choices.append(('good_body', after))
            if 150 < a['area'] < 1024 and after < .05:
                choices.append(('unresolved', a['area']/1024))
            if a['area'] < 1500 and before > .75 and after < .5:
                choices.append(('regression', before-after))
            for kind, priority in choices:
                pools[kind].append(dict(row, kind=kind, priority=priority))
        # Background-like high-confidence predictions: use union coverage,
        # not only per-instance IoU, to avoid labelling merged true objects FP.
        vis, ious = metrics['M1']
        fp_ids = [vis[k] for k in range(len(vis))
                  if ious.shape[1] and float(ious[k].max()) < .05
                  and values['M1'][image_id][vis[k]]['score'] > .7]
        if fp_ids:
            gu = cm.decode(cm.merge(rles)).astype(bool)
            out_rles = [values['M4'][image_id][i]['segmentation'] for i in metrics['M4'][0]]
            pu = cm.decode(cm.merge(out_rles)).astype(bool) if out_rles else np.zeros_like(gu)
            for pi in fp_ids:
                p = values['M1'][image_id][pi]
                m = cm.decode(p['segmentation']).astype(bool)
                area = int(m.sum())
                if not 80 <= area <= 2500:
                    continue
                gt_coverage = float((m & gu).sum()/area)
                remaining = float((m & pu).sum()/area)
                if gt_coverage < .05 and remaining < .1:
                    box = cm.toBbox(p['segmentation']).tolist()
                    pools['fp_reduced'].append(dict(kind='fp_reduced', image_id=image_id,
                        image_index=index, prediction_index=pi, area=area, bbox=box,
                        bounds=bounds(box, minimum=128), gt_coverage=gt_coverage,
                        remaining_coverage=remaining, score=p['score'], priority=1-remaining))
        if (index+1) % 500 == 0:
            print(f'scanned {index+1}/{len(data)}', flush=True)
    shortlist = {}
    for kind, rows in pools.items():
        chosen, seen = [], set()
        for row in sorted(rows, key=lambda a: (-a['priority'], a['image_id'])):
            if row['image_id'] in seen:
                continue
            chosen.append(row)
            seen.add(row['image_id'])
            if len(chosen) == 12:
                break
        shortlist[kind] = chosen
        sheet = Image.new('RGB', (4*320, 3*360), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(chosen):
            x, y = (i % 4)*320, (i//4)*360
            rgb = data.load_rgb(row['image_index'])
            crop = rgb.crop(row['bounds']).resize((300, 280))
            sheet.paste(crop, (x+10,y+45))
            text(draw,(x+10,y+5), f"{i+1}. {row['image_id']}", 19)
            if 'iou_M1' in row:
                note = f"A={row['area']}  {row['iou_M1']:.3f} → {row['iou_M4']:.3f}"
            else:
                note = f"FP A={row['area']} rem={row['remaining_coverage']:.3f}"
            text(draw,(x+10,y+330),note,17)
        sheet.save(OUT/f'selection_{kind}.jpg', quality=93)
    write(OUT/'candidate_cases.json', dict(
        split='test', source='historical local course ZIP, not Kaggle hidden labels',
        selection='posthoc qualitative selection, no training or hyperparameter changes',
        score_threshold=THRESHOLD,
        metric='maximum full-image mask IoU with one selected GT, not COCO AP matching',
        sources={s: dict(path=str(p),sha256=digest(p)) for s,p in SOURCES.items()},
        pool_sizes={k:len(v) for k,v in pools.items()}, candidates=shortlist))
    print(json.dumps({k:len(v) for k,v in pools.items()}), flush=True)

if __name__ == '__main__':
    main()
