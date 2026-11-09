#!/usr/bin/env python3
"""Select near-overhead (85 degree elevation) examples for report figures.

Camera pose is used for illustration selection only, never model inference.
"""
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as cm
from rgbd_seg.course_archive import CourseArchive
from tools.select_report_figure_cases import ARCHIVE, SOURCES, load_records, digest, write
from tools.build_course_showcase import text, bounds

OUT=ROOT/'outputs/report_figures_swinl_v2'

def main():
    OUT.mkdir(exist_ok=True)
    with CourseArchive(ARCHIVE,'test') as data:
        annotations=defaultdict(list)
        for a in data.coco_ground_truth()['annotations']:
            annotations[a['image_id']].append(a)
        records={s:load_records(p) for s,p in SOURCES.items()}
        pool=defaultdict(list)
        for index,image in enumerate(data.images):
            elevation=math.degrees(math.asin(image['camera']['c2w_blender'][2][2]))
            if elevation<80:
                continue
            iid=image['id'];gt=[a for a in annotations[iid] if a['area']>0 and not a.get('iscrowd',0)]
            best={}
            for stage in ('M1','M4'):
                pred=[p for p in records[stage][iid] if p['score']>=.5]
                matrix=cm.iou([p['segmentation'] for p in pred],[a['segmentation'] for a in gt],[0]*len(gt))
                best[stage]=matrix.max(0) if pred else np.zeros(len(gt))
            for j,a in enumerate(gt):
                before,after=float(best['M1'][j]),float(best['M4'][j]);area=a['area']
                if not 100<=area<1024:
                    continue
                choices=[]
                if before<.25 and after>.8:
                    choices.append(('recovered',after-before+.1*area/1024))
                if .5<=before<.85 and after>.94:
                    choices.append(('boundary',after-before+.08*area/1024))
                if before>.92 and after>.98 and area>400:
                    choices.append(('good_small',after+.01*area/1024))
                if after<.1 and area>300:
                    choices.append(('unresolved',area/1024))
                if before>.75 and after<.5:
                    choices.append(('regression',before-after))
                for kind,priority in choices:
                    pool[kind].append(dict(kind=kind,priority=priority,image_id=iid,image_index=index,
                        annotation_id=a['id'],area=area,bbox=a['bbox'],bounds=bounds(a['bbox'],minimum=112),
                        iou_M1=before,iou_M4=after,view_elevation_deg=round(elevation,3),
                        composition=image['composition'],density=image['density']))
        selections={}
        for kind,rows in pool.items():
            chosen=[];seen=set()
            for r in sorted(rows,key=lambda r:-r['priority']):
                if r['image_id'] in seen:continue
                chosen.append(r);seen.add(r['image_id'])
                if len(chosen)>=12:break
            selections[kind]=chosen
            sheet=Image.new('RGB',(4*340,3*390),'white');draw=ImageDraw.Draw(sheet)
            for i,r in enumerate(chosen):
                x,y=(i%4)*340,(i//4)*390
                rgb=data.load_rgb(r['image_index'])
                sheet.paste(rgb.crop(r['bounds']).resize((310,310)),(x+15,y+40))
                text(draw,(x+12,y+4),f"{i+1}. {r['image_id']}",19)
                text(draw,(x+12,y+355),f"A={r['area']}  {r['iou_M1']:.3f} → {r['iou_M4']:.3f}",17)
            sheet.save(OUT/f'contact_{kind}.jpg',quality=94)
        grouped=defaultdict(dict)
        for kind in ('recovered','boundary'):
            for r in pool[kind]:
                old=grouped[r['image_id']].get(kind)
                if old is None or r['priority']>old['priority']:grouped[r['image_id']][kind]=r
        teaser=sorted([dict(image_id=i,**v) for i,v in grouped.items() if len(v)==2],
                     key=lambda r:-(r['recovered']['priority']+r['boundary']['priority']))
        write(OUT/'candidate_cases.json',dict(selection='posthoc, near-overhead visual examples only',
            score_threshold=.5,elevation_min_deg=80,all_candidates=pool,shortlist=selections,teaser_pairs=teaser,
            source_predictions={s:dict(path=str(p),sha256=digest(p)) for s,p in SOURCES.items()}))
        print('Pools', {k:len(v) for k,v in pool.items()})
        print('Teaser image pairs',[(r['image_id'],r['recovered']['area'],r['boundary']['area']) for r in teaser[:12]])

if __name__=='__main__':main()
