#!/usr/bin/env python3
"""Check that report illustrations preserve GT and frozen model predictions."""
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from pycocotools import mask as cm
from tools.select_report_figure_cases import digest,write
from tools.build_report_figures_v2 import OUT,FIG,TITLES

def main():
    rows=json.loads((OUT/'selected_cases.json').read_text())
    metrics=list(csv.DictReader((OUT/'case_metrics.csv').open()))
    count=0;ids={r['image_id'] for r in rows};source_checks=[]
    for row in rows:
        iid=row['image_id'];src=OUT/'source_assets'/str(iid);focus=OUT/'focus_assets'/row['case_key']
        anns=json.loads((src/'ground_truth.json').read_text())['annotations']
        a=next(a for a in anns if a['id']==row['annotation_id'])
        gt=cm.decode(a['segmentation']).astype(bool)
        assert np.array_equal(gt,np.asarray(Image.open(focus/'GT.png'))>0)
        assert int(gt.sum())==row['area']
        meta=json.loads((src/'metadata.json').read_text())
        elevation=math.degrees(math.asin(meta['camera']['c2w_blender'][2][2]))
        assert elevation>=80 and abs(elevation-row['view_elevation_deg'])<.001
        assert Image.open(src/'rgb.png').size==(1024,1024)
        raw_depth=np.array(Image.open(src/'depth_raw.png'))
        assert raw_depth.shape==(1024,1024) and raw_depth.dtype==np.uint16
        for stage in ('M1','M2','M3','M4'):
            preds=json.loads((src/f'{stage}_predictions.json').read_text())
            visible=[p for p in preds if p['score']>=.5]
            values=cm.iou([p['segmentation'] for p in visible],[a['segmentation']],[0])
            expected=float(values.max()) if len(visible) else 0.
            metric=next(m for m in metrics if m['case_key']==row['case_key'] and m['stage']==stage)
            assert abs(expected-float(metric['best_iou']))<1e-12
            mask=np.asarray(Image.open(focus/f'{stage}_best.png'))>0
            if metric['prediction_index']:
                p=preds[int(metric['prediction_index'])]
                assert p['score']>=.5
                assert np.array_equal(mask,cm.decode(p['segmentation']).astype(bool))
            else:assert not mask.any()
            display=np.asarray(Image.open(focus/f'{stage}_display.png'))>0
            assert np.array_equal(display,mask if expected>=.5 else np.zeros_like(mask))
            count+=1
    for stage,record in json.loads((OUT/'sources.json').read_text()).items():
        assert digest(record['path'])==record['sha256'];source_checks.append(stage)
    supp=json.loads((OUT/'supplemental_predictions/run.json').read_text())
    assert supp['status']=='completed' and supp['ground_truth_in_inference'] is False
    assert all(r['identical_native_rles'] and r['max_score_difference']<=1e-6 for r in supp['native_reproduction'])
    trace=OUT/'framework_trace';receipt=json.loads((trace/'run.json').read_text())
    assert receipt['identical_to_locked_M4'] and not receipt['ground_truth_in_inference']
    final=json.loads((trace/'final.json').read_text())
    original=json.loads((OUT/'source_assets'/str(receipt['image_id'])/'M4_predictions.json').read_text())
    assert final==original
    merged=json.loads((trace/'merged.json').read_text());point=json.loads((trace/'point.json').read_text())
    assert len(merged)==len(point)==len(final)
    assert [(r['view_id'],r['query_id'],r['score']) for r in merged]==[(r['view_id'],r['query_id'],r['score']) for r in point]
    for name,records in [('merged',merged),('point',point)]:
        prob=np.load(trace/f'{name}_probability_72.npy')
        assert prob.shape==(1024,1024) and np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1
        assert np.array_equal(prob>.5,cm.decode(records[72]['segmentation']).astype(bool))
    exported=[]
    for stem,title in TITLES:
        with Image.open(FIG/f'{stem}.png') as img:
            assert all(abs(d-300)<.1 for d in img.info['dpi'])
            assert img.width>=3000 and img.height>=1500
        svg=ET.parse(FIG/f'{stem}.svg')
        text=''.join(svg.getroot().itertext())
        assert all(str(iid) not in text for iid in ids),'Visible source ID in figure'
        assert (FIG/f'{stem}.pdf').read_bytes()[:4]==b'%PDF'
        exported.append(stem)
    report=dict(status='passed',gt_and_prediction_correspondence_checks=count,near_overhead_cases=len(rows),
        source_prediction_sha256_verified=source_checks,source_images=len(ids),
        framework_trace='identical to locked M4; same candidate identity through real refinement stages',
        probability_map='actual float32 predictions; threshold reproduces stored mask',
        visible_image_ids=False,png_dpi=300,exported=exported,
        missing='M0 final RGB Swin-Large inference; explicitly marked in figure 3')
    write(OUT/'verification.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
