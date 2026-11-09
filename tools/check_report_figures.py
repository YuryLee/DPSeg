#!/usr/bin/env python3
"""Check source-mask integrity, local metrics and report export metadata."""
from __future__ import annotations
import json
from pathlib import Path
import sys
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from pycocotools import mask as cm
from tools.select_report_figure_cases import OUT, digest, write

def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    supplemental=manifest['supplemental_inference']
    assert supplemental['status']=='completed' and supplemental['tf32'] is False
    for row in supplemental['native_reproduction']:
        assert row['identical_native_rles'] and row['max_score_difference']==0.
    records=0
    for folder in sorted((OUT/'source_assets').iterdir()):
        gt=json.loads((folder/'gt_coco.json').read_text())
        focus=json.loads((folder/'focus.json').read_text())
        annotation=next(a for a in gt['annotations'] if a['id']==focus['annotation_id'])
        target=cm.decode(annotation['segmentation']).astype(bool)
        assert target.shape==(1024,1024)
        assert np.array_equal(target,np.array(Image.open(folder/'focus_gt_mask.png'))>0)
        assert int(target.sum())==focus['gt_mask_area_pixels']==focus['area']
        l,t,r,b=focus['bounds']
        assert 0<=l<r<=1024 and 0<=t<b<=1024 and (r-l)==(b-t)
        for stage in ('M1','M2','M3','M4'):
            preds=json.loads((folder/f'{stage}_predictions.json').read_text())['predictions']
            assert len(preds)<=100
            for p in preds:
                assert p['category_id']==1 and p['segmentation']['size']==[1024,1024]
                assert 0<=p['score']<=1
            ids=[i for i,p in enumerate(preds) if p['score']>=.5]
            ious=cm.iou([preds[i]['segmentation'] for i in ids],[annotation['segmentation']],[0])
            value=float(ious.max()) if ids else 0.
            selection=focus['stages'][stage]
            assert abs(value-selection['mask_iou'])<1e-12
            index=selection['prediction_index']
            expected=cm.decode(preds[index]['segmentation']).astype(bool) if index is not None else np.zeros_like(target)
            assert np.array_equal(expected,np.array(Image.open(folder/f'focus_{stage}_mask.png'))>0)
            rendered=np.array(Image.open(folder/f'focus_{stage}_display_mask.png'))>0
            assert np.array_equal(rendered,expected if value>=.5 else np.zeros_like(target))
            records+=1
    for relative,expected in manifest['figures'].items():
        path=OUT/relative
        assert digest(path)==expected['sha256']
        if path.suffix=='.png':
            with Image.open(path) as img:
                assert img.width>=3000 and min(img.size)>=2000
                assert all(abs(v-300)<.1 for v in img.info['dpi'])
    ElementTree.parse(OUT/'figures/fig02_framework.svg')
    for stage,spec in manifest['source_predictions'].items():
        assert digest(spec['path'])==spec['sha256'],stage
    for name,expected in manifest['generation_scripts'].items():
        assert digest(ROOT/name)==expected,name
    package=OUT.parent/'DPSeg_report_figures_SwinL_v1.zip'
    with ZipFile(package) as archive:
        assert archive.testzip() is None
    result=dict(status='passed',cases=records//4,stage_case_pairs=records,
        native_outputs_reproduced_exactly=len(supplemental['native_reproduction']),
        binary_masks_equal_to_original_rle=True,local_ious_recomputed=True,
        display_correspondence_rule='maximum mask IoU >= 0.5; unmatched displayed empty',
        exports='5 PNGs at 300 dpi, 5 PDFs, 1 editable SVG',zip_integrity='passed')
    write(OUT/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
