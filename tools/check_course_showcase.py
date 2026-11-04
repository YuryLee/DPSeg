#!/usr/bin/env python3
"""Verify exported original assets, browser mask decoding, metrics and controls."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from pycocotools import mask as cm
from tools.build_course_showcase import load_records, decode
from tools.select_course_showcase import ARCHIVE, sha
from rgbd_seg.course_archive import CourseArchive


def digest(array):
    return hashlib.sha256(array.astype(np.uint8).tobytes()).hexdigest()


def expected_metrics(case, threshold):
    output={}
    ref=decode(case['reference_rle'])
    for stage,predictions in case['predictions'].items():
        rles=[p['segmentation'] for p in predictions if p['score'] >= threshold]
        if case['kind']=='false_positive':
            union=decode(cm.merge(rles)) if rles else np.zeros_like(ref)
            output[stage]=float((ref&union).sum()/ref.sum())
        else:
            output[stage]=float(cm.iou(rles,[case['reference_rle']],[0]).max()) if rles else 0.
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    folder=args.demo.parent
    data=json.loads((folder/'cases.json').read_text())
    receipt=json.loads((folder/'provenance.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    for item in receipt['prediction_sources'].values():
        assert sha(item['path'])==item['sha256']
    sources=load_records({c['image_id'] for c in data['cases']},data['split'])
    checked_png=0
    with CourseArchive(ARCHIVE,data['split']) as archive, ZipFile(ARCHIVE) as raw:
        gt=defaultdict(list)
        for ann in archive.coco_ground_truth()['annotations']:
            gt[ann['image_id']].append(ann)
        for c in data['cases']:
            base=folder/'cases'/f"{c['id']}_{c['image_id']}"
            metadata=archive.images[c['image_index']]
            assert json.loads((base/'image_metadata.json').read_text())==metadata
            assert (base/'rgb.png').read_bytes()==raw.read(metadata['file_name'])
            assert (base/'depth_raw.png').read_bytes()==raw.read(metadata['depth_file_name'])
            assert json.loads((base/'gt_coco.json').read_text())['annotations']==gt[c['image_id']]
            assert c['ground_truth']==[a for a in gt[c['image_id']] if a['area']>0 and not a.get('iscrowd',0)]
            for ann in gt[c['image_id']]:
                exported=np.array(Image.open(base/'gt_masks'/f"ann_{ann['id']}.png"))
                assert np.array_equal(exported,cm.decode(ann['segmentation'])*255)
                checked_png+=1
            for stage,preds in c['predictions'].items():
                original=sources[stage][c['image_id']]
                assert [{k:p[k] for k in ('category_id','score','segmentation')} for p in preds]==original
                stage_dir=base/'predictions'/stage
                assert json.loads((stage_dir/'predictions.json').read_text())['predictions']==original
                for j,p in enumerate(preds):
                    exported=np.array(Image.open(stage_dir/'masks'/f'pred_{j:03d}.png'))
                    assert np.array_equal(exported,cm.decode(p['segmentation'])*255)
                    checked_png+=1
    print(json.dumps({'original_assets_verified':checked_png}),flush=True)
    from playwright.sync_api import sync_playwright
    errors,external=[],[]
    checked_browser=0
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=shutil.which('google-chrome') or shutil.which('chromium'),
                                   headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport={'width':1600,'height':1100},offline=True,accept_downloads=True)
        page=context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:external.append(r.url) if r.url.startswith(('http://','https://')) else None)
        page.goto(args.demo.resolve().as_uri())
        page.wait_for_function('window.__showcase?.ready && window.__showcase.lastRender',timeout=60000)
        for i,c in enumerate(data['cases']):
            page.click(f'.case-button[data-index="{i}"]')
            current=page.evaluate('window.__showcase.lastRender')
            assert current['image_id']==c['image_id']
            assert current['before']==c['before'] and current['after']==c['after']
            for stage,v in expected_metrics(c,.5).items():
                assert abs(current['metrics'][stage]-v)<1e-12
            # SHA256 of all independently decoded crop pixels, not just mask areas.
            rles=[a['segmentation'] for a in c['ground_truth']]
            for preds in c['predictions'].values():
                rles.extend(p['segmentation'] for p in preds)
            l,t,r,b=c['bounds']
            expected=[digest(cm.decode(m)[t:b,l:r]) for m in rles]
            actual=page.evaluate('''async ([index,box])=>{
              const c=window.__showcase.data.cases[index];
              const rles=[...c.ground_truth.map(a=>a.segmentation),
                ...Object.values(c.predictions).flat().map(p=>p.segmentation)];
              return Promise.all(rles.map(async r=>{
                const bytes=window.__showcase.decode(r,box);
                const hash=await crypto.subtle.digest('SHA-256',bytes);
                return Array.from(new Uint8Array(hash),n=>n.toString(16).padStart(2,'0')).join('');
              }));
            }''',[i,c['bounds']])
            assert actual==expected
            checked_browser+=len(rles)
            if c['kind']=='false_positive':
                # Suppressed FP region must not mark unrelated nearby GT as missed.
                outside=page.evaluate('''()=>{
                  const d=window.__showcase,c=d.data.cases.find(c=>c.id===d.lastRender.case_id);
                  const m=d.decode(c.reference_rle,c.bounds);
                  const a=document.getElementById('rgb').getContext('2d').getImageData(0,0,c.bounds[2]-c.bounds[0],c.bounds[3]-c.bounds[1]).data;
                  const b=document.getElementById('prediction-after').getContext('2d').getImageData(0,0,c.bounds[2]-c.bounds[0],c.bounds[3]-c.bounds[1]).data;
                  return m.every((v,i)=>v||[0,1,2].every(k=>a[4*i+k]===b[4*i+k]));
                }''')
                assert outside
        page.click('.case-button[data-index="0"]')
        for value in (.01,.99):
            page.locator('#threshold').fill(str(value))
            page.locator('#threshold').dispatch_event('input')
            current=page.evaluate('window.__showcase.lastRender')
            for stage,v in expected_metrics(data['cases'][0],value).items():
                assert abs(current['metrics'][stage]-v)<1e-12
        for stage in data['labels']:
            page.select_option('#after',stage)
            assert page.evaluate('window.__showcase.lastRender.after')==stage
        page.click('#whole')
        page.uncheck('#focus')
        assert page.locator('#gt').evaluate('c=>c.width')==1024
        for mode in ('overlay','outline','mask','error'):
            page.select_option('#mode',mode)
            assert page.evaluate('window.__showcase.lastRender.mode')==mode
        page.click('#source-switch')
        assert '深度' in page.locator('#source-title').inner_text()
        page.click('#reset')
        # Single pixel away from the diagonal catches transposed coordinates.
        m=np.zeros((1024,1024),np.uint8);m[351,137]=1
        rle=cm.encode(np.asfortranarray(m));rle['counts']=rle['counts'].decode('ascii')
        coords=page.evaluate('''r=>{const m=window.__showcase.decode(r,[100,300,200,400]);
          return Array.from(m).flatMap((v,i)=>v?[i]:[])}''',rle)
        assert coords==[51*100+37]
        with page.expect_download() as download:page.click('#download-json')
        download.value.save_as(args.output/'download_masks.json')
        exported=json.loads((args.output/'download_masks.json').read_text())
        for stage,preds in exported['predictions'].items():
            assert preds==sources[stage][data['cases'][0]['image_id']]
        with page.expect_download() as download:page.click('#download-png')
        download.value.save_as(args.output/'download_comparison.png')
        assert Image.open(args.output/'download_comparison.png').size==(2112,670)
        page.click('#whole');page.uncheck('#focus');page.select_option('#mode','overlay')
        page.screenshot(path=str(args.output/'desktop.png'),full_page=True)
        page.click('.case-button[data-index="9"]')
        page.screenshot(path=str(args.output/'boundary_detail.png'),full_page=True)
        page.get_by_role('button',name='成功',exact=True).click()
        assert page.locator('.case-button').count()==sum(c['group']=='成功' for c in data['cases'])
        page.get_by_role('button',name='全部',exact=True).click()
        page.click('.case-button[data-index="0"]')
        page.locator('#title').click();page.keyboard.press('ArrowRight')
        assert page.evaluate('window.__showcase.lastRender.case_id')==data['cases'][1]['id']
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(args.output/'mobile.png'),full_page=True)
        assert not errors and not external
        browser.close()
    result=dict(status='passed',cases=len(data['cases']),variants=len(data['labels']),
        original_rgb_depth_gt_and_predictions_verified=True,exported_binary_masks_checked=checked_png,
        browser_roi_mask_sha256_checked=checked_browser,one_pixel_coordinate_check=True,
        per_case_metrics_checked=True,false_positive_region_isolated=True,
        controls_downloads_mobile_offline_checked=True,javascript_errors=errors,external_requests=external)
    (args.output/'check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
