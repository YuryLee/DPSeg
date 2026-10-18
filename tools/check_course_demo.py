#!/usr/bin/env python3
"""Check the real offline demo, including COCO RLE decoding and downloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright
    html = args.demo.read_text(encoding='utf-8')
    payload = json.loads(re.search(r'<script type="application/json" id="course-data">(.*?)</script>', html, re.S)[1])
    args.output.mkdir(parents=True, exist_ok=False)
    errors, external, checked = [], [], 0
    with sync_playwright() as pw:
        executable = shutil.which('google-chrome') or shutil.which('chromium')
        browser = pw.chromium.launch(executable_path=executable, headless=True, args=['--no-sandbox'])
        context = browser.new_context(viewport={'width': 1440, 'height': 1100}, offline=True, accept_downloads=True)
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: external.append(request.url)
                if request.url.startswith(('http://', 'https://')) else None)
        page.goto(args.demo.resolve().as_uri())
        page.wait_for_function('window.__courseDemo?.ready && window.__courseDemo.lastRender !== null')
        for index, case in enumerate(payload['cases']):
            page.select_option('#case', str(index))
            for method, predictions in case['predictions'].items():
                page.select_option('#method', method)
                current = page.evaluate('window.__courseDemo.lastRender')
                assert current['image_id'] == case['image_id'] and current['method'] == method
                assert current['visible'] == sum(row['score'] >= .5 for row in predictions)
                expected = payload['validation']['metrics'][method]
                assert page.locator('#aps').inner_text() == f"{expected['mask_ap_small']:.5f}"
                decoded = page.evaluate('''([index,method]) => window.__courseDemo.data.cases[index].predictions[method]
                    .map(row=>window.__courseDemo.decodeCounts(row.segmentation))''', [index, method])
                assert len(decoded) == len(predictions)
                for row, runs in zip(predictions, decoded):
                    rle = row['segmentation']
                    canonical = mask_utils.frPyObjects({'size': rle['size'], 'counts': runs}, *rle['size'])
                    assert canonical['counts'].decode('ascii') == rle['counts']
                    checked += 1
        page.select_option('#case', '0')
        method = payload['validation']['selected_variant']
        page.select_option('#method', method)
        initial_ap = page.locator('#aps').inner_text()
        page.locator('#threshold').fill('0.1')
        page.locator('#threshold').dispatch_event('input')
        assert page.evaluate('window.__courseDemo.lastRender.visible') == sum(
            row['score'] >= .1 for row in payload['cases'][0]['predictions'][method])
        assert page.locator('#aps').inner_text() == initial_ap
        page.locator('#zoom').fill('2')
        page.locator('#zoom').dispatch_event('input')
        assert page.locator('#prediction').evaluate('element=>element.style.width') == '200%'
        page.click('#reset')
        assert page.locator('#prediction').evaluate('element=>element.style.width') == '100%'

        # Independent one-pixel mask catches a transposed Fortran-to-canvas mapping.
        single = np.zeros((1024, 1024), dtype=np.uint8)
        single[351, 137] = 1
        encoded = mask_utils.encode(np.asfortranarray(single))
        point = {'category_id': 1, 'score': 1., 'segmentation':
                 {'size': [1024, 1024], 'counts': encoded['counts'].decode('ascii')}}
        page.evaluate('''method=>{window.__savedPredictions=window.__courseDemo.data.cases[0].predictions[method];
            window.__courseDemo.data.cases[0].predictions[method]=[];
            document.getElementById('method').dispatchEvent(new Event('change'));}''', method)
        assert page.locator('#count').inner_text() == '0'
        assert '没有实例' in page.locator('#status').inner_text()
        before = page.evaluate('''()=>{const ctx=document.getElementById('prediction').getContext('2d');
            return [Array.from(ctx.getImageData(137,351,1,1).data),Array.from(ctx.getImageData(351,137,1,1).data)];}''')
        page.evaluate('''([method,point])=>{window.__courseDemo.data.cases[0].predictions[method]=[point];
            document.getElementById('method').dispatchEvent(new Event('change'));}''', [method, point])
        after = page.evaluate('''()=>{const ctx=document.getElementById('prediction').getContext('2d');
            return [Array.from(ctx.getImageData(137,351,1,1).data),Array.from(ctx.getImageData(351,137,1,1).data)];}''')
        assert before[0] != after[0] and before[1] == after[1]
        page.evaluate('''method=>{window.__courseDemo.data.cases[0].predictions[method]=window.__savedPredictions;
            document.getElementById('method').dispatchEvent(new Event('change'));}''', method)
        with page.expect_download() as download:
            page.click('#download-json')
        json_path = args.output / 'download_predictions.json'
        download.value.save_as(json_path)
        exported = json.loads(json_path.read_text())
        assert exported['image_id'] == payload['cases'][0]['image_id']
        assert exported['predictions'] == [row for row in payload['cases'][0]['predictions'][method] if row['score'] >= .5]
        with page.expect_download() as download:
            page.click('#download-image')
        png_path = args.output / 'download_overlay.png'
        download.value.save_as(png_path)
        with Image.open(png_path) as image:
            assert image.size == (1024, 1024)
        page.screenshot(path=str(args.output / 'desktop.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_timeout(200)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(args.output / 'mobile.png'), full_page=True)
        assert not errors and not external
        browser.close()
    result = {'status': 'passed', 'cases': len(payload['cases']), 'variants': len(payload['labels']),
              'rle_masks_checked_against_coco_codec': checked, 'one_pixel_canvas_coordinate_check': True,
              'threshold_preserves_validation_metrics': True, 'empty_view_checked': True,
              'json_and_png_downloads_checked': True, 'mobile_no_horizontal_overflow': True,
              'javascript_errors': errors, 'external_page_requests': external,
              'scope': 'offline_demo_functionality_only'}
    (args.output / 'browser_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
