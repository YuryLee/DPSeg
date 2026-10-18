#!/usr/bin/env python3
"""Exercise a running demo with manifest-selected inputs and real Chromium."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright
from pycocotools import mask as mask_utils


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Base directory for relative paths')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--data', type=Path, default=Path('data/synthetic'))
    parser.add_argument('--server', type=Path, default=Path('reports/server.json'), help='Server JSON containing url')
    parser.add_argument('--server-url', help='Explicit running local server URL; overrides --server')
    parser.add_argument('--output', type=Path, default=Path('reports'), help='Screenshots, exported JSON and test report')
    parser.add_argument('--browser', default='/usr/bin/google-chrome')
    parser.add_argument('--timeout-ms', type=int, default=120000)
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda path: path.resolve() if path.is_absolute() else (root / path).resolve()
    data, output = resolve(args.data), resolve(args.output)
    manifest = json.loads((data / 'manifest.json').read_text())
    cases = [sample for sample in manifest['samples'] if sample['split'] == args.split and sample.get('instances')]
    if not cases:
        raise ValueError('Browser test requires an annotated sample in the selected split')
    sample = cases[0]
    files = {key: (data / sample[key]).resolve() for key in ('rgb', 'depth', 'instances')}
    if any(not path.is_relative_to(data) or not path.is_file() for path in files.values()):
        raise ValueError('Manifest input paths must be files inside the selected dataset')
    url = args.server_url or json.loads(resolve(args.server).read_text())['url']
    parsed = urlsplit(url)
    if parsed.scheme not in ('http', 'https') or parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Browser tests accept a local running demo URL only')
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    exported_instances = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.browser, headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server'])
        try:
            page = browser.new_page(viewport={'width': 1500, 'height': 1250}, device_scale_factor=1)
            page.set_default_timeout(args.timeout_ms)
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(url, wait_until='networkidle')
            page.wait_for_function("!document.querySelector('#run').disabled")
            server_info = page.request.get(url.rstrip('/') + '/api/info').json()
            assert server_info.get('split', 'val') == args.split, 'Server split differs from browser test split'
            page.locator('#sample').select_option(sample['id'])

            def infer():
                page.locator('#run').click()
                page.wait_for_function("document.querySelector('#status').textContent.includes('推理完成') || document.querySelector('#status').classList.contains('error')")
                assert not page.locator('#status').evaluate("node => node.classList.contains('error')"), page.locator('#status').inner_text()

            infer()
            assert page.locator('.image-frame img:visible').count() == 6
            page.screenshot(path=str(output / 'demo_screenshot.png'), full_page=True)
            page.locator('#frame-E3').click()
            assert float(page.locator('#zoom').input_value()) > 1
            with page.expect_download() as download:
                page.locator('#download').click()
            download.value.save_as(str(output / 'browser_result.json'))
            result = json.loads((output / 'browser_result.json').read_text())
            assert result['provenance'] == manifest['provenance']
            assert result.get('split', 'val') == args.split
            assert result['input_source'] == sample['id']
            assert set(result['predictions']) == {'E0', 'E2', 'E3'}
            for predictions in result['predictions'].values():
                for instance in predictions:
                    rle = instance['mask_rle']
                    assert rle['size'] == [result['height'], result['width']]
                    decoded = mask_utils.decode({'size': rle['size'], 'counts': rle['counts'].encode('ascii')})
                    assert int(decoded.sum()) == instance['area']
                    exported_instances += 1
            # Zero confident detections are valid for small exploratory models.
            page.locator('#upload-tab').click()
            assert page.locator('#download').is_disabled()
            page.locator('#rgb').set_input_files(str(files['rgb']))
            page.locator('#depth').set_input_files(str(files['depth']))
            page.locator('#depth-scale').fill(str(manifest['depth_scale']))
            infer()
            assert page.locator('.image-frame img:visible').count() == 5
            page.locator('#gt').set_input_files(str(files['instances']))
            infer()
            assert page.locator('.image-frame img:visible').count() == 6
            page.locator('#depth').set_input_files(str(files['rgb']))
            page.locator('#run').click()
            page.wait_for_function("document.querySelector('#status').classList.contains('error')")
            assert page.locator('#download').is_disabled()
            assert page.locator('.image-frame img:visible').count() == 0
            page.set_viewport_size({'width': 390, 'height': 844})
            page.screenshot(path=str(output / 'demo_mobile.png'), full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            assert not errors, errors
        finally:
            browser.close()
    report = {'status': 'passed', 'provenance': manifest['provenance'], 'dataset': str(data), 'split': args.split,
        'sample_id': sample['id'], 'server_url': url, 'exported_instances_checked': exported_instances,
        'checks': ['sample real inference, six panels', 'synchronized zoom', 'JSON export; every available RLE decoded and area checked',
                   'upload without GT', 'upload with GT', 'invalid depth rejected', 'stale result cleared',
                   'mobile no horizontal overflow', 'no JS page errors']}
    (output / 'browser_test.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
