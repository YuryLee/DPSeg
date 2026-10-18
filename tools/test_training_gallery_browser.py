#!/usr/bin/env python3
"""Check the completed static training comparison in a real local browser."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:33051')
    parser.add_argument('--file', type=Path, help='Verify the generated local HTML without any network or live server')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/v5/browser')
    args = parser.parse_args()
    url = args.url.rstrip('/')
    if not args.file and urlsplit(url).hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Only a local demo server is supported')
    args.output.mkdir(parents=True, exist_ok=True)
    errors, cases = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server', '--disable-gpu'])
        try:
            page = browser.new_page(viewport={'width': 1600, 'height': 1150})
            page.on('pageerror', lambda error: errors.append(str(error)))
            if args.file:
                page.route('http://**/*', lambda route: route.abort())
                page.route('https://**/*', lambda route: route.abort())
                target = args.file.resolve().as_uri()
                page.goto(target, wait_until='networkidle')
            else:
                page.goto(url, wait_until='networkidle')
                assert page.locator('a[href="/training-methods"]').count() == 1
                info = page.request.get(url + '/api/info')
                assert info.status == 200
                target = url + '/training-methods'
                response = page.goto(target, wait_until='networkidle')
                assert response.status == 200
            assert page.locator('#case option').count() == 6
            assert '在线默认模型保持不变' in page.locator('#verdict').inner_text()
            options = page.locator('#case option').evaluate_all('nodes => nodes.map(n => n.value)')
            images_checked = 0
            for value in options:
                page.locator('#case').select_option(value)
                page.wait_for_function('Array.from(document.images).every(x => x.complete && x.naturalWidth > 0)')
                assert page.locator('#full .card').count() == 6
                count = page.locator('#details .detail').count()
                assert page.locator('#details .card').count() == count * 6
                images_checked += page.locator('img').count()
                cases.append({'case': value, 'small_gt': count,
                              'description': page.locator('#case-info').inner_text()})
            assert sum(c['small_gt'] for c in cases) == 13
            for scope, expected in [('graspnet/val', 4), ('graspnet/test', 1), ('tless/test', 1), ('all', 6)]:
                page.locator('#scope').select_option(scope)
                assert page.locator('#case option').count() == expected
            page.locator('#zoom').evaluate("node => { node.value = 2; node.dispatchEvent(new Event('input', {bubbles:true})); }")
            assert page.locator('#zoom-value').inner_text() == '2.00×'
            assert page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--zoom').trim()") == '2'
            page.locator('#reset').click()
            assert page.locator('#zoom-value').inner_text() == '1.00×'
            with page.expect_download() as download:
                page.locator('#download').click()
            export = args.output / 'gallery_export.json'
            download.value.save_as(str(export))
            metadata = json.loads(export.read_text())
            assert len(metadata['cases']) == 6 and len(metadata['evaluation_receipts']) == 28
            assert 'embedded_images' not in metadata
            page.screenshot(path=str(args.output / 'training_comparison.png'), full_page=True)
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            page.screenshot(path=str(args.output / 'training_comparison_mobile.png'))
            assert not errors, errors
        finally:
            browser.close()
    report = {'status': 'passed', 'url': target, 'mode': 'offline_file' if args.file else 'live_server', 'cases': cases,
              'small_gt': 13, 'images_checked': images_checked, 'js_errors': errors,
              'checks': [('local HTML with HTTP(S) requests blocked' if args.file else 'main demo link and API'), 'six fixed cases and all decoded images',
                         'all thirteen small GT diagnostics', 'all scope filters', 'zoom and reset',
                         'metadata download with all28 evaluation receipts', 'mobile no horizontal overflow']}
    (args.output / 'browser_test.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
