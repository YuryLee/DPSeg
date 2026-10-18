#!/usr/bin/env python3
"""Render fixed first-view T-LESS cases from saved binary prediction RLEs only."""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
import torch
from rgbd_seg.data import RGBDDataset, dataset_fingerprint
from rgbd_seg.render import depth_image, overlay, rgb_image


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def main():
    torch.set_num_threads(2)
    data = ROOT / 'data/tless_confirmation'
    base = ROOT / 'reports/v3/tless'
    output = base / 'gallery'
    output.mkdir(exist_ok=True)
    dataset = RGBDDataset(data, 'test')
    fingerprint = dataset_fingerprint(data)
    dataset_ids = {record['id'] for record in dataset.samples}
    sources = [data/'manifest.json', ROOT/'reports/v3/tless_frozen_selection.json',
               ROOT/'reports/v3/tless_data_integrity.json', Path(__file__).resolve(),
               ROOT/'rgbd_seg/data.py', ROOT/'rgbd_seg/render.py']
    predictions, evaluations = {}, {}
    for mode in ('E0', 'E2'):
        prediction_file, evaluation_file = base/mode/'predictions_rle.json', base/mode/'evaluation.json'
        sources.extend([prediction_file, evaluation_file])
        payload = json.loads(prediction_file.read_text())
        evaluation = json.loads(evaluation_file.read_text())
        assert payload['dataset_sha256'] == evaluation['dataset_sha256'] == fingerprint
        assert sha256(prediction_file) == evaluation['predictions_sha256']
        assert payload['checkpoint_sha256'] == evaluation['checkpoint_sha256']
        assert evaluation['manifest_sha256'] == sha256(data/'manifest.json')
        assert payload['mask_threshold'] == evaluation['protocol']['mask_threshold'] == .5
        assert evaluation['protocol']['candidate_score_threshold'] == .01
        assert evaluation['config']['mode'] == mode
        records = payload['samples']
        assert len(records) == len(dataset) == len({r['sample_id'] for r in records})
        assert {r['sample_id'] for r in records} == dataset_ids
        predictions[mode] = {r['sample_id']: r['instances'] for r in records}
        evaluations[mode] = evaluation

    # This fixed data-only rule never inspects predictions or metrics.
    first = {}
    for index, record in enumerate(dataset.samples):
        scene = int(record['scene_id'])
        key = (int(record['source_frame']), record['id'])
        if scene not in first or key < first[scene][0]:
            first[scene] = (key, index)
    assert len(first) == 20
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 25)
    except OSError:
        font = ImageFont.load_default(size=25)
    cases, cards = [], []
    for scene, (_, index) in sorted(first.items()):
        record = dataset.samples[index]
        sample, target = dataset[index]
        rgb = rgb_image(sample)
        width, height = rgb.size
        gt_masks = target['masks'].numpy()
        panels = [('RGB', rgb), ('Depth (per-image scale)', depth_image(sample)),
                  (f'GT: {len(gt_masks)} visible instances', overlay(rgb, gt_masks))]
        count, retained_scores = {}, {}
        for mode in ('E0', 'E2'):
            masks, scores = [], []
            for instance in predictions[mode][record['id']]:
                if instance['score'] < .5:
                    continue
                assert instance['label'] == 1
                encoded = dict(instance['rle'])
                encoded['counts'] = encoded['counts'].encode('ascii')
                decoded = mask_utils.decode(encoded)
                assert decoded.shape == (height, width)
                masks.append(decoded)
                scores.append(instance['score'])
            count[mode], retained_scores[mode] = len(masks), scores
            panels.append((f'{mode}: {len(masks)} predictions (score >= .50)', overlay(rgb, masks)))
        board = Image.new('RGB', (5*width, height+52), '#eef2f7')
        draw = ImageDraw.Draw(board)
        for column, (label, panel) in enumerate(panels):
            x = column*width
            board.paste(panel, (x, 52))
            draw.text((x+15, 12), label, font=font, fill='#192b3c')
            if column:
                draw.line((x, 0, x, height+52), fill='#ffffff', width=3)
        filename = f'scene_{scene:06d}_frame_{record["source_frame"]:06d}.png'
        board.save(output/filename)
        small = int(((target['area'] > 0) & (target['area'] <= 1024)).sum())
        case = {'scene_id': scene, 'sample_id': record['id'], 'source_frame': record['source_frame'],
                'gt_instances': len(gt_masks), 'gt_coco_small_instances': small,
                'displayed_predictions': count, 'displayed_scores': retained_scores,
                'panel_file': filename, 'panel_sha256': sha256(output/filename),
                'rgb_sha256': sha256(record['rgb_path']), 'depth_sha256': sha256(record['depth_path']),
                'gt_mask_sha256': {relative(p):sha256(p) for p in record['instance_mask_paths']}}
        cases.append(case)
        caption = (f'场景 {scene:02d} · 原始帧 {record["source_frame"]:06d} · '
                   f'GT {len(gt_masks)} 个（small {small}） · '
                   f'E0 显示 {count["E0"]} 个 · E2 显示 {count["E2"]} 个')
        cards.append(f'<figure id="scene-{scene}"><figcaption>{html.escape(caption)}</figcaption>'
                     f'<a class="panel-link" href="{filename}" aria-label="放大场景 {scene} 的五列对比图">'
                     f'<img class="case-panel" src="{filename}" width="{board.width}" height="{board.height}" '
                     f'alt="场景 {scene}，依次为 RGB、深度、可见 GT、E0、E2" loading="eager"></a>'
                     f'<p class="case-id">{html.escape(record["id"])} · <a href="{filename}">打开原图</a></p></figure>')

    summary = []
    for mode in ('E0', 'E2'):
        metrics = evaluations[mode]['metrics']
        summary.append(f'<tr><th scope="row">{mode}</th><td>{metrics["mask_ap"]:.2f}</td>'
                       f'<td>{metrics["mask_ap_small"]:.2f}</td><td>{metrics["mask_ap50"]:.2f}</td></tr>')
    page = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>T-LESS 跨数据集检查 · 固定场景案例</title><style>
*{box-sizing:border-box}html{background:#eef2f7;color:#1b2b3b;font:16px/1.7 system-ui,sans-serif}
body{margin:0}main{max-width:1900px;margin:auto;padding:32px}h1{font-size:clamp(1.55rem,3vw,2.4rem);line-height:1.25}
h2{font-size:1.2rem}p{max-width:1000px}a{color:#076b76;text-underline-offset:3px}
.eyebrow{font-size:.85rem;letter-spacing:.12em;color:#426879}.note{background:#fff;padding:20px;border-left:4px solid #168282;border-radius:8px}
.note p{margin:.4em 0}table{border-collapse:collapse;max-width:620px;width:100%;margin:20px 0;background:white}
th,td{text-align:left;border-bottom:1px solid #d5dfe9;padding:8px 14px;font-variant-numeric:tabular-nums}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:24px 0}nav a{padding:3px 10px;border:1px solid #b9ccd3;border-radius:6px}
figure{min-width:0;margin:24px 0;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 6px #223d5010}
figcaption{padding:14px 18px;font-weight:650;font-size:.96rem}.panel-link{display:block;min-width:0}
.case-panel{display:block;width:100%;height:auto;max-width:100%}.case-id{padding:0 18px 12px;margin:10px 0 0;font-size:.84rem;color:#516574;overflow-wrap:anywhere}
footer{font-size:.9rem;margin:32px 0;color:#526471;overflow-wrap:anywhere}
@media(max-width:600px){main{padding:16px}figcaption{padding:12px;font-size:.9rem}.note{padding:14px}th,td{padding:7px 10px}figure{margin:18px 0}}
</style></head><body><main>
<p class="eyebrow">FROZEN-WEIGHT CROSS-DATASET CHECK</p><h1>T-LESS：固定场景案例</h1>
<p>直接使用 GraspNet 训练完成的 E0（RGB）与 E2（RGB-D）权重。这里没有训练、选权重或调整阈值；结果与 GraspNet 分开报告。</p>
<div class="note"><p><strong>固定选择：</strong>100 张冻结测试图中，每个场景取原始帧号最小的一张，共 20 张。选择不依据模型表现；点击图片可打开完整的五列对比。</p>
<p><strong>阅读顺序：</strong>RGB → 深度 → 可见 GT → E0 → E2。深度按每张图有效值的 1%–99% 分位区间着色，仅用于显示；真实深度输入仍以米计。</p>
<p><strong>阈值：</strong>页面仅显示分数 ≥0.50 的预测；COCO 使用分数阈值 0.01 的候选，两者的二值掩码阈值均为 0.50。图中少量高分预测不能替代完整 AP 结果。</p>
<p><strong>真值：</strong>GT 彩色叠加仅供可视化。真实 COCO 评测保留全部独立、可能重叠的可见掩码；不是 amodal（含遮挡部分）标签。颜色按各自实例顺序循环，不表示跨列实例匹配。</p></div>
<h2>全部 100 张图的 COCO 指标（0–100）</h2>
<table><thead><tr><th>模型</th><th>AP</th><th>AP small</th><th>AP50</th></tr></thead><tbody>__SUMMARY__</tbody></table>
<p>共有 688 个非空可见实例，其中 52 次 small 观察来自 9 个场景、12 个物体模型身份；19 次 small 的可见比例不足 10%。关联视角不构成独立重复试验；这些指标也不是 BOP 官方榜单分数。</p>
<nav aria-label="按场景跳转">__NAV__</nav>__CARDS__
<footer><p>本页直接解码已保存的二值预测 RLE，没有再次运行模型。文件哈希、固定样本清单与显示计数见 <a href="provenance.json">可复核来源记录</a>。</p>
<p>完整评测：<a href="../E0/evaluation.json">E0</a> · <a href="../E2/evaluation.json">E2</a> · <a href="../../tless_confirmation_data.md">数据、许可与选择协议</a>。</p>
<p>T-LESS 数据 © Hodan、Haluza、Obdrzalek、Matas、Lourakis、Zabulis，CC BY 4.0。原始来源：<a href="https://cmp.felk.cvut.cz/t-less/">T-LESS</a> / <a href="https://bop.felk.cvut.cz/datasets/">BOP</a>；图像叠加与拼图由本项目生成。</p></footer>
</main></body></html>'''
    page = page.replace('__SUMMARY__', ''.join(summary)).replace('__CARDS__', '\n'.join(cards))
    page = page.replace('__NAV__', ''.join(f'<a href="#scene-{s}">{s:02d}</a>' for s in sorted(first)))
    (output/'index.html').write_text(page, encoding='utf-8')
    receipt = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'dataset_sha256':fingerprint, 'model_inference_performed':False,
               'selection':'One sample per scene: minimum (source_frame, sample_id) in the frozen test set; fixed before rendering.',
               'cases':cases, 'case_count':len(cases), 'source_sha256':{relative(p):sha256(p) for p in sources},
               'html_sha256':sha256(output/'index.html'), 'display_score_threshold':.5,
               'coco_candidate_score_threshold':.01, 'mask_threshold':.5,
               'rendering':'GT and predictions use render.overlay; GT overlaps are blended sequentially for display only. Evaluation retains separate masks.',
               'depth_rendering':'render.depth_image: per-image 1st–99th quantiles over valid depth; invalid pixels dark.',
               'evaluation_metrics':{mode: evaluations[mode]['metrics'] for mode in evaluations},
               'checkpoint_sha256':{mode:evaluations[mode]['checkpoint_sha256'] for mode in evaluations}}
    (output/'provenance.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'gallery':relative(output/'index.html'), 'cases':len(cases), 'dataset_sha256':fingerprint}))


if __name__ == '__main__':
    main()
