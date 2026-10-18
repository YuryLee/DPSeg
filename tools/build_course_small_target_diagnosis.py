#!/usr/bin/env python3
"""Write a validation-only diagnostic report and an explicitly selected example."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'

import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycocotools import mask as masks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_seg.course_archive import CourseArchive
from tools.summarize_course_finetune import read, sha256, check_hash, require


def rle(value):
    return {'size': value['size'], 'counts': value['counts'].encode('ascii')}


def main():
    base = ROOT / 'reports/course_small_target'
    diag_path = base / 'validation_diagnostics/diagnostics.json'
    geometry_path = base / 'validation_geometry/geometry.json'
    diagnostic, geometry = read(diag_path), read(geometry_path)
    require(diagnostic['status'] == geometry['status'] == 'pass', 'Complete diagnoses required')
    require(geometry['source_diagnosis_sha256'] == sha256(diag_path), 'Diagnosis binding differs')
    figure_path = base / 'validation_geometry/failure_example.png'
    case_path = base / 'validation_geometry/failure_example.json'
    report_path = base / 'validation_diagnostics/report.md'
    for path in (figure_path, case_path, report_path):
        require(not path.exists(), 'Refusing to overwrite diagnostic artifacts')
    per_gt = {}
    for name in ('original', 'sampling'):
        path = base / f'validation_geometry/{name}_per_gt.jsonl'
        check_hash(path, geometry['models'][name]['per_gt_sha256'])
        per_gt[name] = {(r['image_id'], r['annotation_id']): r
                        for line in path.read_text().splitlines() if (r := json.loads(line))}
    original, sampling = per_gt['original'], per_gt['sampling']
    require(original.keys() == sampling.keys(), 'Per-GT coverage differs')
    eligible = [key for key in sorted(original)
                if 65 <= original[key]['gt_area'] <= 256 and original[key]['best_iou'] >= .7
                and 0 < sampling[key]['best_iou'] < .25
                and 2 <= sampling[key]['best_prediction_area'] / sampling[key]['gt_area'] <= 10]
    require(bool(eligible), 'No example satisfies the declared rule')
    image_id, annotation_id = eligible[0]
    protocol = read(base / 'protocol.json')
    with CourseArchive(protocol['archive'], 'val') as dataset:
        index = dataset.image_ids.index(image_id)
        ann = next(a for a in dataset.coco_ground_truth()['annotations'] if a['id'] == annotation_id)
        require(ann['image_id'] == image_id, 'Selected GT image differs')
        gt = masks.decode(rle(ann['segmentation'])).astype(bool)
        rgb = np.asarray(dataset.load_rgb(index))
        binary = {'gt': gt}
        details = {}
        for name in ('original', 'sampling'):
            model = diagnostic['models'][name]
            path = Path(model['predictions_path'])
            check_hash(path, model['predictions_sha256'])
            with path.open() as stream:
                record = next(r for line in stream if (r := json.loads(line))['image_id'] == image_id)
            preds = record['predictions']
            ious = masks.iou([rle(p['segmentation']) for p in preds], [rle(ann['segmentation'])], [0])
            selected = int(ious[:, 0].argmax())
            pred = preds[selected]
            binary[name] = masks.decode(rle(pred['segmentation'])).astype(bool)
            expected = per_gt[name][(image_id, annotation_id)]
            require(abs(float(ious[selected, 0]) - expected['best_iou']) < 1e-12, 'Example IoU differs')
            require(int(binary[name].sum()) == expected['best_prediction_area'], 'Example area differs')
            details[name] = {**expected, 'saved_prediction_index': selected, 'score': pred['score'],
                             'intersection_pixels': int((binary[name] & gt).sum()),
                             'outside_gt_pixels': int((binary[name] & ~gt).sum()),
                             'missed_gt_pixels': int((~binary[name] & gt).sum()),
                             'predictions_sha256': model['predictions_sha256']}
        union = binary['gt'] | binary['original'] | binary['sampling']
        ys, xs = np.where(union)
        side = max(64, int(max(xs.max() - xs.min(), ys.max() - ys.min())) + 17)
        side = min(side, 1024)
        left = min(max(0, int((xs.min() + xs.max() - side) // 2)), 1024 - side)
        top = min(max(0, int((ys.min() + ys.max() - side) // 2)), 1024 - side)
        crop = (slice(top, top + side), slice(left, left + side))
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
        titles = ['RGB crop', f'Ground truth: {int(gt.sum())} px',
                  f'Original: {details["original"]["best_prediction_area"]} px\nIoU {details["original"]["best_iou"]:.4f}',
                  f'Sampling: {details["sampling"]["best_prediction_area"]} px\nIoU {details["sampling"]["best_iou"]:.4f}']
        colors = {'gt': (0.1, .75, .35), 'original': (.12, .4, .8), 'sampling': (.9, .35, .1)}
        for ax, key, title in zip(axes, ('rgb', 'gt', 'original', 'sampling'), titles):
            ax.imshow(rgb[crop], interpolation='nearest')
            if key != 'rgb':
                overlay = np.zeros((side, side, 4))
                overlay[..., :3] = colors[key]
                overlay[..., 3] = binary[key][crop] * .5
                ax.imshow(overlay, interpolation='nearest')
                if key != 'gt':
                    ax.contour(gt[crop], levels=[.5], colors=[colors['gt']], linewidths=.9)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        fig.suptitle(f'Validation image {image_id}, GT {annotation_id}: native-pixel crop', fontsize=12)
        fig.text(.5, .04, 'Illustrative failure selected by a stated rule; green outline = GT. Not a representative sample.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .08, 1, .91))
        fig.savefig(figure_path, dpi=200)
        plt.close(fig)
    example = {'image_id': image_id, 'annotation_id': annotation_id, 'split': 'val',
               'selection_rule': 'First sorted(image_id,annotation_id) among GT area65..256, original bestIoU>=0.7, sampling bestIoU in(0,0.25), sampling area/GT in[2,10]',
               'eligible_examples': len(eligible), 'selection_is_posthoc': True,
               'crop_xywh': [left, top, side, side], 'details': details,
               'geometry_sha256': sha256(geometry_path), 'generator_sha256': sha256(__file__),
               'figure_sha256': sha256(figure_path), 'new_inference': False, 'test_accessed': False}
    case_path.write_text(json.dumps(example, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 验证诊断：小目标预测面积偏大', '', '## Material Passport', '',
             '- 来源：课程完整验证集2,584张图的已保存预测；原版选第5轮，采样版选第4轮。',
             '- 性质：单种子、事后描述性诊断；没有重新推理，也没有使用测试集。',
             '- 标准小目标 AP/AR 与原评测记录准确复算一致；下列面积分组用于诊断。', '',
             '| 原图掩码面积 | GT数量 | 原版 AP | 采样版 AP | 原版 Recall@0.50 | 采样版 Recall@0.50 |',
             '|---|---:|---:|---:|---:|---:|']
    for label, title in [('small', '全部small（1–1024）'), ('1_64', '1–64'),
                         ('65_256', '65–256'), ('257_1024', '257–1024')]:
        old = diagnostic['models']['original']['bins'][label]
        new = diagnostic['models']['sampling']['bins'][label]
        lines.append(f'| {title} | {diagnostic["gt_counts"][label]} | {old["ap"]:.6f} | {new["ap"]:.6f} | {old["recall_by_iou"][0]:.6f} | {new["recall_by_iou"][0]:.6f} |')
    old_ratio = geometry['models']['original']['bins']['65_256']['prediction_to_gt_area_ratio_quantiles_on_overlap'][1]
    new_ratio = geometry['models']['sampling']['bins']['65_256']['prediction_to_gt_area_ratio_quantiles_on_overlap'][1]
    lines += ['', '65–256像素组在IoU=0.50时已有明显召回下降，差异并非仅出现在更严格的轮廓匹配要求下。',
        f'对每个GT独立选取保存候选中IoU最高的掩码，在非零重合的目标中，预测面积/GT面积中位数为原版{old_ratio:.3f}、采样版{new_ratio:.3f}。这与输出掩码偏大、难以达到匹配阈值的现象相符。',
        '面积诊断允许不同GT选中同一候选，不是COCO的一对一匹配；它使用阈值0.01、每图最多100个已保存候选。面积比仅统计有非零重合的GT，两模型的入选数量可能不同。',
        '', '## 一个按明确规则选出的失败示例', '',
        f'验证图片{image_id}，标注{annotation_id}。共有{len(eligible)}个实例满足所述失败规则，取按图片ID、标注ID排序的首例。此例用于解释现象，不能代表整体频率。', '',
        '![原图、GT、原版与采样版的局部掩码](../validation_geometry/failure_example.png)', '',
        '| 掩码 | 面积 | 与GT重合像素 | 超出GT像素 | 漏掉GT像素 | IoU |', '|---|---:|---:|---:|---:|---:|']
    for name, title in [('original', '原版'), ('sampling', '采样版')]:
        item = details[name]
        lines.append(f'| {title} | {item["best_prediction_area"]} | {item["intersection_pixels"]} | {item["outside_gt_pixels"]} | {item["missed_gt_pixels"]} | {item["best_iou"]:.6f} |')
    lines += ['', '该例中采样版覆盖了大部分GT，但掩码包含更多背景像素，因而IoU下降。上述输出证据尚不能证明是哪一项训练机制导致扩张；结论范围是本次128个前景点+128个边界带点配置、五轮单种子实验。',
              '', '[完整分组统计](diagnostics.json) · [掩码几何统计](../validation_geometry/geometry.json) · [示例选择与像素计数](../validation_geometry/failure_example.json)']
    report_path.write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'completed', 'report': str(report_path), 'figure': str(figure_path),
                      'example': example}, ensure_ascii=False))


if __name__ == '__main__':
    main()
