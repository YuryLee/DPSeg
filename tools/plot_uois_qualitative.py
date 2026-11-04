#!/usr/bin/env python3
"""Explain saved native UOIS predictions; never run or select a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rgbd_seg.uois_benchmark_metrics import evaluate_label_maps, predictions_to_label_map


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    rows = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['image_id'] in rows:
                raise ValueError('Duplicate image ID')
            rows[row['image_id']] = row
    return rows


def color(label):
    # The same GT ID gets the same color in both instance overlays.
    palette = [(34, 197, 94), (59, 130, 246), (245, 158, 11), (168, 85, 247),
               (6, 182, 212), (244, 63, 94), (132, 204, 22), (249, 115, 22),
               (99, 102, 241), (20, 184, 166), (234, 179, 8), (217, 70, 239)]
    return palette[(int(label) - 1) % len(palette)]


def overlay(rgb, labels, colors):
    image = rgb.astype(np.float32).copy()
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = labels == label
        rgb_color = np.asarray(colors[int(label)], dtype=np.float32)
        image[mask] = .42 * image[mask] + .58 * rgb_color
        interior = np.zeros_like(mask)
        interior[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                                 & mask[2:, 1:-1] & mask[1:-1, :-2]
                                 & mask[1:-1, 2:])
        image[mask & ~interior] = rgb_color
    return image.clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    folder, output = args.prediction_dir.resolve(), args.output.resolve()
    protocol = json.loads((folder / 'protocol.json').read_text())
    scoring = json.loads((folder / 'scores/protocol.json').read_text())
    completion = json.loads((folder / 'completion.json').read_text())
    manifest_path = Path(protocol['manifest'])
    manifest = json.loads(manifest_path.read_text())
    records = {row['id']: row for row in manifest['samples']}
    rows = read_rows(folder / 'scores/per_image.jsonl')
    if not (protocol['family'] == 'dafusion' and protocol['fixed_evaluation_threshold'] == .5
            and protocol['test_used_for_selection'] is False
            and protocol['manifest_sha256'] == sha(manifest_path)
            and completion['predictions_sha256'] == scoring['predictions_sha256']
            and len(rows) == len(records) == protocol['images'] == manifest['images']
            and set(rows) == set(records)):
        raise ValueError('Inputs do not identify a completed, fixed native test')
    paths = [folder / n for n in ['protocol.json', 'completion.json', 'predictions.jsonl',
                                  'scores/protocol.json', 'scores/per_image.jsonl']]
    sources = {str(p): sha(p) for p in paths + [manifest_path, Path(__file__)]}
    if sources[str(folder / 'predictions.jsonl')] != completion['predictions_sha256']:
        raise ValueError('Prediction file changed after scoring')
    nonempty = sorted((r for r in rows.values() if r['metrics']['obj_gt'] > 0),
                      key=lambda r: (r['metrics']['Objects F-measure'], r['image_id']))
    selected = []
    for q in [.25, .5, .75]:
        index = math.floor((len(nonempty) - 1) * q)
        selected.append((f'q{int(q * 100):02d}', nonempty[index],
                         {'quantile': q, 'zero_based_rank': index,
                          'population': 'all nonempty-GT images', 'population_images': len(nonempty)}))
    empty = sorted((r for r in rows.values() if r['metrics']['obj_gt'] == 0),
                   key=lambda r: (-r['metrics']['obj_detected'], r['image_id']))
    if empty:
        selected.append(('empty_gt', empty[0],
                         {'rule': 'most predicted instances among empty-GT images; image_id breaks ties',
                          'population_images': len(empty)}))
    selected_ids = {r['image_id'] for _, r, _ in selected}
    predictions = {}
    seen = set()
    with (folder / 'predictions.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['image_id'] in seen:
                raise ValueError('Duplicate prediction image ID')
            seen.add(row['image_id'])
            if row['image_id'] in selected_ids:
                predictions[row['image_id']] = row['predictions']
    if seen != set(records) or set(predictions) != selected_ids:
        raise ValueError('Incomplete prediction cohort')
    output.mkdir(parents=True, exist_ok=False)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 20)
    small = ImageFont.truetype(font_path, 17)
    samples, plates = [], []
    for tag, saved, selection in selected:
        image_id = saved['image_id']
        record = records[image_id]
        for key in ['rgb', 'instances']:
            if sha(record[key]) != record['sha256'][key]:
                raise ValueError('Example source changed since indexing')
        with Image.open(record['rgb']) as im:
            rgb = np.array(im.convert('RGB'), copy=True)
        with Image.open(record['instances']) as im:
            truth = np.array(im, dtype=np.int32, copy=True)
        truth[np.isin(truth, record['background_labels'])] = 0
        prediction = predictions_to_label_map(predictions[image_id], truth.shape, .5)
        actual = evaluate_label_maps(prediction, truth)
        if actual != {key: saved[key] for key in ['metrics', 'diagnostics', 'assignments']}:
            raise ValueError('Displayed label maps differ from scored result')
        gt_ids = [int(i) for i in np.unique(truth) if i]
        pred_ids = [int(i) for i in np.unique(prediction) if i]
        gt_colors = {i: color(i) for i in gt_ids}
        pred_colors = {i: (236, 72, 153) for i in pred_ids}
        for gt_id, pred_id in actual['assignments']:
            pred_colors[pred_id] = gt_colors[gt_id]
        foreground, predicted = truth > 0, prediction > 0
        regions = {'true_positive': foreground & predicted,
                   'false_positive': ~foreground & predicted,
                   'false_negative': foreground & ~predicted}
        errors = (rgb.astype(np.float32) * .38).astype(np.uint8)
        for key, col in [('true_positive', (34, 197, 94)), ('false_positive', (239, 68, 68)),
                         ('false_negative', (59, 130, 246))]:
            errors[regions[key]] = col
        h, w = truth.shape
        plate = Image.new('RGB', (w * 4, h + 108), 'white')
        draw = ImageDraw.Draw(plate)
        metrics = saved['metrics']
        draw.text((12, 7), f"{manifest['dataset'].upper()} {tag} | Overlap F {metrics['Objects F-measure']*100:.2f}% | "
                  f"Boundary F {metrics['Boundary F-measure']*100:.2f}% | GT {len(gt_ids)}, predicted {len(pred_ids)}",
                  font=font, fill='#111827')
        panels = [rgb, overlay(rgb, truth, gt_colors), overlay(rgb, prediction, pred_colors), errors]
        for j, (title, panel) in enumerate(zip(['RGB', 'Original GT', 'DA-Fusion, score > 0.5',
                                              'Foreground: green TP / red FP / blue FN'], panels)):
            draw.text((j * w + 12, 38), title, font=small, fill='#374151')
            plate.paste(Image.fromarray(panel), (j * w, 66))
        draw.text((12, h + 73), image_id, font=small, fill='#374151')
        path = output / f'{manifest["dataset"]}_{tag}.png'
        plate.save(path)
        plates.append(plate)
        samples.append({'image_id': image_id, 'selection': selection, 'metrics': metrics,
                        'gt_instances': len(gt_ids), 'predicted_instances': len(pred_ids),
                        'foreground_error_pixels': {k: int(v.sum()) for k, v in regions.items()},
                        'display_rescoring_matches_saved_result': True,
                        'rgb': record['rgb'], 'original_gt': record['instances'],
                        'input_sha256': {k: record['sha256'][k] for k in ['rgb', 'instances']},
                        'png': {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}})
    overview = Image.new('RGB', (plates[0].width, sum(p.height for p in plates)), 'white')
    y = 0
    for plate in plates:
        overview.paste(plate, (0, y))
        y += plate.height
    overview_path = output / 'overview.png'
    overview.save(overview_path)
    strata = {}
    for name, group in [('all', list(rows.values())), ('nonempty_gt', nonempty), ('empty_gt', empty)]:
        if not group:
            continue
        strata[name] = {'images': len(group), 'mean_overlap_f': math.fsum(r['metrics']['Objects F-measure'] for r in group) / len(group),
                        'mean_boundary_f': math.fsum(r['metrics']['Boundary F-measure'] for r in group) / len(group),
                        'mean_predicted_instances': math.fsum(r['metrics']['obj_detected'] for r in group) / len(group),
                        'mean_gt_instances': math.fsum(r['metrics']['obj_gt'] for r in group) / len(group),
                        'images_with_predictions': sum(r['metrics']['obj_detected'] > 0 for r in group)}
    # Recheck immutable sources after reading and rendering.
    if any(sha(p) != value for p, value in sources.items()):
        raise ValueError('A source changed while rendering')
    result = {'scope': 'qualitative explanation of completed native predictions; no model or threshold selection',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'dataset': manifest['dataset'],
              'model': 'locally trained DA-Fusion public Swin-L reimplementation',
              'weights_sha256': protocol['weights_sha256'], 'score_threshold': .5,
              'selection_rule': 'sort nonempty GT by (Overlap F, image_id), floor((N-1)*q) at q=.25,.5,.75; optional empty-GT control',
              'official_scores_unchanged': True, 'stratified_metrics_are_diagnostic_only': True,
              'color_note': 'GT and Hungarian-matched prediction IDs share colors; unmatched predictions are pink. Color is not a correctness score.',
              'error_note': 'Foreground errors ignore instance identity and cannot show instance merges/splits by themselves.',
              'sources_sha256': sources, 'diagnostic_strata': strata, 'samples': samples,
              'overview': {'path': str(overview_path), 'sha256': sha(overview_path), 'bytes': overview_path.stat().st_size},
              'visually_inspected': False}
    (output / 'index.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'samples': len(samples), 'strata': strata}, ensure_ascii=False))


if __name__ == '__main__':
    main()
