#!/usr/bin/env python3
"""Check one nominal median camera against EVERY original organized cloud.

No segmentation labels or scores are used. This resolves small differences in
per-frame fits by directly measuring common-camera reprojection error.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.recover_uois_calibration import read_xyz
from rgbd_seg.uois_benchmark_data import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--fit-report', type=Path, required=True)
    p.add_argument('--depth-scale-m', type=float, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.report.exists() or a.output.exists():
        raise FileExistsError('Use new paths for the review and calibration')
    fit = json.loads(a.fit_report.read_text())
    fx, fy, cx, cy = [round(fit['summary'][key]['median'], 2) for key in ('fx', 'fy', 'cx', 'cy')]
    rows, sizes = [], set()
    for index, record in enumerate(fit['per_image']):
        cloud = a.root / record['cloud']
        (x, y, z), _ = read_xyz(cloud)
        size = tuple(z.shape); sizes.add(size)
        depth_path = cloud.parent.parent / 'depth' / cloud.with_suffix('.png').name
        raw = np.array(Image.open(depth_path))
        if raw.shape != z.shape or raw.dtype != np.uint16:
            raise ValueError('Original cloud/depth geometry mismatch')
        yy, xx = np.mgrid[0:z.shape[0]:8, 0:z.shape[1]:8]
        x, y, z, raw = (v[::8, ::8] for v in (x, y, z, raw))
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 0) & (raw > 0)
        if valid.sum() < 100:
            raise ValueError('Insufficient geometry for common-camera validation')
        dx = fx * x[valid].astype(float) / z[valid] + cx - xx[valid]
        dy = fy * y[valid].astype(float) / z[valid] + cy - yy[valid]
        rms_x, rms_y = float(np.sqrt(np.mean(dx**2))), float(np.sqrt(np.mean(dy**2)))
        if max(rms_x, rms_y) >= 1.:
            raise ValueError('Nominal camera reprojection RMS exceeds one pixel: ' + record['cloud'])
        max_depth_error = float(np.max(np.abs(z[valid] - raw[valid] * a.depth_scale_m)))
        if max_depth_error > 1.01 * a.depth_scale_m + 1e-6:
            raise ValueError('Declared depth scale fails the raw quantization check')
        rows.append({'cloud': record['cloud'], 'rms_x_pixels': rms_x, 'rms_y_pixels': rms_y,
                     'p99_x_error_pixels': float(np.quantile(np.abs(dx), .99)),
                     'p99_y_error_pixels': float(np.quantile(np.abs(dy), .99)),
                     'max_depth_difference_m': max_depth_error, 'points': int(valid.sum())})
        if (index + 1) % 300 == 0:
            print(json.dumps({'checked_clouds': index+1, 'clouds': len(fit['per_image'])}), flush=True)
    if len(sizes) != 1 or len(rows) != fit['clouds']:
        raise ValueError('Incomplete geometry coverage or mixed resolutions')
    camera = [[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]
    review = {'status': 'common_nominal_camera_verified', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'clouds': len(rows), 'camera_K': camera, 'depth_scale_m': a.depth_scale_m,
              'fit_report_sha256': file_sha256(a.fit_report), 'source_sha256': file_sha256(Path(__file__)),
              'max_per_frame_rms_x_pixels': max(r['rms_x_pixels'] for r in rows),
              'max_per_frame_rms_y_pixels': max(r['rms_y_pixels'] for r in rows),
              'criterion': 'common median K rounded to .01 pixel; every frame RMS<1 pixel; PNG/Z difference within one raw unit plus float error',
              'instance_labels_used': False, 'frames_excluded': 0, 'per_frame': rows,
              'limitation': 'Recovered nominal pinhole model, not a manufacturer calibration; raw per-frame fit variation remains documented'}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(review, indent=2)+'\n')
    calibration = {'camera_K': camera, 'depth_scale_m': a.depth_scale_m,
                   'image_size': list(next(iter(sizes))), 'depth_semantics': 'camera_axis_z',
                   'invalid_depth_codes': [0], 'status': 'recovered', 'source': str(a.report.resolve()),
                   'source_sha256': file_sha256(a.report),
                   'uncertainty': review['limitation'],
                   'max_per_frame_reprojection_rms_pixels': max(review['max_per_frame_rms_x_pixels'], review['max_per_frame_rms_y_pixels'])}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(calibration, indent=2)+'\n')
    print(json.dumps({key: review[key] for key in ('status', 'clouds', 'camera_K', 'max_per_frame_rms_x_pixels', 'max_per_frame_rms_y_pixels')}))


if __name__ == '__main__':
    main()
