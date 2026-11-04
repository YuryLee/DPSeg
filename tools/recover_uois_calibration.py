#!/usr/bin/env python3
"""Recover pinhole K from original organized XYZ; never use instance labels."""
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
from rgbd_seg.uois_benchmark_data import file_sha256


def read_xyz(path):
    with path.open('rb') as stream:
        header = {}
        for _ in range(50):
            line = stream.readline().decode('ascii').strip()
            if not line or line.startswith('#'):
                continue
            key, *values = line.split()
            header[key] = values
            if key == 'DATA':
                break
        if header.get('DATA') != ['binary']:
            raise ValueError('Expected original binary organized PCD: ' + str(path))
        names, sizes, types = header['FIELDS'], list(map(int, header['SIZE'])), header['TYPE']
        counts = list(map(int, header.get('COUNT', ['1'] * len(names))))
        if len({len(names), len(sizes), len(types), len(counts)}) != 1:
            raise ValueError('Inconsistent PCD field metadata')
        fields = []
        for name, size, kind, count in zip(names, sizes, types, counts):
            if kind not in ('F', 'U', 'I') or count < 1:
                raise ValueError('Unsupported PCD field type/count')
            dtype = '<' + kind.lower() + str(size)
            fields.append((name, dtype) if count == 1 else (name, dtype, (count,)))
        width, height = int(header['WIDTH'][0]), int(header['HEIGHT'][0])
        points = int(header['POINTS'][0])
        if points != width * height or min(width, height) <= 1:
            raise ValueError('Point cloud is not organized')
        offset = stream.tell()
        dtype = np.dtype(fields)
        values = np.fromfile(stream, dtype=dtype, count=points)
        if len(values) != points:
            raise ValueError('Truncated PCD data')
        values = values.reshape(height, width)
        # PCL sometimes leaves unused trailing allocation bytes. Read exactly
        # POINTS records after the header; never shift the binary data start.
        trailing = path.stat().st_size - offset - dtype.itemsize * points
        return tuple(values[name] for name in ('x', 'y', 'z')), trailing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('osd', 'ocid'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--depth-scale-m', type=float, required=True,
                        help='Documented raw depth unit, checked against metric cloud Z')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError('Keep old calibration receipts; use new output paths')
    if not np.isfinite(args.depth_scale_m) or args.depth_scale_m <= 0:
        raise ValueError('A positive documented depth scale is required')
    root = args.root.resolve()
    # Ten official OCID clouds reside in a directory literally named "pd".
    # Keep the original tree and enumerate the actual .pcd files.
    clouds = sorted((root / 'pcd').glob('*.pcd')) if args.dataset == 'osd' else sorted(root.rglob('*.pcd'))
    if not clouds:
        raise ValueError('No original organized point clouds')
    rows = []
    for index, path in enumerate(clouds):
        (x, y, z), trailing = read_xyz(path)
        raw_path = ((root / 'disparity') if args.dataset == 'osd' else path.parent.parent / 'depth') / path.with_suffix('.png').name
        raw = np.array(Image.open(raw_path))
        if raw.shape != z.shape or raw.dtype != np.uint16:
            raise ValueError('Original uint16 depth differs from the organized cloud')
        original_size = list(z.shape)
        yy, xx = np.mgrid[0:z.shape[0]:8, 0:z.shape[1]:8]
        x, y, z, raw = (value[::8, ::8] for value in (x, y, z, raw))
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 0) & (raw > 0)
        if valid.sum() < 100:
            raise ValueError('Insufficient geometry to verify camera: ' + str(path))
        parameters, residuals = [], []
        for coordinate, pixel in ((x, xx), (y, yy)):
            ratio = coordinate[valid].astype(np.float64) / z[valid]
            focal, center = np.linalg.lstsq(np.stack((ratio, np.ones(len(ratio))), 1), pixel[valid], rcond=None)[0]
            residual = float(np.sqrt(np.mean((focal * ratio + center - pixel[valid])**2)))
            if not np.isfinite(focal) or focal <= 0 or residual > 1.:
                raise ValueError('Point geometry does not support a precise pinhole fit: ' + str(path))
            parameters.extend((float(focal), float(center))); residuals.append(residual)
        scale = float(np.median(z[valid] / raw[valid]))
        depth_error = float(np.max(np.abs(z[valid] - raw[valid] * args.depth_scale_m)))
        # PNG is integer quantized; a ratio estimate is biased when a float Z
        # was floored. Verify agreement within one raw unit plus float error.
        if depth_error > 1.01 * args.depth_scale_m + 1e-6:
            raise ValueError('Documented depth unit disagrees with cloud Z: ' + str(path))
        rows.append({'cloud': str(path.relative_to(root)), 'points_used': int(valid.sum()),
                     'fx': parameters[0], 'cx': parameters[1], 'fy': parameters[2], 'cy': parameters[3],
                     'x_rms_pixels': residuals[0], 'y_rms_pixels': residuals[1], 'depth_scale_m': scale,
                     'max_depth_difference_m': depth_error,
                     'pcd_sha256': file_sha256(path), 'depth_sha256': file_sha256(raw_path),
                     'trailing_pcd_bytes_not_used': trailing, 'image_size': original_size})
        if (index + 1) % 200 == 0:
            print(json.dumps({'clouds_fitted': index + 1, 'clouds': len(clouds)}), flush=True)
    summary = {key: {'median': float(np.median([r[key] for r in rows])),
                     'min': min(r[key] for r in rows), 'max': max(r[key] for r in rows)}
               for key in ('fx', 'fy', 'cx', 'cy', 'depth_scale_m', 'x_rms_pixels', 'y_rms_pixels')}
    resolutions = {tuple(r['image_size']) for r in rows}
    if len(resolutions) != 1:
        raise ValueError('Multiple camera resolutions require per-image calibration')
    report = {'status': 'recovered_from_original_geometry', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'dataset': args.dataset, 'clouds': len(rows), 'summary': summary, 'per_image': rows,
              'instance_labels_used': False, 'method': 'stride8 pinhole least-squares from organized metric XYZ; documented depth unit verified',
              'camera_rounding': 'median focal lengths/principal points rounded to .01 pixel',
              'geometry_quality_limits': {'per_frame_rms_pixels': 1., 'per_parameter_frame_spread_pixels': .25,
                                          'depth_max_error': 'one documented raw unit plus floating-point tolerance'},
              'script_sha256': file_sha256(Path(__file__))}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    if any(summary[k]['max'] - summary[k]['min'] > .25 for k in ('fx', 'fy', 'cx', 'cy')):
        report['status'] = 'per_image_calibration_requires_review'
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        raise ValueError('Cameras vary by frame; detailed fits saved, no global K emitted')
    fx, fy, cx, cy = (round(summary[k]['median'], 2) for k in ('fx', 'fy', 'cx', 'cy'))
    calibration = {'camera_K': [[fx, 0., cx], [0., fy, cy], [0., 0., 1.]],
                   'depth_scale_m': args.depth_scale_m, 'image_size': list(next(iter(resolutions))),
                   'depth_semantics': 'camera_axis_z', 'invalid_depth_codes': [0], 'status': 'recovered',
                   'source': str(args.report.resolve()), 'source_sha256': file_sha256(args.report)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration, indent=2) + '\n')
    print(json.dumps({'clouds': len(rows), 'calibration': calibration}), flush=True)


if __name__ == '__main__':
    main()
