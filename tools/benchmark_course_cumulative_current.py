#!/usr/bin/env python3
"""Time the five current cumulative rows without changing frozen experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark_course_swinl as original

read, write, sha, now = original.read, original.write, original.sha, original.now
RUN = original.RUN
NAMES = original.NAMES


def predict(stage, adapter, head, data):
    if stage != 4:
        return original_predict(stage, adapter, head, data)
    import numpy as np
    import torch
    from rgbd_seg.course_dpseg import predict_dpseg
    image, depth, valid = data
    sample = dict(image=torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div_(255),
                  depth_m=depth, depth_valid=valid)
    return predict_dpseg(adapter, head, sample, overlap=.4)[0]


original_predict = original.predict
original_loader = original.load_models


def prepare(args):
    import numpy as np
    import pynvml
    import torch
    from rgbd_seg.course_archive import CourseArchive
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    assert not pynvml.nvmlDeviceGetComputeRunningProcesses(handle), 'Benchmark GPU must be empty'
    gpu_uuid, gpu_name = pynvml.nvmlDeviceGetUUID(handle), pynvml.nvmlDeviceGetName(handle)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / 'weights').mkdir()
    main_path = ROOT / 'docs/results/course_ablation_swinl_current.json'
    main = read(main_path)
    assert main['images'] == 2112 and len(main['results']) == 5
    manifest = read(ROOT / 'configs/dpseg_current.json')
    assert manifest['source_sha256']['rgbd_seg/course_dpseg.py'] == sha(ROOT / 'rgbd_seg/course_dpseg.py')
    rgb_source = ROOT / 'reports/course_swinl_rgb_v1/base15/final.pt'
    rgb_snapshot = out / 'weights/rgb_snapshot.pt'
    os.link(rgb_source, rgb_snapshot)
    rgb = torch.load(rgb_snapshot, map_location='cpu', weights_only=True, mmap=True)
    assert rgb['epoch'] == 15
    base, paired = RUN / 'base15/final.pt', RUN / 'points6/heads_epoch_06.pt'
    saved = torch.load(paired, map_location='cpu', weights_only=True)
    depth = {k.removeprefix('depth.'): v for k, v in saved['heads'].items() if k.startswith('depth.')}
    torch.save(depth, out / 'weights/depth_head_only.pt')
    paths = [('rgb', rgb_snapshot), ('base', base), ('heads', paired),
             ('depth_head', out / 'weights/depth_head_only.pt')]
    weights = {key: dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)) for key, path in paths}
    assert saved['base_checkpoint_sha256'] == weights['base']['sha256']
    assert main['checkpoints']['rgb']['sha256'] == weights['rgb']['sha256']
    assert main['checkpoints']['rgbd']['sha256'] == weights['base']['sha256']
    assert main['checkpoints']['points']['sha256'] == weights['heads']['sha256']
    source = read(RUN / 'protocol.json')
    with CourseArchive(source['archive'], 'val') as archive:
        indices = np.linspace(0, len(archive.image_ids) - 1, args.images, dtype=int).tolist()
        ids = [archive.image_ids[i] for i in indices]
    with CourseArchive(source['archive'], 'test') as archive:
        gate_indices = [0, len(archive.image_ids) // 2, len(archive.image_ids) - 1]
        gate_ids = [archive.image_ids[i] for i in gate_indices]
    references = {
        '0': str(ROOT / 'reports/course_swinl_rgb_v1/base15/val_epoch_15/predictions.jsonl'),
        '1': str(RUN / 'final_evaluation/val_native/predictions.jsonl'),
        '2': str(RUN / 'points6/val_epoch_06/predictions.jsonl'),
    }
    test_references = {str(i): dict(path=row['predictions'], sha256=row['predictions_sha256'])
                       for i, row in enumerate(main['results'])}
    for row in main['results']:
        assert sha(row['evaluation']) == row['evaluation_sha256']
    for entry in test_references.values():
        assert sha(entry['path']) == entry['sha256']
    files = set(manifest['source_sha256']) | {
        'tools/benchmark_course_swinl.py', 'tools/benchmark_course_cumulative_current.py',
        'rgbd_seg/course_swinl_rgb.py', 'configs/dpseg_current.json',
        'tools/run_course_depth_backbone_system.py',
    }
    sources = {name: sha(ROOT / name) for name in sorted(files)}
    for name, expected in manifest['source_sha256'].items():
        assert sources[name] == expected, name
    p = dict(schema='dpseg-current-cumulative-efficiency-v1', created_at=now(), gpu=args.gpu,
             gpu_uuid=gpu_uuid, gpu_name=gpu_name,
             resident_pids=[], cpu_affinity=args.cpu_affinity, weights=weights,
             checkpoint=source['checkpoint'], archive=source['archive'],
             main_table=dict(path=str(main_path), sha256=sha(main_path)),
             rgb_epoch=15, rgb_final_training_complete=True, base_epoch=15, head_epoch=6,
             stages=dict(enumerate(NAMES)), image_ids=ids, image_indices=indices, split='val',
             test_gate_image_ids=gate_ids, test_gate_indices=gate_indices,
             test_gate_scope='Compare three fixed first/middle/last test predictions to already-scored outputs; no scoring, labels, selection, or timing.',
             repeats=args.repeats, warmup_images=5, precision='FP32', tf32=False, batch_size=1,
             threads=4, input_size=[1024, 1024], score_threshold=.01, max_instances=100,
             sampling='32 evenly spaced validation indices; independent of labels and performance',
             tile_size=640, tiles=4, overlap=.4, views_including_whole=5,
             merge_strategy='global_guard_edge_consensus',
             timed_scope='decoded PIL RGB and CPU metric depth -> tensor conversion, preprocessing, H2D, model, applicable refinement/tiling/merging, D2H, COCO RLE',
             excluded='model loading, ZIP/PNG decoding, scoring, file I/O, reference checks',
             size_scope='all inference parameters; actual base checkpoint plus only required depth-head state_dict; no optimizer or unused plain point head',
             gpu_activity_policy='Dedicated idle GPU; fail on any other GPU process or detected other-process compute activity. NVML activity sampling approximately one second.',
             peak_gpu_memory_measured=False, per_stage_timeout_seconds=1800,
             references=references, reference_hashes={k: sha(v) for k, v in references.items()},
             test_references=test_references, sources=sources,
             implementation='Original cumulative S0-S3 prediction paths; current course_dpseg.predict_dpseg for S4. Legacy benchmark helper reused in memory; frozen source files unchanged.')
    write(out / 'protocol.json', p)
    write(out / 'status.json', dict(status='prepared', created_at=now(), protocol_sha256=sha(out / 'protocol.json')))
    print(json.dumps(dict(status='prepared', output=str(out), protocol_sha256=sha(out / 'protocol.json'))))


def engineering_gate(p, stage, adapter, head, output):
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_data import load_depth_record
    entry = p['test_references'][str(stage)]
    assert sha(entry['path']) == entry['sha256']
    expected = original.references(entry['path'], p['test_gate_image_ids'])
    checks = []
    with CourseArchive(p['archive'], 'test') as archive, torch.inference_mode():
        for image_id, index in zip(p['test_gate_image_ids'], p['test_gate_indices']):
            assert archive.image_ids[index] == image_id
            image = archive.load_rgb(index)
            depth, valid = load_depth_record(archive._image_archive(), archive.images[index])
            values = predict(stage, adapter, head, (image, depth, valid))
            reference = expected[image_id]
            check = dict(scope='three fixed test-image engineering checks; excluded from timing', image_id=image_id,
                         exact_masks=original.identity(values) == original.identity(reference),
                         counts_equal=len(values) == len(reference),
                         score_error=max((abs(a['score'] - b['score']) for a, b in zip(values, reference)), default=0.))
            checks.append(check)
            write(output / 'engineering_gate.json', dict(status='checking', checks=checks))
            assert check['exact_masks'] and check['counts_equal'] and check['score_error'] <= 1e-6, check
    write(output / 'engineering_gate.json', dict(status='passed', checks=checks))
    return checks


def worker(args):
    p = read(args.protocol)
    assert p['schema'] == 'dpseg-current-cumulative-efficiency-v1'
    assert p['rgb_epoch'] == 15 and p['merge_strategy'] == 'global_guard_edge_consensus'
    for entry in p['weights'].values():
        assert Path(entry['path']).stat().st_size == entry['bytes'] and sha(entry['path']) == entry['sha256']
    for key, expected in p['reference_hashes'].items():
        assert sha(p['references'][key]) == expected
    checks = []

    def load_checked(plan, stage):
        adapter, head, counts = original_loader(plan, stage)
        checks.extend(engineering_gate(plan, stage, adapter, head, args.output))
        return adapter, head, counts

    original.predict = predict
    original.load_models = load_checked
    original.worker(args)
    result = read(args.output / 'result.json')
    result['reference_checks'] = checks + result['reference_checks']
    result['test_gate_images'] = len(checks)
    result['merge_strategy'] = p['merge_strategy'] if args.stage == 4 else None
    result['rgb_epoch'] = p['rgb_epoch']
    write(args.output / 'result.json', result)


def controller(args):
    out = args.output
    protocol_path = out / 'protocol.json'
    p = read(protocol_path)
    assert p['schema'] == 'dpseg-current-cumulative-efficiency-v1'
    for filename, digest in p['sources'].items():
        assert sha(ROOT / filename) == digest, filename
    exits = []
    for stage in range(5):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=p['gpu_uuid'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
        command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--protocol', str(protocol_path),
                   '--stage', str(stage), '--output', str(out / f'M{stage}')]
        with (out / f'M{stage}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            began = time.monotonic()
            while process.poll() is None:
                write(out / 'status.json', dict(status='benchmarking', stage=stage, controller_pid=os.getpid(),
                      process_pid=process.pid, elapsed_seconds=time.monotonic() - began,
                      last_checked_at_utc=now(), protocol_sha256=sha(protocol_path)))
                if time.monotonic() - began > p['per_stage_timeout_seconds']:
                    write(out / 'timeout.json', dict(stage=stage, pid=process.pid, seconds=p['per_stage_timeout_seconds']))
                    process.kill()
                    process.wait()
                    raise RuntimeError(f'S{stage} exceeded the declared timeout')
                time.sleep(5)
            code = process.returncode
        exits.append(dict(stage=stage, pid=process.pid, exit_code=code))
        write(out / 'exits.json', exits)
        if code:
            raise RuntimeError(f'S{stage} failed; preserved {out}/M{stage}.log')
    results = [read(out / f'M{i}/result.json') for i in range(5)]
    write(out / 'summary.json', dict(status='completed', protocol=p, results=results, exits=exits, completed_at=now()))
    write(out / 'status.json', dict(status='completed', exit_code=0, completed_at=now()))


def launch(args):
    out = args.output
    assert read(out / 'status.json')['status'] == 'prepared'
    command = [sys.executable, '-u', str(Path(__file__).resolve()), 'controller', '--output', str(out)]
    with (out / 'controller.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(out / 'launch.json', dict(pid=process.pid, command=command, launched_at=now()))
    print(json.dumps(dict(status='launched', pid=process.pid, output=str(out))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'launch', 'controller', 'worker'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--images', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cpu-affinity', type=lambda s: [int(x) for x in s.split(',')], default=[32, 33, 34, 35])
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--stage', type=int, choices=range(5))
    args = parser.parse_args()
    args.output = args.output.resolve()
    try:
        globals()[args.mode](args)
    except Exception as error:
        if args.mode == 'controller':
            write(args.output / 'status.json', dict(status='failed', error=str(error), finished_at=now()))
        raise


if __name__ == '__main__':
    main()
