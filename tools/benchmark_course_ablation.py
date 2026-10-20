#!/usr/bin/env python3
"""Measure independent S0-S4 latency and memory on one otherwise idle GPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def gpu_snapshot(gpu):
    fields = 'index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,clocks.sm'
    lines = subprocess.check_output(
        ['nvidia-smi', f'--id={gpu}', f'--query-gpu={fields}', '--format=csv,noheader,nounits'],
        text=True).strip().split(', ')
    info = dict(zip(fields.split(','), lines))
    processes = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'],
        text=True).strip().splitlines()
    info['processes'] = [line.split(', ') for line in processes if line.startswith(info['uuid'])]
    return info


class GpuGuard:
    """Poll NVML every 50 ms and reject intervals overlapping another process."""

    def __init__(self, gpu):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        self.pid = os.getpid()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.conflicts = 0
        self.events = []
        self.others = []
        self.peak = 0
        self.error = None
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def poll(self):
        with self.lock:
            processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            others = sorted(p.pid for p in processes if p.pid != self.pid)
            if others:
                self.conflicts += 1
            if others != self.others:
                self.events.append(dict(time=now(), other_pids=others))
            self.others = others
            for p in processes:
                if p.pid == self.pid and p.usedGpuMemory is not None:
                    self.peak = max(self.peak, p.usedGpuMemory)
            return self.conflicts, bool(others)

    def loop(self):
        try:
            while not self.stop.wait(.05):
                self.poll()
        except Exception as error:
            self.error = error

    def ready(self):
        began = time.monotonic()
        clean = None
        while True:
            if self.error:
                raise RuntimeError('GPU monitor failed') from self.error
            _, busy = self.poll()
            if busy:
                clean = None
            elif clean is None:
                clean = time.monotonic()
            elif time.monotonic() - clean >= .2:
                return
            if time.monotonic() - began > 120:
                raise TimeoutError('GPU did not become idle within 120 seconds')
            time.sleep(.05)

    def begin(self):
        if self.error:
            raise RuntimeError('GPU monitor failed') from self.error
        self.ready()
        with self.lock:
            self.peak = 0
            return self.conflicts

    def end(self, marker):
        self.poll()
        if self.error:
            raise RuntimeError('GPU monitor failed') from self.error
        with self.lock:
            return self.conflicts == marker, self.peak

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def configure(protocol):
    import torch
    os.sched_setaffinity(0, protocol['cpu_affinity'])
    torch.set_num_threads(protocol['torch_threads'])
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(protocol['gpu']):
        raise ValueError('Unexpected GPU mapping')
    if torch.cuda.device_count() != 1:
        raise ValueError('Benchmark must expose exactly one GPU')


def load(stage, protocol):
    import torch
    from rgbd_seg.course_portable import load_model, verify_sources
    from rgbd_seg.course_optimized import tensor_digest
    from rgbd_seg.course_progressive_rgb import build_trainable
    assets = protocol['weights']
    if stage == 'S0':
        checkpoint = torch.load(assets['rgb']['path'], map_location='cpu', weights_only=True)
        verify_sources(checkpoint['source_sha256'])
        adapter, metadata = build_trainable(protocol['coco_checkpoint'], device='cpu', seed=17)
        if metadata != checkpoint['model_metadata']:
            raise ValueError('RGB model construction differs')
        adapter.load_state_dict(checkpoint['model'], strict=True)
        if tensor_digest(adapter.state_dict()) != checkpoint['selection']['model_sha256']:
            raise ValueError('RGB tensor identity differs')
        head = None
        weight_bytes = assets['rgb']['bytes']
    else:
        adapter, heads, _ = load_model(base_weights=assets['rgbd']['path'],
            head_weights=assets['heads']['path'], coco_checkpoint=protocol['coco_checkpoint'],
            device='cpu')
        head = heads.depth if stage in ('S2', 'S3', 'S4') else None
        del heads
        weight_bytes = assets['rgbd']['bytes'] + (assets['heads']['bytes'] if head is not None else 0)
    models = [adapter] + ([] if head is None else [head])
    counts = dict(
        parameters=sum(p.numel() for m in models for p in m.parameters()),
        parameter_bytes=sum(p.numel() * p.element_size() for m in models for p in m.parameters()),
        buffer_bytes=sum(b.numel() * b.element_size() for m in models for b in m.buffers()),
        checkpoint_bytes=weight_bytes,
        depth_head_parameters=0 if head is None else sum(p.numel() for p in head.parameters()),
    )
    for model in models:
        model.eval().requires_grad_(False).to('cuda:0')
    gc.collect()
    torch.cuda.empty_cache()
    return adapter, head, counts


def collect_references(path, wanted):
    references = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['image_id'] in wanted:
                references[row['image_id']] = row['predictions']
            if len(references) == len(wanted):
                break
    if set(references) != set(wanted):
        raise ValueError('Missing original validation predictions')
    return references


def worker(args):
    import numpy as np
    import torch
    import transformers
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_data import load_inference_sample
    from rgbd_seg.course_benchmark import predict
    from tools.check_course_prediction_reload import compare_predictions

    p = read(args.protocol)
    configure(p)
    for relative, digest in p['sources'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f'Benchmark source changed: {relative}')
    for asset in p['weights'].values():
        if sha(asset['path']) != asset['sha256']:
            raise ValueError('Weights changed')
    output = args.output
    output.mkdir()
    began = time.monotonic()
    stage = args.stage
    guard = GpuGuard(p['gpu'])
    guard.ready()
    before = gpu_snapshot(p['gpu'])
    if before['processes']:
        write(output / 'gpu_conflict.json', before)
        raise ValueError('Selected GPU is already in use; see gpu_conflict.json')
    adapter, head, counts = load(stage, p)
    ids = p['image_ids']
    references = collect_references(p['references'][stage], ids)
    with CourseArchive(p['archive'], 'val') as data:
        samples = {i: load_inference_sample(data, i) for i in ids}
    # Decoding ZIP/PNG and reference comparisons are outside the timed interval.
    torch.cuda.reset_peak_memory_stats()
    for image_id in ids[:p['warmup_images']]:
        values = predict(stage, adapter, head, samples[image_id])
        compare_predictions(references[image_id], values)
        del values
    torch.cuda.synchronize()
    warmup_peak = torch.cuda.max_memory_allocated()
    observations, gpu_checks, discarded = [], [], []
    with (output / 'measurements.jsonl').open('x') as stream, torch.inference_mode():
        for repeat in range(p['repeats']):
            guard.ready()
            gpu_checks.append(gpu_snapshot(p['gpu']))
            order = list(ids)
            random.Random(p['seed'] + repeat).shuffle(order)
            for index, image_id in enumerate(order):
                for attempt in range(20):
                    marker = guard.begin()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter_ns()
                    values = predict(stage, adapter, head, samples[image_id])
                    torch.cuda.synchronize()
                    seconds = (time.perf_counter_ns() - started) / 1e9
                    peak = torch.cuda.max_memory_allocated()
                    reserved = torch.cuda.max_memory_reserved()
                    clean, nvml_peak = guard.end(marker)
                    if clean:
                        break
                    discarded.append(dict(repeat=repeat, image_id=image_id, seconds=seconds,
                        reason='Other GPU process detected during this measurement'))
                    write(output / 'discarded.json', discarded)
                    del values
                else:
                    raise RuntimeError('Twenty contaminated attempts for one image')
                score_error = compare_predictions(references[image_id], values)
                row = dict(stage=stage, repeat=repeat, image_id=image_id, seconds=seconds,
                    peak_allocated_bytes=peak, peak_reserved_bytes=reserved,
                    instances=len(values), max_reference_score_error=score_error,
                    nvml_peak_observed_bytes=nvml_peak, gpu_monitor_clean=True)
                observations.append(row)
                stream.write(json.dumps(row) + '\n')
                del values
                if (index + 1) % 8 == 0 or index + 1 == len(ids):
                    stream.flush()
                    progress = dict(stage=stage, repeat=repeat + 1, images=index + 1,
                        per_repeat=len(ids), elapsed_seconds=time.monotonic() - began)
                    write(output / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            guard.ready()
            gpu_checks.append(gpu_snapshot(p['gpu']))
    guard.close()
    write(output / 'gpu_monitor.json', dict(poll_seconds=.05, events=guard.events, discarded=discarded))
    elapsed = [row['seconds'] for row in observations]
    mean = statistics.mean(elapsed)
    result = dict(status='completed', stage=stage, completed_at=now(), **counts,
        samples=len(ids), repeats=p['repeats'], measurements=len(elapsed),
        mean_ms=mean * 1000, median_ms=float(np.median(elapsed)) * 1000,
        p90_ms=float(np.percentile(elapsed, 90)) * 1000,
        std_ms=statistics.stdev(elapsed) * 1000, fps=1 / mean,
        peak_allocated_bytes=max(warmup_peak, max(r['peak_allocated_bytes'] for r in observations)),
        peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in observations),
        warmup_peak_allocated_bytes=warmup_peak,
        mean_instances=statistics.mean(r['instances'] for r in observations),
        repeat_mean_ms=[statistics.mean(r['seconds'] for r in observations if r['repeat'] == k) * 1000
                        for k in range(p['repeats'])],
        exact_reference_masks=True, all_reference_scores_within_tolerance=True,
        reference_score_tolerance=1e-6,
        max_reference_score_error=max(r['max_reference_score_error'] for r in observations),
        hardware_before=before, gpu_exclusivity_checks=gpu_checks,
        worker_pid=os.getpid(), discarded_measurements=len(discarded),
        gpu_monitor_poll_seconds=.05,
        nvml_peak_observed_bytes=max(r['nvml_peak_observed_bytes'] for r in observations),
        gpu_monitor_sha256=sha(output / 'gpu_monitor.json'),
        cuda_device=torch.cuda.get_device_name(), cuda_version=torch.version.cuda,
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        python_version=platform.python_version(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        protocol_sha256=sha(args.protocol), measurements_sha256=sha(output / 'measurements.jsonl'))
    write(output / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('stage', 'mean_ms', 'fps', 'peak_allocated_bytes')}), flush=True)


def launch(args):
    import numpy as np
    from rgbd_seg.course_archive import CourseArchive
    previous = read(ROOT / 'reports/course_progressive_ablation/v1/evaluation/protocol.json')
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    with CourseArchive(previous['archive'], 'val') as data:
        indices = np.linspace(0, len(data.image_ids) - 1, args.images, dtype=int).tolist()
        ids = [data.image_ids[i] for i in indices]
    assets = {}
    for name, path in [
        ('rgb', previous['rgb']['weights']['path']),
        ('rgbd', previous['inputs']['base']['path']),
        ('heads', previous['inputs']['head']['path']),
    ]:
        assets[name] = dict(path=path, sha256=sha(path), bytes=Path(path).stat().st_size)
    arms = ('mask2former_finetune', 'rgbd_fusion', 'depth_point',
            'depth_point_tiled', 'depth_point_tiled_probability')
    references = {f'S{i}': str(ROOT / 'reports/course_progressive_ablation/v1/evaluation' /
        ('rgb_val' if i == 0 else 'rgbd_val') / arm / 'predictions.jsonl') for i, arm in enumerate(arms)}
    protocol = dict(schema='dpseg-inference-benchmark-v1', frozen_at=now(),
        gpu=args.gpu, cpu_affinity=args.cpu_affinity, torch_threads=4,
        archive=previous['archive'], coco_checkpoint=previous['coco_checkpoint'],
        split='val', sampling=f'{args.images} evenly spaced validation image positions, independent of labels',
        image_indices=indices, image_ids=ids, references=references, weights=assets,
        seed=17, repeats=args.repeats, warmup_images=5, batch_size=1,
        input_size=[1024, 1024], precision='FP32', autocast=False, tf32=False,
        timed_scope='decoded CPU inputs -> preprocessing/H2D/model/merge/refinement/D2H/COCO RLE',
        excluded='model loading, ZIP/PNG decoding, evaluation, prediction comparison, disk output',
        memory_scope='model plus inference tensors; allocator reserved separately; CUDA context excluded',
        parameters_scope='all learned parameters used by this stage, including frozen backbone; depth head only',
        checkpoint_scope='actual released files; paired-head file contains both plain and depth heads',
        gpu_guard=dict(poll_seconds=.05, idle_wait_seconds=.2, idle_timeout_seconds=120,
            max_attempts_per_image=20, reject='other compute process detected during measured interval',
            waiting_and_discarded_intervals_included_in_latency=False),
        sources={p: sha(ROOT / p) for p in [
            'tools/benchmark_course_ablation.py', 'rgbd_seg/course_benchmark.py',
            'rgbd_seg/course_progressive_ablation.py', 'rgbd_seg/course_point_refinement.py',
            'rgbd_seg/course_tiled_probability.py', 'rgbd_seg/course_progressive_rgb.py']})
    write(output / 'protocol.json', protocol)
    exits = []
    for stage in ('S0', 'S1', 'S2', 'S3', 'S4'):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
        command = [sys.executable, str(Path(__file__).resolve()), '--worker',
            '--protocol', str(output / 'protocol.json'), '--stage', stage,
            '--output', str(output / stage)]
        with (output / f'{stage}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            write(output / 'process.json', dict(stage=stage, pid=process.pid, started_at=now()))
            try:
                code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise RuntimeError(f'{stage} exceeded the declared 30-minute timeout')
        exits.append(dict(stage=stage, pid=process.pid, exit_code=code))
        write(output / 'exits.json', exits)
        if code:
            raise RuntimeError(f'{stage} failed; inspect {output / (stage + ".log")}')
        print(json.dumps({'completed': stage}), flush=True)
    results = [read(output / stage / 'result.json') for stage in ('S0', 'S1', 'S2', 'S3', 'S4')]
    write(output / 'summary.json', dict(status='completed', completed_at=now(),
        protocol=protocol, protocol_sha256=sha(output / 'protocol.json'), results=results, exits=exits))
    print(json.dumps({'completed': str(output / 'summary.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--images', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cpu-affinity', type=lambda value: [int(x) for x in value.split(',')],
                        default=[12, 13, 14, 15])
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--stage', choices=('S0', 'S1', 'S2', 'S3', 'S4'))
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
    else:
        if args.images < 5 or args.repeats < 1:
            parser.error('Need at least five images and one repeat')
        launch(args)


if __name__ == '__main__':
    main()
