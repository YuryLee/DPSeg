#!/usr/bin/env python3
"""Measure the report's Swin-L/depth-only M0--M4 arms; no peak-memory statistics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'reports/course_swinl_depth_no_k_v1/swinl_depth_no_k'
NAMES = ['Mask2Former Swin-L RGB', '+ RGB-D fusion', '+ depth point refinement',
         '+ depth probability refinement', '+ tiled inference']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def configure(p):
    import torch
    os.sched_setaffinity(0, p['cpu_affinity'])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.device_count() == 1


class ActivityMonitor:
    """Check process activity, allowing an idle resident demo to remain loaded."""
    def __init__(self, gpu, residents):
        import pynvml
        self.api = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        self.allowed = set(residents) | {os.getpid()}
        self.since = int((time.time()-3)*1e6)
        self.samples = []
        self.process_events = []

    def poll(self):
        pids = sorted(x.pid for x in self.api.nvmlDeviceGetComputeRunningProcesses(self.handle))
        self.process_events.append({'time': time.time(), 'pids': pids})
        unexpected = set(pids)-self.allowed
        if unexpected:
            raise RuntimeError(f'Unexpected GPU processes: {sorted(unexpected)}')
        try:
            values = self.api.nvmlDeviceGetProcessUtilization(self.handle, self.since)
        except self.api.NVMLError_NotFound:
            values = []
        for v in values:
            if v.pid != os.getpid():
                self.samples.append(dict(pid=v.pid, timestamp=v.timeStamp/1e6,
                                         sm=v.smUtil, memory_controller=v.memUtil))
        if values:
            self.since = max(v.timeStamp for v in values)

    def contaminated(self, begin, end):
        # NVML activity is sampled at roughly one-second intervals, so use padding.
        return [s for s in self.samples if begin-1.5 <= s['timestamp'] <= end+1.5
                and (s['sm'] or s['memory_controller'])]


def load_models(p, stage):
    import torch
    if stage == 0:
        from rgbd_seg.course_swinl_rgb import build_trainable
        saved = torch.load(p['weights']['rgb']['path'], map_location='cpu', weights_only=True, mmap=True)
        adapter, metadata = build_trainable(p['checkpoint'], device='cpu', seed=17)
        assert metadata == saved['model_metadata']
        adapter.load_state_dict(saved['model'], strict=True)
        head = None
        weight_bytes = p['weights']['rgb']['bytes']
    else:
        from tools.run_course_depth_backbone_system import load_base
        from rgbd_seg.course_depth_backbone_points import PairedPointHeads
        adapter = load_base(read(RUN/'protocol_system.json'), p['weights']['base']['path'], 'cpu')
        head = None
        weight_bytes = p['weights']['base']['bytes']
        if stage >= 2:
            saved = torch.load(p['weights']['heads']['path'], map_location='cpu', weights_only=True)
            assert saved['base_checkpoint_sha256'] == p['weights']['base']['sha256']
            heads = PairedPointHeads(fine_channels=448)
            heads.load_state_dict(saved['heads'], strict=True)
            head = heads.depth
            del heads
            weight_bytes += p['weights']['depth_head']['bytes']
    models = [adapter] + ([] if head is None else [head])
    counts = dict(parameters=sum(x.numel() for m in models for x in m.parameters()),
                  parameter_bytes=sum(x.numel()*x.element_size() for m in models for x in m.parameters()),
                  buffer_bytes=sum(x.numel()*x.element_size() for m in models for x in m.buffers()),
                  checkpoint_bytes=weight_bytes,
                  depth_head_parameters=0 if head is None else sum(x.numel() for x in head.parameters()))
    for model in models:
        model.eval().requires_grad_(False).to('cuda:0')
    gc.collect()
    return adapter, head, counts


def predict(stage, adapter, head, data):
    import numpy as np
    import torch
    from rgbd_seg.course_depth_backbone_points import predict_paired
    from rgbd_seg.course_depth_backbone_system import tiled_pair
    from rgbd_seg.course_tiled_probability import refined_public
    image, depth, valid = data
    if stage == 0:
        return adapter.predict(image, score_threshold=.01, max_detections=100)
    rgb = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div_(255)
    sample = {'image': rgb, 'depth_m': depth, 'depth_valid': valid}
    if stage == 1:
        return adapter.predict_sample(sample, score_threshold=.01, max_detections=100)
    if stage == 4:
        return tiled_pair(adapter, head, sample)[1]
    paired = predict_paired(adapter, SimpleNamespace(depth=head), sample, head_names=('point_depth',))
    if stage == 2:
        return [c.public() for c in paired['point_depth']]
    if stage == 3:
        return refined_public(paired['point_depth'], sample, 'probability', adapter.device)[0]
    raise ValueError(stage)


def identity(values):
    masks = [(x['category_id'], x['segmentation']) for x in values]
    return hashlib.sha256(json.dumps(masks, sort_keys=True).encode()).hexdigest()


def references(path, ids):
    result = {}
    with open(path) as stream:
        for line in stream:
            row = json.loads(line)
            if row['image_id'] in ids:
                result[row['image_id']] = row['predictions']
    assert set(result) == set(ids)
    return result


def worker(args):
    import numpy as np
    import torch
    import transformers
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_data import load_depth_record
    p = read(args.protocol)
    configure(p)
    output = args.output
    output.mkdir()
    for filename, digest in p['sources'].items():
        assert sha(ROOT/filename) == digest, filename
    stage = args.stage
    monitor = ActivityMonitor(p['gpu'], p['resident_pids'])
    monitor.poll()
    adapter, head, counts = load_models(p, stage)
    ids = p['image_ids']
    samples = {}
    with CourseArchive(p['archive'], 'val') as archive:
        for image_id in ids:
            index = archive.image_ids.index(image_id)
            image = archive.load_rgb(index)
            depth, valid = load_depth_record(archive._image_archive(), archive.images[index])
            samples[image_id] = (image, depth, valid)
    ref = references(p['references'][str(stage)], ids) if str(stage) in p['references'] else None
    checks, observations, identities = [], [], {}
    began = time.monotonic()
    with torch.inference_mode():
        for image_id in ids[:5]:
            predict(stage, adapter, head, samples[image_id])
        with (output/'measurements.jsonl').open('x') as stream:
            for repeat in range(p['repeats']):
                order = list(ids)
                random.Random(17+repeat).shuffle(order)
                for j, image_id in enumerate(order):
                    monitor.poll()
                    torch.cuda.synchronize()
                    wall_start = time.time()
                    start = time.perf_counter()
                    values = predict(stage, adapter, head, samples[image_id])
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    wall_end = time.time()
                    digest = identity(values)
                    if image_id in identities:
                        assert digest == identities[image_id], 'Repeated prediction masks differ'
                    identities[image_id] = digest
                    if ref is not None and repeat == 0:
                        previous = ref[image_id]
                        equal = digest == identity(previous)
                        error = max((abs(a['score']-b['score']) for a,b in zip(values, previous)), default=0.)
                        checks.append(dict(image_id=image_id, exact_masks=equal, score_error=error,
                                           counts_equal=len(values)==len(previous)))
                        assert equal and len(values)==len(previous) and error <= 1e-6, checks[-1]
                    row = dict(stage=stage, repeat=repeat, image_id=image_id, seconds=elapsed,
                               wall_start=wall_start, wall_end=wall_end, instances=len(values), mask_sha256=digest)
                    observations.append(row)
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    del values
                    monitor.poll()
                    if (j+1)%8 == 0 or j+1 == len(order):
                        progress = dict(stage=stage, repeat=repeat+1, images=j+1,
                                        total_images=len(order), elapsed_seconds=time.monotonic()-began)
                        write(output/'progress.json', progress)
                        print(json.dumps(progress), flush=True)
    # Allow the activity sampler to publish its final interval.
    time.sleep(2)
    monitor.poll()
    rejected = [r for r in observations if monitor.contaminated(r['wall_start'], r['wall_end'])]
    write(output/'activity.json', dict(resident_pids=p['resident_pids'], samples=monitor.samples,
                                     process_events=monitor.process_events, rejected=rejected))
    if rejected:
        raise RuntimeError('Resident GPU activity overlapped timing; preserve observations and rerun on an idle device')
    times = [r['seconds'] for r in observations]
    result = dict(status='completed', stage=stage, name=NAMES[stage], **counts,
                  mean_ms=1000*statistics.mean(times), median_ms=1000*statistics.median(times),
                  p90_ms=1000*float(np.percentile(times, 90)), fps=1/statistics.mean(times),
                  repeat_mean_ms=[1000*statistics.mean(r['seconds'] for r in observations if r['repeat']==i)
                                  for i in range(p['repeats'])],
                  images=len(ids), repeats=p['repeats'], measurements=len(times),
                  mean_instances=statistics.mean(r['instances'] for r in observations),
                  reference_checks=checks, repeat_masks_exact=True, detected_gpu_activity_conflicts=0,
                  hardware=torch.cuda.get_device_name(), torch=torch.__version__,
                  transformers=transformers.__version__, cuda=torch.version.cuda,
                  protocol_sha256=sha(args.protocol), measurements_sha256=sha(output/'measurements.jsonl'),
                  completed_at=now())
    write(output/'result.json', result)
    print(json.dumps({k:result[k] for k in ('stage','parameters','mean_ms','fps')}), flush=True)


def launch(args):
    import numpy as np
    import torch
    from rgbd_seg.course_archive import CourseArchive
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out/'weights').mkdir()
    rgb_source = ROOT/'reports/course_swinl_rgb_v1/base15'
    rgb_source = rgb_source/('final.pt' if (rgb_source/'final.pt').exists() else 'best.pt')
    # Training checkpoints are atomically replaced: this link pins the current inode.
    rgb_snapshot = out/'weights/rgb_snapshot.pt'
    os.link(rgb_source, rgb_snapshot)
    rgb = torch.load(rgb_snapshot, map_location='cpu', weights_only=True, mmap=True)
    base = RUN/'base15/final.pt'
    heads_path = RUN/'points6/heads_epoch_06.pt'
    heads = torch.load(heads_path, map_location='cpu', weights_only=True)
    depth_head = {k.removeprefix('depth.'):v for k,v in heads['heads'].items() if k.startswith('depth.')}
    torch.save(depth_head, out/'weights/depth_head_only.pt')
    weights = {name:dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)) for name,path in
               [('rgb',rgb_snapshot), ('base',base), ('heads',heads_path),
                ('depth_head',out/'weights/depth_head_only.pt')]}
    source = read(RUN/'protocol.json')
    with CourseArchive(source['archive'], 'val') as archive:
        indices = np.linspace(0, len(archive.image_ids)-1, args.images, dtype=int).tolist()
        ids = [archive.image_ids[i] for i in indices]
    sources = ['tools/benchmark_course_swinl.py', 'rgbd_seg/course_swinl_rgb.py',
               'rgbd_seg/course_depth_backbone.py', 'rgbd_seg/course_depth_backbone_points.py',
               'rgbd_seg/course_depth_backbone_system.py', 'rgbd_seg/course_tiled_probability.py']
    p = dict(schema='dpseg-swinl-efficiency-v1', created_at=now(), gpu=args.gpu,
             resident_pids=args.resident_pids, cpu_affinity=args.cpu_affinity,
             weights=weights, checkpoint=source['checkpoint'], archive=source['archive'],
             rgb_epoch=rgb['epoch'], rgb_final_training_complete=rgb_source.name=='final.pt',
             stages=dict(enumerate(NAMES)), base_epoch=15, head_epoch=6,
             image_ids=ids, image_indices=indices, split='val', repeats=args.repeats,
             sampling='evenly spaced validation indices; independent of labels and performance',
             warmup_images=5, precision='FP32', tf32=False, batch_size=1, threads=4,
             input_size=[1024,1024], score_threshold=.01, max_instances=100,
             timed_scope='decoded PIL RGB and CPU metric depth -> tensor conversion, preprocessing, H2D, model, relevant refinement/tiling, D2H, COCO RLE',
             excluded='model loading, ZIP/PNG decoding, scoring, file I/O, reference checks',
             size_scope='all inference parameters; actual base checkpoint plus only the required depth-head state_dict; no optimizer',
             gpu_activity_policy='allow listed idle resident service; reject any detected other-process SM/memory-controller activity within 1.5 seconds of measurements; NVML approximately 1-second sampling',
             peak_gpu_memory_measured=False, per_stage_timeout_seconds=1800,
             references={'0':str(ROOT/f'reports/course_swinl_rgb_v1/base15/val_epoch_{rgb["epoch"]:02d}/predictions.jsonl'),
                         '1':str(RUN/'final_evaluation/val_native/predictions.jsonl'),
                         '2':str(RUN/'points6/val_epoch_06/predictions.jsonl'),
                         '4':str(RUN/'final_evaluation/val_full/predictions.jsonl')},
             sources={s:sha(ROOT/s) for s in sources})
    write(out/'protocol.json',p)
    exits=[]
    for stage in range(5):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                 OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        command=[sys.executable,__file__,'--worker','--protocol',str(out/'protocol.json'),
                 '--stage',str(stage),'--output',str(out/f'M{stage}')]
        with (out/f'M{stage}.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(out/'process.json',dict(pid=process.pid,stage=stage,started_at=now()))
            try:
                code=process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                write(out/'timeout.json', dict(stage=stage, pid=process.pid, seconds=1800))
                raise RuntimeError(f'M{stage} exceeded the declared 30-minute timeout')
        exits.append(dict(stage=stage,pid=process.pid,exit_code=code))
        write(out/'exits.json',exits)
        if code:
            raise RuntimeError(f'M{stage} failed; inspect {out}/M{stage}.log')
        print(json.dumps({'completed_stage':stage}),flush=True)
    results=[read(out/f'M{i}/result.json') for i in range(5)]
    write(out/'summary.json',dict(status='completed',protocol=p,results=results,exits=exits,completed_at=now()))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=4)
    parser.add_argument('--images',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--resident-pids',type=lambda s:[int(x) for x in s.split(',')],default=[])
    parser.add_argument('--cpu-affinity',type=lambda s:[int(x) for x in s.split(',')],default=[32,33,34,35])
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--stage',type=int,choices=range(5))
    args=parser.parse_args()
    args.output=args.output.resolve()
    worker(args) if args.worker else launch(args)
