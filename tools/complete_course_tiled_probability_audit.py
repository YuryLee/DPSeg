#!/usr/bin/env python3
"""Complete the bounded local audit after the authorized inference job ends."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha
from rgbd_seg.course_tiled_probability import VARIANTS


def main():
    folder = ROOT/'reports/course_tiled_probability/full_v1'
    protocol = ROOT/'reports/course_tiled_probability/protocol_full_v1.json'
    auditor = ROOT/'tools/audit_course_tiled_probability.py'
    report = {'status': 'waiting_inference', 'pid': os.getpid(),
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'auditor_sha256': file_sha(auditor), 'protocol_sha256': file_sha(protocol)}
    path = folder/'audit_run.json'
    write_json(path, report)
    began = time.monotonic()
    try:
        while True:
            run = json.loads((folder/'run.json').read_text())
            if run['status'] == 'completed':
                break
            if run['status'] == 'failed':
                raise RuntimeError('Upstream inference failed; no audit performed')
            if time.monotonic()-began > 14400:
                raise TimeoutError('Inference wait deadline exceeded')
            time.sleep(5)
        if file_sha(auditor) != report['auditor_sha256']:
            raise ValueError('Auditor changed during inference')
        report['status'] = 'scoring_size_bins'
        write_json(path, report)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        command = [sys.executable, '-u', str(auditor), '--protocol', str(protocol), '--output', str(folder)]
        with ExitStack() as stack:
            children = []
            for name in VARIANTS:
                log = stack.enter_context((folder/f'audit_{name}.log').open('x'))
                children.append(subprocess.Popen(command+['--size-bin', name], env=env, stdout=log, stderr=subprocess.STDOUT))
            for child in children:
                if child.wait(timeout=1800) != 0:
                    raise RuntimeError('Area-bin evaluation failed; inspect saved logs')
        report['status'] = 'auditing'
        write_json(path, report)
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
        (folder/'audit_finish.log').write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError('Final output audit failed; inspect audit_finish.log')
        report.update(status='completed', exit_code=0)
    except BaseException as error:
        report.update(status='failed', exit_code=1, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(path, report)
    print(json.dumps({'status': report['status'], 'output': str(folder/'comparison.md')}))


if __name__ == '__main__':
    main()
