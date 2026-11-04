#!/usr/bin/env python3
"""Wait for the live formal queue, then audit CSVs and write verified comparisons."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/course_depth_backbone_v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(data):
    path = OUT/'final_audit_watcher.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def alive(identity):
    try:
        stat = Path(f"/proc/{identity['pid']}/stat").read_text().split(') ')[1].split()
        return stat[0] != 'Z' and stat[19] == identity['start_ticks']
    except FileNotFoundError:
        return False


def main():
    if (OUT/'final_audit_watcher.json').exists():
        raise FileExistsError('Existing audit watcher requires review; no automatic retries')
    identity = read(OUT/'queue_process.json')
    if not alive(identity):
        raise ValueError('Expected a live formal queue at watcher startup')
    reference_sha = sha(OUT/'comparison_reference.json')
    state = {'status': 'waiting_for_formal_queue', 'pid': os.getpid(),
        'start_ticks': Path(f'/proc/{os.getpid()}/stat').read_text().split(') ')[1].split()[19],
        'queue_identity': identity, 'script_sha256': sha(Path(__file__)),
        'reference_sha256': reference_sha}
    write(state)
    try:
        while alive(identity):
            state['last_verified_queue_alive_at_utc'] = datetime.now(timezone.utc).isoformat()
            write(state)
            time.sleep(60)
        queue = read(OUT/'queue.json')
        if (queue['status'] != 'completed'
                or queue['comparison_sha256'] != sha(OUT/'comparison.json')):
            raise ValueError('Formal queue did not complete with verified comparison output')
        state['status'] = 'independent_csv_rescoring'
        write(state)
        command = [sys.executable, str(ROOT/'tools/audit_course_depth_backbone.py'), '--root', str(OUT)]
        state['audit_command'] = command
        with (OUT/'independent_final_audit.log').open('x') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=3600, check=False)
        state['audit_exit_code'] = result.returncode
        if result.returncode:
            raise RuntimeError('Independent CSV audit failed; see independent_final_audit.log')
        if sha(OUT/'comparison_reference.json') != reference_sha:
            raise ValueError('Historical comparison reference changed')
        references = read(OUT/'comparison_reference.json')['rows']
        by_stage = {r['id']: r for r in references}
        rows = []
        for arm in ('swint_depth_no_k', 'swinl_xyz'):
            audit = read(OUT/arm/'independent_audit.json')
            if audit['status'] != 'passed':
                raise ValueError('Missing successful independent audit: '+arm)
            for row in audit['rows']:
                baseline = by_stage['S1' if row['variant'] == 'native' else 'S4']
                rows.append({'arm': arm, **row,
                    'reference_stage': baseline['id'], 'reference_metrics': baseline['test'],
                    'delta_ap': row['metrics']['mask_ap']-baseline['test']['mask_ap'],
                    'delta_aps': row['metrics']['mask_ap_small']-baseline['test']['mask_ap_small']})
        verified = {'status': 'passed', 'scope': 'two_complete_training_and_test_experiments',
            'rows': rows, 'reference_sha256': reference_sha,
            'formal_comparison_sha256': sha(OUT/'comparison.json'),
            'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'limitations': read(OUT/'plan.json')['limitations']}
        (OUT/'verified_comparison.json').write_text(json.dumps(verified, ensure_ascii=False, indent=2)+'\n')
        lines = ['# 无内参与Swin-Large实验对比', '', '## Material Passport', '',
            '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run',
            '- Verification Status: FORMAL_CSV_RESCORED', '',
            '两组均完成15轮基座及6轮配套点头训练。点头仅按完整验证集选择。下表为2,112张本地测试图的COCO掩码指标，范围0–1；已从导出CSV独立复评分，与原报告一致。', '',
            '| 配置 | 推理流程 | AP | APₛ | 相对原XYZ Swin-T的APₛ增量 |',
            '|---|---|---:|---:|---:|']
        for stage, label in (('S1', '整图基座'), ('S4', '完整系统')):
            m = by_stage[stage]['test']
            lines.append(f"| 原XYZ Swin-Tiny | {label} | {m['mask_ap']:.6f} | {m['mask_ap_small']:.6f} | — |")
        for row in rows:
            label = '无内参深度 Swin-Tiny' if row['arm'] == 'swint_depth_no_k' else 'XYZ Swin-Large'
            method = '整图基座' if row['variant'] == 'native' else '点头＋切片＋概率细化'
            m = row['metrics']
            lines.append(f"| {label} | {method} | {m['mask_ap']:.6f} | {m['mask_ap_small']:.6f} | {row['delta_aps']:+.6f} |")
        lines += ['', '## CSV文件', '']
        for row in rows:
            lines.append(f"- {row['arm']} / {row['variant']}：`{row['csv']}`")
        lines += ['', '## 比较范围', '']+['- '+x for x in verified['limitations']]
        (OUT/'verified_comparison.md').write_text('\n'.join(lines)+'\n')
        state.update(status='passed', finished_at_utc=verified['finished_at_utc'],
                     verified_comparison_sha256=sha(OUT/'verified_comparison.json'))
        write(state)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error),
                     stopped_at_utc=datetime.now(timezone.utc).isoformat())
        write(state)
        raise


if __name__ == '__main__':
    main()
