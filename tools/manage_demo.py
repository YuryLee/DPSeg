#!/usr/bin/env python3
"""Start, inspect, or stop this project's local demo (Linux; no model imports)."""
import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
_CHILDREN = {}


def pidfd_open(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    return pidfd_syscall(434, pid, 0)


def pidfd_signal(fd, sig):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(fd, sig)
    return pidfd_syscall(424, fd, sig, 0, 0)


def pidfd_syscall(number, *args):
    # Linux x86_64/aarch64 syscall IDs; Conda Python may omit the wrappers.
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("Safe stop requires Linux pidfd support; no signal sent")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *(ctypes.c_long(arg) for arg in args))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def process_identity(pid):
    """Start ticks distinguish reused PIDs; reject exited/zombie processes."""
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return {'pid': pid, 'start_ticks': fields[19], 'uid': proc.stat().st_uid,
                'cwd': os.readlink(proc / 'cwd'),
                'command': (proc / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')}
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def owns_process(state):
    expected = state.get('process')
    try:
        command = [state['python'], str(Path(state['root']) / 'tools/demo.py'),
                   '--data', state['data'], '--runs', state['runs'], '--split', state['split'],
                   '--device', state['device'], '--threads', str(state['threads']),
                   '--host', '127.0.0.1', '--port', str(state['port'])]
        return bool(expected and expected.get('uid') == os.getuid()
                    and expected.get('command') == command and expected.get('cwd') == state['root']
                    and process_identity(expected.get('pid')) == expected)
    except (KeyError, TypeError):
        return False


def owns_listener(state):
    """Verify the responding port belongs to the recorded process, not a rival."""
    if not owns_process(state):
        return False
    try:
        fds = Path('/proc') / str(state['process']['pid']) / 'fd'
        sockets = {os.readlink(fd) for fd in fds.iterdir()}
        endpoint = f'0100007F:{state["port"]:04X}'
        for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
            fields = line.split()
            if fields[1] == endpoint and fields[3] == '0A':
                return f'socket:[{fields[9]}]' in sockets
    except (OSError, KeyError, ValueError):
        pass
    return False


def file_record(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'inode': stat.st_ino, 'sha256': sha256(path)}



def dataset_identity(root):
    """Same framed v1 digest as rgbd_seg.data.dataset_fingerprint; stdlib only."""
    root = Path(root).resolve()
    manifest = read_json(root / 'manifest.json')
    paths = {'manifest.json': root / 'manifest.json'}
    for sample in manifest['samples']:
        for key in ('rgb', 'depth', 'instances'):
            if sample.get(key):
                path = (root / sample[key]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError('Manifest references a file outside the data directory')
                paths[sample[key]] = path
    digest = hashlib.sha256(b'rgbd-dataset-sha256-v1\0')
    records = {}
    for relative, path in sorted(paths.items()):
        name = relative.encode('utf-8')
        stat = path.stat()
        digest.update(len(name).to_bytes(8, 'big'))
        digest.update(name)
        digest.update(stat.st_size.to_bytes(8, 'big'))
        individual = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
                individual.update(block)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ino) != (stat.st_size, stat.st_mtime_ns, stat.st_ino):
            raise ValueError('Data changed during identity check')
        records[str(path)] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                              'inode': stat.st_ino, 'sha256': individual.hexdigest()}
    return digest.hexdigest(), records


def build_identity(root, data, runs, split):
    """Pin source, manifest and actual checkpoint bytes, without importing Torch."""
    root, data, runs = map(Path, (root, data, runs))
    manifest = read_json(data / 'manifest.json')
    files = [root / 'tools/demo.py', root / 'web/index.html', data / 'manifest.json']
    files += sorted((root / 'rgbd_seg').glob('*.py'))
    actual_dataset_hash, data_files = dataset_identity(data)
    models = []
    for mode in ('E0', 'E2', 'E3'):
        folder = runs / mode
        run = read_json(folder / 'run.json')
        if run.get('manifest_sha256') != sha256(data / 'manifest.json'):
            raise ValueError(f'{mode}: run.json was trained with a different manifest')
        if run.get('dataset_sha256') != actual_dataset_hash:
            raise ValueError(f'{mode}: current data bytes differ from the training dataset')
        checkpoint = folder / 'model.pt'
        actual_hash = sha256(checkpoint)
        if run.get('checkpoint_sha256') != actual_hash:
            raise ValueError(f'{mode}: run.json does not match model.pt')
        models.append({'id': mode, 'checkpoint_sha256': actual_hash, 'config': run['config'],
                       'run_metadata': {key: run.get(key) for key in
                                        ('run_id', 'steps', 'batch_size', 'seed', 'device', 'selection', 'exit_status')}})
        files.extend([folder / 'run.json', checkpoint])
        for name in ('evaluation.json' if split == 'val' else 'evaluation_test.json', f'extra_{split}.json'):
            if (folder / name).exists():
                files.append(folder / name)
    records = {}
    for path in files:
        if path.name == 'model.pt':
            stat = path.stat()
            records[str(path)] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                  'inode': stat.st_ino, 'sha256': read_json(path.parent / 'run.json')['checkpoint_sha256']}
        else:
            records[str(path)] = file_record(path)
    expected = {'provenance': manifest['provenance'], 'split': split,
                'manifest_sha256': sha256(data / 'manifest.json'), 'dataset_sha256': actual_dataset_hash,
                'classes': manifest['classes'], 'class_scheme': manifest.get('class_scheme'),
                'samples': [s['id'] for s in manifest['samples'] if s['split'] == split], 'models': models}
    return {'files': records, 'data_files': data_files, 'api': expected}


def identity_unchanged(identity):
    try:
        for name, record in identity['files'].items():
            stat = Path(name).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (record['size'], record['mtime_ns'], record['inode']):
                return False
            # Hash small metadata and code; checkpoint byte hashes were checked at start.
            if Path(name).name != 'model.pt' and sha256(name) != record['sha256']:
                return False
        for name, record in identity['data_files'].items():
            stat = Path(name).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (record['size'], record['mtime_ns'], record['inode']):
                return False
            if sha256(name) != record['sha256']:
                return False
        return True
    except (OSError, KeyError, TypeError):
        return False


def probe(state, timeout=2):
    if not owns_listener(state):
        return False, 'recorded process does not own the listening port'
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(state['url'] + '/api/info', timeout=timeout) as response:
            if response.status != 200:
                return False, 'HTTP health check failed'
            raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                return False, 'health response too large'
            info = json.loads(raw)
        expected = state['identity']['api']
        for key in ('provenance', 'split', 'classes', 'class_scheme', 'manifest_sha256', 'dataset_sha256'):
            if info.get(key) != expected[key]:
                return False, f'HTTP identity mismatch: {key}'
        if [s['id'] for s in info['samples']] != expected['samples']:
            return False, 'HTTP identity mismatch: samples'
        actual = {m['id']: m for m in info['models']}
        if set(actual) != {m['id'] for m in expected['models']}:
            return False, 'HTTP identity mismatch: model set'
        for model in expected['models']:
            remote = actual[model['id']]
            if remote.get('checkpoint_sha256') != model['checkpoint_sha256']:
                return False, 'HTTP identity mismatch: checkpoint'
            if remote.get('config') != model['config']:
                return False, 'HTTP identity mismatch: model configuration'
            if not isinstance(remote.get('run_metadata'), dict):
                return False, 'HTTP identity missing: run metadata'
            for key, value in model['run_metadata'].items():
                if remote['run_metadata'].get(key) != value:
                    return False, f'HTTP identity mismatch: run {key}'
        if not owns_listener(state):
            return False, 'process changed during health check'
        return True, 'healthy'
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as exc:
        return False, f'health check failed: {type(exc).__name__}'


def inspect(state):
    if state.get('schema') != SCHEMA:
        return {'status': 'unmanaged', 'reason': 'unknown manager state version'}
    if not owns_process(state):
        return {'status': 'stale', 'reason': 'recorded process has exited or its identity changed'}
    if not identity_unchanged(state['identity']):
        return {'status': 'changed', 'reason': 'source, data or run files changed; restart required', 'url': state['url']}
    healthy, reason = probe(state)
    return {'status': 'ready' if healthy else 'unhealthy', 'reason': reason,
            'url': state['url'], 'pid': state['process']['pid'], 'split': state['split'],
            'data': state['data'], 'runs': state['runs'], 'log': state['log']}


def stop_owned(state, timeout=8):
    """Signal through pidfd so PID reuse cannot redirect a signal."""
    if not owns_process(state):
        return False
    try:
        fd = pidfd_open(state['process']['pid'])
    except ProcessLookupError:
        return False
    try:
        if not owns_process(state):
            return False
        pidfd_signal(fd, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while owns_process(state) and time.monotonic() < deadline:
            time.sleep(.05)
        if owns_process(state):
            pidfd_signal(fd, signal.SIGKILL)
            deadline = time.monotonic() + 2
            while owns_process(state) and time.monotonic() < deadline:
                time.sleep(.05)
        return not owns_process(state)
    except ProcessLookupError:
        return True
    finally:
        os.close(fd)
        child = _CHILDREN.pop(state['process']['pid'], None)
        if child is not None:
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _CHILDREN[child.pid] = child


@contextlib.contextmanager
def manager_lock(state_path):
    path = Path(str(state_path) + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another manager command is in progress; retry shortly') from None
        yield


def start(root, state_path, data, runs, split='test', device='cpu', threads=2,
          port=33051, timeout=120, python=sys.executable):
    root, data, runs = (Path(p).resolve() for p in (root, data, runs))
    state_path = Path(state_path).resolve()
    if state_path.exists():
        old = read_json(state_path)
        if owns_process(old):
            report = inspect(old)
            same = all(old.get(key) == value for key, value in
                       [('data', str(data)), ('runs', str(runs)), ('split', split),
                        ('device', device), ('threads', threads)]) and (port == 0 or old.get('port') == port)
            if same and report['status'] == 'ready':
                return report
            raise RuntimeError('A managed process still exists; inspect or stop it before starting another')
    # Fail before loading models when a fixed port is occupied. Bind race is handled by readiness checks.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
        port = sock.getsockname()[1]
    identity = build_identity(root, data, runs, split)
    log = state_path.with_suffix('.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(Path(python).absolute()), str(root / 'tools/demo.py'), '--data', str(data),
               '--runs', str(runs), '--split', split, '--device', device, '--threads', str(threads),
               '--host', '127.0.0.1', '--port', str(port)]
    with log.open('ab', buffering=0) as output:
        child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                                 close_fds=True)
    _CHILDREN[child.pid] = child
    state = {'schema': SCHEMA, 'root': str(root), 'python': command[0], 'status': 'starting', 'url': f'http://127.0.0.1:{port}', 'port': port,
             'data': str(data), 'runs': str(runs), 'split': split, 'device': device, 'threads': threads,
             'log': str(log), 'identity': identity, 'started_at': time.time()}
    # /proc may briefly still describe Popen's pre-exec child; wait for the exact command.
    deadline = time.monotonic() + min(timeout, 5)
    while time.monotonic() < deadline:
        found = process_identity(child.pid)
        if found and found['command'] == command and found['cwd'] == str(root):
            state['process'] = found
            break
        if child.poll() is not None:
            raise RuntimeError(f'Demo exited with code {child.returncode}; see {log}')
        time.sleep(.02)
    if 'process' not in state:
        # Popen still owns the child identity here; no state-file PID is trusted.
        child.terminate()
        child.wait(timeout=5)
        raise RuntimeError(f'Could not verify launched process; see {log}')
    atomic_json(state_path, state)
    try:
        deadline = time.monotonic() + timeout
        reason = 'startup in progress'
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f'Demo exited with code {child.returncode}; see {log}')
            healthy, reason = probe(state, timeout=min(2, max(.1, deadline - time.monotonic())))
            if healthy:
                if not identity_unchanged(identity):
                    raise RuntimeError('Files changed during startup; restart with stable files')
                state['status'] = 'ready'
                state['ready_at'] = time.time()
                atomic_json(state_path, state)
                return inspect(state)
            time.sleep(.2)
        raise RuntimeError(f'Demo did not become healthy: {reason}; see {log}')
    except BaseException:
        stop_owned(state)
        child.wait(timeout=3)
        state['status'] = 'failed'
        atomic_json(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'status', 'stop'))
    parser.add_argument('--state', type=Path, default=ROOT / 'reports/demo_manager.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/graspnet_expanded_v2')
    parser.add_argument('--runs', type=Path, default=ROOT / 'outputs/graspnet_v2')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--device', default='cpu', help='Use cuda:0 for GPU; defaults to CPU')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--port', type=int, default=33051, help='Fixed local port; 0 selects an available port')
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535 or args.timeout <= 0 or args.threads <= 0:
        parser.error('port must be 0–65535; timeout and threads must be positive')
    try:
        with manager_lock(args.state):
            if args.action == 'start':
                result = start(ROOT, args.state, args.data, args.runs, args.split, args.device,
                               args.threads, args.port, args.timeout)
            elif not args.state.exists():
                result = {'status': 'not_managed', 'reason': 'No manager state; old server.json is not proof of a running service'}
            else:
                state = read_json(args.state)
                if args.action == 'status':
                    result = inspect(state)
                elif not owns_process(state):
                    result = {'status': 'stale', 'reason': 'No matching owned process; no signal sent'}
                else:
                    if not stop_owned(state):
                        raise RuntimeError('Could not confirm owned process stopped')
                    state['status'] = 'stopped'
                    atomic_json(args.state, state)
                    result = {'status': 'stopped', 'reason': 'Verified owned process stopped'}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['status'] in ('ready', 'stopped', 'not_managed', 'stale') else 1
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({'status': 'error', 'reason': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
