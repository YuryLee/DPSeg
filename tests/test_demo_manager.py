"""Manager integration checks use tiny fake checkpoints and a CPU-only HTTP child."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('manage_demo', Path(__file__).resolve().parents[1] / 'tools/manage_demo.py')
manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manager)

FAKE_DEMO = r'''
import argparse,json,os,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
p=argparse.ArgumentParser()
for key in ('data','runs','split','device','threads','host','port'): p.add_argument('--'+key)
a=p.parse_args()
root=Path.cwd()
if (root/'exit_early').exists(): raise SystemExit(7)
if (root/'delay').exists(): time.sleep(5)
info=json.loads((root/'info.json').read_text())
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  body=json.dumps(info).encode()
  self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def log_message(self,*args): pass
server=HTTPServer((a.host,int(a.port)),Handler)
server.serve_forever()
'''


class DemoManagerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.runs = self.root / 'runs'
        self.state = self.root / 'reports/manager.json'
        for directory in ('tools', 'web', 'rgbd_seg', 'data'):
            (self.root / directory).mkdir()
        (self.root / 'tools/demo.py').write_text(FAKE_DEMO)
        (self.root / 'web/index.html').write_text('<html>test</html>')
        manifest = {'provenance': 'test_fixture', 'classes': ['object'], 'class_scheme': 'category_agnostic',
                    'samples': [{'id': 'frame_0', 'split': 'test', 'rgb': 'sample.bin'}]}
        (self.data / 'manifest.json').write_text(json.dumps(manifest))
        (self.data / 'sample.bin').write_bytes(b'fixture data bytes')
        dataset_hash, _ = manager.dataset_identity(self.data)
        for mode in ('E0', 'E2', 'E3'):
            directory = self.runs / mode
            directory.mkdir(parents=True)
            raw = ('fake checkpoint ' + mode).encode()
            (directory / 'model.pt').write_bytes(raw)
            run = {'run_id': mode, 'config': {'mode': mode, 'arch': 'dummy'}, 'steps': 0,
                   'checkpoint_sha256': hashlib.sha256(raw).hexdigest(), 'dataset_sha256': dataset_hash,
                   'manifest_sha256': manager.sha256(self.data / 'manifest.json')}
            (directory / 'run.json').write_text(json.dumps(run))
        expected = manager.build_identity(self.root, self.data, self.runs, 'test')['api']
        info = copy.deepcopy(expected)
        info['samples'] = [{'id': value, 'name': value} for value in expected['samples']]
        (self.root / 'info.json').write_text(json.dumps(info))
        self.unrelated = []

    def tearDown(self):
        if self.state.exists():
            saved = manager.read_json(self.state)
            manager.stop_owned(saved)
        for child in self.unrelated:
            child.terminate()
            child.wait(timeout=5)
        # Reap fake children retained by subprocess's detached-child bookkeeping.
        subprocess._cleanup()
        self.temp.cleanup()

    def start(self, **kwargs):
        return manager.start(self.root, self.state, self.data, self.runs, port=0,
                             timeout=kwargs.pop('timeout', 3), **kwargs)

    def test_start_health_identity_idempotence_and_stop(self):
        result = self.start()
        self.assertEqual(result['status'], 'ready')
        saved = manager.read_json(self.state)
        self.assertTrue(manager.owns_listener(saved))
        self.assertEqual(manager.inspect(saved)['status'], 'ready')
        self.assertEqual(self.start()['pid'], result['pid'])
        self.assertTrue(manager.stop_owned(saved))
        self.assertEqual(manager.inspect(saved)['status'], 'stale')
        self.assertFalse(manager.stop_owned(saved))

    def test_stale_pid_cannot_signal_unrelated_process(self):
        self.start()
        saved = manager.read_json(self.state)
        other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.unrelated.append(other)
        bad = copy.deepcopy(saved)
        bad['process']['pid'] = other.pid
        self.assertEqual(manager.inspect(bad)['status'], 'stale')
        self.assertFalse(manager.stop_owned(bad))
        self.assertIsNone(other.poll())
        forged = copy.deepcopy(saved)
        forged['process'] = manager.process_identity(other.pid)
        self.assertFalse(manager.stop_owned(forged))
        self.assertIsNone(other.poll())
        bad = copy.deepcopy(saved)
        bad['process']['start_ticks'] = '0'
        self.assertFalse(manager.stop_owned(bad))
        self.assertEqual(manager.inspect(saved)['status'], 'ready')

    def test_listener_from_other_process_not_accepted(self):
        self.start()
        saved = manager.read_json(self.state)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); sock.listen()
            saved['port'] = sock.getsockname()[1]
            saved['url'] = f'http://127.0.0.1:{saved["port"]}'
            self.assertFalse(manager.owns_listener(saved))
            self.assertFalse(manager.probe(saved)[0])

    def test_fixed_port_collision_fails_before_launch(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); sock.listen()
            with self.assertRaises(OSError):
                manager.start(self.root, self.state, self.data, self.runs, port=sock.getsockname()[1])
        self.assertFalse(self.state.exists())

    def test_runtime_hash_or_split_mismatch_never_ready(self):
        info = manager.read_json(self.root / 'info.json')
        info['models'][0]['checkpoint_sha256'] = 'wrong'
        (self.root / 'info.json').write_text(json.dumps(info))
        with self.assertRaisesRegex(RuntimeError, 'checkpoint'):
            self.start(timeout=.5)
        saved = manager.read_json(self.state)
        self.assertEqual(saved['status'], 'failed')
        self.assertFalse(manager.owns_process(saved))
        info['models'][0]['checkpoint_sha256'] = manager.sha256(self.runs / 'E0/model.pt')
        info['split'] = 'val'
        (self.root / 'info.json').write_text(json.dumps(info))
        with self.assertRaisesRegex(RuntimeError, 'split'):
            self.start(timeout=.5)
        self.assertFalse(manager.owns_process(manager.read_json(self.state)))

    def test_source_edit_requires_restart_but_can_stop(self):
        self.start()
        saved = manager.read_json(self.state)
        (self.root / 'web/index.html').write_text('<html>changed</html>')
        self.assertEqual(manager.inspect(saved)['status'], 'changed')
        with self.assertRaisesRegex(RuntimeError, 'still exists'):
            self.start()
        self.assertTrue(manager.stop_owned(saved))

    def test_checkpoint_changed_before_start_is_rejected(self):
        (self.runs / 'E0/model.pt').write_bytes(b'not recorded')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.start()
        self.assertFalse(self.state.exists())

    def test_startup_failure_and_timeout_clean_up(self):
        (self.root / 'exit_early').touch()
        with self.assertRaisesRegex(RuntimeError, 'exited with code 7'):
            self.start()
        if self.state.exists():
            self.assertFalse(manager.owns_process(manager.read_json(self.state)))
        (self.root / 'exit_early').unlink()
        (self.root / 'delay').touch()
        with self.assertRaisesRegex(RuntimeError, 'did not become healthy'):
            self.start(timeout=.3)
        self.assertFalse(manager.owns_process(manager.read_json(self.state)))

    def test_data_edit_without_manifest_change_is_rejected_before_launch(self):
        (self.data / 'sample.bin').write_bytes(b'modified data')
        with self.assertRaisesRegex(ValueError, 'current data bytes differ'):
            self.start()
        self.assertFalse(self.state.exists())

    def test_equal_length_data_edit_restored_mtime_is_detected_by_status(self):
        self.start()
        saved = manager.read_json(self.state)
        path = self.data / 'sample.bin'
        before = path.stat()
        path.write_bytes(b'X' * before.st_size)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(path.stat().st_size, before.st_size)
        self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertEqual(manager.inspect(saved)['status'], 'changed')
        self.assertTrue(manager.stop_owned(saved))

    def test_conflicting_start_does_not_replace_managed_process(self):
        result = self.start()
        with self.assertRaisesRegex(RuntimeError, 'still exists'):
            self.start(split='val')
        self.assertEqual(manager.inspect(manager.read_json(self.state))['pid'], result['pid'])


if __name__ == '__main__':
    unittest.main()
