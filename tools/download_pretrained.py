#!/usr/bin/env python3
"""Download a pinned COCO Mask R-CNN v2 mirror without proxies; verify SHA256.

The mirror repack is not byte-identical to the official .pth. This tool verifies
its pinned full file hash, then strict-loads the extracted tensors with torchvision.
No pickle fallback: torch.load always uses weights_only=True.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

from xet_range import RangeReader, RangeReadError

REPO = 'LibreYOLO/LibreMaskRCNNr50'
REVISION = '61ac7973f391772dcc1c1775862f9b99f2b95250'
FILENAME = 'LibreMaskRCNNr50.pt'
SIZE = 185836327
SHA256 = '9214933a07cd354265e62c31298d4502f8433d124da6fea1b3c00cf78974cfbd'
XET_HASH = 'b4c4af8918a1de084419fc03c2edd7bee4d44a15e28b3ce07414399725898e73'
BASE = 'https://hf-mirror.com'
ROOT = Path(__file__).resolve().parents[1]


class ModelRangeReader(RangeReader):
    def _authorize(self):
        if self._token and self._expires > time.time() + 30:
            return
        _, _, body = self._request(f'{BASE}/api/models/{REPO}/xet-read-token/{REVISION}')
        try:
            metadata = json.loads(body)
            token, cas_url, expires = metadata['accessToken'], metadata['casUrl'], float(metadata['exp'])
            parsed = urlsplit(cas_url)
            if (parsed.scheme != 'https' or not (parsed.hostname or '').endswith('.xethub.hf.co')
                    or parsed.query or parsed.fragment or not isinstance(token, str) or not token
                    or expires <= time.time()):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise RangeReadError('Invalid public model Xet authorization metadata') from None
        self._token, self._cas_url, self._expires = token, cas_url.rstrip('/'), expires


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract_and_check(path, destination):
    import torch
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    torch.set_num_threads(2)
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError('Expected a checkpoint mapping')
    candidates = [('', obj)] + [(key, obj[key]) for key in ('state_dict', 'model', 'model_state_dict') if key in obj]
    selected = [(key, value) for key, value in candidates if isinstance(value, dict) and value
                and all(isinstance(k, str) and torch.is_tensor(v) for k, v in value.items())]
    if len(selected) != 1:
        raise ValueError('Expected exactly one unambiguous tensor state_dict')
    container, state = selected[0]
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=91)
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all() for value in state.values() if value.is_floating_point()):
        raise ValueError('Checkpoint contains non-finite tensors')
    pure = destination / 'maskrcnn_resnet50_fpn_v2_coco_mirror_state_dict.pt'
    temporary = pure.with_suffix('.partial')
    torch.save(state, temporary)
    temporary.replace(pure)
    return {'checkpoint_top_keys': sorted(obj.keys()), 'state_dict_container': container,
            'state_dict_keys': len(state), 'tensor_dtypes': dict(Counter(str(v.dtype) for v in state.values())),
            'torchvision_version': torchvision.__version__, 'torch_version': str(torch.__version__),
            'strict_load': 'passed; torchvision maskrcnn_resnet50_fpn_v2 num_classes=91',
            'parameters': sum(v.numel() for v in model.parameters()),
            'pure_state_dict': str(pure), 'pure_state_dict_sha256': digest(pure),
            'pure_state_dict_bytes': pure.stat().st_size,
            'class_heads': {key: list(state[key].shape) for key in
                ('roi_heads.box_predictor.cls_score.weight', 'roi_heads.box_predictor.bbox_pred.weight',
                 'roi_heads.mask_predictor.mask_fcn_logits.weight')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'pretrained')
    parser.add_argument('--max-network-mb', type=int, default=450)
    args = parser.parse_args()
    if not 200 <= args.max_network_mb <= 500:
        parser.error('Network budget must be between 200 and 500 MB')
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / FILENAME
    partial = destination / (FILENAME + '.partial')
    receipt_path = destination / 'maskrcnn_resnet50_fpn_v2.source.json'
    receipt = {'status': 'started', 'mirror': BASE, 'repository': REPO, 'revision': REVISION,
               'filename': FILENAME, 'expected_bytes': SIZE, 'expected_sha256': SHA256,
               'xet_hash': XET_HASH, 'proxy': False, 'tls_verification': '/etc/ssl/certs/ca-certificates.crt',
               'upstream_factory': 'torchvision.models.detection.maskrcnn_resnet50_fpn_v2',
               'upstream_weights': 'MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1',
               'upstream_official_url': 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth',
               'upstream_sha256_prefix_confirmed_in_torchvision': '73cbd019',
               'upstream_full_sha256_reported_by_mirror': '73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e',
               'upstream_bitwise_or_tensor_equality_independently_verified': False,
               'scope': 'Mirror file SHA256 and strict torchvision compatibility verified locally; upstream equality remains publisher-reported.',
               'references': [
                   {'url': f'https://huggingface.co/{REPO}/blob/{REVISION}/README.md', 'role': 'Pinned mirror provenance; unchanged tensor claim is publisher-reported'},
                   {'url': 'https://raw.githubusercontent.com/pytorch/vision/v0.22.1/torchvision/models/detection/mask_rcnn.py', 'role': 'Official implementation and upstream weight URL'},
                   {'url': 'https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html', 'role': 'Official v2 architecture and published COCO benchmarks'},
                   {'url': 'https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html', 'role': 'Official COCO transfer and replacing class-specific heads'},
                   {'url': 'https://arxiv.org/abs/1703.06870', 'role': 'Mask R-CNN / RoIAlign / instance mask branch'},
                   {'url': 'https://arxiv.org/abs/1612.03144', 'role': 'FPN multiscale representations'},
                   {'url': 'https://arxiv.org/abs/2111.11429', 'role': 'Detection transfer baseline underlying torchvision v2'},
                   {'url': 'https://arxiv.org/abs/1806.01054', 'role': 'RGB-D multilayer feature fusion precedent; semantic segmentation, not this instance experiment'},
                   {'url': 'https://arxiv.org/abs/2011.06961', 'role': 'Efficient RGB-D feature weighting precedent; semantic segmentation'},
                   {'url': 'https://arxiv.org/abs/2302.05543', 'role': 'Zero-initialized residual condition branch precedent; applying to this detector is an engineering hypothesis'}],
               'started_at_unix': time.time()}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    reader = ModelRangeReader(REPO, REVISION, XET_HASH, SIZE, max_network_bytes=args.max_network_mb * 1024**2)
    try:
        _, _, tree = reader._request(f'{BASE}/api/models/{REPO}/tree/{REVISION}?recursive=true&expand=true')
        matching = [entry for entry in json.loads(tree) if entry.get('path') == FILENAME]
        if len(matching) != 1 or matching[0]['size'] != SIZE or matching[0].get('lfs', {}).get('oid') != SHA256 or matching[0].get('xetHash') != XET_HASH:
            raise ValueError('Pinned mirror metadata does not match expected file identity')
        for name in ('README.md', 'LICENSE', 'NOTICE'):
            _, _, data = reader._request(f'{BASE}/{REPO}/raw/{REVISION}/{name}', limit=65536)
            (destination / ('mirror_' + name)).write_bytes(data)
        if not target.exists():
            start = partial.stat().st_size if partial.exists() else 0
            if start > SIZE:
                raise ValueError('Existing partial file exceeds expected size')
            with partial.open('ab') as stream:
                for offset in range(start, SIZE, 8 * 1024**2):
                    data = reader.read_range(offset, min(8 * 1024**2, SIZE - offset))
                    stream.write(data)
                    stream.flush()
                    print(json.dumps({'downloaded_file_bytes': offset + len(data), 'total_file_bytes': SIZE,
                                      'network_bytes': reader.downloaded_bytes}), flush=True)
            if partial.stat().st_size != SIZE or digest(partial) != SHA256:
                raise ValueError('Downloaded mirror checkpoint size or SHA256 mismatch')
            partial.replace(target)
        if target.stat().st_size != SIZE or digest(target) != SHA256:
            raise ValueError('Existing mirror checkpoint size or SHA256 mismatch')
        receipt.update(actual_bytes=target.stat().st_size, actual_sha256=digest(target),
                       checkpoint_path=str(target), **extract_and_check(target, destination))
        receipt['status'] = 'verified'
        print(json.dumps({'status': 'verified', 'checkpoint': str(target), 'sha256': receipt['actual_sha256'],
                          'pure_state_dict': receipt['pure_state_dict'], 'strict_load': receipt['strict_load']}), flush=True)
    except Exception as exc:
        receipt.update(status='failed', error_type=type(exc).__name__)
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}), flush=True)
        # RangeReader sanitizes signed URL/token details; avoid raw transport traces.
        if isinstance(exc, (RangeReadError, ValueError)):
            print(str(exc), flush=True)
        raise SystemExit(1) from None
    finally:
        receipt.update(network_bytes=reader.downloaded_bytes, finished_at_unix=time.time())
        reader.close()
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
