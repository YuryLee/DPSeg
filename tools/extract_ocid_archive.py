#!/usr/bin/env python3
"""Verify the official OCID archive, then extract original files in one pass."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

EXPECTED_BYTES = 6822716319
EXPECTED_MD5 = 'aa0eff01f75104c40e4243e6f892a757'


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--destination-parent', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    if (archive.name != 'OCID-dataset.tar.gz' or archive.stat().st_size != EXPECTED_BYTES
            or archive.with_name(archive.name + '.aria2').exists()):
        raise ValueError('The official download is incomplete or has the wrong name/size')
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    md5, sha = hashlib.md5(), hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            md5.update(chunk); sha.update(chunk)
    if md5.hexdigest() != EXPECTED_MD5:
        raise ValueError('Official OCID MD5 mismatch; archive retained')
    parent = args.destination_parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / 'OCID-dataset'
    if destination.exists():
        raise FileExistsError(destination)
    staging = parent / ('.extracting-OCID-' + EXPECTED_MD5[:12])
    staging.mkdir(exist_ok=False)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    state = {'status': 'extracting', 'archive': str(archive), 'archive_bytes': EXPECTED_BYTES,
             'official_md5': md5.hexdigest(), 'archive_sha256': sha.hexdigest(),
             'started_utc': datetime.now(timezone.utc).isoformat(), 'destination': str(destination),
             'files': 0, 'uncompressed_bytes': 0, 'archive_deleted': False}
    write(args.receipt, state)
    inventory, seen = [], set()
    reserve = 8 * 1024**3
    last = time.monotonic()
    try:
        with gzip.open(archive, 'rb') as compressed:
            with tarfile.open(fileobj=compressed, mode='r|') as source:
                for member in source:
                    name = PurePosixPath(member.name)
                    if (name.is_absolute() or '..' in name.parts or '\\' in member.name
                            or not name.parts or name.parts[0] != 'OCID-dataset'):
                        raise ValueError('Unsafe archive path: ' + member.name)
                    target = staging.joinpath(*name.parts)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile() or member.name in seen:
                        raise ValueError('Expected unique original regular files: ' + member.name)
                    seen.add(member.name)
                    if shutil.disk_usage(parent).free < member.size + reserve:
                        raise OSError('Insufficient space with 8 GiB reserve')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest, count = hashlib.sha256(), 0
                    with source.extractfile(member) as stream, target.open('xb') as output:
                        for chunk in iter(lambda: stream.read(1024**2), b''):
                            output.write(chunk); digest.update(chunk); count += len(chunk)
                    if count != member.size or target.stat().st_size != count:
                        raise ValueError('Extracted size mismatch: ' + member.name)
                    inventory.append({'name': member.name, 'bytes': count, 'sha256': digest.hexdigest()})
                    state['files'] += 1
                    state['uncompressed_bytes'] += count
                    if time.monotonic() - last > 15:
                        write(args.receipt, state)
                        print(json.dumps({key: state[key] for key in ('status', 'files', 'uncompressed_bytes')}), flush=True)
                        last = time.monotonic()
            # Consume the gzip trailer even if TAR stopped at its end markers.
            while compressed.read(1024**2):
                pass
        if not inventory:
            raise ValueError('Archive contained no regular dataset files')
        (staging / 'OCID-dataset').rename(destination)
        staging.rmdir()
        inventory_path = args.receipt.with_name(args.receipt.stem + '_inventory.json')
        write(inventory_path, {'files': inventory})
        state.update(status='completed', gzip_crc_checked=True,
                     completed_utc=datetime.now(timezone.utc).isoformat(), inventory=str(inventory_path))
        write(args.receipt, state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status='failed', error=str(error))
        write(args.receipt, state)
        raise


if __name__ == '__main__':
    main()
