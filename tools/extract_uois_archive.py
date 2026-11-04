#!/usr/bin/env python3
"""CRC-check UOAIS/OSD while extracting; remove an archive only after validation.

Uses the already saved author ZIP directory inventory and never downloads data.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--expected-bytes', type=int, required=True)
    parser.add_argument('--destination-parent', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--delete-after-verification', action='store_true')
    args = parser.parse_args()
    archive = args.archive.resolve()
    allowed = {'UOAIS-Sim.zip': 'UOAIS-Sim', 'OSD-0.2-depth.zip': 'OSD-0.2-depth'}
    if archive.name not in allowed or archive.stat().st_size != args.expected_bytes:
        raise ValueError('Archive filename or expected download size does not match')
    inventory = json.loads(args.inventory.read_text())
    args.destination_parent.mkdir(parents=True, exist_ok=True)
    destination = args.destination_parent.resolve() / allowed[archive.name]
    if destination.exists() or args.receipt.exists():
        raise FileExistsError('Use a new destination and receipt; never overwrite an existing dataset')
    total = inventory['summary']['uncompressed_bytes']
    reserve = 8 * 1024**3
    overhead = inventory['summary']['files'] * 4096
    if shutil.disk_usage(args.destination_parent).free < total + overhead + reserve:
        raise OSError('Insufficient free space for extraction plus an 8 GiB reserve')
    digest = sha256(archive)
    staging = args.destination_parent.resolve() / ('.extracting-' + archive.stem + '-' + digest[:12])
    staging.mkdir(exist_ok=False)
    state = {'status': 'extracting', 'archive': str(archive), 'archive_bytes': args.expected_bytes,
             'archive_sha256': digest, 'inventory_sha256': sha256(args.inventory),
             'destination': str(destination), 'started_utc': datetime.now(timezone.utc).isoformat(),
             'verified_files': 0, 'verified_uncompressed_bytes': 0, 'archive_deleted': False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.receipt, state)
    try:
        with zipfile.ZipFile(archive) as source:
            actual = [{'name': i.filename, 'compressed_bytes': i.compress_size,
                       'uncompressed_bytes': i.file_size, 'directory': i.is_dir()} for i in source.infolist()]
            if actual != inventory['entries']:
                raise ValueError('Archive directory differs from the author metadata inventory')
            seen = set()
            for info in source.infolist():
                name = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (name.is_absolute() or '..' in name.parts or '\\' in info.filename
                        or not name.parts or name.parts[0] != allowed[archive.name]
                        or stat.S_ISLNK(mode) or info.filename in seen):
                    raise ValueError('Unsafe or duplicate archive member: ' + info.filename)
                seen.add(info.filename)
                target = staging.joinpath(*name.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if shutil.disk_usage(staging).free < reserve:
                    raise OSError('Free disk reserve fell below 8 GiB')
                target.parent.mkdir(parents=True, exist_ok=True)
                # ZipExtFile checks each member CRC on the final read.
                with source.open(info) as stream, target.open('xb') as output:
                    shutil.copyfileobj(stream, output, length=1024 * 1024)
                if target.stat().st_size != info.file_size:
                    raise ValueError('Extracted member has wrong size: ' + info.filename)
                state['verified_files'] += 1
                state['verified_uncompressed_bytes'] += info.file_size
                if state['verified_files'] % 1000 == 0:
                    write_json(args.receipt, state)
        if (state['verified_files'] != inventory['summary']['files']
                or state['verified_uncompressed_bytes'] != total):
            raise ValueError('Extracted file count or total size differs')
        (staging / allowed[archive.name]).rename(destination)
        staging.rmdir()
        state.update(status='extraction_verified', all_member_crc_checked=True,
                     completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.receipt, state)
        if args.delete_after_verification:
            # The verified archive itself is the only file this tool deletes.
            if archive.stat().st_size != args.expected_bytes or sha256(archive) != digest:
                raise ValueError('Archive changed after extraction; refusing deletion')
            archive.unlink()
            state['archive_deleted'] = True
        state['status'] = 'completed'
        write_json(args.receipt, state)
        print(json.dumps(state, ensure_ascii=False))
    except BaseException as error:
        state.update(status='failed', error=str(error))
        write_json(args.receipt, state)
        raise


if __name__ == '__main__':
    main()
