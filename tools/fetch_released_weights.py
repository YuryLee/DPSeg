"""Download the published course weights and verify their original bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("main", "all"), default="main",
                        help="main: three files for S0–S4; all: include historical ablations")
    parser.add_argument("--directory", type=Path, default=ROOT / "pretrained/dpseg_weights")
    parser.add_argument("--endpoint", choices=("https://hf-mirror.com", "https://huggingface.co"),
                        default="https://hf-mirror.com", help="Download mirror, or the official Hub")
    parser.add_argument("--verify-only", action="store_true", help="Check local files without network access")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/released_weights.json").read_text())
    assets = [a for a in manifest["weights"] if args.group == "all" or a["group"] == "main"]
    args.directory.mkdir(parents=True, exist_ok=True)
    # Keep downloads direct, consistent with the project's download policy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for asset in assets:
        path = args.directory / asset["filename"]
        if path.exists():
            if path.stat().st_size != asset["bytes"] or digest(path) != asset["sha256"]:
                raise ValueError(f"Existing file differs from release; choose another directory: {path}")
        elif args.verify_only:
            raise FileNotFoundError(path)
        else:
            temporary = path.with_suffix(path.suffix + ".partial")
            print(f"Downloading {asset['filename']} ({asset['bytes'] / 2**20:.2f} MiB)", flush=True)
            size, checksum = 0, hashlib.sha256()
            try:
                url = asset["url"].replace("https://huggingface.co", args.endpoint, 1)
                with opener.open(url, timeout=120) as response, temporary.open("wb") as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        size += len(block)
                        if size > asset["bytes"]:
                            raise ValueError(f"Oversized response: {asset['filename']}")
                        checksum.update(block)
                        stream.write(block)
                if size != asset["bytes"] or checksum.hexdigest() != asset["sha256"]:
                    raise ValueError(f"Download checksum differs: {asset['filename']}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Verified {asset['filename']}", flush=True)
    print(f"Verified {len(assets)} weight files in {args.directory.resolve()}")


if __name__ == "__main__":
    main()
