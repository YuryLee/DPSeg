"""Fetch the pinned COCO instance checkpoint via hf-mirror, without proxies."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests
from xet_range import RangeReader, RangeReadError


MODEL_ID = "facebook/mask2former-swin-tiny-coco-instance"
REVISION = "22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6"
DEST = Path(__file__).resolve().parents[1] / "pretrained/mask2former_swin_tiny_coco_instance"
FILES = ("README.md", "config.json", "preprocessor_config.json", "model.safetensors")


class ModelRangeReader(RangeReader):
    def _authorize(self):
        if self._token and self._expires > time.time() + 30:
            return
        url = f"{self.endpoint}/api/models/{self.repo}/xet-read-token/{self.revision}?refresh={time.time_ns()}"
        _, _, body = self._request(url, headers={"Cache-Control": "no-cache"})
        metadata = json.loads(body)
        token, cas_url, expires = metadata["accessToken"], metadata["casUrl"], float(metadata["exp"])
        parsed = urlsplit(cas_url)
        if (parsed.scheme != "https" or not (parsed.hostname or "").endswith(".xethub.hf.co")
                or parsed.query or parsed.fragment or not isinstance(token, str) or not token or expires <= time.time()):
            raise RangeReadError("Invalid public model Xet metadata")
        self._token, self._cas_url, self._expires = token, cas_url.rstrip("/"), expires


def main():
    session = requests.Session()
    session.trust_env = False
    session.verify = "/etc/ssl/certs/ca-certificates.crt"
    DEST.mkdir(parents=True, exist_ok=True)
    api_url = f"https://hf-mirror.com/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
    response = session.get(api_url, timeout=(20, 120))
    response.raise_for_status()
    metadata = response.json()
    if metadata["sha"] != REVISION or metadata["id"] != MODEL_ID:
        raise RuntimeError("Unexpected model identity/revision from mirror")
    (DEST / "mirror_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    siblings = {item["rfilename"]: item for item in metadata["siblings"]}
    receipt = {
        "model_id": MODEL_ID, "revision": REVISION,
        "official_model_card": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
        "upstream_repository": "https://github.com/facebookresearch/Mask2Former",
        "task": "COCO instance segmentation; Swin tiny; 80 foreground classes",
        "mirror_metadata_url": api_url, "download_origin": "https://hf-mirror.com",
        "trust_env": False, "proxy_used": False,
        "tls_verify": "/etc/ssl/certs/ca-certificates.crt",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "license_metadata": metadata.get("cardData", {}).get("license"),
        "license_note": "HF model card says 'other'; this revision contains no separate weight LICENSE. Preserve model card; do not infer a weight license from upstream code license.",
        "files": {},
    }
    for filename in FILES:
        expected = siblings[filename]
        url = f"https://hf-mirror.com/{MODEL_ID}/resolve/{REVISION}/{filename}"
        destination = DEST / filename
        temporary = destination.with_suffix(destination.suffix + ".partial")
        h = hashlib.sha256()
        size = 0
        if filename == "model.safetensors":
            tree_url = f"https://hf-mirror.com/api/models/{MODEL_ID}/tree/{REVISION}?recursive=true&expand=true"
            response = session.get(tree_url, timeout=(20, 120))
            response.raise_for_status()
            entry, = [item for item in response.json() if item.get("path") == filename]
            if entry["size"] != expected["size"] or entry["lfs"]["oid"] != expected["lfs"]["sha256"]:
                raise RuntimeError("Xet tree identity disagrees with pinned model metadata")
            start = temporary.stat().st_size if temporary.exists() else 0
            if start > expected["size"]:
                raise RuntimeError("Partial file exceeds expected size")
            with ModelRangeReader(MODEL_ID, REVISION, entry["xetHash"], expected["size"],
                                  max_network_bytes=400 * 1024 ** 2) as reader:
                with temporary.open("ab") as handle:
                    for offset in range(start, expected["size"], 16 * 1024 ** 2):
                        block = reader.read_range(offset, min(16 * 1024 ** 2, expected["size"] - offset))
                        handle.write(block)
                        handle.flush()
                        print(f"Weight bytes {offset + len(block)}/{expected['size']}", flush=True)
                receipt["xet_download"] = {"xet_hash": entry["xetHash"],
                    "network_bytes_this_attempt": reader.downloaded_bytes,
                    "resumed_file_bytes": start, "metadata_url": tree_url,
                    "method": "mirror-authorized Xet reconstruction; HTTPS CAS/CDN; no proxy",
                    "reason": "direct mirror resolve redirected to cas-bridge; TLS certificate verification failed; kept TLS verification enabled and used range protocol"}
            size = temporary.stat().st_size
            with temporary.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(block)
            chain = ["hf-mirror.com", "mirror-advertised *.xethub.hf.co / *.cdn.hf.co (Xet range)"]
        else:
            with session.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                chain = [urlsplit(item.url).hostname for item in response.history] + [urlsplit(response.url).hostname]
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            size += len(chunk)
                            if size > expected["size"]:
                                raise RuntimeError(f"Oversized response: {filename}")
                            handle.write(chunk)
                            h.update(chunk)
        if size != expected["size"]:
            raise RuntimeError(f"Size mismatch: {filename}")
        digest = h.hexdigest()
        if "lfs" in expected and digest != expected["lfs"]["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {filename}")
        if "lfs" not in expected:
            git_blob = hashlib.sha1(f"blob {size}\0".encode() + temporary.read_bytes()).hexdigest()
            if git_blob != expected["blobId"]:
                raise RuntimeError(f"Git blob mismatch: {filename}")
        temporary.replace(destination)
        receipt["files"][filename] = {"sha256": digest, "bytes": size, "url": url,
            "redirect_hosts": chain, "expected": expected, "verified": True}
        print(f"Verified {filename}: {size} bytes, sha256={digest}", flush=True)
        (DEST / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["verified_file_bytes"] = sum(item["bytes"] for item in receipt["files"].values())
    (DEST / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
