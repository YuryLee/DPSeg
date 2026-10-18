"""Manifest-based RGB-D data, with explicit depth units and instance annotations."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def _scale(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("depth_scale must be finite and positive (raw depth to metres)")
    return value


def _contained(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Sample paths must be nonempty relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Sample path escapes dataset root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_rgbd(rgb_path, depth_path, depth_scale):
    """Load RGB and raw single-channel depth; invalid depth becomes zero."""
    scale = _scale(depth_scale)
    with Image.open(rgb_path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    depth_path = Path(depth_path)
    if depth_path.suffix.lower() == ".npy":
        raw = np.load(depth_path, allow_pickle=False)
    elif depth_path.suffix.lower() == ".png":
        with Image.open(depth_path) as image:
            raw = np.array(image)
        if raw.dtype.kind not in "iu" or raw.dtype.itemsize < 2:
            raise ValueError("PNG depth must be a raw 16-bit single-channel image")
    else:
        raise ValueError("Depth must be raw 16-bit PNG or numeric NPY")
    if raw.ndim != 2 or raw.dtype.kind not in "iuf":
        raise ValueError("Depth must be a numeric single-channel H×W array, not a color image")
    if raw.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth size mismatch: {rgb.shape[:2]} vs {raw.shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        depth = raw.astype(np.float64) * scale
    valid = np.isfinite(depth) & (depth > 0) & (depth <= np.finfo(np.float32).max)
    depth = np.where(valid, depth, 0).astype(np.float32)
    return {
        "image": torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0,
        "depth": torch.from_numpy(depth.copy())[None],
        "depth_valid": torch.from_numpy(valid.astype(np.float32))[None],
    }


def load_instance_target(instance_path, labels, image_id, size=None):
    """Decode instance IDs; zero is background and boxes use exclusive x2/y2."""
    if instance_path is None:
        if size is None:
            raise ValueError("size is required for an unannotated sample")
        instances = np.zeros(size, dtype=np.uint16)
    else:
        if Path(instance_path).suffix.lower() == ".npy":
            instances = np.load(instance_path, allow_pickle=False)
        else:
            with Image.open(instance_path) as image:
                instances = np.array(image)
        if instances.ndim != 2 or instances.dtype.kind not in "iu" or np.any(instances < 0):
            raise ValueError("Instance mask must be a nonnegative integer H×W array")
        if size is not None and instances.shape != tuple(size):
            raise ValueError("RGB/instance mask size mismatch")
    h, w = instances.shape
    boxes, masks, classes, areas = [], [], [], []
    for instance_id in np.unique(instances):
        if instance_id == 0:
            continue
        label = labels.get(str(int(instance_id)), labels.get(int(instance_id)))
        if isinstance(label, bool) or not isinstance(label, int) or label <= 0:
            raise ValueError(f"Missing or invalid class label for instance {instance_id}")
        mask = instances == instance_id
        y, x = np.nonzero(mask)
        boxes.append([int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1])
        masks.append(mask)
        classes.append(label)
        areas.append(int(mask.sum()))
    count = len(boxes)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(count, 4),
        "masks": torch.from_numpy(np.stack(masks).astype(np.uint8)) if masks else torch.zeros((0, h, w), dtype=torch.uint8),
        "labels": torch.tensor(classes, dtype=torch.int64),
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "area": torch.tensor(areas, dtype=torch.float32),
        "iscrowd": torch.zeros(count, dtype=torch.int64),
    }


def load_separate_instance_target(mask_paths, labels, image_id, size):
    """Preserve binary per-instance masks, including overlapping visible masks."""
    h, w = size
    boxes, masks, areas = [], [], []
    if len(mask_paths) != len(labels):
        raise ValueError("Separate mask paths and labels must have equal lengths")
    for path in mask_paths:
        with Image.open(path) as image:
            raw = np.array(image)
        if raw.ndim != 2 or raw.dtype.kind not in "iub" or raw.shape != (h, w):
            raise ValueError("Each instance mask must be a single-channel integer image matching RGB dimensions")
        if not (np.isin(raw, [0, 1]).all() or np.isin(raw, [0, 255]).all()):
            raise ValueError("Each instance mask must be binary (0/1 or 0/255)")
        mask = raw != 0
        if not mask.any():
            raise ValueError("Separate instance masks must be nonempty; omit invisible instances")
        y, x = np.nonzero(mask)
        boxes.append([int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1])
        masks.append(mask)
        areas.append(int(mask.sum()))
    count = len(boxes)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(count, 4),
        "masks": torch.from_numpy(np.stack(masks).astype(np.uint8)) if masks else torch.zeros((0, h, w), dtype=torch.uint8),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "area": torch.tensor(areas, dtype=torch.float32),
        "iscrowd": torch.zeros(count, dtype=torch.int64),
    }


class RGBDDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.depth_scale = _scale(self.manifest["depth_scale"])
        self.classes = self.manifest["classes"]
        if not isinstance(self.classes, list) or not self.classes or any(not isinstance(x, str) or not x for x in self.classes):
            raise ValueError("classes must be a nonempty list of class names")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("Class names must be unique")
        if self.manifest.get("provenance") not in ("synthetic_smoke", "course_data", "public_graspnet_subset", "public_tless_subset"):
            raise ValueError("provenance must be synthetic_smoke, course_data, public_graspnet_subset or public_tless_subset")
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        self.split = split
        self.samples = []
        ids, rgb_paths = set(), set()
        records = self.manifest["samples"]
        if not isinstance(records, list):
            raise ValueError("samples must be a list")
        for index, record in enumerate(records):
            sid = record.get("id")
            if not isinstance(sid, str) or not sid or sid in ids:
                raise ValueError("Sample IDs must be nonempty and globally unique")
            ids.add(sid)
            if record.get("split") not in ("train", "val", "test"):
                raise ValueError(f"Invalid split for sample {sid}")
            rgb = _contained(self.root, record["rgb"])
            depth = _contained(self.root, record["depth"])
            instance = _contained(self.root, record["instances"]) if record.get("instances") else None
            separate_paths, separate_labels = None, None
            if "instance_masks" in record:
                separate = record["instance_masks"]
                if instance is not None or not isinstance(separate, list):
                    raise ValueError("instance_masks must be a list and cannot accompany an instances ID image")
                separate_paths, separate_labels = [], []
                for item in separate:
                    if not isinstance(item, dict):
                        raise ValueError("instance_masks entries must contain path and label")
                    label = item.get("label")
                    if isinstance(label, bool) or not isinstance(label, int) or not 1 <= label <= len(self.classes):
                        raise ValueError("Invalid separate instance mask class index")
                    mask_path = _contained(self.root, item.get("path"))
                    if mask_path in separate_paths:
                        raise ValueError("Separate instance mask paths must be unique within each sample")
                    separate_paths.append(mask_path)
                    separate_labels.append(label)
            if rgb in rgb_paths:
                raise ValueError("RGB files must not be reused across samples/splits")
            rgb_paths.add(rgb)
            labels = record.get("labels", {})
            if not isinstance(labels, dict):
                raise ValueError("labels must map instance IDs to positive class indices")
            for key, label in labels.items():
                if not str(key).isdigit() or int(key) <= 0 or isinstance(label, bool) or not isinstance(label, int) or not 1 <= label <= len(self.classes):
                    raise ValueError("Invalid instance ID/class index in labels")
            if record["split"] == split:
                self.samples.append({**record, "rgb_path": rgb, "depth_path": depth, "instance_path": instance,
                                     "instance_mask_paths": separate_paths, "instance_mask_labels": separate_labels,
                                     "image_id": index + 1})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        if self.split == "train" and record["instance_path"] is None and record["instance_mask_paths"] is None:
            raise ValueError(f"Training sample {record['id']} requires instance annotations")
        sample = load_rgbd(record["rgb_path"], record["depth_path"], self.depth_scale)
        if record["instance_mask_paths"] is not None:
            target = load_separate_instance_target(record["instance_mask_paths"], record["instance_mask_labels"],
                                                   record["image_id"], sample["image"].shape[-2:])
        else:
            target = load_instance_target(record["instance_path"], record.get("labels", {}), record["image_id"], sample["image"].shape[-2:])
        return sample, target


def generate_synthetic(root, n_train=48, n_val=12, n_test=8, seed=41, size=128):
    """Persist a deterministic smoke-test dataset; never overwrite existing data."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty dataset directory: {root}")
    if size < 32 or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in (n_train, n_val, n_test)):
        raise ValueError("size must be >=32 and split counts must be nonnegative integers")
    for folder in ("rgb", "depth", "instances"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    manifest = {"provenance": "synthetic_smoke", "depth_scale": 0.001, "classes": ["object"], "seed": seed, "samples": []}
    index = 0
    yy, xx = np.mgrid[:size, :size]
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        for item in range(count):
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            base = rng.uniform(90, 160, size=3)
            rgb = base + rng.normal(0, 8, (size, size, 3))
            depth = rng.normal(3000, 12, (size, size))
            instances = np.zeros((size, size), dtype=np.uint16)
            occupied = np.zeros((size, size), dtype=bool)
            labels = {}
            wanted = int(rng.integers(2, 6))
            for attempt in range(100):
                if len(labels) >= wanted:
                    break
                low, high = max(4, size // 24), max(7, size // 7)
                width, height = rng.integers(low, high, size=2)
                x = int(rng.integers(2, size - width - 2))
                y = int(rng.integers(2, size - height - 2))
                if occupied[y - 2:y + height + 2, x - 2:x + width + 2].any():
                    continue
                if rng.random() < 0.5:
                    mask = ((xx - (x + (width - 1) / 2)) / (width / 2)) ** 2 + ((yy - (y + (height - 1) / 2)) / (height / 2)) ** 2 <= 1
                else:
                    mask = (xx >= x) & (xx < x + width) & (yy >= y) & (yy < y + height)
                instance_id = len(labels) + 1
                instances[mask] = instance_id
                labels[str(instance_id)] = 1
                occupied[y:y + height, x:x + width] = True
                rgb[mask] = base + rng.uniform(-28, 28, 3) + rng.normal(0, 5, (int(mask.sum()), 3))
                depth[mask] = rng.uniform(700, 2000) + rng.normal(0, 8, int(mask.sum()))
            depth[rng.random((size, size)) < 0.025] = 0
            sid = f"{split}_{item:04d}"
            paths = {name: f"{name}/{sid}.png" for name in ("rgb", "depth", "instances")}
            Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(root / paths["rgb"])
            Image.fromarray(np.clip(depth, 0, 65535).astype(np.uint16)).save(root / paths["depth"])
            Image.fromarray(instances).save(root / paths["instances"])
            manifest["samples"].append({"id": sid, "split": split, **paths, "labels": labels})
            index += 1
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root


def dataset_fingerprint(root):
    """SHA256 of manifest bytes and every referenced data file, independent of root.

    Relative paths and byte lengths frame each file to avoid ambiguous concatenation.
    The manifest itself is included verbatim, so changes to splits, units, or labels
    also change the fingerprint. Files not referenced by the manifest are excluded.
    """
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {"manifest.json": manifest_path}
    for record in manifest["samples"]:
        for kind in ("rgb", "depth", "instances"):
            if record.get(kind):
                paths[record[kind]] = _contained(root, record[kind])
        if "instance_masks" in record:
            if not isinstance(record["instance_masks"], list):
                raise ValueError("instance_masks must be a list")
            for item in record["instance_masks"]:
                if not isinstance(item, dict):
                    raise ValueError("instance_masks entries must contain path and label")
                relative = item.get("path")
                path = _contained(root, relative)
                paths[relative] = path
    digest = hashlib.sha256()
    digest.update(b"rgbd-dataset-sha256-v1\0")
    for relative, path in sorted(paths.items()):
        name = relative.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(root):
    """Fully load every sample and report data quality without fabricated scores."""
    report = {"root": str(Path(root).resolve()), "dataset_sha256": dataset_fingerprint(root), "splits": {}, "instances": 0, "all_invalid_depth_images": 0, "per_class_instances": {}}
    depth_min, depth_max = float("inf"), 0.0
    total_pixels, valid_pixels = 0, 0
    content_groups = {}
    for split in ("train", "val", "test"):
        dataset = RGBDDataset(root, split)
        report["provenance"] = dataset.manifest["provenance"]
        report["classes"] = dataset.classes
        report["depth_scale"] = dataset.depth_scale
        entry = {"images": len(dataset), "annotated_images": 0, "instances": 0, "small_instances": 0, "medium_instances": 0, "large_instances": 0, "per_class_instances": {name: 0 for name in dataset.classes}}
        for name in dataset.classes:
            report["per_class_instances"].setdefault(name, 0)
        for index in range(len(dataset)):
            sample, target = dataset[index]
            record = dataset.samples[index]
            # Hash decoded inputs: metadata changes or PNG re-encoding should not
            # hide identical images copied into another split.
            rgb_hash = hashlib.sha256()
            rgb_hash.update(str(tuple(sample["image"].shape)).encode("ascii"))
            rgb_hash.update(sample["image"].numpy().tobytes())
            rgbd_hash = rgb_hash.copy()
            rgbd_hash.update(sample["depth"].numpy().tobytes())
            for kind, digest in (("rgb", rgb_hash.hexdigest()), ("rgbd", rgbd_hash.hexdigest())):
                content_groups.setdefault((kind, digest), []).append({"id": record["id"], "split": split})
            entry["annotated_images"] += int(record["instance_path"] is not None or record["instance_mask_paths"] is not None)
            area = target["area"]
            entry["instances"] += len(area)
            entry["small_instances"] += int((area < 32 ** 2).sum())
            entry["medium_instances"] += int(((area >= 32 ** 2) & (area < 96 ** 2)).sum())
            entry["large_instances"] += int((area >= 96 ** 2).sum())
            for label, name in enumerate(dataset.classes, start=1):
                count = int((target["labels"] == label).sum())
                entry["per_class_instances"][name] += count
                report["per_class_instances"][name] += count
            valid = sample["depth_valid"].bool()
            total_pixels += valid.numel()
            valid_pixels += int(valid.sum())
            if valid.any():
                depth_min = min(depth_min, float(sample["depth"][valid].min()))
                depth_max = max(depth_max, float(sample["depth"][valid].max()))
            else:
                report["all_invalid_depth_images"] += 1
        report["splits"][split] = entry
        report["instances"] += entry["instances"]
    report["valid_depth_fraction"] = valid_pixels / total_pixels if total_pixels else None
    report["depth_range_m"] = [depth_min, depth_max] if valid_pixels else None
    report["duplicate_content_across_splits"] = [
        {"inputs": kind, "sha256": digest, "samples": records}
        for (kind, digest), records in sorted(content_groups.items())
        if len({record["split"] for record in records}) > 1
    ]
    return report
