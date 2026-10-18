"""Read course RGB/depth and optional COCO annotations directly from its ZIP.

No files are extracted. Inference samples contain only RGB and identity fields;
GT and depth are separate, explicit APIs. Official IDs and annotation areas are
preserved. No support mask, camera, composition, or GT enters __getitem__.
"""
from __future__ import annotations

import copy
import csv
import gzip
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch


class CourseArchive:
    def __init__(self, zip_path, split="test"):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.zip_path = Path(zip_path).resolve()
        self.split = split
        self._archive = None
        self._archive_pid = None
        self.annotation_members = []
        images, categories = {}, None
        with ZipFile(self.zip_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate ZIP member names are ambiguous")
            self.members = frozenset(names)
            self.annotation_members = sorted(n for n in names if n.startswith(f"annotations/{split}/") and n.endswith(".json.gz"))
            for member in self.annotation_members:
                document = self._read_annotation(archive, member)
                current = document.get("categories", [])
                if categories is not None and categories != current:
                    raise ValueError("Categories differ across annotation shards")
                categories = current
                for record in document["images"]:
                    image_id = record["id"]
                    if not isinstance(image_id, int) or isinstance(image_id, bool) or image_id in images:
                        raise ValueError("Image IDs must be unique integers")
                    images[image_id] = record
            if not images and split == "test" and "test_images.json" in self.members:
                metadata = json.loads(archive.read("test_images.json"))
                records = metadata["images"] if isinstance(metadata, dict) else metadata
                for record in records:
                    if record["id"] in images:
                        raise ValueError("Duplicate image ID")
                    images[record["id"]] = record
                categories = metadata.get("categories", []) if isinstance(metadata, dict) else []
            if not images:
                raise ValueError(f"No image metadata for {split}")
            self.submission_rows = None
            if split == "test" and "submission.csv" in self.members:
                reader = csv.DictReader(io.StringIO(archive.read("submission.csv").decode("utf-8-sig")))
                if reader.fieldnames != ["image_id", "PredictionString"]:
                    raise ValueError("Unexpected submission.csv columns")
                self.submission_rows = list(reader)
                order = [int(row["image_id"]) for row in self.submission_rows]
                if len(order) != len(set(order)) or set(order) != set(images):
                    raise ValueError("Submission image IDs and metadata must match exactly")
            else:
                order = sorted(images)
            self.images = [copy.deepcopy(images[i]) for i in order]
            self.image_ids = list(order)
            self.categories = copy.deepcopy(categories or [])
            rgb_names = set()
            for record in self.images:
                for field in ("file_name", "depth_file_name"):
                    if field not in record:
                        if field == "depth_file_name":
                            continue
                        raise ValueError("Missing RGB file_name")
                    name = record[field]
                    path = PurePosixPath(name)
                    if path.is_absolute() or ".." in path.parts or name not in self.members:
                        raise ValueError(f"Missing or unsafe ZIP image member: {name}")
                    expected = "rgb" if field == "file_name" else "depth"
                    if path.parts[:2] != (expected, split):
                        raise ValueError("Image member does not belong to requested split")
                if record["file_name"] in rgb_names:
                    raise ValueError("Multiple image IDs reference the same RGB")
                rgb_names.add(record["file_name"])
                if record["width"] <= 0 or record["height"] <= 0:
                    raise ValueError("Invalid image dimensions")
                if "depth_file_name" in record:
                    scale = record.get("depth_scale_m")
                    if scale is None or not math.isfinite(scale) or scale <= 0:
                        raise ValueError("Depth scale must be finite and positive")

    def _image_archive(self):
        """Reuse a read-only handle within one process; reopen after a fork."""
        pid = os.getpid()
        if self._archive is None or self._archive_pid != pid:
            self.close()
            self._archive = ZipFile(self.zip_path, mode="r")
            self._archive_pid = pid
        return self._archive

    def close(self):
        if self._archive is not None:
            self._archive.close()
            self._archive = None
        self._archive_pid = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _read_annotation(archive, member):
        return json.loads(gzip.decompress(archive.read(member)))

    def __len__(self):
        return len(self.images)

    def load_rgb(self, index):
        """Return an RGB PIL image at its original dimensions; no GT is read."""
        record = self.images[index]
        archive = self._image_archive()
        with Image.open(io.BytesIO(archive.read(record["file_name"]))) as original:
            if original.size != (record["width"], record["height"]):
                raise ValueError("RGB dimensions disagree with official metadata")
            return original.convert("RGB")

    def __getitem__(self, index):
        record = self.images[index]
        image = np.array(self.load_rgb(index), dtype=np.uint8, copy=True)
        return {"image": torch.from_numpy(image).permute(2, 0, 1).float().div_(255),
                "image_id": record["id"], "file_name": record["file_name"]}

    def load_depth(self, index):
        """Explicit optional raw-depth input: float meters + bool valid, H×W.

        Uses each image's depth scale and invalid code. This is never part of
        the default RGB inference item.
        """
        record = self.images[index]
        archive = self._image_archive()
        with Image.open(io.BytesIO(archive.read(record["depth_file_name"]))) as image:
            raw = np.array(image)
        if raw.ndim != 2 or raw.shape != (record["height"], record["width"]) or raw.dtype.kind not in "iu":
            raise ValueError("Expected aligned integer raw depth")
        valid = (raw != record.get("depth_invalid_code", 0)) & (raw > 0)
        depth = raw.astype(np.float32) * record["depth_scale_m"]
        depth[~valid] = 0
        return depth, valid

    def coco_ground_truth(self):
        """Return independent COCO dict; never remap IDs/areas/iscrowd/categories.

        Raises if no annotation shards exist. Presence of this API does not
        authorize using test labels for tuning, training, or input construction.
        """
        if not self.annotation_members:
            raise ValueError("No ground-truth annotation shards are present")
        annotations, seen = [], set()
        image_ids = {record["id"] for record in self.images}
        category_ids = {category["id"] for category in self.categories}
        with ZipFile(self.zip_path) as archive:
            for member in self.annotation_members:
                document = self._read_annotation(archive, member)
                shard_ids = {image["id"] for image in document["images"]}
                for annotation in document["annotations"]:
                    if annotation["id"] in seen:
                        raise ValueError("Duplicate annotation ID")
                    if annotation["image_id"] not in image_ids or annotation["image_id"] not in shard_ids:
                        raise ValueError("Annotation points to an unknown image")
                    if annotation["category_id"] not in category_ids:
                        raise ValueError("Annotation points to an unknown category")
                    seen.add(annotation["id"])
                    annotations.append(annotation)
        return {"info": {"source": str(self.zip_path), "split": self.split},
                "images": copy.deepcopy(self.images), "annotations": annotations,
                "categories": copy.deepcopy(self.categories)}
