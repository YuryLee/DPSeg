"""Convert a selected, scene-disjoint GraspNet RGB-D subset without resizing.

Object identities are the default classes: raw label 1 is GraspNet object 000.
GraspNet labels do not distinguish two instances of the same object identity in
one frame; such duplicates cannot be represented by this converter.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rgbd_seg.data import load_rgbd


def parse_indices(value):
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            result = [int(item.strip()) for item in value.split(",")]
        except ValueError as exc:
            raise ValueError("Scene/frame indices must be comma-separated integers") from exc
    else:
        result = list(value or [])
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise ValueError("Scene/frame indices must be integers")
    if len(set(result)) != len(result):
        raise ValueError("Duplicate scene/frame indices are not allowed")
    return sorted(result)


def _source_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Source path escapes source directory: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing selected source file: {path}")
    return path


def _positive_scale(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("depth_scale must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError("depth_scale must be finite and positive")
    return result


def _metadata_scale(path):
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ValueError("Supply --depth-scale explicitly: reading MAT metadata requires an already installed scipy") from exc
    try:
        value = np.asarray(loadmat(path)["factor_depth"])
    except (KeyError, ValueError, OSError) as exc:
        raise ValueError(f"Cannot read factor_depth from {path}; supply --depth-scale explicitly") from exc
    if value.size != 1:
        raise ValueError(f"Expected scalar factor_depth in {path}")
    return 1.0 / _positive_scale(value.item())


def prepare_graspnet(source, output, train_scenes, val_scenes, test_scenes=None,
                     camera="realsense", frames=None, depth_scale=None,
                     class_mode="identity", max_frames_per_scene=None):
    """Copy all requested triples and publish manifest.json only after validation."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if camera not in ("realsense", "kinect"):
        raise ValueError("camera must be realsense or kinect")
    if class_mode not in ("identity", "object"):
        raise ValueError("class_mode must be identity or object")
    available_mode = isinstance(frames, str) and frames.strip() == "available"
    if max_frames_per_scene is not None:
        if isinstance(max_frames_per_scene, bool) or not isinstance(max_frames_per_scene, int) or max_frames_per_scene <= 0:
            raise ValueError("max_frames_per_scene must be a positive integer")
        if not available_mode:
            raise ValueError("max_frames_per_scene is supported only with --frames available; explicit lists are always used in full")
    if not available_mode:
        frames = parse_indices(frames)
        if not frames or any(frame < 0 or frame > 255 for frame in frames):
            raise ValueError("Provide at least one frame index in [0, 255], or 'available'")
    splits = {"train": parse_indices(train_scenes), "val": parse_indices(val_scenes), "test": parse_indices(test_scenes)}
    seen = set()
    for split, scenes in splits.items():
        if seen.intersection(scenes):
            raise ValueError("Scenes must be disjoint across train, val, and test")
        seen.update(scenes)
        allowed = range(100, 190) if split == "test" else range(100)
        if any(scene not in allowed for scene in scenes):
            raise ValueError("Train/val scenes must be 0–99; test scenes must be official test scenes 100–189")
    if not seen:
        raise ValueError("Select at least one scene")

    scale = _positive_scale(depth_scale) if depth_scale is not None else None
    scale_source = "explicit" if scale is not None else "factor_depth_metadata"
    jobs = []
    selected_frames_by_scene = {}
    discovered_frames_by_scene = {}
    for split, scenes in splits.items():
        for scene in scenes:
            prefix = Path("scenes") / f"scene_{scene:04d}" / camera
            if available_mode:
                rgb_folder = (source / prefix / "rgb").resolve()
                if not rgb_folder.is_relative_to(source):
                    raise ValueError("RGB directory escapes source directory")
                if not rgb_folder.is_dir():
                    raise FileNotFoundError(f"Missing RGB directory: {rgb_folder}")
                discovered = sorted(int(path.stem) for path in rgb_folder.iterdir()
                                    if path.is_file() and re.fullmatch(r"\d{4}\.png", path.name))
                if not discovered or any(frame > 255 for frame in discovered):
                    raise ValueError(f"Expected available RGB frames in [0, 255] for scene {scene}")
                discovered_frames_by_scene[str(scene)] = discovered
                selected = discovered[:max_frames_per_scene] if max_frames_per_scene is not None else discovered
            else:
                selected = frames
            selected_frames_by_scene[str(scene)] = list(selected)
            for frame in selected:
                name = f"{frame:04d}.png"
                files = {kind: _source_file(source, prefix / folder / name)
                         for kind, folder in (("rgb", "rgb"), ("depth", "depth"), ("instances", "label"))}
                if depth_scale is None:
                    metadata = _source_file(source, prefix / "meta" / f"{frame:04d}.mat")
                    frame_scale = _metadata_scale(metadata)
                    if scale is not None and not math.isclose(scale, frame_scale, rel_tol=1e-9):
                        raise ValueError("Selected frames have inconsistent depth units")
                    scale = frame_scale
                sample = load_rgbd(files["rgb"], files["depth"], scale)
                with Image.open(files["instances"]) as image:
                    instances = np.array(image)
                if instances.ndim != 2 or instances.dtype.kind not in "iu" or instances.shape != tuple(sample["image"].shape[-2:]):
                    raise ValueError("Label PNG must be a single-channel integer image matching RGB resolution")
                ids = [int(value) for value in np.unique(instances) if value != 0]
                if any(value < 1 or value > 88 for value in ids):
                    raise ValueError("GraspNet object label IDs must be in [1, 88], with zero background")
                sid = f"scene_{scene:04d}_{camera}_{frame:04d}"
                record = {"id": sid, "split": split,
                          "labels": {str(value): value if class_mode == "identity" else 1 for value in ids},
                          "source_scene": scene, "source_frame": frame, "source_camera": camera,
                          "source_files": {kind: str(path.relative_to(source)) for kind, path in files.items()}}
                for kind in files:
                    record[kind] = f"{kind}/{sid}.png"
                jobs.append((files, record))

    manifest = {
        "provenance": "public_graspnet_subset", "source_dataset": "GraspNet-1Billion",
        "source_url": "https://graspnet.net/datasets.html", "source_root": str(source),
        "depth_scale": scale, "depth_scale_source": scale_source,
        "classes": [f"graspnet_object_{index:03d}" for index in range(88)] if class_mode == "identity" else ["object"],
        "class_scheme": "object_identity" if class_mode == "identity" else "category_agnostic",
        "class_mode": class_mode, "camera": camera,
        "selected_frames": "available" if available_mode else frames,
        "selected_frames_by_scene": selected_frames_by_scene,
        "frame_selection": {"mode": "available" if available_mode else "explicit", "strategy": "first_sorted" if max_frames_per_scene is not None else "all_selected", "max_frames_per_scene": max_frames_per_scene},
        "scene_splits": splits,
        "split_protocol": "Custom scene-disjoint subset; train/val from official scenes 0–99; test from 100–189",
        "annotation_note": "PNG ID n maps to GraspNet object n-1. Same-identity duplicate instances within one frame cannot be distinguished. RGB/depth/mask bytes are copied without resizing.",
        "samples": [record for _, record in jobs],
    }
    if available_mode:
        manifest["discovered_frames_by_scene"] = discovered_frames_by_scene
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for kind in ("rgb", "depth", "instances"):
            (staging / kind).mkdir()
        for files, record in jobs:
            for kind, path in files.items():
                shutil.copyfile(path, staging / record[kind])
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Output appeared during conversion: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-scenes", default="", help="Comma-separated scene IDs, 0–99")
    parser.add_argument("--val-scenes", default="", help="Comma-separated scene IDs, 0–99")
    parser.add_argument("--test-scenes", default="", help="Optional official test scenes, 100–189")
    parser.add_argument("--frames", required=True, help="Comma-separated frame IDs, 0–255, or 'available' to discover each scene's RGB frames; every selected RGB/depth/label triple must exist")
    parser.add_argument("--max-frames-per-scene", type=int, help="With --frames available, take at most this many frames per scene in ascending frame order")
    parser.add_argument("--camera", choices=("realsense", "kinect"), default="realsense")
    parser.add_argument("--depth-scale", type=float, help="Raw depth to metres; otherwise read factor_depth from every selected MAT file")
    parser.add_argument("--class-mode", choices=("identity", "object"), default="identity")
    args = parser.parse_args()
    try:
        manifest = prepare_graspnet(**vars(args))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(Path(args.output).resolve()), "images": len(manifest["samples"]), "class_scheme": manifest["class_scheme"], "depth_scale": manifest["depth_scale"]}, indent=2))


if __name__ == "__main__":
    main()
