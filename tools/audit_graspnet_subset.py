"""Audit original paired GraspNet frames and suggest data-only 2-train/1-val splits.

All sizes use original mask pixel areas. Label/class ID n means GraspNet object
index n-1. No model predictions, resized masks, or synthesized labels are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
from scipy.io import loadmat


def _frame_files(directory, suffix):
    return {int(path.stem): path for path in directory.glob(f"*{suffix}")
            if re.fullmatch(r"\d{4}" + re.escape(suffix), path.name)}


def audit_graspnet_subset(source="data/graspnet_original_subset", camera="realsense"):
    root = Path(source).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if camera not in ("realsense", "kinect"):
        raise ValueError("camera must be realsense or kinect")
    scenes = []
    for directory in sorted((root / "scenes").glob("scene_*")):
        if not re.fullmatch(r"scene_\d{4}", directory.name):
            continue
        scene_id = int(directory.name[-4:])
        base = directory / camera
        files = {kind: _frame_files(base / kind, ".mat" if kind == "meta" else ".png")
                 for kind in ("rgb", "depth", "label", "meta")}
        all_frames = sorted(set().union(*(set(entries) for entries in files.values())))
        if not all_frames:
            continue
        entry = {"scene_id": scene_id, "paired_frames": 0, "incomplete_frames": [],
                 "instances": 0, "small_instances": 0, "medium_instances": 0,
                 "large_instances": 0, "frames": []}
        counts = Counter()
        for frame_id in all_frames:
            missing = [kind for kind, entries in files.items() if frame_id not in entries]
            if missing:
                entry["incomplete_frames"].append({"frame_id": frame_id, "missing": missing})
                continue
            paths = {kind: entries[frame_id] for kind, entries in files.items()}
            if any(not path.resolve().is_relative_to(root) for path in paths.values()):
                raise ValueError("A selected input path escapes the source directory")
            with Image.open(paths["rgb"]) as image:
                size = image.size
            with Image.open(paths["depth"]) as image:
                depth = np.array(image)
            with Image.open(paths["label"]) as image:
                masks = np.array(image)
            if masks.ndim != 2 or masks.dtype.kind not in "iu" or masks.shape != (size[1], size[0]):
                raise ValueError(f"Invalid mask dimensions/type: {paths['label']}")
            if depth.ndim != 2 or depth.dtype.kind not in "iu" or depth.dtype.itemsize < 2 or depth.shape != masks.shape:
                raise ValueError(f"Invalid raw depth dimensions/type: {paths['depth']}")
            metadata = loadmat(paths["meta"])
            factor = np.asarray(metadata["factor_depth"])
            if factor.size != 1 or not np.isfinite(factor.item()) or factor.item() <= 0:
                raise ValueError(f"Invalid factor_depth: {paths['meta']}")
            raw_meta_ids = np.asarray(metadata["cls_indexes"]).reshape(-1)
            if not np.isfinite(raw_meta_ids).all() or not np.equal(raw_meta_ids, np.floor(raw_meta_ids)).all():
                raise ValueError(f"Noninteger class IDs: {paths['meta']}")
            meta_ids = sorted({int(value) for value in raw_meta_ids})
            ids, areas = np.unique(masks, return_counts=True)
            instance_areas = {int(i): int(a) for i, a in zip(ids, areas) if i != 0}
            if any(i < 1 or i > 88 for i in set(instance_areas) | set(meta_ids)):
                raise ValueError(f"Object label ID outside 1–88: {paths['label']}")
            unexpected = sorted(set(instance_areas) - set(meta_ids))
            if unexpected:
                raise ValueError(f"Mask IDs absent from metadata in scene {scene_id}, frame {frame_id}: {unexpected}")
            small = {str(i): a for i, a in instance_areas.items() if a < 32 ** 2}
            medium_count = sum(32 ** 2 <= a < 96 ** 2 for a in instance_areas.values())
            large_count = sum(a >= 96 ** 2 for a in instance_areas.values())
            frame = {"frame_id": frame_id, "width": size[0], "height": size[1],
                     "factor_depth": float(factor.item()), "depth_scale_m": 1.0 / float(factor.item()),
                     "valid_depth_fraction": float((depth > 0).mean()),
                     "metadata_label_ids": meta_ids,
                     "visible_label_ids": sorted(instance_areas),
                     "metadata_ids_without_visible_pixels": sorted(set(meta_ids) - set(instance_areas)),
                     "instance_areas_px": {str(i): a for i, a in instance_areas.items()},
                     "small_instance_areas_px": small,
                     "instances": len(instance_areas), "small_instances": len(small),
                     "medium_instances": medium_count, "large_instances": large_count}
            entry["frames"].append(frame)
            entry["paired_frames"] += 1
            for key in ("instances", "small_instances", "medium_instances", "large_instances"):
                entry[key] += frame[key]
            counts.update(instance_areas.keys())
        entry["visible_label_ids"] = sorted(counts)
        entry["graspnet_object_indices"] = [i - 1 for i in sorted(counts)]
        entry["per_identity_instance_counts"] = {str(i): counts[i] for i in sorted(counts)}
        scenes.append(entry)

    candidates = []
    # Training and validation candidates stay within official training scenes.
    eligible = [scene for scene in scenes if scene["paired_frames"] and scene["instances"] and scene["scene_id"] < 100]
    for val in eligible:
        for train in combinations([scene for scene in eligible if scene is not val], 2):
            train_ids = set().union(*(set(scene["visible_label_ids"]) for scene in train))
            val_ids = set(val["visible_label_ids"])
            covered, unseen = sorted(val_ids & train_ids), sorted(val_ids - train_ids)
            candidates.append({"train_scenes": [scene["scene_id"] for scene in train],
                               "val_scenes": [val["scene_id"]],
                               "train_visible_label_ids": sorted(train_ids),
                               "val_visible_label_ids": sorted(val_ids),
                               "covered_val_label_ids": covered, "unseen_val_label_ids": unseen,
                               "unseen_val_graspnet_object_indices": [i - 1 for i in unseen],
                               "val_class_coverage": len(covered) / len(val_ids),
                               "val_small_instances": val["small_instances"],
                               "val_instances": val["instances"], "val_paired_frames": val["paired_frames"],
                               "train_instances": sum(scene["instances"] for scene in train),
                               "train_paired_frames": sum(scene["paired_frames"] for scene in train)})
    candidates.sort(key=lambda item: (-item["val_class_coverage"], len(item["unseen_val_label_ids"]),
                                      -item["val_small_instances"], -item["train_instances"],
                                      item["val_scenes"], item["train_scenes"]))
    all_ids = sorted(set().union(*(set(scene["visible_label_ids"]) for scene in scenes)))
    return {"source": str(root), "camera": camera,
            "label_convention": "PNG/metadata/class ID n (1–88) identifies GraspNet object index n-1 (0–87); object identities are not generic semantic categories.",
            "area_convention": "Original visible mask pixels: small<1024; medium1024–9215; large>=9216. No resizing.",
            "summary": {"scenes_with_paired_frames": sum(scene["paired_frames"] > 0 for scene in scenes),
                        "paired_frames": sum(scene["paired_frames"] for scene in scenes),
                        "incomplete_frames": sum(len(scene["incomplete_frames"]) for scene in scenes),
                        **{key: sum(scene[key] for scene in scenes) for key in ("instances", "small_instances", "medium_instances", "large_instances")},
                        "visible_label_ids": all_ids, "graspnet_object_indices": [i - 1 for i in all_ids]},
            "scenes": scenes, "candidate_split_count": len(candidates), "candidate_splits": candidates,
            "recommendation": candidates[0] if candidates else None,
            "recommendation_status": "Data-only ranking; no model evaluation used" if candidates else "Need three annotated nonempty scenes in official training scenes 0–99",
            "ranking_rule": "Highest validation class coverage, fewest unseen validation identities, most original-area small validation instances, most training instances, then scene IDs.",
            "limitation": "Tiny scene subsets and few small instances only support a demo. Selecting a split by annotation coverage must be disclosed and fixed before training; this is not the official GraspNet test protocol."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/graspnet_original_subset")
    parser.add_argument("--camera", choices=("realsense", "kinect"), default="realsense")
    parser.add_argument("--output", default="reports/graspnet_raw_audit.json")
    args = parser.parse_args()
    report = audit_graspnet_subset(args.source, args.camera)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "summary": report["summary"], "recommendation": report["recommendation"], "recommendation_status": report["recommendation_status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
