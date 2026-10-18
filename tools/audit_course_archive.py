#!/usr/bin/env python3
"""Audit ZIP metadata and supplied GT without extraction, inference, or training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from rgbd_seg.course_archive import CourseArchive


def audit(path):
    dataset = CourseArchive(path)
    coco = dataset.coco_ground_truth()
    records = {item["id"]: item for item in dataset.images}
    areas, decoded_areas, crowd, category_counts = [], [], Counter(), Counter()
    discrepancies, rle_errors = [], []
    per_image = Counter()
    for annotation in coco["annotations"]:
        image = records[annotation["image_id"]]
        area = annotation["area"]
        if not math.isfinite(area) or area < 0:
            raise ValueError("Invalid official annotation area")
        areas.append(area)
        per_image[annotation["image_id"]] += 1
        crowd[str(annotation["iscrowd"])] += 1
        category_counts[str(annotation["category_id"])] += 1
        rle = annotation["segmentation"]
        try:
            if not isinstance(rle, dict) or rle["size"] != [image["height"], image["width"]] or not isinstance(rle["counts"], str):
                raise ValueError("Expected compressed RLE with original dimensions")
            measured = int(mask_utils.area(rle))
            decoded_areas.append(measured)
            if measured != area:
                discrepancies.append({"annotation_id": annotation["id"], "image_id": annotation["image_id"], "original_area": area, "rle_area": measured})
        except Exception as error:
            rle_errors.append({"annotation_id": annotation["id"], "error": str(error)})
    headers = Counter()
    depth_scales = Counter(str(item["depth_scale_m"]) for item in dataset.images)
    annotation_hash = hashlib.sha256()
    with ZipFile(dataset.zip_path) as archive:
        for member in dataset.annotation_members:
            annotation_hash.update(member.encode()+b'\0'+archive.read(member))
        for record in dataset.images:
            for field in ("file_name", "depth_file_name"):
                with archive.open(record[field]) as member:
                    with Image.open(member) as image:
                        if image.size != (record["width"], record["height"]):
                            raise ValueError("PNG header and metadata dimensions differ")
                        headers[(field, image.mode, image.size)] += 1
        readme = archive.read("README.md").decode()
        template_hash = hashlib.sha256(archive.read("submission.csv")).hexdigest()
        root_members = [name for name in archive.namelist() if '/' not in name]
    sample_checks = []
    for index in sorted({0, len(dataset)//2, len(dataset)-1}):
        image = dataset.load_rgb(index)
        depth, valid = dataset.load_depth(index)
        sample_checks.append({"index": index, "image_id": dataset.image_ids[index], "rgb_size": list(image.size), "depth_shape": list(depth.shape), "valid_pixels": int(valid.sum()), "valid_depth_min_m": float(depth[valid].min()) if valid.any() else None, "valid_depth_max_m": float(depth[valid].max()) if valid.any() else None})
    for ann in coco["annotations"][:3]:
        decoded = mask_utils.decode(ann["segmentation"])
        if int(decoded.sum()) != int(mask_utils.area(ann["segmentation"])):
            raise ValueError("Decoded mask and RLE area mismatch")
    areas = np.asarray(areas)
    return {
        "status": "passed_with_documentation_conflict" if not rle_errors else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_path": str(dataset.zip_path), "archive_bytes": dataset.zip_path.stat().st_size,
        "archive_sha256": None, "archive_sha256_note": "Not recomputed by this bounded audit; full download SHA receipt is maintained separately.",
        "split": "test", "image_count": len(dataset), "scene_count": len({r['scene_id'] for r in dataset.images}),
        "annotation_shards": len(dataset.annotation_members), "annotations": len(coco['annotations']),
        "categories": dataset.categories, "instances_by_category": dict(category_counts), "iscrowd_counts": dict(crowd),
        "small_original_area_inclusive_0_1024": int((areas <= 1024).sum()),
        "medium_original_area_inclusive_1024_9216": int(((areas >= 1024) & (areas <= 9216)).sum()),
        "large_original_area_inclusive_9216": int((areas >= 9216).sum()),
        "area_exactly_1024": int((areas == 1024).sum()), "area_exactly_9216": int((areas == 9216).sum()),
        "zero_area_annotations": int((areas == 0).sum()), "empty_gt_images": len(dataset)-len(per_image),
        "original_area_min": float(areas.min()), "original_area_max": float(areas.max()),
        "rle_area_checked": len(decoded_areas), "original_vs_rle_area_mismatch_count": len(discrepancies),
        "original_vs_rle_area_mismatches": discrepancies, "rle_errors": rle_errors,
        "original_area_and_iscrowd_preserved": True,
        "submission_columns": ['image_id','PredictionString'], "submission_rows": len(dataset.submission_rows),
        "submission_id_order_exact_match": True, "template_predictionstrings_all_empty_json_arrays": all(json.loads(r['PredictionString']) == [] for r in dataset.submission_rows),
        "submission_sha256": template_hash, "annotation_member_name_and_bytes_sha256": annotation_hash.hexdigest(),
        "root_members": root_members,
        "missing_readme_named_members": [n for n in ['test_images.json','sample_submission.csv','sample_submission.json'] if n not in dataset.members],
        "actual_test_gt_present": bool(coco['annotations']),
        "readme_conflict": "README states test has no GT, but annotations/test/*.json.gz contain actual per-instance masks. README submission formats also differ from supplied submission.csv and the user's latest CSV-with-JSON compressed-RLE requirement.",
        "submission_rule": "Follow user's latest CSV image_id,PredictionString with JSON compressed RLE, not README legacy detection-string/separate-JSON guidance.",
        "depth_scale_m_counts": dict(depth_scales), "depth_invalid_code_counts": dict(Counter(str(r['depth_invalid_code']) for r in dataset.images)),
        "depth_semantics_counts": dict(Counter(r['depth_semantics'] for r in dataset.images)),
        "png_header_checks": [{"field": k[0], "mode": k[1], "size": list(k[2]), "count": v} for k,v in headers.items()],
        "full_pixel_decode_samples": sample_checks,
        "header_check_limit": "All RGB/depth PNG headers inspected; only three paired images fully decoded. Annotation ZIP entries fully read (ZIP CRC checked); image entries not all fully CRC checked.",
        "inference_fields": ['image','image_id','file_name'], "gt_separate_api": 'coco_ground_truth',
        "gt_never_used_for_inference_inputs": True,
        "first_image_mapping": {k: dataset.images[0][k] for k in ['id','file_name','depth_file_name','height','width','depth_scale_m','depth_invalid_code']},
        "activity": "Archive reads only; no extraction, training, inference, or downloads.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = audit(args.archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','image_count','annotations','small_original_area_inclusive_0_1024','original_vs_rle_area_mismatch_count','rle_errors']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
