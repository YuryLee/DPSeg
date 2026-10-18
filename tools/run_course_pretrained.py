#!/usr/bin/env python3
"""Run unchanged public RGB weights on every course test image, then score separately."""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def score_predictions(dataset, prediction_path, output):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    truth = dataset.coco_ground_truth()
    if not truth.get("annotations"):
        result = {"status": "unavailable", "reason": "No ground-truth annotations in requested split"}
        write_json(output / "evaluation.json", result)
        return result
    truth.setdefault("info", {})
    names = ("mask_ap", "mask_ap50", "mask_ap75", "mask_ap_small", "mask_ap_medium", "mask_ap_large",
             "mask_ar_1", "mask_ar_10", "mask_ar_100", "mask_ar_small", "mask_ar_medium", "mask_ar_large")
    results = []
    with prediction_path.open() as stream:
        for line in stream:
            record = json.loads(line)
            results.extend({"image_id": record["image_id"], **item} for item in record["predictions"])
    coco = COCO()
    coco.dataset = truth
    started = time.monotonic()
    with (output / "coco_evaluation.log").open("w") as log, contextlib.redirect_stdout(log):
        coco.createIndex()
        if results:
            detections = coco.loadRes(results)
        else:
            detections = COCO()
            detections.dataset = {"info": {}, "images": truth["images"], "categories": truth["categories"], "annotations": []}
            detections.createIndex()
        evaluator = COCOeval(coco, detections, "segm")
        evaluator.params.imgIds = list(dataset.image_ids)
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    metrics = {name: float(value) if value >= 0 else None for name, value in zip(names, evaluator.stats)}
    result = {"status": "completed", "metrics": metrics, "scale": "0–1; no multiplication by 100",
              "split": dataset.split, "images": len(dataset), "gt_instances": len(truth["annotations"]),
              "gt_small_including_crowd": sum(0 <= a["area"] <= 1024 for a in truth["annotations"]),
              "predicted_instances": len(results), "iou_type": "segm", "iou_thresholds": evaluator.params.iouThrs.tolist(),
              "max_detections": [1, 10, 100], "elapsed_seconds": time.monotonic() - started,
              "ground_truth_area": "original annotation area preserved",
              "prediction_area": "COCO decoded RLE area; no bbox passed to loadRes",
              "category_mapping": "All COCO foreground predictions mapped to the single component category 1",
              "interpretation": "Unfine-tuned RGB class-agnostic transfer diagnostic, not trained RGB-D course performance"}
    write_json(output / "evaluation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["maskrcnn", "mask2former"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Engineering smoke only; final run must leave zero")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_submission import SubmissionWriter, validate_csv
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; do not silently switch the full run to CPU")
        torch.cuda.set_per_process_memory_fraction(0.55, args.device)
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 4 * 1024**3:
        raise OSError("Less than 4 GiB free; original files remain untouched")
    dataset = CourseArchive(args.archive, split="test")
    if dataset.categories != [{"id": 1, "name": "component", "supercategory": "component"}]:
        raise ValueError("The declared class-agnostic mapping requires the verified single component category")
    count = len(dataset) if not args.limit else min(args.limit, len(dataset))
    expected_ids = list(dataset.image_ids[:count])
    if len(expected_ids) != count or len(set(expected_ids)) != count:
        raise ValueError("Test image ID coverage is inconsistent")
    run = {"status": "loading_model", "started_at_utc": datetime.now(timezone.utc).isoformat(),
           "pid": os.getpid(), "command": sys.argv, "model": args.model, "expected_images": count,
           "full_test_images": len(dataset), "scope": "full_test" if count == len(dataset) else "engineering_smoke",
           "archive": str(args.archive.resolve()), "archive_bytes": args.archive.stat().st_size,
           "device": args.device, "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "python": platform.python_version(), "torch": str(torch.__version__), "threads": args.threads,
           "package_versions": {name: importlib.metadata.version(name) for name in ("torchvision", "transformers", "pycocotools", "numpy", "Pillow")},
           "training_performed": False, "depth_used": False, "score_threshold": 0.01,
           "max_detections": 100, "category_mapping": "All predicted COCO foreground labels -> component category_id=1",
           "test_ground_truth_used_for_selection": False,
           "fixed_gallery_indices": [0, 400, 800, 1200, 1600, 2000]}
    if args.device.startswith("cuda"):
        run["device_name"] = torch.cuda.get_device_name(args.device)
    sources = [Path(__file__), ROOT / "rgbd_seg/course_archive.py", ROOT / "rgbd_seg/course_submission.py",
               ROOT / f"rgbd_seg/course_{args.model}.py"]
    run["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    run["protocol_sha256"] = sha256(ROOT / "reports/course_pretrained/protocol.json")
    run["dataset_audit_sha256"] = sha256(ROOT / "reports/course_pretrained/dataset_audit.json")
    write_json(out / "run.json", run)
    try:
        if args.model == "maskrcnn":
            from rgbd_seg.course_maskrcnn import CourseMaskRCNN
            predictor = CourseMaskRCNN(args.checkpoint, device=args.device, score_threshold=0.01)
        else:
            from rgbd_seg.course_mask2former import CourseMask2Former
            predictor = CourseMask2Former(args.checkpoint, device=args.device, score_threshold=0.01)
        run["model_metadata"] = predictor.metadata
        first = dataset.load_rgb(0)
        for _ in range(3):
            predictor.predict(first)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
        run["status"] = "inferring"
        write_json(out / "run.json", run)
        times = []
        n_predictions = 0
        start = time.monotonic()
        raw_path = out / "predictions.jsonl"
        with SubmissionWriter(out / "submission.csv", expected_ids) as writer, raw_path.open("x") as raw:
            for index in range(count):
                rgb = dataset.load_rgb(index)
                if rgb.size != (1024, 1024):
                    raise ValueError("Unexpected course RGB dimensions")
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                began = time.perf_counter()
                original = predictor.predict(rgb)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                elapsed = time.perf_counter() - began
                mapped = [{"category_id": 1, "score": p["score"], "segmentation": p["segmentation"]} for p in original]
                if len(mapped) > 100:
                    raise ValueError("Predictor exceeded the declared instance cap")
                image_id = expected_ids[index]
                writer.write(image_id, mapped)
                record = {"image_id": image_id, "file_name": dataset.images[index]["file_name"],
                          "original_category_ids": [p["category_id"] for p in original],
                          "predictions": mapped, "inference_seconds": elapsed}
                raw.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                raw.flush()
                times.append(elapsed)
                n_predictions += len(mapped)
                if (index + 1) % 25 == 0 or index + 1 == count or index == 0:
                    progress = {"status": "inferring", "images_completed": index + 1, "expected_images": count,
                                "predicted_instances": n_predictions, "elapsed_seconds": time.monotonic() - start,
                                "mean_inference_ms": float(np.mean(times) * 1000)}
                    write_json(out / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                    if shutil.disk_usage(out).free < 2 * 1024**3:
                        raise OSError("Disk reserve reached; predictions already written are preserved")
        run.update(status="validating_submission", images_completed=count, predictions=n_predictions,
                   inference_elapsed_seconds=time.monotonic() - start,
                   timing={"mean_ms": float(np.mean(times) * 1000), "median_ms": float(np.median(times) * 1000),
                           "p95_ms": float(np.percentile(times, 95) * 1000), "fps": count / sum(times),
                           "warmup": 3, "batch": 1, "precision": "float32", "threads": args.threads,
                           "scope": "PIL in memory -> preprocessing/device/model/independent masks/CPU/RLE; excludes ZIP reads, CSV validation/writing and COCO"})
        if args.device.startswith("cuda"):
            run["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            run["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
        write_json(out / "run.json", run)
        validation = validate_csv(out / "submission.csv", expected_ids)
        write_json(out / "submission_validation.json", validation)
        run["submission_sha256"] = sha256(out / "submission.csv")
        run["predictions_sha256"] = sha256(raw_path)
        if not args.skip_evaluation and not args.limit:
            run["status"] = "evaluating"
            write_json(out / "run.json", run)
            metrics = score_predictions(dataset, raw_path, out)
            run["evaluation_status"] = metrics["status"]
        run.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "run.json", run)
        print(json.dumps({"status": "completed", "output": str(out), "images": count,
                          "metrics": metrics.get("metrics") if 'metrics' in locals() else None}), flush=True)
    except BaseException as error:
        run.update(status="failed", error_type=type(error).__name__, error=str(error),
                   stopped_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "run.json", run)
        raise
    finally:
        close = getattr(dataset, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    main()
