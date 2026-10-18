"""RGB-only full-split evaluation for course fine-tuning adapters.

Prediction never receives targets. Original annotation areas and independent
instance RLEs are preserved. This module scores checkpoints; it never selects one.
"""
from __future__ import annotations

import contextlib
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .course_archive import CourseArchive
from . import course_submission as submission

METRICS = ("mask_ap", "mask_ap50", "mask_ap75", "mask_ap_small", "mask_ap_medium", "mask_ap_large",
           "mask_ar_1", "mask_ar_10", "mask_ar_100", "mask_ar_small", "mask_ar_medium", "mask_ar_large")


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def _sources():
    return {p.name: submission._sha256(p) for p in
            (Path(__file__), Path(submission.__file__), Path(__file__).with_name("course_archive.py"))}


def validate_csv_fast(path, image_ids, *, category_ids=(1,), image_size=(1024, 1024), max_detections=100):
    """Same CSV/schema/run guards as the frozen writer, with linear native checks.

    Native decoding is called only AFTER safe pure-Python run validation. Every
    uint8 mask is checked using max and count_nonzero, avoiding np.isin's costly
    integer lookup path. One decoded instance is retained at a time.
    """
    spec = submission._Spec(image_ids, category_ids, image_size, max_detections)
    rows = predictions = empty = area = 0
    began = time.monotonic()
    with submission._CSV_LIMIT_LOCK:
        previous_limit = csv.field_size_limit()
        csv.field_size_limit(max(previous_limit, spec.max_json_chars))
        try:
            with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                if next(reader, None) != submission.HEADER:
                    raise ValueError("CSV header must be image_id,PredictionString in this order")
                for row in reader:
                    if len(row) != 2:
                        raise ValueError("CSV row must contain exactly two columns")
                    submission._check_id(row[0], rows, spec)
                    if len(row[1]) > spec.max_json_chars:
                        raise ValueError("PredictionString exceeds bounded image/count size")
                    try:
                        values = json.loads(row[1], object_pairs_hook=submission._unique_object)
                    except (ValueError, RecursionError) as error:
                        raise ValueError(f"Invalid PredictionString JSON at row {rows + 1}") from error
                    if not isinstance(values, list) or len(values) > spec.max_detections:
                        raise ValueError("Prediction count exceeds the declared cap or is not a list")
                    for value in values:
                        normalized, checked_area = submission._predictions([value], spec, decode=False)
                        rle = normalized[0]["segmentation"]
                        binary = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")})
                        if (binary.shape != spec.size or binary.dtype != np.uint8
                                or binary.max() > 1 or int(np.count_nonzero(binary)) != checked_area):
                            raise ValueError("Decoded RLE shape/binary values/area differ from checked runs")
                        del binary
                        area += checked_area
                    rows += 1
                    predictions += len(values)
                    empty += not values
        except csv.Error as error:
            raise ValueError(f"Malformed submission CSV: {error}") from error
        finally:
            csv.field_size_limit(previous_limit)
    if rows != len(spec.ids):
        raise ValueError(f"Missing predictions: found {rows} of {len(spec.ids)} rows")
    receipt = submission._receipt(path, spec, rows, predictions, empty, area, True)
    receipt.update(validator="course_finetune_eval.validate_csv_fast", source_sha256=_sources(),
                   native_checks="uint8, original shape, max <= 1, count_nonzero == checked RLE area",
                   elapsed_seconds=time.monotonic() - began)
    return receipt


def score_predictions(dataset, prediction_path, output_dir, *, image_ids=None):
    """Score complete ordered predictions; safe checks precede COCO native RLE use."""
    output_dir = Path(output_dir)
    ids = dataset.image_ids if image_ids is None else list(image_ids)
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(dataset.image_ids):
        raise ValueError("Invalid evaluation image subset")
    spec = submission._Spec(ids, (1,), (1024, 1024), 100)
    results, rows = [], 0
    with Path(prediction_path).open() as stream:
        for line in stream:
            record = json.loads(line, object_pairs_hook=submission._unique_object)
            submission._check_id(record["image_id"], rows, spec)
            values, _ = submission._predictions(record["predictions"], spec, decode=False)
            results.extend({"image_id": ids[rows], **p} for p in values)
            rows += 1
    if rows != len(spec.ids):
        raise ValueError("Missing prediction rows before COCO evaluation")
    truth = dataset.coco_ground_truth()
    if {image["id"] for image in truth["images"]} != set(dataset.image_ids):
        raise ValueError("Ground-truth image IDs differ from evaluated split")
    if {category["id"] for category in truth["categories"]} != {1}:
        raise ValueError("Expected component category 1")
    if image_ids is not None:
        truth["images"] = [r for r in truth["images"] if r["id"] in set(ids)]
        truth["annotations"] = [r for r in truth["annotations"] if r["image_id"] in set(ids)]
    truth.setdefault("info", {})
    began = time.monotonic()
    with (output_dir / "coco_evaluation.log").open("x") as log, contextlib.redirect_stdout(log):
        coco = COCO()
        coco.dataset = truth
        coco.createIndex()
        if results:
            detections = coco.loadRes(results)
        else:
            detections = COCO()
            detections.dataset = {"info": {}, "images": truth["images"], "categories": truth["categories"], "annotations": []}
            detections.createIndex()
        evaluator = COCOeval(coco, detections, "segm")
        evaluator.params.imgIds = list(ids)
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    result = {"status": "completed", "split": dataset.split, "images": rows,
              "metrics": {name: float(value) if value >= 0 else None for name, value in zip(METRICS, evaluator.stats)},
              "scale": "0–1; no multiplication by 100", "iou_type": "segm", "max_detections": [1, 10, 100],
              "iou_thresholds": evaluator.params.iouThrs.tolist(), "area_ranges": evaluator.params.areaRng,
              "gt_instances": len(truth["annotations"]),
              "gt_area_groups_including_crowd": {label: sum(lo <= a["area"] <= hi for a in truth["annotations"])
                  for label, (lo, hi) in zip(evaluator.params.areaRngLbl, evaluator.params.areaRng)},
              "gt_small_including_crowd": sum(0 <= a["area"] <= 1024 for a in truth["annotations"]),
              "predicted_instances": len(results), "ground_truth_area": "original annotation area preserved",
              "prediction_area": "independent COCO RLE; no bbox passed to loadRes",
              "predictions_sha256": submission._sha256(prediction_path), "source_sha256": _sources(),
              "elapsed_seconds": time.monotonic() - began, "depth_used": False}
    _write_json(output_dir / "evaluation.json", result)
    return result


def evaluate(adapter, archive_path, split, output_dir, *, export_submission=False,
             progress_callback=None, expected_ids=None, max_images=None, warmup=3):
    """Evaluate every official val/test image, with no filtering or GT in predict.

    The output directory must be new or empty. The callback receives a dict at
    each stage and every 25 images (also first/final); exceptions abort evaluation.
    Adapter train/eval flags are restored, including mixed-mode frozen modules.
    Timing synchronizes the adapter's CUDA device, excludes ZIP reads and uses
    three warmups by default. No device movement or optimizer changes occur here.
    """
    if split not in {"val", "test"}:
        raise ValueError("Fine-tune evaluation supports val/test only")
    if export_submission and split != "test":
        raise ValueError("CSV export is reserved for test")
    if max_images is not None and (isinstance(max_images, bool) or not isinstance(max_images, int) or max_images <= 0):
        raise ValueError("max_images must be None or a positive smoke-only integer")
    if max_images is not None and export_submission:
        raise ValueError("Smoke subsets cannot export a submission")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Evaluation output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    states = [(module, module.training) for module in adapter.modules()]
    devices = {p.device for p in adapter.parameters() if p.device.type == "cuda"}
    devices.update(b.device for b in adapter.buffers() if b.device.type == "cuda")

    def synchronize():
        for device in devices:
            torch.cuda.synchronize(device)

    started = time.monotonic()
    receipt = {"status": "starting", "split": split, "archive": str(Path(archive_path).resolve()),
               "source_sha256": _sources(), "warmup": warmup, "depth_used": False,
               "timing_scope": "PIL RGB -> adapter preprocessing/model/postprocessing/CPU RLE; excludes ZIP I/O"}

    def progress(stage, **fields):
        value = {"status": stage, "elapsed_seconds": time.monotonic() - started, **fields}
        _write_json(output / "progress.json", value)
        if progress_callback is not None:
            progress_callback(value)

    try:
        with CourseArchive(archive_path, split=split) as dataset:
            if {c["id"] for c in dataset.categories} != {1}:
                raise ValueError("Expected course component category 1")
            if expected_ids is not None and list(expected_ids) != dataset.image_ids:
                raise ValueError("Expected IDs/order differ from complete official split")
            if any((r["height"], r["width"]) != (1024, 1024) for r in dataset.images):
                raise ValueError("Course evaluation requires original 1024 x 1024 images")
            ids = dataset.image_ids if max_images is None else dataset.image_ids[:max_images]
            count_images = len(ids)
            spec = submission._Spec(ids, (1,), (1024, 1024), 100)
            receipt.update(images=count_images, full_split_images=len(dataset),
                           scope="full_split" if max_images is None else "engineering_smoke")
            adapter.eval()
            progress("warming_up", expected_images=count_images)
            with torch.inference_mode():
                first = dataset.load_rgb(0)
                for _ in range(warmup):
                    adapter.predict(first)
                synchronize()
                times, count = [], 0
                writer_context = (submission.SubmissionWriter(output / "submission.csv", dataset.image_ids)
                                  if export_submission else contextlib.nullcontext())
                with writer_context as writer, (output / "predictions.jsonl").open("x") as stream:
                    for index, image_id in enumerate(ids):
                        image = dataset.load_rgb(index)
                        synchronize()
                        began = time.perf_counter()
                        values = adapter.predict(image)
                        synchronize()
                        elapsed = time.perf_counter() - began
                        values, _ = submission._predictions(values, spec, decode=False)
                        if writer is not None:
                            writer.write(image_id, values)
                        record = {"image_id": image_id, "file_name": dataset.images[index]["file_name"],
                                  "predictions": values, "inference_seconds": elapsed}
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        times.append(elapsed)
                        count += len(values)
                        if index == 0 or (index + 1) % 25 == 0 or index + 1 == count_images:
                            stream.flush()
                            progress("inferring", images_completed=index + 1, expected_images=count_images, predicted_instances=count)
            if export_submission:
                progress("validating_submission", images_completed=count_images)
                validation = validate_csv_fast(output / "submission.csv", dataset.image_ids)
                _write_json(output / "submission_validation.json", validation)
                receipt["submission_sha256"] = validation["sha256"]
                receipt["submission_validation_sha256"] = submission._sha256(output / "submission_validation.json")
            progress("evaluating", images_completed=count_images)
            result = score_predictions(dataset, output / "predictions.jsonl", output,
                                       image_ids=None if max_images is None else ids)
            result.update(scope=receipt["scope"], full_split_images=len(dataset))
            _write_json(output / "evaluation.json", result)
            receipt.update(status="completed", predictions_sha256=result["predictions_sha256"],
                           evaluation_sha256=submission._sha256(output / "evaluation.json"),
                           elapsed_seconds=time.monotonic() - started,
                           timing={"mean_ms": float(np.mean(times) * 1000), "median_ms": float(np.median(times) * 1000),
                                   "p95_ms": float(np.percentile(times, 95) * 1000), "warmup": warmup, "batch": 1})
            _write_json(output / "evaluation_run.json", receipt)
            progress("completed", images_completed=count_images, expected_images=count_images)
            return {**result, "timing": receipt["timing"], "output_dir": str(output.resolve())}
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        _write_json(output / "evaluation_run.json", receipt)
        raise
    finally:
        for module, training in states:
            module.training = training
