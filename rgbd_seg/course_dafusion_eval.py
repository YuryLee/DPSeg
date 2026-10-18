"""Full-split RGB-D evaluation for DA-Fusion; no annotations enter prediction."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .course_archive import CourseArchive
from .course_depth_data import load_inference_sample
from . import course_submission as submission
from . import course_finetune_eval as shared


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def _sources():
    from .course_dafusion import upstream_source_sha256
    names = ['course_dafusion_eval.py', 'course_depth_data.py', 'course_dafusion.py',
             'course_finetune_eval.py', 'course_training_data.py', 'course_archive.py',
             'course_finetune_mask2former.py', 'course_mask2former.py', 'course_submission.py']
    sources = {name: submission._sha256(Path(__file__).with_name(name)) for name in names}
    sources.update(upstream_source_sha256())
    return sources


def validate_csv_fast(path, image_ids, **kwargs):
    receipt = shared.validate_csv_fast(path, image_ids, **kwargs)
    receipt.update(validator='course_dafusion_eval.validate_csv_fast', source_sha256=_sources())
    return receipt


def score_predictions(dataset, prediction_path, output_dir, **kwargs):
    result = shared.score_predictions(dataset, prediction_path, output_dir, **kwargs)
    result.update(depth_used=True, source_sha256=_sources())
    _write_json(Path(output_dir) / 'evaluation.json', result)
    return result


def evaluate(adapter, archive_path, split, output_dir, *, export_submission=False,
             progress_callback=None, expected_ids=None, expected_count=None, max_images=None, warmup=3):
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
               "source_sha256": _sources(), "warmup": warmup, "depth_used": True,
               "timing_scope": "RGB tensor + metric depth + validity + camera K -> adapter preprocessing/model/postprocessing/CPU RLE; excludes ZIP I/O"}

    def progress(stage, **fields):
        value = {"status": stage, "elapsed_seconds": time.monotonic() - started, **fields}
        _write_json(output / "progress.json", value)
        if progress_callback is not None:
            progress_callback(value)

    try:
        with CourseArchive(archive_path, split=split) as dataset:
            if {c["id"] for c in dataset.categories} != {1}:
                raise ValueError("Expected course component category 1")
            if expected_count is not None and len(dataset) != expected_count:
                raise ValueError(f"Expected {expected_count} official {split} images, found {len(dataset)}")
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
                first = load_inference_sample(dataset, ids[0])
                for _ in range(warmup):
                    adapter.predict_sample(first, score_threshold=.01, max_detections=100)
                synchronize()
                times, count = [], 0
                writer_context = (submission.SubmissionWriter(output / "submission.csv", dataset.image_ids)
                                  if export_submission else contextlib.nullcontext())
                with writer_context as writer, (output / "predictions.jsonl").open("x") as stream:
                    for index, image_id in enumerate(ids):
                        sample = load_inference_sample(dataset, image_id)
                        synchronize()
                        began = time.perf_counter()
                        values = adapter.predict_sample(sample, score_threshold=.01, max_detections=100)
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
