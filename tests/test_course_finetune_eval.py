"""Real tiny ZIP/COCO evaluation and adversarial native-decoder guard tests."""
import copy
import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch

from rgbd_seg.course_finetune_eval import evaluate, validate_csv_fast, score_predictions
from rgbd_seg.course_submission import SubmissionWriter, validate_csv
from rgbd_seg.course_archive import CourseArchive


def prediction():
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[10:42, 20:52] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")
    return {"category_id": 1, "score": .9, "segmentation": rle}


def archive(root):
    path = Path(root) / "fixture.zip"
    with ZipFile(path, "w") as z:
        for split in ("val", "test"):
            records, annotations = [], []
            for i in (1, 2):
                filename = f"rgb/{split}/{i}.png"
                records.append({"id": i, "width": 1024, "height": 1024, "file_name": filename,
                                "support_rle": {"must_not_reach_predict": True}})
                annotations.append({"id": i, "image_id": i, "category_id": 1,
                                    "segmentation": prediction()["segmentation"], "bbox": [20, 10, 32, 32],
                                    "area": 1024 if i == 1 else 1025, "iscrowd": 0})
                image = io.BytesIO()
                Image.new("RGB", (1024, 1024), (i, 0, 0)).save(image, format="PNG")
                z.writestr(filename, image.getvalue())
            z.writestr(f"annotations/{split}/coco.json.gz", gzip.compress(json.dumps({
                "images": records, "annotations": annotations, "categories": [{"id": 1, "name": "component"}]}).encode()))
        z.writestr("submission.csv", "image_id,PredictionString\n2,[]\n1,[]\n")
    return path


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.BatchNorm1d(1)
        self.child.eval()
        self.inputs = []

    def predict(self, image):
        assert isinstance(image, Image.Image) and image.mode == "RGB"
        assert not self.training and torch.is_inference_mode_enabled()
        assert image.size == (1024, 1024)
        self.inputs.append(image.getpixel((0, 0))[0])
        return [prediction()]


class EvaluationTests(unittest.TestCase):
    def test_full_val_original_area_rgb_only_and_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            path = archive(folder)
            adapter = Adapter()
            updates = []
            output = Path(folder) / "val"
            result = evaluate(adapter, path, "val", output, progress_callback=updates.append)
            self.assertEqual(adapter.inputs, [1, 1, 1, 1, 2])
            self.assertTrue(adapter.training)
            self.assertFalse(adapter.child.training)
            self.assertAlmostEqual(result["metrics"]["mask_ap"], 1)
            self.assertAlmostEqual(result["metrics"]["mask_ap_small"], 1)
            self.assertEqual(result["gt_small_including_crowd"], 1)
            self.assertEqual(result["gt_area_groups_including_crowd"]["medium"], 2)
            self.assertEqual(result["images"], 2)
            self.assertEqual(result["scope"], "full_split")
            self.assertFalse((output / "submission.csv").exists())
            self.assertEqual(updates[-1]["status"], "completed")
            with self.assertRaises(FileExistsError):
                evaluate(adapter, path, "val", output)

    def test_test_export_order_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = archive(folder)
            output = Path(folder) / "test"
            result = evaluate(Adapter(), path, "test", output, export_submission=True, warmup=0)
            rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual([r["image_id"] for r in rows], [2, 1])
            receipt = json.loads((output / "submission_validation.json").read_text())
            old = validate_csv(output / "submission.csv", [2, 1])
            for key in ("rows", "predictions", "summed_instance_area", "sha256"):
                self.assertEqual(receipt[key], old[key])
            self.assertEqual(result["images"], 2)
            self.assertTrue(receipt["rle_native_decode_verified"])

    def test_smoke_explicit_and_callback_failure_restores_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = archive(folder)
            result = evaluate(Adapter(), path, "val", Path(folder) / "smoke", max_images=1, warmup=0)
            self.assertEqual((result["images"], result["full_split_images"], result["scope"]), (1, 2, "engineering_smoke"))
            adapter = Adapter()
            def stop(update):
                if update["status"] == "inferring":
                    raise TimeoutError("runner deadline")
            with self.assertRaises(TimeoutError):
                evaluate(adapter, path, "val", Path(folder) / "failed", progress_callback=stop, warmup=0)
            self.assertTrue(adapter.training)
            self.assertFalse(adapter.child.training)
            self.assertFalse((Path(folder) / "failed/evaluation.json").exists())

    def test_missing_or_duplicate_raw_id_rejected_before_coco(self):
        with tempfile.TemporaryDirectory() as folder:
            path = archive(folder)
            raw = Path(folder) / "raw.jsonl"
            with CourseArchive(path, "val") as dataset:
                for ids in ([1], [1, 1], [2, 1], [1, 99]):
                    raw.write_text("".join(json.dumps({"image_id": i, "predictions": []}) + "\n" for i in ids))
                    with self.subTest(ids=ids), self.assertRaises(ValueError):
                        score_predictions(dataset, raw, folder)

    def test_explicit_empty_predictions_score_zero_and_csv_missing_is_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = archive(folder)
            raw = Path(folder) / "raw.jsonl"
            raw.write_text(''.join(json.dumps({"image_id": i, "predictions": []}) + "\n" for i in (1, 2)))
            with CourseArchive(path, "val") as dataset:
                result = score_predictions(dataset, raw, folder)
            self.assertEqual(result["metrics"]["mask_ap"], 0)
            self.assertEqual(result["metrics"]["mask_ap_small"], 0)
            csv_path = Path(folder) / "missing.csv"
            for rows in ([[1, "[]"]], [[1, "[]"], [1, "[]"]], [[1, "[]"], [99, "[]"]]):
                with csv_path.open("w", newline="") as stream:
                    csv.writer(stream).writerows([["image_id", "PredictionString"], *rows])
                with self.assertRaises(ValueError):
                    validate_csv_fast(csv_path, [1, 2])

    def test_fast_validator_guards_native_and_detects_bad_decoded_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "submission.csv"
            with SubmissionWriter(path, [1, 2]) as writer:
                writer.write(1, [prediction()])
                writer.write(2, [])
            receipt = validate_csv_fast(path, [1, 2])
            self.assertEqual((receipt["empty_images"], receipt["summed_instance_area"]), (1, 1024))
            invalid = np.zeros((1024, 1024), dtype=np.uint8)
            invalid[10:42, 20:52] = 2
            with patch("rgbd_seg.course_finetune_eval.mask_utils.decode", return_value=invalid):
                with self.assertRaises(ValueError):
                    validate_csv_fast(path, [1, 2])
            cases = []
            for update in ({"score": float("nan")}, {"category_id": 2},
                           {"segmentation": {"size": [1024, 1024], "counts": "P"}}):
                value = copy.deepcopy(prediction()); value.update(update); cases.append(value)
            for value in cases:
                with path.open("w", newline="") as stream:
                    csv.writer(stream).writerows([["image_id", "PredictionString"], [1, json.dumps([value])], [2, "[]"]])
                with patch("rgbd_seg.course_finetune_eval.mask_utils.decode") as decoder:
                    with self.assertRaises(ValueError):
                        validate_csv_fast(path, [1, 2])
                    decoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
