"""Check course metric scale, empty predictions and the original-area grouping."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from pycocotools import mask as mask_utils
from tools.run_course_pretrained import score_predictions


def exercise_case(tmp_path, has_prediction, area):
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[20:52, 30:62] = 1
    encoded = mask_utils.encode(np.asfortranarray(mask))
    encoded["counts"] = encoded["counts"].decode("ascii")

    class Fixture:
        split = "test"
        image_ids = [987654]

        def __len__(self):
            return 1

        def coco_ground_truth(self):
            return {"images": [{"id": 987654, "width": 1024, "height": 1024}],
                    "categories": [{"id": 1, "name": "component"}],
                    "annotations": [{"id": 123, "image_id": 987654, "category_id": 1,
                                     "segmentation": encoded, "area": area, "bbox": [30, 20, 32, 32], "iscrowd": 0}]}

    predictions = [{"category_id": 1, "score": 0.9, "segmentation": encoded}] if has_prediction else []
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"image_id": 987654, "predictions": predictions}) + "\n")
    result = score_predictions(Fixture(), path, tmp_path)
    return result["metrics"]


class CourseEvaluationTests(unittest.TestCase):
    def test_course_coco_scale_empty_predictions_and_area(self):
        for has_prediction, area, expected_small in [(True, 1024, 1.0), (False, 1024, 0.0), (True, 1025, None)]:
            with self.subTest(prediction=has_prediction, area=area), tempfile.TemporaryDirectory() as folder:
                metrics = exercise_case(Path(folder), has_prediction, area)
                self.assertAlmostEqual(metrics["mask_ap"], 1.0 if has_prediction else 0.0)
                if expected_small is None:
                    self.assertIsNone(metrics["mask_ap_small"])
                else:
                    self.assertAlmostEqual(metrics["mask_ap_small"], expected_small)


if __name__ == "__main__":
    unittest.main()
