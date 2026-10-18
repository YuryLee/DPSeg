import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
from PIL import Image
from rgbd_seg.course_archive import CourseArchive


class CourseArchiveTests(unittest.TestCase):
    def make_archive(self, root, *, missing_rgb=False, duplicate_id=False, bad_dimensions=False):
        path = Path(root) / "course.zip"
        records = [{"id": i, "file_name": f"rgb/test/{i}.png", "depth_file_name": f"depth/test/{i}.png", "height": 8, "width": 11 if bad_dimensions else 10, "depth_scale_m": .001, "depth_invalid_code": 65535, "support_rle": {"secret": True}} for i in (7, 9)]
        if duplicate_id:
            records[1]["id"] = 7
        annotations = [{"id": 1, "image_id": 7, "category_id": 1, "area": 123, "iscrowd": 1, "segmentation": {"size": [8, 10], "counts": "`2"}, "bbox": [0, 0, 10, 8]}]
        with ZipFile(path, "w") as z:
            z.writestr("submission.csv", 'image_id,PredictionString\n9,[]\n7,[]\n')
            z.writestr("annotations/test/coco_x.json.gz", gzip.compress(json.dumps({"images": records, "categories": [{"id": 1, "name": "component"}], "annotations": annotations}).encode()))
            for i in (7, 9):
                if not missing_rgb or i != 7:
                    b = io.BytesIO();Image.fromarray(np.full((8, 10, 3), i, dtype=np.uint8)).save(b, format="PNG");z.writestr(f"rgb/test/{i}.png", b.getvalue())
                raw = np.full((8, 10), 2000, dtype=np.uint16);raw[0, 0] = 65535
                b = io.BytesIO();Image.fromarray(raw).save(b, format="PNG");z.writestr(f"depth/test/{i}.png", b.getvalue())
        return path, annotations

    def test_order_pixels_and_gt_separation(self):
        with tempfile.TemporaryDirectory() as root:
            path, annotations = self.make_archive(root)
            dataset = CourseArchive(path)
            self.assertEqual(dataset.image_ids, [9, 7])
            sample = dataset[0]
            self.assertEqual(set(sample), {"image", "image_id", "file_name"})
            self.assertEqual(tuple(sample["image"].shape), (3, 8, 10))
            self.assertAlmostEqual(sample["image"][0, 0, 0].item(), 9/255)
            self.assertEqual(dataset.load_rgb(0).size, (10, 8))
            self.assertEqual(dataset.coco_ground_truth()["annotations"], annotations)
            self.assertEqual(dataset.coco_ground_truth()["annotations"][0]["area"], 123)
            self.assertEqual(dataset.coco_ground_truth()["annotations"][0]["iscrowd"], 1)
            self.assertEqual(sorted(p.name for p in Path(root).iterdir()), ["course.zip"])

    def test_raw_depth_uses_invalid_code_and_meter_scale(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.make_archive(root)
            meters, valid = CourseArchive(path).load_depth(0)
            self.assertEqual(meters[0, 0], 0)
            self.assertFalse(valid[0, 0])
            self.assertAlmostEqual(meters[1, 1], 2)

    def test_missing_member_and_duplicate_id_rejected(self):
        for case in ("missing_rgb", "duplicate_id"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as root:
                path, _ = self.make_archive(root, **{case: True})
                with self.assertRaises(ValueError):
                    CourseArchive(path)

    def test_dimension_mismatch_rejected_on_lazy_read(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.make_archive(root, bad_dimensions=True)
            dataset = CourseArchive(path)
            with self.assertRaises(ValueError):
                dataset.load_rgb(0)


if __name__ == "__main__":
    unittest.main()
