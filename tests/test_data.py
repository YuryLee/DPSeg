import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from rgbd_seg.data import RGBDDataset, audit_dataset, dataset_fingerprint, generate_synthetic, load_instance_target, load_rgbd
from rgbd_seg.evaluation import evaluate_predictions


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def images(self, depth):
        rgb = self.root / "rgb.png"
        path = self.root / "depth.npy"
        Image.fromarray(np.full((8, 9, 3), 128, dtype=np.uint8)).save(rgb)
        np.save(path, depth)
        return rgb, path

    def test_depth_units_and_invalid_values(self):
        raw = np.full((8, 9), 2000.0)
        raw[0, :4] = [0, -1, np.nan, np.inf]
        sample = load_rgbd(*self.images(raw), 0.001)
        self.assertEqual(sample["depth"][0, 1, 1].item(), 2.0)
        self.assertEqual(sample["depth"][0, 0, :4].sum().item(), 0)
        self.assertEqual(sample["depth_valid"].sum().item(), 68)
        self.assertTrue(torch.isfinite(sample["depth"]).all())

    def test_all_invalid_depth(self):
        sample = load_rgbd(*self.images(np.full((8, 9), np.nan)), 1.0)
        self.assertEqual(sample["depth"].sum().item(), 0)
        self.assertEqual(sample["depth_valid"].sum().item(), 0)

    def test_corrupt_depth_and_scale(self):
        for raw in (np.ones((8, 9, 3)), np.ones((9, 8))):
            with self.assertRaises(ValueError):
                load_rgbd(*self.images(raw), 1)
        paths = self.images(np.ones((8, 9)))
        for scale in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                load_rgbd(*paths, scale)
        depth_png = self.root / "color_depth.png"
        Image.fromarray(np.zeros((8, 9, 3), dtype=np.uint8)).save(depth_png)
        with self.assertRaises(ValueError):
            load_rgbd(paths[0], depth_png, 1)

    def test_empty_and_one_pixel_annotations(self):
        target = load_instance_target(None, {}, 3, (8, 9))
        self.assertEqual(tuple(target["boxes"].shape), (0, 4))
        self.assertEqual(tuple(target["masks"].shape), (0, 8, 9))
        mask = np.zeros((8, 9), dtype=np.uint16)
        mask[7, 8] = 42
        path = self.root / "instances.png"
        Image.fromarray(mask).save(path)
        target = load_instance_target(path, {"42": 1}, 3, (8, 9))
        self.assertEqual(target["boxes"].tolist(), [[8, 7, 9, 8]])
        self.assertEqual(target["area"].tolist(), [1])
        with self.assertRaises(ValueError):
            load_instance_target(path, {}, 3, (8, 9))
        with self.assertRaises(ValueError):
            load_instance_target(path, {"42": 1}, 3, (9, 8))

    def test_synthetic_split_reproducibility_and_overwrite(self):
        root = generate_synthetic(self.root / "first", 2, 1, 1, size=64)
        other = generate_synthetic(self.root / "second", 2, 1, 1, size=64)
        train, val, test = (RGBDDataset(root, split) for split in ("train", "val", "test"))
        self.assertEqual([len(train), len(val), len(test)], [2, 1, 1])
        ids = [int(ds[i][1]["image_id"].item()) for ds in (train, val, test) for i in range(len(ds))]
        self.assertEqual(len(set(ids)), 4)
        for path in root.rglob("*.png"):
            self.assertEqual(path.read_bytes(), (other / path.relative_to(root)).read_bytes())
        self.assertFalse(torch.equal(train[0][0]["image"], val[0][0]["image"]))
        self.assertEqual(audit_dataset(root)["splits"]["test"]["images"], 1)
        with self.assertRaises(FileExistsError):
            generate_synthetic(root)

    def test_manifest_prevents_escape_and_split_leakage(self):
        root = generate_synthetic(self.root / "data", 1, 1, 0, size=64)
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        original = manifest["samples"][1]["rgb"]
        manifest["samples"][1]["rgb"] = manifest["samples"][0]["rgb"]
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            RGBDDataset(root, "train")
        manifest["samples"][1]["rgb"] = "../../outside.png"
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            RGBDDataset(root, "train")
        manifest["samples"][1]["rgb"] = original
        manifest["samples"][1]["id"] = manifest["samples"][0]["id"]
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            RGBDDataset(root, "train")

    def test_missing_train_annotations_fail_but_val_can_load(self):
        root = generate_synthetic(self.root / "data", 1, 1, 0, size=64)
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        for record in manifest["samples"]:
            record.pop("instances")
            record.pop("labels")
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "requires instance annotations"):
            RGBDDataset(root, "train")[0]
        sample, target = RGBDDataset(root, "val")[0]
        self.assertEqual(tuple(sample["image"].shape), (3, 64, 64))
        self.assertEqual(len(target["labels"]), 0)

    def test_fingerprint_counts_and_duplicate_content_report(self):
        root = generate_synthetic(self.root / "first", 1, 1, 0, size=64)
        other = generate_synthetic(self.root / "second", 1, 1, 0, size=64)
        initial = dataset_fingerprint(root)
        self.assertEqual(initial, dataset_fingerprint(other))
        report = audit_dataset(root)
        self.assertEqual(report["dataset_sha256"], initial)
        self.assertEqual(report["per_class_instances"]["object"], report["instances"])
        self.assertEqual(report["duplicate_content_across_splits"], [])
        train, val = RGBDDataset(root, "train"), RGBDDataset(root, "val")
        for kind in ("rgb_path", "depth_path"):
            val.samples[0][kind].write_bytes(train.samples[0][kind].read_bytes())
        changed = dataset_fingerprint(root)
        self.assertNotEqual(initial, changed)
        duplicates = audit_dataset(root)["duplicate_content_across_splits"]
        self.assertEqual({item["inputs"] for item in duplicates}, {"rgb", "rgbd"})
        self.assertEqual({item["split"] for item in duplicates[0]["samples"]}, {"train", "val"})
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["depth_scale"] = 0.002
        manifest_path.write_text(json.dumps(manifest))
        self.assertNotEqual(changed, dataset_fingerprint(root))

    def test_official_coco_perfect_empty_and_missing_sizes(self):
        try:
            import pycocotools  # noqa: F401
        except ImportError:
            self.skipTest("pycocotools not installed")
        root = generate_synthetic(self.root / "data", 0, 2, 0, size=64)
        ds = RGBDDataset(root, "val")
        predictions = []
        for _, target in ds:
            predictions.append({"boxes": target["boxes"], "labels": target["labels"], "scores": torch.ones(len(target["labels"])), "masks": target["masks"][:, None].float()})
        metrics = evaluate_predictions(ds, predictions)
        self.assertAlmostEqual(metrics["mask_ap"], 100.0, places=5)
        self.assertAlmostEqual(metrics["mask_ap_small"], 100.0, places=5)
        self.assertIsNone(metrics["mask_ap_large"])
        self.assertIsNone(metrics["mask_ap_medium"])
        empty = [{"labels": torch.zeros(0, dtype=torch.long), "scores": torch.zeros(0), "masks": torch.zeros(0, 1, 64, 64)} for _ in ds]
        metrics = evaluate_predictions(ds, empty)
        self.assertEqual(metrics["mask_ap"], 0)
        for record in ds.samples:
            Image.fromarray(np.zeros((64, 64), dtype=np.uint16)).save(record["instance_path"])
        metrics = evaluate_predictions(ds, empty)
        self.assertTrue(all(value is None for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
