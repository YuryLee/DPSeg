import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.prepare_graspnet import prepare_graspnet


class GraspNetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.output = self.root / "converted"
        for scene in (0, 1, 100):
            prefix = self.source / "scenes" / f"scene_{scene:04d}" / "realsense"
            label = np.zeros((11, 17), dtype=np.uint16)
            if scene != 100:
                label[2:4, 3:6] = 1
                label[8, 16] = 88
            for kind, data in (("rgb", np.full((11, 17, 3), scene, dtype=np.uint8)),
                               ("depth", np.full((11, 17), 1500, dtype=np.uint16)),
                               ("label", label)):
                (prefix / kind).mkdir(parents=True)
                Image.fromarray(data).save(prefix / kind / "0000.png")

    def tearDown(self):
        self.temp.cleanup()

    def convert(self, **kwargs):
        args = {"source": self.source, "output": self.output, "train_scenes": "0", "val_scenes": "1", "test_scenes": "100", "frames": "0", "depth_scale": 0.001}
        args.update(kwargs)
        return prepare_graspnet(**args)

    def test_identity_offset_resolution_bytes_and_empty_mask(self):
        manifest = self.convert()
        self.assertEqual(len(manifest["classes"]), 88)
        self.assertEqual(manifest["classes"][0], "graspnet_object_000")
        self.assertEqual(manifest["classes"][87], "graspnet_object_087")
        self.assertEqual(manifest["samples"][0]["labels"], {"1": 1, "88": 88})
        self.assertEqual(manifest["samples"][2]["labels"], {})
        self.assertEqual(manifest["provenance"], "public_graspnet_subset")
        self.assertEqual(manifest["depth_scale_source"], "explicit")
        for record in manifest["samples"]:
            for kind in ("rgb", "depth", "instances"):
                self.assertEqual((self.output / record[kind]).read_bytes(), (self.source / record["source_files"][kind]).read_bytes())
                with Image.open(self.output / record[kind]) as image:
                    self.assertEqual(image.size, (17, 11))
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), manifest)
        with self.assertRaises(FileExistsError):
            self.convert()

    def test_category_agnostic_is_explicit(self):
        manifest = self.convert(class_mode="object")
        self.assertEqual(manifest["classes"], ["object"])
        self.assertEqual(manifest["class_scheme"], "category_agnostic")
        self.assertEqual(manifest["samples"][0]["labels"], {"1": 1, "88": 1})

    def test_split_safety_and_missing_frames(self):
        for kwargs in ({"val_scenes": "0"}, {"train_scenes": "100"}, {"test_scenes": "1"}, {"train_scenes": "0,0"}, {"frames": "0,0"}):
            with self.assertRaises(ValueError):
                self.convert(**kwargs)
            self.assertFalse(self.output.exists())
        with self.assertRaises(FileNotFoundError):
            self.convert(frames="0,1")
        self.assertFalse(self.output.exists())

    def test_missing_depth_and_invalid_units(self):
        for scale in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                self.convert(depth_scale=scale)
        with self.assertRaises(FileNotFoundError):
            self.convert(depth_scale=None)
        (self.source / "scenes/scene_0001/realsense/depth/0000.png").unlink()
        with self.assertRaises(FileNotFoundError):
            self.convert()
        self.assertFalse(self.output.exists())

    def test_invalid_object_id_rejected(self):
        Image.fromarray(np.full((11, 17), 89, dtype=np.uint16)).save(self.source / "scenes/scene_0000/realsense/label/0000.png")
        with self.assertRaisesRegex(ValueError, "label IDs"):
            self.convert()
        self.assertFalse(self.output.exists())

    def test_mat_depth_scale_and_inconsistent_units(self):
        try:
            from scipy.io import savemat
        except ImportError:
            self.skipTest("scipy is optional; explicit depth scale remains supported")
        for scene in (0, 1, 100):
            path = self.source / "scenes" / f"scene_{scene:04d}" / "realsense/meta"
            path.mkdir()
            savemat(path / "0000.mat", {"factor_depth": 1000.0})
        manifest = self.convert(depth_scale=None)
        self.assertEqual(manifest["depth_scale"], 0.001)
        self.assertEqual(manifest["depth_scale_source"], "factor_depth_metadata")
        savemat(self.source / "scenes/scene_0001/realsense/meta/0000.mat", {"factor_depth": 10000.0})
        other = self.root / "inconsistent"
        with self.assertRaisesRegex(ValueError, "inconsistent depth units"):
            self.convert(output=other, depth_scale=None)
        self.assertFalse(other.exists())

    def test_available_varying_frames_and_selection_record(self):
        for scene, ids in ((0, (2, 9)), (1, (4, 17)), (100, (5, 30))):
            for folder in ("rgb", "depth", "label"):
                directory = self.source / "scenes" / f"scene_{scene:04d}" / "realsense" / folder
                content = (directory / "0000.png").read_bytes()
                (directory / "0000.png").unlink()
                for frame in ids:
                    (directory / f"{frame:04d}.png").write_bytes(content)
        manifest = self.convert(frames="available")
        self.assertEqual(manifest["selected_frames_by_scene"], {"0": [2, 9], "1": [4, 17], "100": [5, 30]})
        self.assertEqual(len(manifest["samples"]), 6)
        limited = self.convert(output=self.root / "limited", frames="available", max_frames_per_scene=1)
        self.assertEqual(limited["selected_frames_by_scene"], {"0": [2], "1": [4], "100": [5]})
        self.assertEqual(limited["discovered_frames_by_scene"], manifest["selected_frames_by_scene"])
        self.assertEqual(limited["frame_selection"]["strategy"], "first_sorted")

    def test_available_never_silently_skips_missing_labels(self):
        label = self.source / "scenes/scene_0001/realsense/label/0000.png"
        label.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "label/0000.png"):
            self.convert(frames="available")
        self.assertFalse(self.output.exists())

    def test_available_frame_limit_validation(self):
        for limit in (0, -2, 1.5, True):
            with self.assertRaises(ValueError):
                self.convert(frames="available", max_frames_per_scene=limit)
        with self.assertRaisesRegex(ValueError, "explicit lists"):
            self.convert(frames="0", max_frames_per_scene=1)


if __name__ == "__main__":
    unittest.main()
