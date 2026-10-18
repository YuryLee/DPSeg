"""Unmodified COCO Mask R-CNN weights for the course transfer baseline."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import pil_to_tensor
from pycocotools import mask as mask_utils


class CourseMaskRCNN:
    def __init__(self, checkpoint, device="cpu", score_threshold=0.01):
        self.device = torch.device(device)
        self.score_threshold = float(score_threshold)
        checkpoint = Path(checkpoint)
        with checkpoint.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != "a015f8e1988c0d3171679bb32ff9ef2204f544cebd7d29d18b59be5001957064":
            raise ValueError("Checkpoint differs from the locally verified mirror state_dict")
        self.model = maskrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=None, num_classes=91,
            min_size=800, max_size=1333, box_score_thresh=self.score_threshold,
            box_detections_per_img=100,
        )
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.model.to(self.device).eval()
        self.metadata = {
            "name": "Mask R-CNN ResNet50-FPN v2 COCO_V1",
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checksum,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "input": "RGB only; no course training or depth fusion",
            "preprocess": {"shortest_edge": 800, "longest_edge": 1333,
                           "mean": self.model.transform.image_mean, "std": self.model.transform.image_std},
            "postprocess": {"score_threshold": self.score_threshold, "mask_threshold": 0.5,
                            "max_detections": 100, "nms": "native COCO class-wise box NMS; no additional mapped-class NMS",
                            "overlapping_masks_preserved": True},
            "original_category_ids": "COCO sparse category IDs, as emitted by torchvision",
            "weight_source": json.loads(checkpoint.with_name("maskrcnn_resnet50_fpn_v2.source.json").read_text()),
        }

    @torch.inference_mode()
    def predict(self, image):
        tensor = pil_to_tensor(image.convert("RGB")).to(self.device, dtype=torch.float32).div_(255)
        output = self.model([tensor])[0]
        # Transfer binary masks instead of storing full-resolution float masks on CPU.
        masks = (output["masks"][:, 0] >= 0.5).to(torch.uint8).cpu().numpy()
        scores = output["scores"].cpu().tolist()
        labels = output["labels"].cpu().tolist()
        results = []
        for binary, score, label in zip(masks, scores, labels):
            if score < self.score_threshold or not binary.any():
                continue
            encoded = mask_utils.encode(np.asfortranarray(binary))
            encoded["counts"] = encoded["counts"].decode("ascii")
            results.append({"category_id": int(label), "score": float(score), "segmentation": encoded})
        return results
