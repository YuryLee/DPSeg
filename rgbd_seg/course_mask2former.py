"""Pinned COCO-instance Mask2Former adapter, preserving overlapping masks.

The instance rule follows facebookresearch/Mask2Former/mask2former/
maskformer_model.py: foreground query/class top-k, logit > 0 masks, and class
probability multiplied by mean sigmoid probability inside each positive mask.
This independently written adapter avoids Transformers 4.53.2's hardcoded
384-pixel postprocessing and its flattened instance map / RLE path.
Only RGB enters this model; no targets, depth, or course class mapping is used.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
from torch.nn import functional as F
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


MODEL_ID = "facebook/mask2former-swin-tiny-coco-instance"
REVISION = "22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "pretrained/mask2former_swin_tiny_coco_instance"
# COCO's 80 contiguous thing indices map to these original dataset category IDs.
COCO_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_size(size):
    if len(size) != 2 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in size):
        raise ValueError("Sizes must be (positive integer height, positive integer width)")
    return tuple(size)


@torch.inference_mode()
def instance_predictions(class_logits, mask_logits, *, input_size, original_size,
                         score_threshold=0.01, max_detections=100, id2label=None,
                         chunk_size=8):
    """Return independent original-size COCO RLEs for one RGB image.

    Shapes: Qx81 class logits (last class = no object), Qxhxw mask logits.
    input_size is the processor's actual resized image size, without batch pad.
    Single-image processor inputs have no batch padding. Spatial logits are
    bilinearly restored to input_size and then original_size before thresholding
    or computing the mask quality score, as in the author's instance path.
    category_id uses original sparse COCO IDs; coco_contiguous_id is 0..79.
    No pixel ownership competition, connected-component removal, or NMS occurs.
    """
    input_size, original_size = _positive_size(input_size), _positive_size(original_size)
    if not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must lie in [0, 1]")
    if not isinstance(max_detections, int) or not 1 <= max_detections <= 100:
        raise ValueError("max_detections must lie in 1..100")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if class_logits.ndim != 2 or class_logits.shape[1] != 81:
        raise ValueError("Expected Qx81 COCO instance class logits including no-object")
    if mask_logits.ndim != 3 or mask_logits.shape[0] != class_logits.shape[0] or min(mask_logits.shape[-2:]) <= 0:
        raise ValueError("Mask logits must align with the class queries")
    if class_logits.device != mask_logits.device:
        raise ValueError("Class and mask logits must be on the same device")
    if not torch.isfinite(class_logits).all() or not torch.isfinite(mask_logits).all():
        raise ValueError("Model outputs contain non-finite logits")
    if class_logits.shape[0] == 0:
        return []
    # Keep no-object in the softmax denominator, then remove its output column.
    foreground = class_logits.float().softmax(dim=-1)[:, :-1]
    scores, flattened_indices = foreground.flatten().topk(
        min(max_detections, foreground.numel()), sorted=False)
    query_indices = torch.div(flattened_indices, 80, rounding_mode="floor")
    labels = flattened_indices % 80
    candidates = []
    for start in range(0, len(scores), chunk_size):
        end = min(start + chunk_size, len(scores))
        logits = mask_logits[query_indices[start:end]].float()[None]
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)[0]
        binary = logits > 0  # Author's rule; zero logit (= 0.5) is background.
        areas = binary.flatten(1).sum(1)
        quality = (logits.sigmoid() * binary).flatten(1).sum(1) / (areas + 1e-6)
        final_scores = scores[start:end] * quality
        for offset in range(end - start):
            score = float(final_scores[offset])
            if int(areas[offset]) == 0 or score < score_threshold:
                continue
            candidate = start + offset
            contiguous_id = int(labels[candidate])
            mask = np.asfortranarray(binary[offset].cpu().numpy(), dtype=np.uint8)
            rle = mask_utils.encode(mask)
            bbox = mask_utils.toBbox(rle).tolist()
            rle["counts"] = rle["counts"].decode("ascii")
            item = {"category_id": COCO_CATEGORY_IDS[contiguous_id],
                    "coco_contiguous_id": contiguous_id,
                    "score": score, "segmentation": rle,
                    "area": int(areas[offset]), "bbox": bbox,
                    "query_id": int(query_indices[candidate])}
            if id2label is not None:
                item["category_name"] = id2label[contiguous_id]
            candidates.append(item)
    # Reordering by final confidence changes no candidate or mask. Tie-break by
    # query/class keeps serialized results stable for equal confidence values.
    candidates.sort(key=lambda item: (-item["score"], item["query_id"], item["coco_contiguous_id"]))
    return candidates


class CourseMask2Former:
    """Load local safetensors only; expose model/processor and predict(PIL RGB)."""

    def __init__(self, checkpoint_dir=DEFAULT_CHECKPOINT, device="cpu", *,
                 score_threshold=0.01, max_detections=100):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.device = torch.device(device)
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        receipt_path = self.checkpoint_dir / "source_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("revision") != REVISION or receipt.get("model_id") != MODEL_ID:
            raise ValueError("Checkpoint receipt does not match the pinned COCO-instance model")
        for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
            if _sha256(self.checkpoint_dir / name) != receipt["files"][name]["sha256"]:
                raise ValueError(f"Checkpoint file hash mismatch: {name}")
        self.processor = Mask2FormerImageProcessor.from_pretrained(self.checkpoint_dir, local_files_only=True)
        published_processor_config = self.processor.to_dict()
        # Frozen course-test protocol: common scale for both pretrained models.
        # The HF checkpoint's published processor default is square 384x384.
        self.processor.size = {"shortest_edge": 800, "longest_edge": 1333}
        self.model, loading_info = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.checkpoint_dir, local_files_only=True, use_safetensors=True,
            torch_dtype=torch.float32, output_loading_info=True)
        if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Checkpoint loading was not exact: {loading_info}")
        if self.model.config.num_labels != 80:
            raise ValueError("Expected the 80-class COCO instance checkpoint")
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.last_timing = None
        self.metadata = {
            "model_id": MODEL_ID, "revision": REVISION,
            "checkpoint_directory": str(self.checkpoint_dir),
            "checkpoint_sha256": receipt["files"]["model.safetensors"]["sha256"],
            "source_receipt_sha256": _sha256(receipt_path),
            "adapter_sha256": _sha256(__file__),
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "device": str(self.device), "dtype": "float32", "input": "single PIL RGB image",
            "preprocess": self.processor.to_dict(),
            "published_hf_preprocess": published_processor_config,
            "preprocess_override": "shared fixed course-test protocol: shortest edge 800, longest edge 1333; replaces HF default square 384x384; selected without GT or results",
            "postprocess": {
                "source": "https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py",
                "candidate_selection": "top100 foreground query-class softmax probabilities; no-object remains in softmax denominator",
                "max_detections": max_detections, "score_threshold": score_threshold,
                "mask_threshold": 0.5, "mask_comparison": "> (author's logit > 0)",
                "spatial_restore": "bilinear logits: mask resolution -> processor input size -> original image size; align_corners=False",
                "score": "class softmax probability * mean sigmoid over positive original-resolution mask; denominator area + 1e-6",
                "mask_format": "independent compressed COCO RLE; overlaps retained",
                "category_id": "original COCO dataset ID (1..90 with gaps)",
                "coco_contiguous_id": "0..79 model class index",
                "additional_nms": False, "empty_masks": "removed", "score_rounding": False,
            },
            "loading_info": loading_info,
            "limitations": "COCO vocabulary; RGB only; no course-set training or tuning; HF-converted model with author-style instance postprocessing, not a bitwise Detectron2 reproduction",
        }

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(self, image: Image.Image):
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("predict requires a PIL image in RGB mode")
        self._synchronize()
        started = perf_counter()
        inputs = self.processor(images=image, return_tensors="pt")
        input_size = tuple(inputs["pixel_values"].shape[-2:])
        if not bool(inputs["pixel_mask"].all()):
            raise ValueError("Single-image adapter requires no batch padding")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        self._synchronize()
        preprocessed = perf_counter()
        outputs = self.model(**inputs)
        self._synchronize()
        forwarded = perf_counter()
        result = instance_predictions(outputs.class_queries_logits[0], outputs.masks_queries_logits[0],
            input_size=input_size, original_size=(image.height, image.width),
            score_threshold=self.score_threshold, max_detections=self.max_detections,
            id2label=self.model.config.id2label)
        self._synchronize()
        finished = perf_counter()
        self.last_timing = {
            "preprocess_and_transfer_seconds": preprocessed - started,
            "model_forward_seconds": forwarded - preprocessed,
            "postprocess_and_rle_seconds": finished - forwarded,
            "total_seconds": finished - started,
            "processor_input_size_hw": list(input_size),
            "original_size_hw": [image.height, image.width],
            "timing_scope": "decoded PIL RGB through serialized instance RLEs; excludes file I/O and model load; CUDA synchronized",
        }
        return result
