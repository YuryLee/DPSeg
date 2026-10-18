"""Official COCO segmentation AP; results are percentages, unavailable bins None."""
from __future__ import annotations

import contextlib
import io

import numpy as np


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def evaluate_predictions(dataset, predictions):
    """Evaluate one prediction dictionary per dataset image, in dataset order."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_utils

    if len(predictions) != len(dataset):
        raise ValueError("Exactly one prediction is required per dataset image")
    names = ("mask_ap", "mask_ap50", "mask_ap75", "mask_ap_small", "mask_ap_medium", "mask_ap_large",
             "mask_ar_1", "mask_ar_10", "mask_ar_100", "mask_ar_small", "mask_ar_medium", "mask_ar_large")
    if not len(dataset):
        return {name: None for name in names}
    images, annotations, results = [], [], []
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(dataset.classes)]
    category_ids = {item["id"] for item in categories}
    annotation_id = 1
    for index, prediction in enumerate(predictions):
        if (hasattr(dataset, "samples") and dataset.samples[index].get("instance_path") is None
                and dataset.samples[index].get("instance_mask_paths") is None):
            raise ValueError("COCO evaluation requires instance annotations for every image")
        sample, target = dataset[index]
        h, w = sample["image"].shape[-2:]
        image_id = int(_numpy(target["image_id"]).reshape(-1)[0])
        images.append({"id": image_id, "height": h, "width": w})
        for mask, label, box in zip(_numpy(target["masks"]), _numpy(target["labels"]), _numpy(target["boxes"])):
            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            annotations.append({"id": annotation_id, "image_id": image_id, "category_id": int(label), "segmentation": rle, "area": float(mask_utils.area(rle)), "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])], "iscrowd": 0})
            annotation_id += 1
        masks = _numpy(prediction["masks"])
        scores = _numpy(prediction["scores"])
        labels = _numpy(prediction["labels"])
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3 or masks.shape[1:] != (h, w) or scores.shape != (len(masks),) or labels.shape != (len(masks),):
            raise ValueError("Prediction masks/scores/labels have inconsistent shapes")
        if not np.isfinite(masks).all() or not np.isfinite(scores).all() or np.any((masks < 0) | (masks > 1)) or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Prediction masks and scores must be finite probabilities in [0, 1]")
        for mask, score, label in zip(masks, scores, labels):
            if label not in category_ids:
                raise ValueError("Prediction contains an unknown class index")
            binary = mask >= 0.5
            if not binary.any():
                continue
            rle = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
            results.append({"image_id": image_id, "category_id": int(label), "score": float(score), "segmentation": rle})
    coco = COCO()
    coco.dataset = {"info": {}, "images": images, "annotations": annotations, "categories": categories}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
        if results:
            detections = coco.loadRes(results)
        else:
            detections = COCO()
            detections.dataset = {"info": {}, "images": images, "annotations": [], "categories": categories}
            detections.createIndex()
        evaluator = COCOeval(coco, detections, "segm")
        evaluator.params.imgIds = [item["id"] for item in images]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {name: float(value * 100) if value >= 0 else None for name, value in zip(names, evaluator.stats)}
