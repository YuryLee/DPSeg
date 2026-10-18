#!/usr/bin/env python3
"""Plot complete, SHA-bound five-epoch validation histories; never infer or score.

The final DA comparison embeds its own validation history and binds the RGB
comparison by SHA256. Read that bound summary for the two RGB histories. Test
metric fields are not used; prediction, image, and checkpoint files are never opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("maskrcnn", "mask2former", "dafusion")
NAMES = {"maskrcnn": "Mask R-CNN (RGB)", "mask2former": "Mask2Former (RGB)",
         "dafusion": "DA-Fusion course adaptation (RGB-D)"}
COLORS = {"maskrcnn": "#1764AB", "mask2former": "#C65B22", "dafusion": "#21835C"}
COUNTS = {"train_per_epoch": 14432, "epochs": 5, "val": 2584, "test": 2112}
METRICS = ("mask_ap_small", "mask_ap")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    require(Path(path).is_file(), f"Required completed input is missing: {path}")
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def unit_metric(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            f"{name}: expected a finite metric on the 0–1 scale")
    return float(value)


def load_verified(summary_path):
    """Validate all inputs before creating directories or importing pyplot."""
    summary_path = Path(summary_path).resolve()
    summary = read(summary_path)
    require(summary.get("status") == "complete_and_verified"
            and summary.get("schema") == "course_dafusion_comparison_v1",
            "A completed and verified final DA comparison is required; partial results cannot be plotted")
    require(summary.get("counts") == COUNTS and summary.get("metric_scale") == "0–1",
            "Expected complete official five-epoch comparison on the 0–1 scale")
    require(set(summary["models"]) == set(MODELS), "Expected all three comparison models")
    binding = summary["source_binding"]
    baseline_path = Path(binding["baseline_summary_path"])
    require(baseline_path.is_absolute(), "Baseline summary binding must use an absolute path")
    baseline = read(baseline_path)
    require(sha256(baseline_path) == binding["baseline_summary_sha256"],
            "Bound RGB comparison SHA256 differs")
    require(baseline.get("status") == "complete_and_verified"
            and baseline.get("schema") == "course_finetune_comparison_v1"
            and baseline.get("counts") == COUNTS,
            "Complete five-epoch RGB comparison required")
    for flag in ("identical_epoch_orders", "complete_train_val_test", "validation_only_checkpoint_selection"):
        require(baseline.get("checks", {}).get(flag) is True, f"RGB audit lacks {flag}")
    plotted, gt_identity = {}, None
    for name in MODELS:
        final = summary["models"][name]
        model = final if name == "dafusion" else baseline["models"][name]
        require(model["best_epoch"] == final["best_epoch"]
                and model["best_checkpoint_sha256"] == final["best_checkpoint_sha256"],
                f"{name}: selected model identity differs across summaries")
        history = model["validation"]
        require(isinstance(history, list) and len(history) == 5
                and all(type(row.get("epoch")) is int for row in history)
                and [row["epoch"] for row in history] == [1, 2, 3, 4, 5],
                f"{name}: exactly five complete ordered validation epochs are required")
        values = {key: [] for key in METRICS}
        sources = []
        for row in history:
            path = Path(row["path"]) / "evaluation.json"
            evaluation = read(path)
            require(sha256(path) == row["evaluation_sha256"], f"Validation evaluation SHA256 differs: {path}")
            require(evaluation.get("status") == "completed" and evaluation.get("split") == "val"
                    and evaluation.get("scope") == "full_split"
                    and evaluation.get("images") == evaluation.get("full_split_images") == 2584,
                    f"{name} epoch {row['epoch']}: complete 2,584-image validation required")
            require(evaluation.get("depth_used") is (name == "dafusion"), "Validation modality differs")
            require(evaluation.get("iou_type") == "segm" and evaluation.get("max_detections") == [1, 10, 100]
                    and evaluation.get("ground_truth_area") == "original annotation area preserved"
                    and evaluation.get("scale") == "0–1; no multiplication by 100",
                    "Validation COCO metric protocol differs")
            require(evaluation["metrics"] == row["metrics"], f"{name}: summary and validation metrics differ")
            identity = {key: evaluation[key] for key in ("gt_instances", "gt_area_groups_including_crowd",
                                                         "iou_thresholds", "area_ranges")}
            if gt_identity is None:
                gt_identity = identity
            require(identity == gt_identity, "Validation GT coverage or metric configuration differs")
            for key in METRICS:
                values[key].append(unit_metric(row["metrics"][key], f"{name} epoch {row['epoch']} {key}"))
            sources.append({"epoch": row["epoch"], "path": str(path.resolve()),
                            "sha256": row["evaluation_sha256"]})
        chosen = max(range(5), key=lambda i: (values["mask_ap_small"][i], values["mask_ap"][i], -i)) + 1
        require(type(model["best_epoch"]) is int and model["best_epoch"] == chosen,
                f"{name}: chosen epoch differs from APsmall, AP, earlier-epoch rule")
        plotted[name] = {"name": NAMES[name], "validation": values, "selected_epoch": chosen,
                         "evaluation_sources": sources}
    return plotted, {"summary_path": str(summary_path), "summary_sha256": sha256(summary_path),
                     "baseline_summary_path": str(baseline_path),
                     "baseline_summary_sha256": binding["baseline_summary_sha256"],
                     "validation_gt": gt_identity}


def plot(data, source, output, receipt_path):
    require(output.resolve() != receipt_path.resolve(), "Figure and receipt paths must differ")
    require(not output.exists() and not receipt_path.exists(), "Output already exists; choose new paths")
    require(output.suffix.lower() == ".png", "Figure output must be a PNG")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for axis, key, title in zip(axes, METRICS, ("Validation mask APsmall", "Validation mask AP")):
        for name, marker in zip(MODELS, ("o", "s", "^")):
            row = data[name]
            epoch = row["selected_epoch"]
            axis.plot(range(1, 6), row["validation"][key], color=COLORS[name], marker=marker,
                      linewidth=2, markersize=5, label=f"{NAMES[name]} · selected epoch {epoch}")
            axis.scatter([epoch], [row["validation"][key][epoch - 1]], marker="*", s=175,
                         color=COLORS[name], edgecolor="black", linewidth=.7, zorder=5)
        axis.set(title=title, xlabel="Training epoch", ylabel="COCO mask AP (0–1)",
                 xlim=(.8, 5.2), ylim=(0, 1), xticks=range(1, 6))
        axis.yaxis.set_major_locator(MultipleLocator(.2))
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Five-epoch learning curves · full validation split (2,584 images)", fontsize=14, y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .055), ncol=1,
               frameon=False, fontsize=9)
    fig.text(.5, .018, "Stars: selected by validation APsmall, then AP, then earlier epoch. "
             "Single run per model; no test scores plotted.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .23, 1, .94))
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)
    receipt = {"status": "completed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "full_validation_only", "images_per_epoch": 2584, "epochs": [1, 2, 3, 4, 5],
               "metric_scale": "0–1", "axis_limits": {key: [0, 1] for key in METRICS},
               "selection_rule": "max validation APsmall, then AP, then earlier epoch; same star epoch on both panels",
               "test_scores_plotted": False, "new_training_or_inference": False,
               "limitations": ["Single run per model; no uncertainty intervals or statistical significance claim.",
                               "Different model and training configurations; curves do not isolate a causal depth effect."],
               "source_binding": source, "models": data, "generator_sha256": sha256(__file__),
               "matplotlib_version": matplotlib.__version__,
               "figure": {"path": str(output.resolve()), "sha256": sha256(output), "bytes": output.stat().st_size}}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "reports/course_dafusion/comparison_summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/course_dafusion/learning_curves.png")
    parser.add_argument("--receipt", type=Path, help="Defaults to OUTPUT.with_suffix('.receipt.json')")
    parser.add_argument("--check-only", action="store_true", help="Verify complete inputs; create no artifacts")
    args = parser.parse_args()
    try:
        data, source = load_verified(args.summary)
        if args.check_only:
            print(json.dumps({"status": "verified_inputs", "models": list(data), "epochs": 5, "figures_created": False}))
            return
        receipt = args.receipt or args.output.with_suffix(".receipt.json")
        result = plot(data, source, args.output, receipt)
        print(json.dumps({"status": result["status"], "figure": str(args.output.resolve()),
                          "receipt": str(receipt.resolve())}))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
