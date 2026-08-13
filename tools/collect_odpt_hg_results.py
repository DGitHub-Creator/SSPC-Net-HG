#!/usr/bin/env python
"""Validate and summarize completed SSPC-Net ODPT-HG official runs.

The final metrics are independently recomputed from ``pointwise_cm.npy``.
Rows of that matrix are ground-truth classes and columns are predictions.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import os
import re
import shlex
import sys

import h5py
import numpy as np


DEFAULT_CLASSES = [
    "pipeline",
    "steel_frame",
    "elbow_pipe",
    "valve_guardrail",
    "gate_valve",
    "Christmas-tree_body",
]
EXPECTED_SCENES = [
    "Area_3_conferenceRoom_20",
    "Area_3_conferenceRoom_21",
    "Area_3_conferenceRoom_22",
]
PROTOCOL_MARKER = (
    "ODPT-HG protocol: training complete; loading held-out Area 3 for final evaluation"
)
EXTENSION_PATTERN = re.compile(r"extension points: torch\.Size\(\[(\d+)\]\)")
PROTOTYPE_SEED_PATTERN = re.compile(r"prototype seed points: torch\.Size\(\[(\d+)\]\)")


def fail(message):
    raise ValueError(message)


def load_json(path):
    with open(path, "r") as stream:
        return json.load(stream)


def derive_dataset_evidence(manifest_path, validation_path):
    manifest = load_json(manifest_path)
    validation = load_json(validation_path)
    classes = manifest.get("classes", DEFAULT_CLASSES)
    if len(classes) != 6:
        fail("expected 6 ODPT-HG classes, found {}".format(len(classes)))
    scenes = validation.get("scenes", {})
    actual_test_scenes = sorted(name for name in scenes if name.startswith("Area_3_"))
    if actual_test_scenes != EXPECTED_SCENES:
        fail("unexpected Area 3 scenes: {}".format(actual_test_scenes))
    expected_points = sum(int(scenes[name]["raw_points"]) for name in EXPECTED_SCENES)
    expected_support = np.sum(
        np.asarray([scenes[name]["class_points"] for name in EXPECTED_SCENES], dtype=np.int64),
        axis=0,
    )
    if int(expected_support.sum()) != expected_points:
        fail("Area 3 class support does not sum to raw point count")
    expected_superpoints = {
        name: int(scenes[name]["superpoints"]) for name in EXPECTED_SCENES
    }
    return classes, expected_points, expected_support, expected_superpoints


def compute_metrics(confusion, classes):
    confusion = np.asarray(confusion)
    if confusion.shape != (len(classes), len(classes)):
        fail("confusion matrix has shape {}, expected {}".format(
            confusion.shape, (len(classes), len(classes))))
    if not np.all(np.isfinite(confusion)) or np.any(confusion < 0):
        fail("confusion matrix contains non-finite or negative values")
    if not np.allclose(confusion, np.rint(confusion), atol=1e-6):
        fail("confusion matrix contains non-integral point counts")
    confusion = np.rint(confusion).astype(np.int64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    union = support + predicted - true_positive
    iou = np.divide(
        true_positive, union, out=np.zeros(len(classes), dtype=np.float64), where=union != 0
    )
    accuracy = np.divide(
        true_positive,
        support,
        out=np.zeros(len(classes), dtype=np.float64),
        where=support != 0,
    )
    seen = union != 0
    total = int(confusion.sum())
    return {
        "confusion_matrix": confusion.tolist(),
        "evaluated_points": total,
        "OA": float(true_positive.sum()) / total if total else 0.0,
        "mIoU": float(iou[seen].mean()) if np.any(seen) else 0.0,
        "mAcc": float(accuracy.mean()),
        "classes": [
            {
                "name": name,
                "support": int(support[index]),
                "predicted": int(predicted[index]),
                "true_positive": int(true_positive[index]),
                "IoU": float(iou[index]),
                "accuracy": float(accuracy[index]),
            }
            for index, name in enumerate(classes)
        ],
    }


def parse_cmdline(path):
    with open(path, "r") as stream:
        tokens = shlex.split(stream.read().strip())
    parsed = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--") and index + 1 < len(tokens):
            parsed[token[2:]] = tokens[index + 1]
            index += 2
        else:
            index += 1
    return parsed


def assert_close(label, actual, expected, tolerance=5e-7):
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        fail("{} mismatch: recomputed={} saved={}".format(label, actual, expected))


def _summarize_counts(pattern, log):
    counts = [int(value) for value in pattern.findall(log)]
    return {
        "checks": len(counts),
        "generated_points": sum(counts),
        "nonempty_checks": sum(value > 0 for value in counts),
    }


def extract_extension_evidence(log):
    return {
        "final_extension": _summarize_counts(EXTENSION_PATTERN, log),
        "prototype_seeds": _summarize_counts(PROTOTYPE_SEED_PATTERN, log),
    }


def validate_protocol(run_dir, budget, require_improved=False):
    cmdline = parse_cmdline(os.path.join(run_dir, "cmdline.txt"))
    expected = {
        "dataset": "odpt_hg",
        "odpt_budget": str(budget),
        "epochs": "400",
        "seed": "1",
        "test_nth_epoch": "0",
        "test_multisamp_n": "10",
    }
    if require_improved:
        expected.update({
            "odpt_steps_per_epoch": "18",
            "odpt_class_balance_power": "0.5",
            "odpt_class_weight_max": "4.0",
            "odpt_prototype_seeds": "1",
            "odpt_seed_confidence": "0.85",
            "odpt_seed_margin": "0.05",
            "odpt_seed_max_per_class": "3",
        })
    for key, value in expected.items():
        if cmdline.get(key) != value:
            fail("{} command line must have --{} {} (found {})".format(
                run_dir, key, value, cmdline.get(key)))

    stats = load_json(os.path.join(run_dir, "trainlog.txt"))
    if not stats or int(stats[-1].get("epoch", -1)) != 399:
        fail("{} did not record completed epoch 399".format(run_dir))
    if not math.isfinite(float(stats[-1].get("loss", float("nan")))):
        fail("{} ended with a non-finite loss".format(run_dir))
    if not os.path.isfile(os.path.join(run_dir, "epoch399_model.pth.tar")):
        fail("{} is missing the final epoch399 checkpoint".format(run_dir))

    log_path = os.path.join(run_dir, "train.log")
    with open(log_path, "r", errors="replace") as stream:
        log = stream.read()
    marker_at = log.find(PROTOCOL_MARKER)
    if marker_at < 0:
        fail("{} is missing the final-only Area 3 protocol marker".format(run_dir))
    if "Area_3" in log[:marker_at]:
        fail("{} mentions Area_3 before final-only evaluation".format(run_dir))
    if "Traceback (most recent call last)" in log or "CUDA out of memory" in log:
        fail("{} log contains a fatal runtime marker".format(run_dir))
    extension = extract_extension_evidence(log)
    if require_improved:
        seeds = extension["prototype_seeds"]
        final_extension = extension["final_extension"]
        if seeds["checks"] == 0 or seeds["generated_points"] == 0:
            fail("{} produced no prototype seed evidence".format(run_dir))
        if final_extension["generated_points"] == 0:
            fail("{} produced no final extension points".format(run_dir))
    return {
        "command_line": cmdline,
        "label_extension": extension,
        "final_epoch": 399,
        "final_loss": float(stats[-1]["loss"]),
    }


def validate_saved_scores(run_dir, metrics):
    scores = load_json(os.path.join(run_dir, "scores_val.txt"))
    if not isinstance(scores, list) or len(scores) != 1:
        fail("{} scores_val.txt must contain one final record".format(run_dir))
    saved = scores[0]
    assert_close("OA", metrics["OA"], saved["oacc_test"])
    assert_close("mIoU", metrics["mIoU"], saved["avg_iou_test"])
    assert_close("mAcc", metrics["mAcc"], saved["avg_acc_test"])


def validate_predictions(path, expected_superpoints):
    datasets = {}
    with h5py.File(path, "r") as stream:
        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                scene = os.path.basename(name)
                if scene in datasets:
                    fail("duplicate prediction dataset leaf: {}".format(scene))
                datasets[scene] = list(item.shape)
        stream.visititems(visitor)
    if sorted(datasets) != sorted(expected_superpoints):
        fail("{} contains prediction scenes {}, expected {}".format(
            path, sorted(datasets), sorted(expected_superpoints)))
    for scene, superpoints in expected_superpoints.items():
        if datasets[scene] != [superpoints]:
            fail("{} prediction has shape {}, expected [{}]".format(
                scene, datasets[scene], superpoints))
    return datasets


def read_baselines(path):
    if not path or not os.path.isfile(path):
        return {}
    baselines = {}
    with open(path, "r", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                budget_value = row.get("budget_alias", row.get("budget"))
                budget = int(str(budget_value).rstrip("%"))
                baselines[budget] = {
                    "mIoU": float(row["mIoU"]),
                    "mAcc": float(row["mAcc"]),
                    "OA": float(row["OA"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return baselines


def collect(args):
    classes, expected_points, expected_support, expected_superpoints = derive_dataset_evidence(
        args.manifest, args.validation)
    baselines = read_baselines(args.baseline_summary)
    references = read_baselines(args.reference_summary)
    runs = []
    for budget in args.budgets:
        run_dir = os.path.join(args.results_root, "{}pct".format(budget), args.run_id)
        training_evidence = validate_protocol(run_dir, budget, args.require_improved)
        metrics = compute_metrics(np.load(os.path.join(run_dir, "pointwise_cm.npy")), classes)
        if metrics["evaluated_points"] != expected_points:
            fail("{} evaluated {} points, expected {}".format(
                run_dir, metrics["evaluated_points"], expected_points))
        support = np.asarray([item["support"] for item in metrics["classes"]])
        if not np.array_equal(support, expected_support):
            fail("{} ground-truth class support differs from the Area 3 manifest".format(run_dir))
        validate_saved_scores(run_dir, metrics)
        prediction_shapes = validate_predictions(
            os.path.join(run_dir, "predictions_val.h5"), expected_superpoints)
        runs.append({
            "method": args.method,
            "dataset": "ODPT-HG",
            "budget": budget,
            "seed": 1,
            "epochs": 400,
            "run_id": args.run_id,
            "run_dir": os.path.abspath(run_dir),
            "protocol": "Area 1 train; Area 3 final-only; 10-sample evaluation",
            "training_evidence": training_evidence,
            "prediction_shapes": prediction_shapes,
            "metrics": metrics,
            "SemiGMMPoint": baselines.get(budget),
            "reference": references.get(budget),
            "reference_name": args.reference_name if budget in references else None,
        })
    return {
        "status": "ok",
        "classes": classes,
        "test_scenes": EXPECTED_SCENES,
        "expected_test_points": expected_points,
        "runs": runs,
    }


def write_csv(path, report):
    fields = ["method", "dataset", "budget", "seed", "epochs", "run_id", "OA", "mAcc", "mIoU"]
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run in report["runs"]:
            row = {key: run[key] for key in fields if key in run}
            row.update({key: run["metrics"][key] for key in ("OA", "mAcc", "mIoU")})
            writer.writerow(row)


def pct(value):
    return "{:.2f}%".format(100.0 * value)


def write_markdown(path, report):
    method = report["runs"][0]["method"] if report["runs"] else "SSPC-Net"
    lines = [
        "# {} ODPT-HG official results".format(method),
        "",
        "Status: **verified**. Metrics were independently recomputed from the saved point-weighted confusion matrices.",
        "",
        "Protocol: canonical Area 1 10%/20% scene splits, seed 1, 400 epochs, no validation during training, then one final Area 3 evaluation using 10 deterministic samples.",
        "",
        "Area 3: {} scenes and {:,} original points.".format(
            len(report["test_scenes"]), report["expected_test_points"]),
        "",
        "| Method | Budget | OA | mAcc | mIoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        metrics = run["metrics"]
        lines.append("| {} | {}% | {} | {} | {} |".format(
            run["method"], run["budget"], pct(metrics["OA"]),
            pct(metrics["mAcc"]), pct(metrics["mIoU"])))
        baseline = run.get("SemiGMMPoint")
        if baseline:
            lines.append("| SemiGMMPoint | {}% | {} | {} | {} |".format(
                run["budget"], pct(baseline["OA"]), pct(baseline["mAcc"]), pct(baseline["mIoU"])))
        reference = run.get("reference")
        if reference:
            lines.append("| {} | {}% | {} | {} | {} |".format(
                run["reference_name"], run["budget"], pct(reference["OA"]),
                pct(reference["mAcc"]), pct(reference["mIoU"])))
    for run in report["runs"]:
        lines.extend([
            "",
            "## {} {}% per-class metrics".format(run["method"], run["budget"]),
            "",
            "| Class | Support | Accuracy | IoU |",
            "|---|---:|---:|---:|",
        ])
        for item in run["metrics"]["classes"]:
            lines.append("| {} | {:,} | {} | {} |".format(
                item["name"], item["support"], pct(item["accuracy"]), pct(item["IoU"])))
        baseline = run.get("SemiGMMPoint")
        if baseline:
            lines.extend([
                "",
                "Difference versus SemiGMMPoint ({} minus baseline): OA {:+.2f} pp, mAcc {:+.2f} pp, mIoU {:+.2f} pp.".format(
                    run["method"],
                    100 * (run["metrics"]["OA"] - baseline["OA"]),
                    100 * (run["metrics"]["mAcc"] - baseline["mAcc"]),
                    100 * (run["metrics"]["mIoU"] - baseline["mIoU"]),
                ),
            ])
        reference = run.get("reference")
        if reference:
            lines.extend([
                "",
                "Difference versus {} ({} minus reference): OA {:+.2f} pp, mAcc {:+.2f} pp, mIoU {:+.2f} pp.".format(
                    run["reference_name"], run["method"],
                    100 * (run["metrics"]["OA"] - reference["OA"]),
                    100 * (run["metrics"]["mAcc"] - reference["mAcc"]),
                    100 * (run["metrics"]["mIoU"] - reference["mIoU"]),
                ),
            ])
        extension = run["training_evidence"]["label_extension"]["final_extension"]
        seeds = run["training_evidence"]["label_extension"]["prototype_seeds"]
        lines.extend([
            "",
            "Label extension evidence: {} checks, {} non-empty checks, {:,} generated extension points.".format(
                extension["checks"], extension["nonempty_checks"], extension["generated_points"]
            ),
        ])
        if seeds["checks"]:
            lines.append(
                "Prototype seed evidence: {} checks, {} non-empty checks, {:,} generated seeds.".format(
                    seeds["checks"], seeds["nonempty_checks"], seeds["generated_points"]
                )
            )
    lines.extend([
        "",
        "## Validation evidence",
        "",
        "For each budget, the collector verified epoch 399 completion, finite final loss, the final checkpoint, command-line protocol flags, Area 3 appearing only after the final-evaluation marker, all three prediction vectors and their manifest-derived superpoint counts, confusion shape/count/support, agreement with `scores_val.txt`, and label-extension counts parsed from the training log.",
        "",
    ])
    with open(path, "w") as stream:
        stream.write("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/odpt_hg")
    parser.add_argument("--run-id", default="official-seed1")
    parser.add_argument("--method", default="SSPC-Net")
    parser.add_argument("--require-improved", action="store_true")
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--manifest", default="/path/to/odpt-hg-sspc/preprocess_manifest.json")
    parser.add_argument("--validation", default="/path/to/odpt-hg-sspc/validation_all.json")
    parser.add_argument(
        "--baseline-summary",
        default="/path/to/semigmmpoint/experiments/odpt_hg/summary.csv",
    )
    parser.add_argument("--reference-summary", default="")
    parser.add_argument("--reference-name", default="SSPC-Net original")
    parser.add_argument("--json-out", default="results/odpt_hg/summary.json")
    parser.add_argument("--csv-out", default="results/odpt_hg/summary.csv")
    parser.add_argument("--markdown-out", default="reports/ODPT_HG_OFFICIAL_RESULTS.md")
    args = parser.parse_args(argv)

    report = collect(args)
    for path in (args.json_out, args.csv_out, args.markdown_out):
        directory = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(directory):
            os.makedirs(directory)
    with open(args.json_out, "w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    write_csv(args.csv_out, report)
    write_markdown(args.markdown_out, report)
    print("ODPT_HG_RESULTS_OK")
    print("wrote {}".format(args.json_out))
    print("wrote {}".format(args.csv_out))
    print("wrote {}".format(args.markdown_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
