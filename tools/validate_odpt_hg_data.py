#!/usr/bin/env python
from __future__ import print_function

import argparse
import json
import os

import h5py
import numpy as np


def count_lines(path):
    count = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            count += block.count(b"\n")
    return count


def validate_scene(input_root, output_root, scene):
    raw_path = os.path.join(input_root, scene + ".txt")
    graph_path = os.path.join(output_root, "graph_v", scene + ".h5")
    cloud_path = os.path.join(output_root, "sp_voxel_pc", scene + ".h5")
    feature_path = os.path.join(output_root, "features", scene + ".h5")
    for path in (raw_path, graph_path, cloud_path, feature_path):
        if not os.path.isfile(path):
            raise ValueError("missing {}".format(path))

    raw_points = count_lines(raw_path)
    with h5py.File(graph_path, "r") as graph:
        labels = graph["sp_labels"][:]
        components = len(graph["components"].keys())
        if labels.shape != (components, 6):
            raise ValueError("{}: invalid sp_labels shape {}".format(scene, labels.shape))
        if int(labels.sum()) != raw_points:
            raise ValueError("{}: graph labels {} != raw points {}".format(scene, labels.sum(), raw_points))
        if graph["sp_point_count"].shape != (components, 1):
            raise ValueError("{}: invalid sp_point_count".format(scene))
        edges = int(graph["source"].shape[0])

    with h5py.File(feature_path, "r") as features:
        voxels = int(features["xyz"].shape[0])
        if features["labels"].shape != (voxels, 7):
            raise ValueError("{}: invalid voxel label histogram".format(scene))
        if int(features["labels"][:, :6].sum()) != raw_points:
            raise ValueError("{}: voxel labels do not preserve raw point count".format(scene))

    with h5py.File(cloud_path, "r") as clouds:
        if len(clouds.keys()) != components:
            raise ValueError("{}: cloud/component count mismatch".format(scene))
        for component_id in range(components):
            dataset = clouds["{}_data".format(component_id)]
            if dataset.ndim != 2 or dataset.shape[1] != 14 or dataset.shape[0] == 0:
                raise ValueError("{}: invalid component cloud {}".format(scene, dataset.shape))
            if not np.isfinite(dataset[: min(32, dataset.shape[0])]).all():
                raise ValueError("{}: non-finite point features".format(scene))
    return {
        "raw_points": raw_points,
        "voxels": voxels,
        "superpoints": components,
        "edges": edges,
        "class_points": labels.sum(axis=0).astype("int64").tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/path/to/odpt-hg-dataset")
    parser.add_argument("--output-root", default="/path/to/odpt-hg-sspc")
    parser.add_argument("--split-root", default="/path/to/odpt-hg-data/splits")
    parser.add_argument("--area", type=int, choices=[1, 3])
    parser.add_argument("--report")
    args = parser.parse_args()

    scenes = sorted(
        os.path.splitext(name)[0] for name in os.listdir(args.input_root)
        if name.startswith("Area_") and name.endswith(".txt")
        and (args.area is None or name.startswith("Area_{}_".format(args.area)))
    )
    expected = 18 if args.area is None else (15 if args.area == 1 else 3)
    if len(scenes) != expected:
        raise ValueError("expected {} scenes, found {}".format(expected, len(scenes)))
    results = {scene: validate_scene(args.input_root, args.output_root, scene) for scene in scenes}

    for budget, expected_count in ((10, 2), (20, 3)):
        split_path = os.path.join(args.split_root, "{}.txt".format(budget))
        with open(split_path, "r") as handle:
            split = [line.strip() for line in handle if line.strip()]
        if len(split) != expected_count or len(set(split)) != expected_count:
            raise ValueError("{}: invalid split size".format(split_path))
        if any(not scene.startswith("Area_1_") or scene not in results and args.area != 3 for scene in split):
            raise ValueError("{}: split contains unknown/non-training scene".format(split_path))

    report = {
        "status": "ok",
        "area": args.area,
        "scene_count": len(results),
        "total_points": sum(item["raw_points"] for item in results.values()),
        "total_voxels": sum(item["voxels"] for item in results.values()),
        "total_superpoints": sum(item["superpoints"] for item in results.values()),
        "scenes": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        with open(args.report, "w") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
