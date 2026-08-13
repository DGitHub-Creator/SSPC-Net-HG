#!/usr/bin/env python
"""Build SSPC-Net superpoint graphs from the canonical ODPT-HG text files.

The geometric partition follows the official SPGraph release implementation.
Labels in the graph retain original (pre-voxelization) point counts so final
metrics are weighted over the same Area 3 points as the SemiGMMPoint protocol.
"""
from __future__ import print_function

import argparse
import json
import os
import random
import sys
import time

import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors

from spgraph_graphs import compute_sp_graph


N_CLASSES = 6
CLASS_NAMES = [
    "pipeline",
    "steel_frame",
    "elbow_pipe",
    "valve_guardrail",
    "gate_valve",
    "Christmas_tree_body",
]


def compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof):
    if k_nn_adj > k_nn_geof:
        raise ValueError("k_nn_adj must be <= k_nn_geof")
    nn = NearestNeighbors(n_neighbors=k_nn_geof + 1, algorithm="kd_tree").fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    target_geof = neighbors[:, 1:].reshape(-1).astype("uint32")
    neighbors = neighbors[:, 1:k_nn_adj + 1]
    distances = distances[:, 1:k_nn_adj + 1]
    source = np.repeat(np.arange(xyz.shape[0], dtype="uint32"), k_nn_adj)
    return {
        "source": source,
        "target": neighbors.reshape(-1).astype("uint32"),
        "distances": distances.reshape(-1).astype("float32"),
    }, target_geof


def atomic_h5(path, writer):
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    with h5py.File(tmp, "w") as handle:
        writer(handle)
    os.replace(tmp, path)


def write_features(path, xyz, rgb, label_hist, geof, graph_nn):
    def writer(handle):
        for i, name in enumerate(("linearity", "planarity", "scattering", "verticality")):
            handle.create_dataset(name, data=geof[:, i], dtype="float32")
        for name in ("source", "target", "distances"):
            handle.create_dataset(name, data=graph_nn[name])
        handle.create_dataset("xyz", data=xyz, dtype="float32")
        handle.create_dataset("rgb", data=rgb, dtype="uint8")
        handle.create_dataset("labels", data=label_hist, dtype="uint32")
    atomic_h5(path, writer)


def write_graph(path, graph, components, in_component):
    def writer(handle):
        group = handle.create_group("components")
        for component_id, indices in enumerate(components):
            group.create_dataset(str(component_id), data=indices, dtype="uint32")
        handle.create_dataset("in_component", data=in_component, dtype="uint32")
        for name in (
            "sp_labels", "sp_centroids", "sp_length", "sp_surface", "sp_volume",
            "sp_point_count", "source", "target", "se_delta_mean", "se_delta_std",
            "se_delta_norm", "se_delta_centroid", "se_length_ratio", "se_surface_ratio",
            "se_volume_ratio", "se_point_count_ratio",
        ):
            handle.create_dataset(name, data=graph[name])
    atomic_h5(path, writer)


def write_superpoint_clouds(path, xyz, rgb, geof, components, seed):
    elevation = xyz[:, 2:3] / 4.0 - 0.5
    local_geof = geof.copy()
    local_geof[:, :4] -= 0.5
    xyz_min = xyz.min(axis=0, keepdims=True)
    xyz_max = xyz.max(axis=0, keepdims=True)
    xyz_normalized = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-8)
    points = np.concatenate(
        [xyz, rgb.astype("float32") / 255.0 - 0.5, elevation, local_geof, xyz_normalized],
        axis=1,
    ).astype("float32")
    rng = random.Random(seed)

    def writer(handle):
        for component_id, indices in enumerate(components):
            indices = np.asarray(indices, dtype="int64")
            if indices.size > 10000:
                indices = np.asarray(rng.sample(indices.tolist(), 10000), dtype="int64")
            handle.create_dataset("{:d}_data".format(component_id), data=points[indices], dtype="float32")
    atomic_h5(path, writer)


def validate_labels(label_hist, scene):
    if label_hist.ndim != 2 or label_hist.shape[1] != N_CLASSES + 1:
        raise ValueError("{}: unexpected voxel label histogram {}".format(scene, label_hist.shape))
    if label_hist[:, N_CLASSES].sum() != 0:
        raise ValueError("{}: found labels outside 0..{}".format(scene, N_CLASSES - 1))


def process_scene(args, scene, libply_c, libcp):
    source_path = os.path.join(args.input_root, scene + ".txt")
    feature_path = os.path.join(args.output_root, "features", scene + ".h5")
    graph_path = os.path.join(args.output_root, "graph_v", scene + ".h5")
    cloud_path = os.path.join(args.output_root, "sp_voxel_pc", scene + ".h5")
    if not args.overwrite and all(os.path.isfile(p) for p in (feature_path, graph_path, cloud_path)):
        print("{}: outputs already exist; skipping".format(scene), flush=True)
        return

    started = time.time()
    raw = np.loadtxt(source_path, dtype="float32")
    if raw.ndim != 2 or raw.shape[1] != 7:
        raise ValueError("{}: expected seven columns, got {}".format(scene, raw.shape))
    xyz = np.ascontiguousarray(raw[:, :3], dtype="float32")
    rgb = np.ascontiguousarray(raw[:, 3:6], dtype="uint8")
    raw_labels = raw[:, 6]
    if raw_labels.min() < 0 or raw_labels.max() >= N_CLASSES or not np.all(raw_labels == np.floor(raw_labels)):
        raise ValueError("{}: labels must be in 0..{}".format(scene, N_CLASSES - 1))
    labels = np.ascontiguousarray(raw_labels, dtype="uint8")
    original_points = int(raw.shape[0])
    del raw

    xyz, rgb, label_hist = libply_c.prune(xyz, args.voxel_width, rgb, labels, N_CLASSES)
    xyz = np.ascontiguousarray(xyz, dtype="float32")
    rgb = np.ascontiguousarray(rgb, dtype="uint8")
    label_hist = np.ascontiguousarray(label_hist, dtype="uint32")
    validate_labels(label_hist, scene)
    if int(label_hist[:, :N_CLASSES].sum()) != original_points:
        raise ValueError("{}: voxel labels do not preserve original point count".format(scene))

    graph_nn, target_geof = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_geof)
    geof = libply_c.compute_geof(xyz, target_geof, args.k_nn_geof).astype("float32")
    write_features(feature_path, xyz, rgb, label_hist, geof, graph_nn)

    partition_features = np.hstack((geof, rgb.astype("float32") / 255.0)).astype("float32")
    partition_features[:, 3] *= 2.0
    distances = graph_nn["distances"]
    edge_weights = 1.0 / (args.lambda_edge_weight + distances / distances.mean())
    components, in_component = libcp.cutpursuit(
        partition_features,
        graph_nn["source"],
        graph_nn["target"],
        edge_weights.astype("float32"),
        args.reg_strength,
    )
    components = np.asarray(components, dtype=object)
    in_component = np.asarray(in_component, dtype="uint32")
    graph = compute_sp_graph(xyz, args.d_se_max, in_component, components, label_hist, N_CLASSES)
    # Raw ODPT labels are already zero based. The seventh column is the empty
    # overflow bin introduced by SPGraph's n_classes+1 histogram convention.
    graph["sp_labels"] = graph["sp_labels"][:, :N_CLASSES]
    if int(graph["sp_labels"].sum()) != original_points:
        raise ValueError("{}: superpoint labels do not preserve original point count".format(scene))
    write_graph(graph_path, graph, components, in_component)
    write_superpoint_clouds(cloud_path, xyz, rgb, geof, components, args.seed)
    print(
        "{}: original={} voxels={} superpoints={} edges={} elapsed={:.1f}s".format(
            scene, original_points, xyz.shape[0], len(components), graph["source"].shape[0],
            time.time() - started,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/path/to/odpt-hg-dataset")
    parser.add_argument("--output-root", default="/path/to/odpt-hg-sspc")
    parser.add_argument("--spgraph-root", default="/path/to/superpoint-graph")
    parser.add_argument("--scene", action="append", help="scene basename; repeat to process a subset")
    parser.add_argument("--voxel-width", type=float, default=0.03)
    parser.add_argument("--k-nn-geof", type=int, default=45)
    parser.add_argument("--k-nn-adj", type=int, default=10)
    parser.add_argument("--lambda-edge-weight", type=float, default=1.0)
    parser.add_argument("--reg-strength", type=float, default=0.03)
    parser.add_argument("--d-se-max", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(args.spgraph_root, "partition", "ply_c"))
    sys.path.insert(0, os.path.join(args.spgraph_root, "partition", "cut-pursuit", "build", "src"))
    import libcp
    import libply_c

    for directory in ("features", "graph_v", "sp_voxel_pc"):
        path = os.path.join(args.output_root, directory)
        if not os.path.isdir(path):
            os.makedirs(path)
    scenes = args.scene
    if not scenes:
        scenes = sorted(
            os.path.splitext(name)[0] for name in os.listdir(args.input_root)
            if name.startswith("Area_") and name.endswith(".txt")
        )
    manifest = {
        "input_root": os.path.abspath(args.input_root),
        "spgraph_upstream_commit": "1684a420aeadf8d00e13a7a7a70a6f4259209402",
        "spgraph_compat_commit": "fce4f56",
        "cutpursuit_compat_commit": "f12f663",
        "classes": CLASS_NAMES,
        "voxel_width": args.voxel_width,
        "k_nn_geof": args.k_nn_geof,
        "k_nn_adj": args.k_nn_adj,
        "reg_strength": args.reg_strength,
        "scenes": scenes,
    }
    with open(os.path.join(args.output_root, "preprocess_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    for scene in scenes:
        process_scene(args, scene, libply_c, libcp)


if __name__ == "__main__":
    main()
