from __future__ import division, print_function

import functools
import os
import random

import numpy as np
import torch
import torchnet as tnt

import spg


CLASS_NAMES = [
    "pipeline",
    "steel_frame",
    "elbow_pipe",
    "valve_guardrail",
    "gate_valve",
    "Christmas_tree_body",
]


def class_balanced_weights(class_points, power=0.5, max_weight=4.0):
    """Return stable inverse-frequency weights from labeled point counts only."""
    counts = np.asarray(class_points, dtype=np.float64)
    if counts.ndim != 1 or counts.size != len(CLASS_NAMES):
        raise ValueError("expected one point count for each ODPT-HG class")
    if np.any(counts <= 0):
        raise ValueError("every ODPT-HG class must occur in the labeled split")
    if not 0.0 <= power <= 1.0:
        raise ValueError("class balance power must be in [0, 1]")
    if max_weight < 1.0:
        raise ValueError("maximum class weight must be >= 1")
    weights = np.power(counts.mean() / counts, power)
    weights = np.clip(weights, 1.0 / max_weight, max_weight)
    weights /= weights.mean()
    return weights.astype(np.float32)


class LabeledSceneBatchSampler(torch.utils.data.Sampler):
    """Cover every unlabeled scene while ensuring each batch has supervision."""

    def __init__(self, labeled_indices, unlabeled_indices, batch_size, seed,
                 steps_per_epoch=None):
        if not labeled_indices:
            raise ValueError("ODPT-HG training requires at least one labeled scene")
        if batch_size < 2 and unlabeled_indices:
            raise ValueError("batch_size must be >=2 when unlabeled scenes are present")
        self.labeled = list(labeled_indices)
        self.unlabeled = list(unlabeled_indices)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        minimum_steps = int(np.ceil(
            len(self.unlabeled) / float(max(1, self.batch_size - 1))))
        self.steps_per_epoch = minimum_steps if steps_per_epoch is None else int(steps_per_epoch)
        if self.steps_per_epoch < minimum_steps:
            raise ValueError(
                "steps_per_epoch={} cannot cover {} unlabeled scenes with batch_size={}".format(
                    self.steps_per_epoch, len(self.unlabeled), self.batch_size))

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        labeled = list(self.labeled)
        unlabeled = list(self.unlabeled)
        rng.shuffle(labeled)
        rng.shuffle(unlabeled)
        if not unlabeled:
            for start in range(0, len(labeled), self.batch_size):
                yield labeled[start:start + self.batch_size]
            return
        for step in range(self.steps_per_epoch):
            batch = [labeled[step % len(labeled)]]
            for offset in range(self.batch_size - 1):
                position = step * (self.batch_size - 1) + offset
                batch.append(unlabeled[position % len(unlabeled)])
            yield batch


def _graph_dir(args):
    return os.path.join(args.ODPT_HG_PATH, "graph_v" if args.data_mode == "voxel" else "graph_p")


def _read_split(args):
    split_path = args.odpt_split_file
    if not split_path:
        split_path = os.path.join(args.ODPT_HG_SPLIT_ROOT, "{}.txt".format(args.odpt_budget))
    with open(split_path, "r") as handle:
        names = {line.strip() for line in handle if line.strip()}
    if not names:
        raise ValueError("empty labeled split: {}".format(split_path))
    if any(not name.startswith("Area_1_") for name in names):
        raise ValueError("labeled split may contain Area 1 scenes only")
    return names, os.path.abspath(split_path)


def _read_graphs(args, area, labeled_names=None):
    graphs = []
    names = []
    for filename in sorted(os.listdir(_graph_dir(args))):
        if not filename.endswith(".h5") or not filename.startswith("Area_{}_".format(area)):
            continue
        name = filename[:-3]
        weakly_labeled = labeled_names is not None and name in labeled_names
        graphs.append(spg.spg_reader(args, os.path.join(_graph_dir(args), filename), True, weakly_labeled))
        names.append(name)
    if not graphs:
        raise ValueError("no Area {} graphs found under {}".format(area, _graph_dir(args)))
    return graphs, names


def _fit_edge_scaler(args, train_graphs):
    edge_features = np.concatenate([graph[4] for graph in train_graphs], axis=0)
    args.odpt_edge_mean = edge_features.mean(axis=0).astype("float32")
    args.odpt_edge_scale = edge_features.std(axis=0).astype("float32")
    args.odpt_edge_scale[args.odpt_edge_scale == 0] = 1.0


def _scale_graphs(args, graphs):
    if not args.spg_attribs01:
        return
    for graph in graphs:
        graph[4][:] = (graph[4] - args.odpt_edge_mean) / args.odpt_edge_scale


def get_train_dataset(args):
    labeled_names, split_path = _read_split(args)
    train_graphs, train_names = _read_graphs(args, area=1, labeled_names=labeled_names)
    missing = labeled_names.difference(train_names)
    if missing:
        raise ValueError("labeled scenes missing graph files: {}".format(sorted(missing)))
    _fit_edge_scaler(args, train_graphs)
    _scale_graphs(args, train_graphs)
    dataset = tnt.dataset.ListDataset(
        [spg.spg_to_igraph(*graph) for graph in train_graphs],
        functools.partial(spg.loader, train=True, args=args, db_path=args.ODPT_HG_PATH),
    )
    dataset.labeled_indices = [i for i, name in enumerate(train_names) if name in labeled_names]
    dataset.unlabeled_indices = [i for i, name in enumerate(train_names) if name not in labeled_names]
    dataset.scene_names = train_names
    dataset.split_path = split_path
    labeled_graphs = [
        graph for graph, name in zip(train_graphs, train_names) if name in labeled_names
    ]
    dataset.class_point_counts = np.sum(
        [graph[1][:, :len(CLASS_NAMES)].sum(axis=0) for graph in labeled_graphs], axis=0
    ).astype(np.int64)
    dataset.class_weights = class_balanced_weights(
        dataset.class_point_counts,
        power=args.odpt_class_balance_power,
        max_weight=args.odpt_class_weight_max,
    )
    return dataset


def get_test_dataset(args, test_seed_offset=0):
    # This function is deliberately the only path that opens Area 3 graphs.
    test_graphs, _ = _read_graphs(args, area=3, labeled_names=None)
    _scale_graphs(args, test_graphs)
    return tnt.dataset.ListDataset(
        [spg.spg_to_igraph(*graph) for graph in test_graphs],
        functools.partial(
            spg.loader,
            train=False,
            args=args,
            db_path=args.ODPT_HG_PATH,
            test_seed_offset=test_seed_offset,
        ),
    )


def get_info(args):
    edge_features = 0
    for attribute in args.edge_attribs.split(","):
        name = attribute.split("/")[0]
        edge_features += 3 if name in ("delta_avg", "delta_std", "xyz") else 1
    return {
        "node_feats": 14 if args.pc_attribs == "" else len(args.pc_attribs),
        "edge_feats": edge_features,
        "classes": len(CLASS_NAMES),
        "inv_class_map": dict(enumerate(CLASS_NAMES)),
    }
