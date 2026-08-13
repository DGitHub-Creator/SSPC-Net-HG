#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import tempfile
import unittest

import h5py
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

from collect_odpt_hg_results import (
    compute_metrics,
    extract_extension_evidence,
    read_baselines,
    validate_predictions,
)


class ComputeMetricsTest(unittest.TestCase):
    def test_point_weighted_metrics_and_orientation(self):
        confusion = np.asarray([[8, 2], [1, 9]])
        result = compute_metrics(confusion, ["a", "b"])
        self.assertEqual(20, result["evaluated_points"])
        self.assertAlmostEqual(0.85, result["OA"])
        self.assertAlmostEqual((0.8 + 0.9) / 2, result["mAcc"])
        self.assertAlmostEqual((8.0 / 11 + 9.0 / 12) / 2, result["mIoU"])
        self.assertEqual(10, result["classes"][0]["support"])
        self.assertEqual(9, result["classes"][0]["predicted"])

    def test_rejects_invalid_shape(self):
        with self.assertRaises(ValueError):
            compute_metrics(np.zeros((2, 3)), ["a", "b"])

    def test_rejects_fractional_counts(self):
        with self.assertRaises(ValueError):
            compute_metrics(np.asarray([[1.5, 0], [0, 1]]), ["a", "b"])

    def test_extracts_label_extension_evidence(self):
        log = "\n".join([
            "extension points: torch.Size([0])",
            "prototype seed points: torch.Size([3])",
            "unrelated",
            "extension points: torch.Size([17])",
        ])
        self.assertEqual(
            {
                "final_extension": {
                    "checks": 2, "generated_points": 17, "nonempty_checks": 1
                },
                "prototype_seeds": {
                    "checks": 1, "generated_points": 3, "nonempty_checks": 1
                },
            },
            extract_extension_evidence(log),
        )

    def test_validates_prediction_scene_shapes(self):
        handle, path = tempfile.mkstemp(suffix=".h5")
        os.close(handle)
        try:
            with h5py.File(path, "w") as stream:
                stream.create_dataset("Area_3_conferenceRoom_20", data=np.zeros(3))
                stream.create_dataset("Area_3_conferenceRoom_21", data=np.zeros(2))
            self.assertEqual(
                {
                    "Area_3_conferenceRoom_20": [3],
                    "Area_3_conferenceRoom_21": [2],
                },
                validate_predictions(
                    path,
                    {
                        "Area_3_conferenceRoom_20": 3,
                        "Area_3_conferenceRoom_21": 2,
                    },
                ),
            )
        finally:
            os.unlink(path)

    def test_reads_budget_and_budget_alias_csv_formats(self):
        for header, value in (("budget", "10"), ("budget_alias", "10%")):
            handle, path = tempfile.mkstemp(suffix=".csv")
            os.close(handle)
            try:
                with open(path, "w") as stream:
                    stream.write("{},OA,mAcc,mIoU\n".format(header))
                    stream.write("{},0.8,0.7,0.6\n".format(value))
                self.assertEqual(
                    {"OA": 0.8, "mAcc": 0.7, "mIoU": 0.6},
                    read_baselines(path)[10],
                )
            finally:
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()
