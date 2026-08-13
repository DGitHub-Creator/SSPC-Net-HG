#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "semantic_seg"))

from odpt_hg_dataset import LabeledSceneBatchSampler, class_balanced_weights


class LabeledSceneBatchSamplerTest(unittest.TestCase):
    def test_every_unlabeled_scene_is_covered_with_supervision(self):
        sampler = LabeledSceneBatchSampler([0, 1], list(range(2, 15)), 2, 1)
        batches = list(sampler)
        self.assertEqual(set(range(2, 15)), {batch[1] for batch in batches})
        self.assertTrue(all(batch[0] in (0, 1) for batch in batches))

    def test_fixed_steps_equalize_budgets_and_labeled_exposure(self):
        sampler10 = LabeledSceneBatchSampler([0, 1], list(range(2, 15)), 2, 1, 18)
        sampler20 = LabeledSceneBatchSampler([0, 1, 2], list(range(3, 15)), 2, 1, 18)
        batches10 = list(sampler10)
        batches20 = list(sampler20)
        self.assertEqual(18, len(batches10))
        self.assertEqual(18, len(batches20))
        self.assertEqual([9, 9], [sum(b[0] == i for b in batches10) for i in (0, 1)])
        self.assertEqual([6, 6, 6], [sum(b[0] == i for b in batches20) for i in (0, 1, 2)])
        self.assertTrue(set(range(2, 15)).issubset({batch[1] for batch in batches10}))
        self.assertTrue(set(range(3, 15)).issubset({batch[1] for batch in batches20}))

    def test_rejects_too_few_steps_for_unlabeled_coverage(self):
        with self.assertRaises(ValueError):
            LabeledSceneBatchSampler([0, 1], list(range(2, 15)), 2, 1, 12)

    def test_legacy_minimum_twenty_percent_split_shape(self):
        sampler = LabeledSceneBatchSampler([0, 1, 2], list(range(3, 15)), 2, 1)
        batches = list(sampler)
        self.assertEqual(12, len(batches))
        self.assertTrue(all(len(batch) == 2 for batch in batches))


class ClassBalancedWeightsTest(unittest.TestCase):
    def test_uniform_counts_produce_unit_weights(self):
        weights = class_balanced_weights([10] * 6, power=0.5)
        np.testing.assert_allclose(np.ones(6), weights)

    def test_rare_classes_receive_larger_but_bounded_weights(self):
        weights = class_balanced_weights([1000, 500, 250, 100, 50, 10], power=0.5, max_weight=4)
        self.assertGreater(weights[-1], weights[0])
        self.assertAlmostEqual(1.0, float(weights.mean()), places=6)
        self.assertLessEqual(float(weights.max() / weights.min()), 16.0 + 1e-6)

    def test_rejects_missing_labeled_class(self):
        with self.assertRaises(ValueError):
            class_balanced_weights([10, 10, 10, 10, 10, 0])


if __name__ == "__main__":
    unittest.main()
