#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import unittest

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "semantic_seg"))

from extension import extension_accum2, prototype_seed_indices


class PrototypeSeedTest(unittest.TestCase):
    def setUp(self):
        self.features = torch.tensor([
            [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9],
            [0.95, 0.05], [0.85, 0.15], [0.05, 0.95], [0.5, 0.5],
        ])
        self.scores = torch.tensor([
            [5.0, 0.0], [5.0, 0.0], [0.0, 5.0], [0.0, 5.0],
            [5.0, 0.0], [4.0, 0.0], [0.0, 5.0], [5.0, 0.0],
        ])
        self.labels = torch.tensor([0, 0, 1, 1, 255, 255, 255, 255])

    def test_seeds_unlabeled_scene_by_confidence_and_prototype_agreement(self):
        indices, labels = prototype_seed_indices(
            self.scores, self.features, self.labels, [4, 4],
            confidence_th=0.85, prototype_margin=0.1, max_per_class=1)
        self.assertEqual([4, 6], sorted(indices.tolist()))
        self.assertEqual({4: 0, 6: 1}, dict(zip(indices.tolist(), labels.tolist())))

    def test_never_targets_labeled_scene(self):
        indices, _ = prototype_seed_indices(
            self.scores, self.features, self.labels, [4, 4],
            confidence_th=0.5, prototype_margin=0.0, max_per_class=4)
        self.assertTrue(all(index >= 4 for index in indices.tolist()))

    def test_rejects_prediction_prototype_disagreement(self):
        scores = self.scores.clone()
        scores[4] = torch.tensor([0.0, 5.0])
        indices, _ = prototype_seed_indices(
            scores, self.features, self.labels, [4, 4],
            confidence_th=0.85, prototype_margin=0.1, max_per_class=4)
        self.assertNotIn(4, indices.tolist())

    def test_rejects_low_confidence(self):
        scores = self.scores.clone()
        scores[4] = torch.tensor([0.1, 0.0])
        indices, _ = prototype_seed_indices(
            scores, self.features, self.labels, [4, 4],
            confidence_th=0.85, prototype_margin=0.1, max_per_class=4)
        self.assertNotIn(4, indices.tolist())

    def test_requires_aligned_scene_sizes(self):
        with self.assertRaises(ValueError):
            prototype_seed_indices(
                self.scores, self.features, self.labels, [3, 4], max_per_class=1)

    def test_original_extension_empty_result_is_normalizable(self):
        clouds = torch.zeros((2, 1, 1))
        scores = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
        features = torch.zeros((2, 2))
        labels = torch.tensor([0, 255])
        edges = torch.zeros((0, 2), dtype=torch.long)
        _, _, output2, weak_label2, extend_idx, _ = extension_accum2(
            clouds, scores, features, labels, edges, th=0.9)
        normalized_idx = torch.as_tensor(extend_idx, dtype=torch.long).reshape(-1)
        normalized_label = torch.as_tensor(weak_label2, dtype=torch.long).reshape(-1)
        self.assertEqual(0, normalized_idx.numel())
        self.assertEqual(0, normalized_label.numel())
        self.assertEqual([], output2)

    def test_accumulated_mask_includes_class_zero_and_excludes_sentinel(self):
        ext_mask = torch.tensor([-1, 0, 1, 5])
        self.assertEqual(
            [False, True, True, True], (ext_mask >= 0).tolist())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_accumulated_and_current_indices_share_cuda_device(self):
        ext_mask = torch.tensor([-1, 0, 1, -1])
        accumulated = torch.nonzero(
            ext_mask >= 0, as_tuple=False).reshape(-1).long().cuda()
        current = torch.tensor([0, 3], dtype=torch.long, device="cuda")
        combined = torch.cat((accumulated, current), dim=0)
        self.assertTrue(combined.is_cuda)
        self.assertEqual([1, 2, 0, 3], combined.cpu().tolist())


if __name__ == "__main__":
    unittest.main()
