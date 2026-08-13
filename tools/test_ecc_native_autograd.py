#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import types

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "semantic_seg"))
try:
    import igraph  # noqa: F401
except ImportError:
    # The numerical kernel test does not construct GraphConvInfo/igraph data.
    sys.modules["igraph"] = types.ModuleType("igraph")

from ecc import GraphConvFunction


def run(device, full_matrix):
    inputs = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device, requires_grad=True
    )
    source_indices = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
    degrees_cpu = torch.tensor([2, 1], dtype=torch.long)
    degrees_device = degrees_cpu.to(device)
    if full_matrix:
        weights = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0]],
                [[1.0, 1.0], [0.0, 1.0]],
            ],
            device=device,
            requires_grad=True,
        )
        expected = torch.tensor([[3.5, 5.0], [5.0, 11.0]], device=device)
    else:
        weights = torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 4.0]], device=device, requires_grad=True
        )
        expected = torch.tensor([[3.5, 5.0], [15.0, 24.0]], device=device)

    output = GraphConvFunction(
        2, 2, source_indices, None, degrees_cpu, degrees_device, edge_mem_limit=100
    )(inputs, weights)
    if not torch.allclose(output, expected):
        raise AssertionError("unexpected output {} != {}".format(output, expected))
    output.sum().backward()
    if inputs.grad is None or weights.grad is None:
        raise AssertionError("missing autograd gradients")


def main():
    run(torch.device("cpu"), False)
    run(torch.device("cpu"), True)
    if torch.cuda.is_available():
        run(torch.device("cuda"), False)
        run(torch.device("cuda"), True)
    print("ECC_NATIVE_AUTOGRAD_OK")


if __name__ == "__main__":
    main()
