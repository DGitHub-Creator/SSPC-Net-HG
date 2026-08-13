#!/usr/bin/env bash
set -euo pipefail

spgraph_root=${1:-/path/to/superpoint-graph}
env_root=${CONDA_PREFIX:?activate the sspc-net Conda environment first}
numpy_include=$(python -c 'import numpy; print(numpy.get_include())')
common=(
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  "-DPYTHON_LIBRARY=${env_root}/lib/libpython3.7m.so"
  "-DPYTHON_INCLUDE_DIR=${env_root}/include/python3.7m"
  "-DBOOST_INCLUDEDIR=${env_root}/include"
  "-DEIGEN3_INCLUDE_DIR=${env_root}/include/eigen3"
)

cmake -S "${spgraph_root}/partition/ply_c" -B "${spgraph_root}/partition/ply_c" \
  "${common[@]}" "-DPYTHON_NUMPY_INCLUDE_DIR=${numpy_include}"
cmake --build "${spgraph_root}/partition/ply_c" --parallel 8

cmake -S "${spgraph_root}/partition/cut-pursuit" \
  -B "${spgraph_root}/partition/cut-pursuit/build" \
  "${common[@]}" "-DPYTHON_EXECUTABLE=${env_root}/bin/python"
cmake --build "${spgraph_root}/partition/cut-pursuit/build" --parallel 8
