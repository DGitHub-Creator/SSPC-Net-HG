#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python -u tools/preprocess_odpt_hg.py \
  --input-root /path/to/odpt-hg-dataset \
  --output-root /path/to/odpt-hg-sspc \
  --spgraph-root /path/to/superpoint-graph \
  --voxel-width 0.03 \
  --reg-strength 0.03
