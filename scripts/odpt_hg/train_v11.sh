#!/usr/bin/env bash
set -euo pipefail

budget=${1:?usage: train_budget.sh BUDGET GPU [RUN_NAME]}
gpu=${2:?usage: train_budget.sh BUDGET GPU [RUN_NAME]}
run_name=${3:-official-seed1}

case "${budget}" in
  10|20) ;;
  *) echo "budget must be 10 or 20" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
data_root=/path/to/odpt-hg-sspc
split_root=/path/to/odpt-hg-data/splits
output_dir="${repo_root}/results/odpt_hg/${budget}pct/${run_name}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate sspc-4090

mkdir -p "${output_dir}"
cd "${repo_root}/semantic_seg"
export PYTHONPATH=./
export PYTHONHASHSEED=1
export CUDA_VISIBLE_DEVICES="${gpu}"

python -u main.py \
  --dataset odpt_hg \
  --ODPT_HG_PATH "${data_root}" \
  --ODPT_HG_SPLIT_ROOT "${split_root}" \
  --odpt_budget "${budget}" \
  --data_mode voxel \
  --epochs 400 \
  --batch_size 2 \
  --odpt_steps_per_epoch 18 \
  --odpt_class_balance_power 0.8 \
  --odpt_class_weight_max 5.0 \
  --lr 0.02 \
  --lr_steps '[330,380]' \
  --nworkers 4 \
  --test_nth_epoch 0 \
  --test_multisamp_n 10 \
  --save_nth_epoch 40 \
  --model_config 'gru_10,f_6' \
  --ptn_nfeat_stn 14 \
  --extension_th 0.9 \
  --odpt_prototype_seeds 1 \
  --odpt_seed_confidence 0.85 \
  --odpt_seed_margin 0.05 \
  --odpt_seed_max_per_class 3 \
  --odpt_class_quota 1 \
  --ext_epoch_gap 40 \
  --ext_drop 0.95 \
  --seed 1 \
  --odir "${output_dir}" 2>&1 | tee "${output_dir}/train.log"
