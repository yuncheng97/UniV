#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/../Data/EchoCP/preprocessed/}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results}"
BACKBONE="${BACKBONE:-vit_base_patch16_dinov3}"
TRAIN_SIZE="${TRAIN_SIZE:-352}"
CLIP_SIZE="${CLIP_SIZE:-3}"
NUM_CLASSES="${NUM_CLASSES:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-60}"
GPU_ID="${GPU_ID:-0,1}"
NOTE="${NOTE:-univ_dinov3_test}"
MASTER_PORT="${MASTER_PORT:-29503}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${GPU_ID}"
  NPROC_PER_NODE="${#GPU_IDS[@]}"
fi

cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

CMD=(
  /root/miniconda3/envs/univ/bin/torchrun
  --nproc_per_node "${NPROC_PER_NODE}"
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --master_port "${MASTER_PORT}"
  train.py
  --data_root "${DATA_ROOT}"
  --save_path "${OUTPUT_ROOT}"
  --backbone "${BACKBONE}"
  --train_size "${TRAIN_SIZE}"
  --clip_size "${CLIP_SIZE}"
  --batchsize "${BATCH_SIZE}"
  --epoch "${EPOCHS}"
  --num_classes "${NUM_CLASSES}"
  --gpu_id "${GPU_ID}"
  --distributed
  --optimizer adamw
  --scheduler cos
  --use_mode_pseudo_labels
  --ssm 
  --note "${NOTE}"
  "$@"
)

"${CMD[@]}"  > ./train_log5.log 2>&1 &
