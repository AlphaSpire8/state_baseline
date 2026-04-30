#!/usr/bin/env bash
set -euo pipefail

# 启动 10 个 Tahoe scVIDR 进程；每个进程负责一个 holdout target cell_type。
# 默认所有进程使用当前可见 GPU 0。多 GPU 时可传入 CUDA_DEVICE_LIST，例如：
#   CUDA_DEVICE_LIST=0,1,2,3 bash run_scvidr_tahoe_10_targets.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCVIDR_TAHOE_LOG_DIR:-${SCRIPT_DIR}/../logs/scvidr_tahoe}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${CUDA_DEVICE_LIST:-0}"

mkdir -p "${LOG_DIR}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "CUDA_DEVICE_LIST 不能为空" >&2
  exit 1
fi

echo "log dir: ${LOG_DIR}"
echo "gpu list: ${GPU_LIST}"

for target_index in {0..9}; do
  gpu="${GPUS[$((target_index % ${#GPUS[@]}))]}"
  log_file="${LOG_DIR}/target_${target_index}.log"

  echo "start target_index=${target_index} on gpu=${gpu}, log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PYTHON_BIN}" "${SCRIPT_DIR}/scVIDR_pipeline.py" \
    --target-index "${target_index}" \
    > "${log_file}" 2>&1 &
done

wait
echo "all Tahoe scVIDR target jobs finished"
