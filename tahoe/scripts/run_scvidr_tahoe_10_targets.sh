#!/usr/bin/env bash
set -euo pipefail

# 启动 10 个 Tahoe scVIDR target 任务；每个任务负责一个 holdout target cell_type。
#
# 这个 launcher 采用“分批队列”策略：
#   1. CUDA_DEVICE_LIST 指定当前可用 GPU，例如 3,5,7。
#   2. 每一批最多启动 len(CUDA_DEVICE_LIST) 个任务。
#   3. 同一批中，每张 GPU 只绑定一个 scVIDR 进程，避免多个训练进程抢同一张 GPU。
#   4. 当前批次全部结束后，再启动下一批。
#
# 示例：
#   CUDA_DEVICE_LIST=0,1,2,3 bash run_scvidr_tahoe_10_targets.sh
#
# 如果你当前只有 3/5/7 三张 GPU 空闲，推荐：
#   CUDA_DEVICE_LIST=3,5,7 bash run_scvidr_tahoe_10_targets.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCVIDR_TAHOE_LOG_DIR:-${SCRIPT_DIR}/logs/scvidr_tahoe}"
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

TARGET_INDICES=(0 1 2 3 4 5 6 7 8 9)
NUM_GPUS="${#GPUS[@]}"

# 按 NUM_GPUS 个 target 为一组分批运行。
# 例如 CUDA_DEVICE_LIST=3,5,7 时：
#   batch 1: target 0->GPU3, target 1->GPU5, target 2->GPU7
#   batch 2: target 3->GPU3, target 4->GPU5, target 5->GPU7
#   batch 3: target 6->GPU3, target 7->GPU5, target 8->GPU7
#   batch 4: target 9->GPU3
for ((batch_start = 0; batch_start < ${#TARGET_INDICES[@]}; batch_start += NUM_GPUS)); do
  batch_end=$((batch_start + NUM_GPUS - 1))
  if ((batch_end >= ${#TARGET_INDICES[@]})); then
    batch_end=$((${#TARGET_INDICES[@]} - 1))
  fi

  echo "start batch: target position ${batch_start}..${batch_end}"

  # 启动当前批次。slot 是当前批次内的位置，同时用于选择 GPU。
  for ((slot = 0; slot < NUM_GPUS; slot += 1)); do
    target_position=$((batch_start + slot))
    if ((target_position >= ${#TARGET_INDICES[@]})); then
      break
    fi

    target_index="${TARGET_INDICES[${target_position}]}"
    gpu="${GPUS[${slot}]}"
    log_file="${LOG_DIR}/target_${target_index}.log"

    echo "  start target_index=${target_index} on gpu=${gpu}, log=${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PYTHON_BIN}" "${SCRIPT_DIR}/scVIDR_pipeline.py" \
      --target-index "${target_index}" \
      > "${log_file}" 2>&1 &
  done

  # 等当前批次全部完成，再进入下一批；这样能保证同一张 GPU 同时只有一个 scVIDR 进程。
  wait
  echo "finish batch: target position ${batch_start}..${batch_end}"
done

wait
echo "all Tahoe scVIDR target jobs finished"
