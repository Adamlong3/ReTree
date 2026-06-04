#!/usr/bin/env bash
set -euo pipefail

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29731}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
DRAFT_PATH="${DRAFT_PATH:-z-lab/Qwen3-8B-DFlash-b16}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
TREE_BUDGET="${TREE_BUDGET:-32}"
CORRECTION_FREQ_THRESHOLD="${CORRECTION_FREQ_THRESHOLD:-6}"
CORRECTION_THRESHOLD="${CORRECTION_THRESHOLD:-0.01}"
CORRECTION_RECORD_TOP_K="${CORRECTION_RECORD_TOP_K:-8}"
CORRECTION_RECOVER_TOP_K="${CORRECTION_RECOVER_TOP_K:-8}"
METHODS="${METHODS:-dflash,retree}"
MEMORY_FILE="${MEMORY_FILE:-logs/retree_memory_calibrated.json}"

TASKS="${TASKS:-gsm8k:128}"
read -r -a TASK_ARRAY <<< "$TASKS"

mkdir -p logs

# ============================
# Step 1: Calibrate ReTree memory (skip if exists)
# ============================
if [ ! -f "$MEMORY_FILE" ]; then
  echo "========================================================"
  echo "[All] Running ReTree memory calibration on gsm8k (2000 samples, T=0.6, tb=${TREE_BUDGET})"
  echo "========================================================"

  python calibrate.py \
    --model-name-or-path "$MODEL_PATH" \
    --draft-name-or-path "$DRAFT_PATH" \
    --block-size "$BLOCK_SIZE" \
    --tree-budget "$TREE_BUDGET" \
    --dataset gsm8k \
    --max-samples 2000 \
    --temperature 0.6 \
    --max-new-tokens 512 \
    --record-top-k "$CORRECTION_RECORD_TOP_K" \
    --output-file "$MEMORY_FILE"
else
  echo "ReTree memory file already exists: $MEMORY_FILE (skipping calibration)"
fi

# ============================
# Step 2: Run selected methods in one benchmark
# ============================
for task in "${TASK_ARRAY[@]}"; do
  IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "$task"

  echo "========================================================"
  echo "[All] $DATASET_NAME with $MAX_SAMPLES samples (${METHODS})"
  echo "========================================================"

  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    benchmark.py \
    --dataset "$DATASET_NAME" \
    --max-samples "$MAX_SAMPLES" \
    --model-name-or-path "$MODEL_PATH" \
    --draft-name-or-path "$DRAFT_PATH" \
    --block-size "$BLOCK_SIZE" \
    --tree-budget "$TREE_BUDGET" \
    --max-new-tokens 2048 \
    --temperature 0.0 \
    --methods "$METHODS" \
    --correction-freq-threshold "$CORRECTION_FREQ_THRESHOLD" \
    --correction-threshold "$CORRECTION_THRESHOLD" \
    --correction-record-top-k "$CORRECTION_RECORD_TOP_K" \
    --correction-recover-top-k "$CORRECTION_RECOVER_TOP_K" \
    --correction-memory-file "$MEMORY_FILE" \
    2>&1 | tee "logs/all_tb${TREE_BUDGET}_tau${CORRECTION_THRESHOLD}_${DATASET_NAME}.log"
done
