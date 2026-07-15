#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Full recovery-memory pipeline for Qwen3-4B / Qwen3-8B ReTree experiments
#
# What this script does:
#   For MODEL in 4B,8B and TREE_BUDGET in 16,32,64,128,256:
#     1) Calibrate NEW ngram recovery memory on five tasks with 5 GPUs in parallel.
#     2) Merge OLD recovery memory + NEW ngram recovery memory.
#     3) Benchmark five tasks at T=0 and T=1.
#        For each task/TB/model/temp, benchmark runs:
#          a) DDTREE_TREE_STRATEGY=heap + OLD recovery memory:
#               dflash / old DDTree / old ReTree
#          b) DDTREE_TREE_STRATEGY=rank_gated_ngram + MERGED recovery memory:
#               dflash / DDTree+ngram / ReTree
#
# Important:
#   - This is offline recovery-memory inference by default. It does NOT use --recovery-online-update.
#   - Set MODEL_PATH_4B, DRAFT_PATH_4B, MODEL_PATH_8B, and DRAFT_PATH_8B if
#     you use local checkpoints instead of Hugging Face model IDs.
#   - Set REUSE_4B_TB64_MERGED=1 only if ${MERGE_ROOT}/recovery_merged_64_4B.json
#     already exists and should be reused.
# =============================================================================

# -----------------------------
# Basic environment
# -----------------------------
if [[ -n "${ENV_BIN:-}" ]]; then
  export PATH="${ENV_BIN}:${PATH}"
fi
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"
cd "${PROJECT_ROOT}"

LOCAL_DATASETS_ROOT="${LOCAL_DATASETS_ROOT:-}"
if [[ -n "${LOCAL_DATASETS_ROOT}" ]]; then
  export LOCAL_DATASETS_ROOT
fi

RECOVERY_ROOT="${RECOVERY_ROOT:-${PROJECT_ROOT}/recovery_memory}"
OLD_ROOT="${OLD_ROOT:-${RECOVERY_ROOT}/old}"
NEW_ROOT="${NEW_ROOT:-${RECOVERY_ROOT}/new}"
MERGE_ROOT="${MERGE_ROOT:-${RECOVERY_ROOT}/merge}"
RUN_ROOT="${RUN_ROOT:-${RECOVERY_ROOT}/run}"
CALIB_LOG_ROOT="${CALIB_LOG_ROOT:-${RUN_ROOT}/calib_logs}"
MERGE_LOG_ROOT="${MERGE_LOG_ROOT:-${RUN_ROOT}/merge_logs}"
BENCH_LOG_ROOT="${BENCH_LOG_ROOT:-${RUN_ROOT}/bench_logs}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${RUN_ROOT}/summaries}"

mkdir -p "${OLD_ROOT}" "${NEW_ROOT}" "${MERGE_ROOT}" "${RUN_ROOT}" \
         "${CALIB_LOG_ROOT}" "${MERGE_LOG_ROOT}" "${BENCH_LOG_ROOT}" "${SUMMARY_ROOT}" logs

# -----------------------------
# Model paths
# -----------------------------
MODEL_PATH_4B="${MODEL_PATH_4B:-Qwen/Qwen3-4B}"
DRAFT_PATH_4B="${DRAFT_PATH_4B:-z-lab/Qwen3-4B-DFlash-b16}"

MODEL_PATH_8B="${MODEL_PATH_8B:-Qwen/Qwen3-8B}"
DRAFT_PATH_8B="${DRAFT_PATH_8B:-z-lab/Qwen3-8B-DFlash-b16}"

# Override to restrict, e.g. MODEL_LIST="4B" or MODEL_LIST="8B"
MODEL_LIST="${MODEL_LIST:-4B 8B}"
read -r -a MODELS <<< "${MODEL_LIST}"

# -----------------------------
# Tree budgets / tasks
# -----------------------------
TB_LIST="${TB_LIST:-16 32 64 128 256}"
read -r -a TREE_BUDGETS <<< "${TB_LIST}"

CALIB_TASKS=(
  "gsm8k:2000"
  "math500:500"
  "humaneval:164"
  "mbpp:374"
  "mt-bench:80"
)

BENCH_TASKS=(
  "gsm8k:128"
  "math500:128"
  "humaneval:164"
  "mbpp:128"
  "mt-bench:80"
)

# T=0 / T=1. Use 0.0 and 1.0 for clearer log names.
BENCH_TEMPS="${BENCH_TEMPS:-0.0 1.0}"
read -r -a TEMPS <<< "${BENCH_TEMPS}"

# -----------------------------
# Decode / recovery config
# -----------------------------
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_NEW_TOKENS_BENCH="${MAX_NEW_TOKENS_BENCH:-2048}"
MAX_NEW_TOKENS_CALIB="${MAX_NEW_TOKENS_CALIB:-512}"
CALIB_TEMPERATURE="${CALIB_TEMPERATURE:-0.6}"

export DDTREE_NGRAM_BETA="${DDTREE_NGRAM_BETA:-0.15}"
export DDTREE_NGRAM_RANK_CAP="${DDTREE_NGRAM_RANK_CAP:-8}"
export DDTREE_NGRAM_MAX_N="${DDTREE_NGRAM_MAX_N:-4}"
export DDTREE_NGRAM_CONTEXT_WINDOW="${DDTREE_NGRAM_CONTEXT_WINDOW:-2048}"
export DDTREE_NGRAM_MIN_COUNT="${DDTREE_NGRAM_MIN_COUNT:-1}"
export DDTREE_NGRAM_USE_LOG_COUNT="${DDTREE_NGRAM_USE_LOG_COUNT:-1}"
export DDTREE_NGRAM_LONGER_WEIGHT="${DDTREE_NGRAM_LONGER_WEIGHT:-1.5}"
export DDTREE_NGRAM_MAX_TOKEN_BONUS="${DDTREE_NGRAM_MAX_TOKEN_BONUS:-6.0}"

RECOVERY_FREQ_THRESHOLD="${RECOVERY_FREQ_THRESHOLD:-6}"
RECOVERY_THRESHOLD="${RECOVERY_THRESHOLD:-0.01}"
RECOVERY_RECORD_TOP_K="${RECOVERY_RECORD_TOP_K:-8}"
RECOVERY_RESCUE_TOP_K="${RECOVERY_RESCUE_TOP_K:-8}"

# -----------------------------
# GPU / control config
# -----------------------------
# Calibrate five tasks concurrently, one task per GPU.
CALIB_GPUS=( ${CALIB_GPUS:-0 1 2 3 4} )

# Benchmark uses torchrun.
BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-31000}"

# Force switches
FORCE_RECALIB="${FORCE_RECALIB:-0}"
FORCE_MERGE="${FORCE_MERGE:-0}"
FORCE_BENCH="${FORCE_BENCH:-0}"
REUSE_4B_TB64_MERGED="${REUSE_4B_TB64_MERGED:-0}"

# By default do offline recovery-memory inference. Set BENCH_ONLINE_UPDATE=1 only for an ablation.
BENCH_ONLINE_UPDATE="${BENCH_ONLINE_UPDATE:-0}"

# -----------------------------
# Helpers
# -----------------------------
safe_name() {
  echo "$1" | tr '-' '_' | tr '/' '_'
}

safe_float() {
  echo "$1" | sed 's/\./p/g' | sed 's/-/m/g'
}

model_paths() {
  local model_tag="$1"
  case "${model_tag}" in
    4B)
      echo "${MODEL_PATH_4B}|${DRAFT_PATH_4B}"
      ;;
    8B)
      echo "${MODEL_PATH_8B}|${DRAFT_PATH_8B}"
      ;;
    *)
      echo "[ERROR] Unknown model tag: ${model_tag}" >&2
      exit 1
      ;;
  esac
}

validate_json() {
  local json_file="$1"
  python - "$json_file" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    raise FileNotFoundError(p)
if p.stat().st_size <= 0:
    raise RuntimeError(f"empty json: {p}")
with open(p, "r") as f:
    json.load(f)
print(f"[OK] valid json: {p}")
PY
}

resolve_json_from_path_or_dir() {
  local p="$1"

  if [[ -f "${p}" ]]; then
    echo "${p}"
    return 0
  fi

  if [[ -d "${p}" ]]; then
    local preferred="${p}/recovery_combined_gsm8k_math500_humaneval_mbpp_mtbench.json"
    if [[ -f "${preferred}" ]]; then
      echo "${preferred}"
      return 0
    fi

    local any_json
    any_json="$(find "${p}" -maxdepth 1 -type f -name "*.json" | sort | head -n 1 || true)"
    if [[ -n "${any_json}" ]]; then
      echo "${any_json}"
      return 0
    fi
  fi

  return 1
}

old_memory_for_model_tb() {
  local model_tag="$1"
  local tb="$2"

  local base
  if [[ "${model_tag}" == "4B" ]]; then
    base="${OLD_ROOT}/recovery_multitask_qwen3_4b_dflash_b16_tb${tb}_top${RECOVERY_RECORD_TOP_K}_nostop_bidir"
  else
    base="${OLD_ROOT}/recovery_multitask_tb${tb}_top${RECOVERY_RECORD_TOP_K}_nostop_bidir"
  fi

  if resolved="$(resolve_json_from_path_or_dir "${base}")"; then
    echo "${resolved}"
    return 0
  fi

  # Try a few fallback patterns in case old recovery memories were copied as json files.
  local pattern
  if [[ "${model_tag}" == "4B" ]]; then
    pattern="${OLD_ROOT}/recovery_multitask_qwen3_4b_dflash_b16_tb${tb}_top${RECOVERY_RECORD_TOP_K}_nostop_bidir*.json"
  else
    pattern="${OLD_ROOT}/recovery_multitask_tb${tb}_top${RECOVERY_RECORD_TOP_K}_nostop_bidir*.json"
  fi

  local file
  file="$(ls ${pattern} 2>/dev/null | sort | head -n 1 || true)"
  if [[ -n "${file}" ]]; then
    echo "${file}"
    return 0
  fi

  echo "[ERROR] Cannot find OLD recovery memory for model=${model_tag}, tb=${tb}" >&2
  echo "        Tried base: ${base}" >&2
  echo "        Tried pattern: ${pattern}" >&2
  return 1
}

new_memory_dir_for_model_tb() {
  local model_tag="$1"
  local tb="$2"
  echo "${NEW_ROOT}/${model_tag}/tb${tb}_ngram_top${RECOVERY_RECORD_TOP_K}_rankcap${DDTREE_NGRAM_RANK_CAP}"
}

new_task_memory_file() {
  local model_tag="$1"
  local tb="$2"
  local dataset_name="$3"
  local max_samples="$4"
  local safe_dataset
  safe_dataset="$(safe_name "${dataset_name}")"
  echo "$(new_memory_dir_for_model_tb "${model_tag}" "${tb}")/recovery_${safe_dataset}_${max_samples}_${model_tag}_ngram_tb${tb}_rankcap${DDTREE_NGRAM_RANK_CAP}.json"
}

new_combined_memory_file() {
  local model_tag="$1"
  local tb="$2"
  echo "$(new_memory_dir_for_model_tb "${model_tag}" "${tb}")/recovery_new_${model_tag}_ngram_alltask_tb${tb}.json"
}

merged_memory_file() {
  local model_tag="$1"
  local tb="$2"
  echo "${MERGE_ROOT}/recovery_merged_${tb}_${model_tag}.json"
}

calib_log_file() {
  local model_tag="$1"
  local tb="$2"
  local dataset_name="$3"
  local max_samples="$4"
  local gpu_id="$5"
  local safe_dataset
  safe_dataset="$(safe_name "${dataset_name}")"
  echo "${CALIB_LOG_ROOT}/${model_tag}/tb${tb}/calib_${safe_dataset}_${max_samples}_gpu${gpu_id}_${model_tag}_tb${tb}.log"
}

merge_log_file() {
  local model_tag="$1"
  local tb="$2"
  echo "${MERGE_LOG_ROOT}/merge_${model_tag}_tb${tb}.log"
}

bench_log_file() {
  local model_tag="$1"
  local tb="$2"
  local temp="$3"
  local dataset_name="$4"
  local max_samples="$5"
  local run_kind="$6"  # heap uses OLD recovery memory; ngram_retree uses MERGED recovery memory
  local safe_dataset
  safe_dataset="$(safe_name "${dataset_name}")"
  local temp_safe
  temp_safe="$(safe_float "${temp}")"
  echo "${BENCH_LOG_ROOT}/${model_tag}/tb${tb}/T${temp_safe}/bench_${model_tag}_tb${tb}_T${temp_safe}_${safe_dataset}_${max_samples}_${run_kind}.log"
}

check_project_files() {
  echo "[CHECK] Python syntax..."
  python -m py_compile benchmark.py ddtree.py retree.py retree_calibrate.py
}

print_config() {
  cat <<EOF2

################################################################################
# Full Qwen3 4B/8B TB sweep: calibrate new ngram recovery memory, merge old+new, benchmark
################################################################################
PROJECT_ROOT                 = ${PROJECT_ROOT}
LOCAL_DATASETS_ROOT          = ${LOCAL_DATASETS_ROOT}
RECOVERY_ROOT                     = ${RECOVERY_ROOT}
OLD_ROOT                     = ${OLD_ROOT}
NEW_ROOT                     = ${NEW_ROOT}
MERGE_ROOT                   = ${MERGE_ROOT}
RUN_ROOT                     = ${RUN_ROOT}
MODEL_LIST                   = ${MODEL_LIST}
TB_LIST                      = ${TB_LIST}
BENCH_TEMPS                  = ${BENCH_TEMPS}
MODEL_PATH_4B                = ${MODEL_PATH_4B}
DRAFT_PATH_4B                = ${DRAFT_PATH_4B}
MODEL_PATH_8B                = ${MODEL_PATH_8B}
DRAFT_PATH_8B                = ${DRAFT_PATH_8B}
BLOCK_SIZE                   = ${BLOCK_SIZE}
MAX_NEW_TOKENS_BENCH         = ${MAX_NEW_TOKENS_BENCH}
MAX_NEW_TOKENS_CALIB         = ${MAX_NEW_TOKENS_CALIB}
CALIB_TEMPERATURE            = ${CALIB_TEMPERATURE}
DDTREE_NGRAM_BETA            = ${DDTREE_NGRAM_BETA}
DDTREE_NGRAM_RANK_CAP        = ${DDTREE_NGRAM_RANK_CAP}
DDTREE_NGRAM_MAX_N           = ${DDTREE_NGRAM_MAX_N}
RECOVERY_FREQ_THRESHOLD                   = ${RECOVERY_FREQ_THRESHOLD}
RECOVERY_THRESHOLD                      = ${RECOVERY_THRESHOLD}
RECOVERY_RECORD_TOP_K             = ${RECOVERY_RECORD_TOP_K}
RECOVERY_RESCUE_TOP_K             = ${RECOVERY_RESCUE_TOP_K}
CALIB_GPUS                   = ${CALIB_GPUS[*]}
BENCH_CUDA_VISIBLE_DEVICES   = ${BENCH_CUDA_VISIBLE_DEVICES}
NPROC_PER_NODE               = ${NPROC_PER_NODE}
MASTER_PORT_BASE             = ${MASTER_PORT_BASE}
FORCE_RECALIB                = ${FORCE_RECALIB}
FORCE_MERGE                  = ${FORCE_MERGE}
FORCE_BENCH                  = ${FORCE_BENCH}
REUSE_4B_TB64_MERGED         = ${REUSE_4B_TB64_MERGED}
BENCH_ONLINE_UPDATE          = ${BENCH_ONLINE_UPDATE}
################################################################################
EOF2
}

# -----------------------------
# Calibration
# -----------------------------
run_calibration_for_model_tb() {
  local model_tag="$1"
  local tb="$2"

  if [[ "${REUSE_4B_TB64_MERGED}" == "1" && "${model_tag}" == "4B" && "${tb}" == "64" ]]; then
    local premerged
    premerged="$(merged_memory_file "${model_tag}" "${tb}")"
    echo "[SKIP CALIB] model=4B tb=64 reuses existing merged recovery memory: ${premerged}"
    validate_json "${premerged}" >/dev/null
    return 0
  fi

  if (( ${#CALIB_GPUS[@]} < ${#CALIB_TASKS[@]} )); then
    echo "[ERROR] CALIB_GPUS has ${#CALIB_GPUS[@]} GPUs, but CALIB_TASKS has ${#CALIB_TASKS[@]} tasks." >&2
    exit 1
  fi

  local paths model_path draft_path
  paths="$(model_paths "${model_tag}")"
  model_path="${paths%%|*}"
  draft_path="${paths##*|}"

  local new_dir
  new_dir="$(new_memory_dir_for_model_tb "${model_tag}" "${tb}")"
  mkdir -p "${new_dir}" "${CALIB_LOG_ROOT}/${model_tag}/tb${tb}"

  echo ""
  echo "################################################################################"
  echo "# Calibrate NEW ngram recovery memory: model=${model_tag}, tb=${tb}"
  echo "################################################################################"

  local pids=()
  local names=()
  local logs=()
  local outputs=()

  local idx=0
  for task in "${CALIB_TASKS[@]}"; do
    IFS=':' read -r dataset_name max_samples <<< "${task}"
    local gpu_id="${CALIB_GPUS[$idx]}"
    local output_file log_file done_file
    output_file="$(new_task_memory_file "${model_tag}" "${tb}" "${dataset_name}" "${max_samples}")"
    log_file="$(calib_log_file "${model_tag}" "${tb}" "${dataset_name}" "${max_samples}" "${gpu_id}")"
    done_file="${output_file}.done"

    if [[ "${FORCE_RECALIB}" == "1" ]]; then
      rm -f "${output_file}" "${done_file}" "${log_file}"
    fi

    if [[ -s "${output_file}" ]] && validate_json "${output_file}" >/dev/null 2>&1; then
      echo "[SKIP CALIB] valid recovery memory exists: ${output_file}"
      touch "${done_file}"
    else
      echo "[LAUNCH CALIB] model=${model_tag}, tb=${tb}, dataset=${dataset_name}, samples=${max_samples}, gpu=${gpu_id}"
      echo "               output=${output_file}"
      echo "               log=${log_file}"

      (
        set -euo pipefail
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        export DDTREE_TREE_STRATEGY="rank_gated_ngram"

        python retree_calibrate.py \
          --model-name-or-path "${model_path}" \
          --draft-name-or-path "${draft_path}" \
          --block-size "${BLOCK_SIZE}" \
          --tree-budget "${tb}" \
          --dataset "${dataset_name}" \
          --max-samples "${max_samples}" \
          --temperature "${CALIB_TEMPERATURE}" \
          --max-new-tokens "${MAX_NEW_TOKENS_CALIB}" \
          --record-top-k "${RECOVERY_RECORD_TOP_K}" \
          --output-file "${output_file}"

        validate_json "${output_file}" >/dev/null
        touch "${done_file}"
      ) > "${log_file}" 2>&1 &

      pids+=("$!")
      names+=("${dataset_name}:${max_samples}")
      logs+=("${log_file}")
      outputs+=("${output_file}")
    fi

    idx=$((idx + 1))
  done

  local failed=0
  for i in "${!pids[@]}"; do
    local pid="${pids[$i]}"
    local name="${names[$i]}"
    local log_file="${logs[$i]}"
    local output_file="${outputs[$i]}"

    if wait "${pid}"; then
      echo "[DONE CALIB] model=${model_tag}, tb=${tb}, ${name}, output=${output_file}"
    else
      echo "[FAILED CALIB] model=${model_tag}, tb=${tb}, ${name}, pid=${pid}" >&2
      echo "               log=${log_file}" >&2
      echo "               last 100 log lines:" >&2
      tail -n 100 "${log_file}" >&2 || true
      failed=1
    fi
  done

  if [[ "${failed}" != "0" ]]; then
    echo "[ERROR] Calibration failed for model=${model_tag}, tb=${tb}" >&2
    exit 1
  fi

  echo "[CHECK] verify expected new recovery memory files..."
  local missing=0
  for task in "${CALIB_TASKS[@]}"; do
    IFS=':' read -r dataset_name max_samples <<< "${task}"
    local output_file
    output_file="$(new_task_memory_file "${model_tag}" "${tb}" "${dataset_name}" "${max_samples}")"
    if ! validate_json "${output_file}" >/dev/null 2>&1; then
      echo "[MISSING/INVALID] model=${model_tag}, tb=${tb}, task=${dataset_name}:${max_samples}, file=${output_file}" >&2
      missing=1
    else
      echo "[OK recovery memory] ${output_file}"
    fi
  done

  if [[ "${missing}" != "0" ]]; then
    echo "[ERROR] Missing or invalid new recovery memory files for model=${model_tag}, tb=${tb}" >&2
    exit 1
  fi
}

# -----------------------------
# Merge
# -----------------------------
merge_old_new_for_model_tb() {
  local model_tag="$1"
  local tb="$2"

  if [[ "${REUSE_4B_TB64_MERGED}" == "1" && "${model_tag}" == "4B" && "${tb}" == "64" ]]; then
    local premerged
    premerged="$(merged_memory_file "${model_tag}" "${tb}")"
    echo "[SKIP MERGE] model=4B tb=64 uses existing merged recovery memory: ${premerged}"
    validate_json "${premerged}" >/dev/null
    return 0
  fi

  local old_memory new_combined merged log_file
  old_memory="$(old_memory_for_model_tb "${model_tag}" "${tb}")"
  new_combined="$(new_combined_memory_file "${model_tag}" "${tb}")"
  merged="$(merged_memory_file "${model_tag}" "${tb}")"
  log_file="$(merge_log_file "${model_tag}" "${tb}")"

  mkdir -p "$(dirname "${new_combined}")" "$(dirname "${merged}")" "$(dirname "${log_file}")"

  echo ""
  echo "################################################################################"
  echo "# Merge NEW all-task recovery memory: model=${model_tag}, tb=${tb}"
  echo "################################################################################"

  if [[ "${FORCE_MERGE}" == "1" ]]; then
    rm -f "${new_combined}" "${merged}" "${log_file}" "${log_file}.new"
  fi

  local new_inputs=()
  for task in "${CALIB_TASKS[@]}"; do
    IFS=':' read -r dataset_name max_samples <<< "${task}"
    new_inputs+=("$(new_task_memory_file "${model_tag}" "${tb}" "${dataset_name}" "${max_samples}")")
  done

  if [[ -s "${new_combined}" ]] && validate_json "${new_combined}" >/dev/null 2>&1; then
    echo "[SKIP MERGE NEW] valid new combined recovery memory exists: ${new_combined}"
  else
    python - "${new_combined}" "${new_inputs[@]}" <<'PY' 2>&1 | tee "${log_file}.new"
import json
import sys
from pathlib import Path
from collections import defaultdict

out = Path(sys.argv[1])
inputs = [Path(x) for x in sys.argv[2:]]
merged = defaultdict(int)

for path in inputs:
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"[Load new] {path}")
    with open(path, "r") as f:
        data = json.load(f)
    pairs = data.get("rescue_pairs", data) if isinstance(data, dict) else None
    if not isinstance(pairs, dict):
        raise TypeError(f"Unsupported recovery memory format: {path}")
    for k, v in pairs.items():
        merged[str(k)] += int(v)

out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(dict(merged), f)

print(f"[DONE] saved new combined: {out}")
print(f"unique pairs: {len(merged)}")
print(f"total counts: {sum(merged.values())}")
print("top10:")
for k, v in sorted(merged.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(k, v)
PY
    validate_json "${new_combined}" >/dev/null
  fi

  echo ""
  echo "################################################################################"
  echo "# Merge OLD + NEW recovery memory: model=${model_tag}, tb=${tb}"
  echo "# OLD=${old_memory}"
  echo "# NEW=${new_combined}"
  echo "# OUT=${merged}"
  echo "################################################################################"

  validate_json "${old_memory}" >/dev/null
  validate_json "${new_combined}" >/dev/null

  if [[ -s "${merged}" ]] && validate_json "${merged}" >/dev/null 2>&1; then
    echo "[SKIP MERGE OLD+NEW] valid merged recovery memory exists: ${merged}"
    return 0
  fi

  python - "${merged}" "${old_memory}" "${new_combined}" <<'PY' 2>&1 | tee "${log_file}"
import json
import sys
from pathlib import Path
from collections import defaultdict

out = Path(sys.argv[1])
inputs = [Path(x) for x in sys.argv[2:]]
merged = defaultdict(int)

for path in inputs:
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"[Load] {path}")
    with open(path, "r") as f:
        data = json.load(f)
    pairs = data.get("rescue_pairs", data) if isinstance(data, dict) else None
    if not isinstance(pairs, dict):
        raise TypeError(f"Unsupported recovery memory format: {path}")
    for k, v in pairs.items():
        merged[str(k)] += int(v)

out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(dict(merged), f)

print(f"[DONE] saved merged old+new: {out}")
print(f"unique pairs: {len(merged)}")
print(f"total counts: {sum(merged.values())}")
print("top10:")
for k, v in sorted(merged.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(k, v)
PY

  validate_json "${merged}" >/dev/null
  echo "[DONE MERGE] valid merged recovery memory: ${merged}"
}

# -----------------------------
# Benchmark
# -----------------------------
run_benchmark_one() {
  local model_tag="$1"
  local tb="$2"
  local temp="$3"
  local dataset_name="$4"
  local max_samples="$5"
  local run_kind="$6"

  local paths model_path draft_path
  paths="$(model_paths "${model_tag}")"
  model_path="${paths%%|*}"
  draft_path="${paths##*|}"

  local log_file done_file port merged old_memory memory_for_run
  log_file="$(bench_log_file "${model_tag}" "${tb}" "${temp}" "${dataset_name}" "${max_samples}" "${run_kind}")"
  done_file="${log_file}.done"
  port=$((MASTER_PORT_BASE + GLOBAL_RUN_IDX))
  GLOBAL_RUN_IDX=$((GLOBAL_RUN_IDX + 1))

  merged="$(merged_memory_file "${model_tag}" "${tb}")"
  old_memory="$(old_memory_for_model_tb "${model_tag}" "${tb}")"

  mkdir -p "$(dirname "${log_file}")"

  if [[ "${FORCE_BENCH}" == "1" ]]; then
    rm -f "${log_file}" "${done_file}"
  fi

  if [[ -f "${done_file}" ]]; then
    echo "[SKIP BENCH] model=${model_tag}, tb=${tb}, T=${temp}, dataset=${dataset_name}, kind=${run_kind}, log=${log_file}"
    return 0
  fi

  local methods strategy
  local recovery_args=()

  case "${run_kind}" in
    heap)
      # Old-tree ablation:
      #   DFlash / old DDTree heap / old ReTree recovery
      # Use OLD recovery memory here, not merged ngram recovery memory.
      strategy="heap"
      methods="dflash,ddtree,retree"
      memory_for_run="${old_memory}"
      validate_json "${memory_for_run}" >/dev/null
      recovery_args=(
        --recovery-freq-threshold "${RECOVERY_FREQ_THRESHOLD}"
        --recovery-threshold "${RECOVERY_THRESHOLD}"
        --recovery-memory-file "${memory_for_run}"
        --recovery-record-top-k "${RECOVERY_RECORD_TOP_K}"
        --recovery-rescue-top-k "${RECOVERY_RESCUE_TOP_K}"
      )
      if [[ "${BENCH_ONLINE_UPDATE}" == "1" ]]; then
        recovery_args+=(--recovery-online-update)
      fi
      ;;

    ngram_retree)
      # Ngram-tree ablation:
      #   DFlash / DDTree+ngram / ReTree
      # Use MERGED old+new recovery memory here.
      strategy="rank_gated_ngram"
      methods="dflash,ddtree,retree"
      memory_for_run="${merged}"
      validate_json "${memory_for_run}" >/dev/null
      recovery_args=(
        --recovery-freq-threshold "${RECOVERY_FREQ_THRESHOLD}"
        --recovery-threshold "${RECOVERY_THRESHOLD}"
        --recovery-memory-file "${memory_for_run}"
        --recovery-record-top-k "${RECOVERY_RECORD_TOP_K}"
        --recovery-rescue-top-k "${RECOVERY_RESCUE_TOP_K}"
      )
      if [[ "${BENCH_ONLINE_UPDATE}" == "1" ]]; then
        recovery_args+=(--recovery-online-update)
      fi
      ;;

    *)
      echo "[ERROR] Unknown run_kind=${run_kind}" >&2
      exit 1
      ;;
  esac

  echo ""
  echo "============================================================"
  echo "[Benchmark] model=${model_tag}, tb=${tb}, T=${temp}, dataset=${dataset_name}, samples=${max_samples}, kind=${run_kind}"
  echo "strategy=${strategy}"
  echo "methods=${methods}"
  echo "memory_for_run=${memory_for_run}"
  echo "old_memory=${old_memory}"
  echo "merged_memory=${merged}"
  echo "log=${log_file}"
  echo "============================================================"

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES}"
    export DDTREE_TREE_STRATEGY="${strategy}"

    torchrun \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port="${port}" \
      benchmark.py \
      --dataset "${dataset_name}" \
      --max-samples "${max_samples}" \
      --model-name-or-path "${model_path}" \
      --draft-name-or-path "${draft_path}" \
      --block-size "${BLOCK_SIZE}" \
      --tree-budget "${tb}" \
      --max-new-tokens "${MAX_NEW_TOKENS_BENCH}" \
      --temperature "${temp}" \
      --methods "${methods}" \
      "${recovery_args[@]}"
  ) 2>&1 | tee "${log_file}"

  touch "${done_file}"
  echo "[DONE BENCH] model=${model_tag}, tb=${tb}, T=${temp}, dataset=${dataset_name}, kind=${run_kind}, log=${log_file}"
}

run_benchmarks_for_model_tb() {
  local model_tag="$1"
  local tb="$2"

  local merged old_memory
  merged="$(merged_memory_file "${model_tag}" "${tb}")"
  old_memory="$(old_memory_for_model_tb "${model_tag}" "${tb}")"

  validate_json "${old_memory}" >/dev/null
  validate_json "${merged}" >/dev/null

  echo ""
  echo "################################################################################"
  echo "# Benchmark all tasks/temperatures: model=${model_tag}, tb=${tb}"
  echo "# OLD recovery memory for heap=${old_memory}"
  echo "# MERGED recovery memory for ngram_retree=${merged}"
  echo "################################################################################"

  for temp in "${TEMPS[@]}"; do
    for task in "${BENCH_TASKS[@]}"; do
      IFS=':' read -r dataset_name max_samples <<< "${task}"

      # heap: old DDTree + old ReTree, using OLD recovery memory
      run_benchmark_one "${model_tag}" "${tb}" "${temp}" "${dataset_name}" "${max_samples}" "heap"

      # ngram_retree: DDTree+ngram + ReTree, using MERGED recovery memory
      run_benchmark_one "${model_tag}" "${tb}" "${temp}" "${dataset_name}" "${max_samples}" "ngram_retree"
    done
  done
}

# -----------------------------
# Summary
# -----------------------------
summarize_logs() {
  local out_tsv="${SUMMARY_ROOT}/summary_$(date +%Y%m%d_%H%M%S).tsv"
  mkdir -p "${SUMMARY_ROOT}"

  python - "${BENCH_LOG_ROOT}" "${out_tsv}" <<'PY'
import re
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
out_tsv = Path(sys.argv[2])
logs = sorted(log_root.rglob("bench_*.log"))

method_re = re.compile(r"--- (.*?) ---")
speed_re = re.compile(r"Speedup vs baseline:\s*([0-9.]+)x")
accept_re = re.compile(r"Average Acceptance length:\s*([0-9.]+)")
accuracy_re = re.compile(r"Accuracy:\s*([0-9.]+)%\s*\((.*?)\)")
rescued_re = re.compile(r"Recovery Total Rescued Tokens:\s*([0-9]+)")

def infer_run_kind(path: Path) -> str:
    name = path.name
    if name.endswith("_heap.log"):
        return "heap"
    if name.endswith("_ngram_retree.log"):
        return "ngram_retree"
    if "_heap." in name:
        return "heap"
    if "_ngram_retree." in name:
        return "ngram_retree"
    if "_heap" in name:
        return "heap"
    if "_ngram_retree" in name:
        return "ngram_retree"
    return ""

def normalized_variant(run_kind: str, method: str) -> str:
    if method == "DFlash (linear SD)":
        return "dflash"
    if run_kind == "heap" and method == "DDTree":
        return "ddtree"
    if run_kind == "heap" and method == "ReTree":
        return "retree_old_memory"
    if run_kind == "ngram_retree" and method == "DDTree":
        return "ddtree_ngram"
    if run_kind == "ngram_retree" and method == "ReTree":
        return "ddtree_ngram_retree_merged_memory"
    return method

rows = []
for path in logs:
    current = None
    rescued = ""
    run_kind = infer_run_kind(path)

    for line in path.read_text(errors="ignore").splitlines():
        m = method_re.search(line)
        if m:
            method = m.group(1)
            current = {
                "log": str(path),
                "run_kind": run_kind,
                "variant": normalized_variant(run_kind, method),
                "method": method,
                "speedup": "",
                "avg_accept": "",
                "accuracy": "",
                "evaluable": "",
                "rescued": "",
            }
            rows.append(current)
            continue

        if current is not None:
            m = speed_re.search(line)
            if m:
                current["speedup"] = m.group(1)
            m = accept_re.search(line)
            if m:
                current["avg_accept"] = m.group(1)
            m = accuracy_re.search(line)
            if m:
                current["accuracy"] = m.group(1)
                current["evaluable"] = m.group(2)

        m = rescued_re.search(line)
        if m:
            rescued = m.group(1)

    for r in rows:
        if r["log"] == str(path) and r["method"] == "ReTree":
            r["rescued"] = rescued

header = ["log", "run_kind", "variant", "method", "speedup", "avg_accept", "accuracy", "evaluable", "rescued"]
out_tsv.parent.mkdir(parents=True, exist_ok=True)
with out_tsv.open("w") as f:
    f.write("\t".join(header) + "\n")
    for r in rows:
        f.write("\t".join(str(r.get(k, "")) for k in header) + "\n")

print(f"[DONE] summary saved: {out_tsv}")
print("\t".join(header))
for r in rows:
    print("\t".join(str(r.get(k, "")) for k in header))
PY
}

# -----------------------------
# Main
# -----------------------------
main() {
  print_config
  check_project_files

  echo ""
  echo "################################################################################"
  echo "# Preflight recovery memory paths"
  echo "################################################################################"
  for model_tag in "${MODELS[@]}"; do
    for tb in "${TREE_BUDGETS[@]}"; do
      echo "[CHECK] model=${model_tag} tb=${tb} old_memory=$(old_memory_for_model_tb "${model_tag}" "${tb}")"
      validate_json "$(old_memory_for_model_tb "${model_tag}" "${tb}")" >/dev/null

      if [[ "${REUSE_4B_TB64_MERGED}" == "1" && "${model_tag}" == "4B" && "${tb}" == "64" ]]; then
        echo "[CHECK] model=4B tb=64 premerged=$(merged_memory_file "${model_tag}" "${tb}")"
        validate_json "$(merged_memory_file "${model_tag}" "${tb}")" >/dev/null
      fi
    done
  done

  for model_tag in "${MODELS[@]}"; do
    for tb in "${TREE_BUDGETS[@]}"; do
      run_calibration_for_model_tb "${model_tag}" "${tb}"
      merge_old_new_for_model_tb "${model_tag}" "${tb}"
    done
  done

  GLOBAL_RUN_IDX=0
  for model_tag in "${MODELS[@]}"; do
    for tb in "${TREE_BUDGETS[@]}"; do
      run_benchmarks_for_model_tb "${model_tag}" "${tb}"
    done
  done

  echo ""
  echo "################################################################################"
  echo "# Summary"
  echo "################################################################################"
  summarize_logs

  cat <<EOF2

################################################################################
# DONE
################################################################################
New recovery memory root:
  ${NEW_ROOT}

Merged recovery memory root:
  ${MERGE_ROOT}

Calibration logs:
  ${CALIB_LOG_ROOT}

Merge logs:
  ${MERGE_LOG_ROOT}

Benchmark logs:
  ${BENCH_LOG_ROOT}

Summaries:
  ${SUMMARY_ROOT}
################################################################################
EOF2
}

main "$@"
