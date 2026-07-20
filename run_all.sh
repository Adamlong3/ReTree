#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Full ReTree experiment pipeline for Qwen3-4B / Qwen3-8B.
#
# For each model and tree budget, this script:
#   1) initializes task-domain recovery priors with the path-guided tree;
#   2) combines those domain priors into one fixed evaluation prior;
#   3) benchmarks the base DDTree verifier and the full ReTree configuration.
#
# Reported ReTree runs reload a fixed offline prior for each dataset and then
# evolve an online delta causally. Set BENCH_ONLINE_UPDATE=0 for the ablation.
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
RECOVERY_PRIOR_TAG="${RECOVERY_PRIOR_TAG:-train_disjoint_v1}"
CALIB_ROOT="${CALIB_ROOT:-${RECOVERY_ROOT}/calibration/${RECOVERY_PRIOR_TAG}}"
ALLTASK_ROOT="${ALLTASK_ROOT:-${RECOVERY_ROOT}/alltask/${RECOVERY_PRIOR_TAG}}"
RUN_ROOT="${RUN_ROOT:-${RECOVERY_ROOT}/run}"
CALIB_LOG_ROOT="${CALIB_LOG_ROOT:-${RUN_ROOT}/calib_logs/${RECOVERY_PRIOR_TAG}}"
COMBINE_LOG_ROOT="${COMBINE_LOG_ROOT:-${RUN_ROOT}/combine_logs/${RECOVERY_PRIOR_TAG}}"
BENCH_LOG_ROOT="${BENCH_LOG_ROOT:-${RUN_ROOT}/bench_logs/${RECOVERY_PRIOR_TAG}}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${RUN_ROOT}/summaries/${RECOVERY_PRIOR_TAG}}"

mkdir -p "${CALIB_ROOT}" "${ALLTASK_ROOT}" "${RUN_ROOT}" \
         "${CALIB_LOG_ROOT}" "${COMBINE_LOG_ROOT}" "${BENCH_LOG_ROOT}" \
         "${SUMMARY_ROOT}" logs

# -----------------------------
# Model paths
# -----------------------------
MODEL_PATH_4B="${MODEL_PATH_4B:-Qwen/Qwen3-4B}"
DRAFT_PATH_4B="${DRAFT_PATH_4B:-z-lab/Qwen3-4B-DFlash-b16}"

MODEL_PATH_8B="${MODEL_PATH_8B:-Qwen/Qwen3-8B}"
DRAFT_PATH_8B="${DRAFT_PATH_8B:-z-lab/Qwen3-8B-DFlash-b16}"

MODEL_LIST="${MODEL_LIST:-4B 8B}"
read -r -a MODELS <<< "${MODEL_LIST}"

# -----------------------------
# Tree budgets / tasks
# -----------------------------
TB_LIST="${TB_LIST:-16 32 64 128 256}"
read -r -a TREE_BUDGETS <<< "${TB_LIST}"

CALIB_TASKS=(
  "gsm8k_train:2000"
  "math_train:7500"
  "mbpp_train:374"
  "cnn_dailymail_train:2000"
)

BENCH_TASKS=(
  "gsm8k:128"
  "math500:128"
  "aime25:30"
  "humaneval:164"
  "mbpp:128"
  "livecodebench:128"
  "mt-bench:80"
)

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
CALIB_DEDUPE_DATASETS="${CALIB_DEDUPE_DATASETS:-gsm8k,math500,aime25,humaneval,mbpp,livecodebench,mt-bench}"

# -----------------------------
# GPU / control config
# -----------------------------
CALIB_GPUS=( ${CALIB_GPUS:-0 1 2 3} )
BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-31000}"

FORCE_RECALIB="${FORCE_RECALIB:-0}"
FORCE_COMBINE="${FORCE_COMBINE:-0}"
FORCE_BENCH="${FORCE_BENCH:-0}"
BENCH_ONLINE_UPDATE="${BENCH_ONLINE_UPDATE:-1}"

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

memory_dir_for_model_tb() {
  local model_tag="$1"
  local tb="$2"
  echo "${CALIB_ROOT}/${model_tag}/tb${tb}_rankcap${DDTREE_NGRAM_RANK_CAP}"
}

task_memory_file() {
  local model_tag="$1"
  local tb="$2"
  local dataset_name="$3"
  local max_samples="$4"
  local safe_dataset
  safe_dataset="$(safe_name "${dataset_name}")"
  echo "$(memory_dir_for_model_tb "${model_tag}" "${tb}")/recovery_${safe_dataset}_${max_samples}_${model_tag}_tb${tb}.json"
}

alltask_memory_file() {
  local model_tag="$1"
  local tb="$2"
  echo "${ALLTASK_ROOT}/${model_tag}/tb${tb}/recovery_alltask_${model_tag}_tb${tb}.json"
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

combine_log_file() {
  local model_tag="$1"
  local tb="$2"
  echo "${COMBINE_LOG_ROOT}/combine_${model_tag}_tb${tb}.log"
}

bench_log_file() {
  local model_tag="$1"
  local tb="$2"
  local temp="$3"
  local dataset_name="$4"
  local max_samples="$5"
  local run_kind="$6"
  local safe_dataset temp_safe
  safe_dataset="$(safe_name "${dataset_name}")"
  temp_safe="$(safe_float "${temp}")"
  echo "${BENCH_LOG_ROOT}/${model_tag}/tb${tb}/T${temp_safe}/bench_${model_tag}_tb${tb}_T${temp_safe}_${safe_dataset}_${max_samples}_${run_kind}.log"
}

check_project_files() {
  echo "[CHECK] Python syntax..."
  python -m py_compile benchmark.py ddtree.py retree.py retree_calibrate.py \
    dflash.py model/dflash.py model/recovery.py model/utils.py
}

print_config() {
  cat <<EOF2

################################################################################
# ReTree Qwen3 4B/8B sweep
################################################################################
PROJECT_ROOT                 = ${PROJECT_ROOT}
LOCAL_DATASETS_ROOT          = ${LOCAL_DATASETS_ROOT}
RECOVERY_ROOT                = ${RECOVERY_ROOT}
RECOVERY_PRIOR_TAG           = ${RECOVERY_PRIOR_TAG}
CALIB_ROOT                   = ${CALIB_ROOT}
ALLTASK_ROOT                 = ${ALLTASK_ROOT}
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
RECOVERY_FREQ_THRESHOLD      = ${RECOVERY_FREQ_THRESHOLD}
RECOVERY_THRESHOLD           = ${RECOVERY_THRESHOLD}
RECOVERY_RECORD_TOP_K        = ${RECOVERY_RECORD_TOP_K}
RECOVERY_RESCUE_TOP_K        = ${RECOVERY_RESCUE_TOP_K}
CALIB_DEDUPE_DATASETS        = ${CALIB_DEDUPE_DATASETS}
CALIB_GPUS                   = ${CALIB_GPUS[*]}
BENCH_CUDA_VISIBLE_DEVICES   = ${BENCH_CUDA_VISIBLE_DEVICES}
NPROC_PER_NODE               = ${NPROC_PER_NODE}
MASTER_PORT_BASE             = ${MASTER_PORT_BASE}
FORCE_RECALIB                = ${FORCE_RECALIB}
FORCE_COMBINE                = ${FORCE_COMBINE}
FORCE_BENCH                  = ${FORCE_BENCH}
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

  if (( ${#CALIB_GPUS[@]} < ${#CALIB_TASKS[@]} )); then
    echo "[ERROR] CALIB_GPUS has ${#CALIB_GPUS[@]} GPUs, but CALIB_TASKS has ${#CALIB_TASKS[@]} tasks." >&2
    exit 1
  fi

  local paths model_path draft_path memory_dir
  paths="$(model_paths "${model_tag}")"
  model_path="${paths%%|*}"
  draft_path="${paths##*|}"
  memory_dir="$(memory_dir_for_model_tb "${model_tag}" "${tb}")"
  mkdir -p "${memory_dir}" "${CALIB_LOG_ROOT}/${model_tag}/tb${tb}"

  echo ""
  echo "################################################################################"
  echo "# Initialize ReTree recovery prior: model=${model_tag}, tb=${tb}"
  echo "################################################################################"

  local pids=()
  local names=()
  local logs=()
  local outputs=()

  local idx=0
  for task in "${CALIB_TASKS[@]}"; do
    IFS=':' read -r dataset_name max_samples <<< "${task}"
    local gpu_id output_file log_file done_file
    gpu_id="${CALIB_GPUS[$idx]}"
    output_file="$(task_memory_file "${model_tag}" "${tb}" "${dataset_name}" "${max_samples}")"
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
          --dedupe-against "${CALIB_DEDUPE_DATASETS}" \
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
    local pid name log_file output_file
    pid="${pids[$i]}"
    name="${names[$i]}"
    log_file="${logs[$i]}"
    output_file="${outputs[$i]}"

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

  local missing=0
  for task in "${CALIB_TASKS[@]}"; do
    IFS=':' read -r dataset_name max_samples <<< "${task}"
    local output_file
    output_file="$(task_memory_file "${model_tag}" "${tb}" "${dataset_name}" "${max_samples}")"
    if ! validate_json "${output_file}" >/dev/null 2>&1; then
      echo "[MISSING/INVALID] model=${model_tag}, tb=${tb}, task=${dataset_name}:${max_samples}, file=${output_file}" >&2
      missing=1
    else
      echo "[OK recovery memory] ${output_file}"
    fi
  done

  if [[ "${missing}" != "0" ]]; then
    echo "[ERROR] Missing or invalid recovery memory files for model=${model_tag}, tb=${tb}" >&2
    exit 1
  fi
}

# -----------------------------
# Combine task-domain priors
# -----------------------------
combine_memories_for_model_tb() {
  local model_tag="$1"
  local tb="$2"

  local output_file log_file
  output_file="$(alltask_memory_file "${model_tag}" "${tb}")"
  log_file="$(combine_log_file "${model_tag}" "${tb}")"
  mkdir -p "$(dirname "${output_file}")" "$(dirname "${log_file}")"

  echo ""
  echo "################################################################################"
  echo "# Combine all-task recovery memory: model=${model_tag}, tb=${tb}"
  echo "################################################################################"

  if [[ "${FORCE_COMBINE}" == "1" ]]; then
    rm -f "${output_file}" "${log_file}"
  fi

  local inputs=()
  for task in "${CALIB_TASKS[@]}"; do
    IFS=':' read -r dataset_name max_samples <<< "${task}"
    inputs+=("$(task_memory_file "${model_tag}" "${tb}" "${dataset_name}" "${max_samples}")")
  done

  if [[ -s "${output_file}" ]] && validate_json "${output_file}" >/dev/null 2>&1; then
    echo "[SKIP COMBINE] valid all-task recovery memory exists: ${output_file}"
    return 0
  fi

  python - "${output_file}" "${inputs[@]}" <<'PY' 2>&1 | tee "${log_file}"
import json
import sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
inputs = [Path(x) for x in sys.argv[2:]]
combined = defaultdict(int)

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
        combined[str(k)] += int(v)

out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_name(out.name + ".tmp")
with open(tmp, "w") as f:
    json.dump(dict(combined), f)
tmp.replace(out)

print(f"[DONE] saved all-task recovery memory: {out}")
print(f"unique pairs: {len(combined)}")
print(f"total counts: {sum(combined.values())}")
print("top10:")
for k, v in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(k, v)
PY

  validate_json "${output_file}" >/dev/null
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

  local log_file done_file port memory_for_run
  log_file="$(bench_log_file "${model_tag}" "${tb}" "${temp}" "${dataset_name}" "${max_samples}" "${run_kind}")"
  done_file="${log_file}.done"
  port=$((MASTER_PORT_BASE + GLOBAL_RUN_IDX))
  GLOBAL_RUN_IDX=$((GLOBAL_RUN_IDX + 1))
  memory_for_run="$(alltask_memory_file "${model_tag}" "${tb}")"

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
    base)
      strategy="heap"
      methods="dflash,ddtree"
      ;;

    retree)
      strategy="rank_gated_ngram"
      methods="dflash,ddtree,retree"
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
      else
        recovery_args+=(--no-recovery-online-update)
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
  if [[ "${run_kind}" == "retree" ]]; then
    echo "recovery_memory=${memory_for_run}"
  fi
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
  local memory_for_run
  memory_for_run="$(alltask_memory_file "${model_tag}" "${tb}")"
  validate_json "${memory_for_run}" >/dev/null

  echo ""
  echo "################################################################################"
  echo "# Benchmark all tasks/temperatures: model=${model_tag}, tb=${tb}"
  echo "# ReTree recovery memory=${memory_for_run}"
  echo "################################################################################"

  for temp in "${TEMPS[@]}"; do
    for task in "${BENCH_TASKS[@]}"; do
      IFS=':' read -r dataset_name max_samples <<< "${task}"
      run_benchmark_one "${model_tag}" "${tb}" "${temp}" "${dataset_name}" "${max_samples}" "base"
      run_benchmark_one "${model_tag}" "${tb}" "${temp}" "${dataset_name}" "${max_samples}" "retree"
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
    if name.endswith("_base.log") or "_base." in name or "_base" in name:
        return "base"
    if name.endswith("_retree.log") or "_retree." in name or "_retree" in name:
        return "retree"
    return ""

def normalized_variant(run_kind: str, method: str) -> str:
    if method == "DFlash (linear SD)":
        return "dflash"
    if run_kind == "base" and method == "DDTree":
        return "ddtree_base"
    if run_kind == "retree" and method == "DDTree":
        return "path_tree_only"
    if run_kind == "retree" and method == "ReTree":
        return "retree"
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

  for model_tag in "${MODELS[@]}"; do
    for tb in "${TREE_BUDGETS[@]}"; do
      run_calibration_for_model_tb "${model_tag}" "${tb}"
      combine_memories_for_model_tb "${model_tag}" "${tb}"
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
Task-domain recovery priors:
  ${CALIB_ROOT}

All-task recovery memories:
  ${ALLTASK_ROOT}

Calibration logs:
  ${CALIB_LOG_ROOT}

Combine logs:
  ${COMBINE_LOG_ROOT}

Benchmark logs:
  ${BENCH_LOG_ROOT}

Summaries:
  ${SUMMARY_ROOT}
################################################################################
EOF2
}

main "$@"
