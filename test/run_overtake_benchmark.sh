#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-0}"
MATRIX_SIZE="${MATRIX_SIZE:-4096}"
DTYPE="${DTYPE:-float16}"
BATCH_ITERS="${BATCH_ITERS:-10}"
REPORT_INTERVAL="${REPORT_INTERVAL:-1.0}"
WARMUP_SECONDS="${WARMUP_SECONDS:-2}"
PRE_SECONDS="${PRE_SECONDS:-8}"
INVADE_SECONDS="${INVADE_SECONDS:-10}"
POST_SECONDS="${POST_SECONDS:-2}"
NICE_VALUE="${NICE_VALUE:-0}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/test/overtake_output}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_DIR}/${RUN_ID}"

resident_py="${REPO_ROOT}/test/overtake_resident.py"
invader_py="${REPO_ROOT}/test/overtake_invader.py"
analyze_py="${REPO_ROOT}/test/overtake_analyze.py"
table_py="${REPO_ROOT}/test/overtake_make_table.py"

mkdir -p "${RUN_DIR}"

PYTHONPATH_ENV="${REPO_ROOT}"
if [[ -n "${PYTHONPATH-}" ]]; then
  PYTHONPATH_ENV="${PYTHONPATH_ENV}:${PYTHONPATH}"
fi

cuda_ok() {
  PYTHONPATH="${PYTHONPATH_ENV}" "${PYTHON_BIN}" - <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(1)
sys.exit(0 if torch.cuda.is_available() else 2)
PY
}

if ! cuda_ok; then
  echo "CUDA is not available in this environment; skipping overtake benchmark." >&2
  echo "Run on a machine with an NVIDIA GPU + CUDA-enabled PyTorch." >&2
  exit 0
fi

epoch_now() {
  "${PYTHON_BIN}" - <<'PY'
import time
print(time.time())
PY
}

run_case() {
  local label="$1"
  local use_nvertake="$2"

  local resident_log="${RUN_DIR}/resident_${label}.jsonl"
  local summary_json="${RUN_DIR}/summary_${label}.json"

  echo "[case=${label}] starting resident..."
  PYTHONPATH="${PYTHONPATH_ENV}" "${PYTHON_BIN}" -u "${resident_py}" \
    --device "${DEVICE}" \
    --matrix-size "${MATRIX_SIZE}" \
    --dtype "${DTYPE}" \
    --batch-iters "${BATCH_ITERS}" \
    --report-interval "${REPORT_INTERVAL}" \
    --warmup-seconds "${WARMUP_SECONDS}" \
    --output "${resident_log}" &
  local resident_pid="$!"

  cleanup() {
    if kill -0 "${resident_pid}" 2>/dev/null; then
      kill -TERM "${resident_pid}" 2>/dev/null || true
      wait "${resident_pid}" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT

  sleep "${WARMUP_SECONDS}"

  echo "[case=${label}] baseline window (${PRE_SECONDS}s)..."
  local pre_start
  pre_start="$(epoch_now)"
  sleep "${PRE_SECONDS}"
  local invader_start
  invader_start="$(epoch_now)"

  echo "[case=${label}] starting invader (${INVADE_SECONDS}s, use_nvertake=${use_nvertake})..."
  if [[ "${use_nvertake}" == "1" ]]; then
    PYTHONPATH="${PYTHONPATH_ENV}" "${PYTHON_BIN}" -u "${invader_py}" \
      --device "${DEVICE}" \
      --matrix-size "${MATRIX_SIZE}" \
      --dtype "${DTYPE}" \
      --batch-iters "${BATCH_ITERS}" \
      --duration-seconds "${INVADE_SECONDS}" \
      --use-nvertake \
      --nice "${NICE_VALUE}" &
  else
    PYTHONPATH="${PYTHONPATH_ENV}" "${PYTHON_BIN}" -u "${invader_py}" \
      --device "${DEVICE}" \
      --matrix-size "${MATRIX_SIZE}" \
      --dtype "${DTYPE}" \
      --batch-iters "${BATCH_ITERS}" \
      --duration-seconds "${INVADE_SECONDS}" &
  fi
  local invader_pid="$!"
  wait "${invader_pid}"

  local invader_end
  invader_end="$(epoch_now)"
  echo "[case=${label}] invader done."

  echo "[case=${label}] post window (${POST_SECONDS}s)..."
  sleep "${POST_SECONDS}"

  echo "[case=${label}] stopping resident..."
  kill -TERM "${resident_pid}" 2>/dev/null || true
  wait "${resident_pid}" 2>/dev/null || true
  trap - EXIT

  PYTHONPATH="${PYTHONPATH_ENV}" "${PYTHON_BIN}" -u "${analyze_py}" \
    --label "${label}" \
    --native-log "${resident_log}" \
    --pre-start "${pre_start}" \
    --pre-end "${invader_start}" \
    --during-start "${invader_start}" \
    --during-end "${invader_end}" \
    --json-out "${summary_json}" >/dev/null

  echo "[case=${label}] summary written: ${summary_json}"
}

run_case "invader_no_nvertake" "0"
run_case "invader_with_nvertake" "1"

echo
echo "Run dir: ${RUN_DIR}"
echo
PYTHONPATH="${PYTHONPATH_ENV}" "${PYTHON_BIN}" -u "${table_py}" \
  "${RUN_DIR}/summary_invader_no_nvertake.json" \
  "${RUN_DIR}/summary_invader_with_nvertake.json"
