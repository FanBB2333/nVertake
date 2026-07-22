#!/usr/bin/env bash
set -euo pipefail

# Shell wrapper for running the test suite with the Python summarizer.
# GPU-oriented modes target real NVIDIA GPUs with CUDA-enabled PyTorch.
#
# Usage:
#   # Run everything (GPU tests are opt-in; will be skipped by default)
#   bash test/run_tests_summary.sh
#
#   # Convenience modes:
#   bash test/run_tests_summary.sh unit
#   bash test/run_tests_summary.sh gpu --device 0
#   bash test/run_tests_summary.sh pmon --device 0
#   bash test/run_tests_summary.sh strict --device 0
#   bash test/run_tests_summary.sh strict-pmon --device 0
#   bash test/run_tests_summary.sh report --device 0
#   bash test/run_tests_summary.sh overtake
#   bash test/run_tests_summary.sh mps-share --device 0
#   bash test/run_tests_summary.sh green-share --device 0
#
#   # Or forward any args directly to the Python runner:
#   bash test/run_tests_summary.sh --json test/test_summary.json
#   bash test/run_tests_summary.sh --enable-gpu-tests --device 0 --slowest 10
#
# You can override the Python interpreter:
#   PYTHON_BIN=python3 bash test/run_tests_summary.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage:
  bash test/run_tests_summary.sh [mode] [args...]

Modes:
  all          Run all tests (default; GPU tests skipped unless enabled)
  unit         Run CPU-only unit tests (test_nvertake*.py)
  gpu          Run real-GPU tests with --enable-gpu-tests
  pmon         Run real-GPU tests with --enable-gpu-tests --enable-pmon-tests
  strict       Run stricter real-GPU tests with --enable-gpu-tests --strict-gpu-priority
  strict-pmon  Run stricter real-GPU tests with --enable-gpu-tests --enable-pmon-tests --strict-gpu-pmon
  report       Run tests with --enable-gpu-tests --print-markdown-table
  overtake     Run the resident-vs-invader benchmark on a real CUDA GPU
  mps-share    Run the NVIDIA MPS 50/50 vs 25/75 share experiment
  green-share  Run the CUDA Green Context 50/50 vs 25/75 SM experiment

Any extra args are forwarded to test/run_tests_summary.py.
Examples:
  bash test/run_tests_summary.sh unit
  bash test/run_tests_summary.sh gpu --device 1 --slowest 10
  bash test/run_tests_summary.sh --json test/test_summary.json
EOF
}

MODE="${1:-}"

# Backwards-compatible: if the first arg looks like a flag, forward everything.
if [[ -z "${MODE}" || "${MODE}" == -* ]]; then
  exec "${PYTHON_BIN}" "${REPO_ROOT}/test/run_tests_summary.py" "$@"
fi

shift
case "${MODE}" in
  all)
    ;;
  unit)
    set -- --pattern 'test_nvertake*.py' "$@"
    ;;
  gpu)
    set -- --enable-gpu-tests "$@"
    ;;
  pmon)
    set -- --enable-gpu-tests --enable-pmon-tests "$@"
    ;;
  strict)
    set -- --enable-gpu-tests --strict-gpu-priority "$@"
    ;;
  strict-pmon)
    set -- --enable-gpu-tests --enable-pmon-tests --strict-gpu-pmon "$@"
    ;;
  report)
    set -- --enable-gpu-tests --print-markdown-table "$@"
    ;;
  overtake)
    exec bash "${REPO_ROOT}/test/run_overtake_benchmark.sh" "$@"
    ;;
  mps-share)
    exec "${PYTHON_BIN}" "${REPO_ROOT}/verification/run_mps_share_experiment.py" "$@"
    ;;
  green-share)
    exec "${PYTHON_BIN}" "${REPO_ROOT}/verification/run_green_context_share_experiment.py" "$@"
    ;;
  help|-h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

exec "${PYTHON_BIN}" "${REPO_ROOT}/test/run_tests_summary.py" "$@"
