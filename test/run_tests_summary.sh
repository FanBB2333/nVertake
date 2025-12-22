#!/usr/bin/env bash
set -euo pipefail

# Shell wrapper for running the test suite with the Python summarizer.
#
# Usage:
#   bash test/run_tests_summary.sh
#   bash test/run_tests_summary.sh --json test/test_summary.json
#   bash test/run_tests_summary.sh --enable-gpu-tests --device 0
#
# You can override the Python interpreter:
#   PYTHON_BIN=python3 bash test/run_tests_summary.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${REPO_ROOT}/test/run_tests_summary.py" "$@"

