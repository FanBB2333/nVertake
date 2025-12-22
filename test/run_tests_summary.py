#!/usr/bin/env python3
"""
Run the project's test suite and print a structured summary.

This is a small convenience wrapper around `unittest` discovery that also:
  - groups skipped tests by reason
  - lists failing/errored tests
  - optionally writes a JSON summary (useful for CI logs)

GPU integration tests are opt-in via env flags; this script provides CLI switches
to set them before discovery.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unittest
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent


def _set_env_flag(name: str, enabled: bool) -> None:
    if enabled:
        os.environ[name] = "1"


def _get_env_flag(name: str) -> str:
    return os.environ.get(name, "")


class _SummaryResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successes: List[unittest.case.TestCase] = []
        self.durations_seconds: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    def startTest(self, test):  # noqa: N802 (unittest API)
        test_id = getattr(test, "id", lambda: str(test))()
        self._start_times[test_id] = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test):  # noqa: N802 (unittest API)
        test_id = getattr(test, "id", lambda: str(test))()
        start_time = self._start_times.pop(test_id, None)
        if start_time is not None:
            self.durations_seconds[test_id] = time.perf_counter() - start_time
        super().stopTest(test)

    def addSuccess(self, test):  # noqa: N802 (unittest API)
        self.successes.append(test)
        super().addSuccess(test)


@dataclass(frozen=True)
class Summary:
    duration_seconds: float
    tests_run: int
    passed: int
    failures: int
    errors: int
    skipped: int
    expected_failures: int
    unexpected_successes: int
    skip_reasons: Dict[str, int]
    failed_tests: List[str]
    errored_tests: List[str]
    env: Dict[str, str]
    cuda_available: Optional[bool]
    cuda_device_count: Optional[int]


def _cuda_info() -> Dict[str, Optional[int]]:
    try:
        import torch  # type: ignore
    except Exception:
        return {"cuda_available": None, "cuda_device_count": None}
    # Some environments emit warnings when CUDA is not properly configured; keep
    # the summary output clean and treat it as "not available".
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
    return {"cuda_available": available, "cuda_device_count": count}


def _build_summary(result: _SummaryResult, duration_seconds: float) -> Summary:
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    expected_failures = len(getattr(result, "expectedFailures", []))
    unexpected_successes = len(getattr(result, "unexpectedSuccesses", []))

    passed = result.testsRun - (
        failures + errors + skipped + expected_failures + unexpected_successes
    )
    if passed < 0:
        passed = 0

    skip_reasons = Counter(reason for _test, reason in result.skipped)
    failed_tests = [test.id() for test, _err in result.failures]
    errored_tests = [test.id() for test, _err in result.errors]

    env = {
        "NVERTAKE_TEST_DEVICE": _get_env_flag("NVERTAKE_TEST_DEVICE"),
        "NVERTAKE_ENABLE_GPU_TESTS": _get_env_flag("NVERTAKE_ENABLE_GPU_TESTS"),
        "NVERTAKE_ENABLE_PMON_TESTS": _get_env_flag("NVERTAKE_ENABLE_PMON_TESTS"),
        "NVERTAKE_STRICT_GPU_PRIORITY": _get_env_flag("NVERTAKE_STRICT_GPU_PRIORITY"),
        "NVERTAKE_STRICT_GPU_PMON": _get_env_flag("NVERTAKE_STRICT_GPU_PMON"),
        "NVERTAKE_PRINT_MARKDOWN_TABLE": _get_env_flag("NVERTAKE_PRINT_MARKDOWN_TABLE"),
    }

    cuda = _cuda_info()
    return Summary(
        duration_seconds=duration_seconds,
        tests_run=result.testsRun,
        passed=passed,
        failures=failures,
        errors=errors,
        skipped=skipped,
        expected_failures=expected_failures,
        unexpected_successes=unexpected_successes,
        skip_reasons=dict(skip_reasons),
        failed_tests=failed_tests,
        errored_tests=errored_tests,
        env=env,
        cuda_available=cuda["cuda_available"],
        cuda_device_count=cuda["cuda_device_count"],
    )


def _print_summary(summary: Summary, slowest: int, durations: Dict[str, float]) -> None:
    print("\n========== nVertake 测试汇总 ==========")
    print(f"耗时: {summary.duration_seconds:.3f}s")
    print(
        "总数: {tests_run} 通过: {passed} 失败: {failures} 错误: {errors} 跳过: {skipped} "
        "XFAIL: {expected_failures} XPASS: {unexpected_successes}".format(**asdict(summary))
    )

    cuda_text = "未知"
    if summary.cuda_available is True:
        cuda_text = f"可用 (device_count={summary.cuda_device_count})"
    elif summary.cuda_available is False:
        cuda_text = "不可用"
    print(f"CUDA: {cuda_text}")

    if any(value for value in summary.env.values()):
        enabled_env = {key: value for key, value in summary.env.items() if value}
        print(f"环境变量: {enabled_env}")

    if summary.skip_reasons:
        print("\n跳过原因汇总:")
        for reason, count in sorted(summary.skip_reasons.items(), key=lambda item: (-item[1], item[0])):
            print(f"- {count}x: {reason}")

    if summary.failed_tests:
        print("\n失败用例:")
        for test_id in summary.failed_tests:
            print(f"- {test_id}")

    if summary.errored_tests:
        print("\n错误用例:")
        for test_id in summary.errored_tests:
            print(f"- {test_id}")

    if slowest > 0 and durations:
        print(f"\n最慢用例 Top {slowest}:")
        for test_id, seconds in sorted(durations.items(), key=lambda item: item[1], reverse=True)[:slowest]:
            print(f"- {seconds:.3f}s  {test_id}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run nVertake tests and print a summary.")
    parser.add_argument(
        "--pattern",
        default="test*.py",
        help="unittest discovery pattern (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=2,
        help="unittest verbosity (default: %(default)s)",
    )
    parser.add_argument("--failfast", action="store_true", help="stop on first failure/error")
    parser.add_argument("--buffer", action="store_true", help="buffer stdout/stderr during tests")
    parser.add_argument(
        "--slowest",
        type=int,
        default=0,
        help="print N slowest tests from this run (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="write summary JSON to this path",
    )

    # Convenience switches for GPU integration tests (opt-in).
    parser.add_argument("--device", type=int, default=None, help="set NVERTAKE_TEST_DEVICE")
    parser.add_argument("--enable-gpu-tests", action="store_true", help="set NVERTAKE_ENABLE_GPU_TESTS=1")
    parser.add_argument("--enable-pmon-tests", action="store_true", help="set NVERTAKE_ENABLE_PMON_TESTS=1")
    parser.add_argument(
        "--strict-gpu-priority",
        action="store_true",
        help="set NVERTAKE_STRICT_GPU_PRIORITY=1",
    )
    parser.add_argument(
        "--strict-gpu-pmon",
        action="store_true",
        help="set NVERTAKE_STRICT_GPU_PMON=1",
    )
    parser.add_argument(
        "--print-markdown-table",
        action="store_true",
        help="set NVERTAKE_PRINT_MARKDOWN_TABLE=1",
    )

    args = parser.parse_args(argv)

    # Ensure repo root is importable when running from anywhere.
    sys.path.insert(0, str(REPO_ROOT))

    if args.device is not None:
        os.environ["NVERTAKE_TEST_DEVICE"] = str(args.device)
    _set_env_flag("NVERTAKE_ENABLE_GPU_TESTS", args.enable_gpu_tests)
    _set_env_flag("NVERTAKE_ENABLE_PMON_TESTS", args.enable_pmon_tests)
    _set_env_flag("NVERTAKE_STRICT_GPU_PRIORITY", args.strict_gpu_priority)
    _set_env_flag("NVERTAKE_STRICT_GPU_PMON", args.strict_gpu_pmon)
    _set_env_flag("NVERTAKE_PRINT_MARKDOWN_TABLE", args.print_markdown_table)

    suite = unittest.defaultTestLoader.discover(str(TEST_DIR), pattern=args.pattern)
    runner = unittest.TextTestRunner(
        verbosity=args.verbosity,
        failfast=args.failfast,
        buffer=args.buffer,
        resultclass=_SummaryResult,
    )

    start_time = time.perf_counter()
    result: _SummaryResult = runner.run(suite)  # type: ignore[assignment]
    duration_seconds = time.perf_counter() - start_time

    summary = _build_summary(result, duration_seconds=duration_seconds)
    _print_summary(summary, slowest=args.slowest, durations=result.durations_seconds)

    if args.json_path:
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n已写入 JSON 汇总: {json_path}")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
