"""
GPU integration tests for nVertake.

These tests are intentionally opt-in because they require a CUDA-capable NVIDIA GPU.

Enable with:
  NVERTAKE_ENABLE_GPU_TESTS=1 pytest -q

Optional (requires `nvidia-smi pmon`):
  NVERTAKE_ENABLE_PMON_TESTS=1 pytest -q

Stricter "prove priority" assertions:
  NVERTAKE_STRICT_GPU_PRIORITY=1 pytest -q
  NVERTAKE_STRICT_GPU_PMON=1 pytest -q
"""

from __future__ import annotations

import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import unittest
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Dict, Iterable, List, Optional, Tuple

import torch

# Add parent directory to path to import nvertake when running tests directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nvertake.scheduler import PriorityScheduler


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _get_test_device() -> int:
    raw = os.environ.get("NVERTAKE_TEST_DEVICE", "0").strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def _cuda_available(device: int) -> bool:
    if not torch.cuda.is_available():
        return False
    return 0 <= device < torch.cuda.device_count()


def _gpu_tests_enabled(device: int) -> bool:
    return _env_truthy("NVERTAKE_ENABLE_GPU_TESTS") and _cuda_available(device)


def _pmon_tests_enabled(device: int) -> bool:
    if not _gpu_tests_enabled(device):
        return False
    if not _env_truthy("NVERTAKE_ENABLE_PMON_TESTS"):
        return False
    return shutil.which("nvidia-smi") is not None


def _strict_priority_asserts_enabled() -> bool:
    return _env_truthy("NVERTAKE_STRICT_GPU_PRIORITY")


def _strict_pmon_asserts_enabled() -> bool:
    return _env_truthy("NVERTAKE_STRICT_GPU_PMON")


def _report_table_enabled() -> bool:
    return _env_truthy("NVERTAKE_PRINT_MARKDOWN_TABLE")


def _pct(delta: float) -> float:
    return delta * 100.0


def _fmt_seconds(value: float) -> str:
    return f"{value:.4f}"


def _fmt_ratio(value: float) -> str:
    return f"{value:.2f}x"


def _print_markdown_table(rows: List[Tuple[str, str]]) -> None:
    # Keep formatting stable for copy-paste into README/issues.
    print("\n| Metric | Value |")
    print("|---|---:|")
    for metric, value in rows:
        print(f"| {metric} | {value} |")


@dataclass(frozen=True)
class _MatmulWorkload:
    matrix_size: int
    dtype: torch.dtype
    background_iters: int
    probe_iters: int


def _estimate_mm_bytes(matrix_size: int, dtype: torch.dtype) -> int:
    bytes_per_element = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float64: 8,
    }.get(dtype, 4)
    # left + right + out
    return 3 * matrix_size * matrix_size * bytes_per_element


def _pick_matmul_size(device: int, dtype: torch.dtype) -> int:
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    # Keep plenty of headroom for driver/workspace/other processes.
    budget_bytes = int(free_bytes * 0.15)
    for candidate in (8192, 6144, 4096, 3072, 2048, 1536, 1024):
        if _estimate_mm_bytes(candidate, dtype) <= budget_bytes:
            return candidate
    return 1024


def _calibrate_iters(
    device: int,
    matrix_size: int,
    dtype: torch.dtype,
    target_background_seconds: float = 0.8,
    target_probe_seconds: float = 0.2,
) -> _MatmulWorkload:
    torch.manual_seed(0)
    with torch.cuda.device(device):
        left = torch.randn((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)
        right = torch.randn((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)
        out = torch.empty((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)

    # Warm up (init cuBLAS handles, autotuning, etc.)
    for _ in range(3):
        torch.mm(left, right, out=out)
    torch.cuda.synchronize(device)

    calibration_iters = 10
    start_time = time.perf_counter()
    for _ in range(calibration_iters):
        torch.mm(left, right, out=out)
    torch.cuda.synchronize(device)
    elapsed = max(1e-6, time.perf_counter() - start_time)
    seconds_per_iter = elapsed / calibration_iters

    probe_iters = int(target_probe_seconds / seconds_per_iter)
    background_iters = int(target_background_seconds / seconds_per_iter)

    # Clamp to keep tests bounded.
    probe_iters = max(5, min(probe_iters, 400))
    background_iters = max(probe_iters * 4, min(background_iters, 4000))

    return _MatmulWorkload(
        matrix_size=matrix_size,
        dtype=dtype,
        background_iters=background_iters,
        probe_iters=probe_iters,
    )


def _allocate_mm_tensors(
    device: int,
    matrix_size: int,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    with torch.cuda.device(device):
        left = torch.randn((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)
        right = torch.randn((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)
        out = torch.empty((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)
    return left, right, out


def _measure_probe_latency_seconds(
    device: int,
    workload: _MatmulWorkload,
    background_stream: torch.cuda.Stream,
    probe_stream: torch.cuda.Stream,
    background_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    probe_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> float:
    bg_left, bg_right, bg_out = background_tensors
    probe_left, probe_right, probe_out = probe_tensors

    torch.cuda.synchronize(device)

    with torch.cuda.stream(background_stream):
        for _ in range(workload.background_iters):
            torch.mm(bg_left, bg_right, out=bg_out)

    start_time = time.perf_counter()
    with torch.cuda.stream(probe_stream):
        for _ in range(workload.probe_iters):
            torch.mm(probe_left, probe_right, out=probe_out)

    probe_stream.synchronize()
    elapsed = time.perf_counter() - start_time

    # Clean up pending background work (do not include in measurement).
    torch.cuda.synchronize(device)
    return elapsed


def _parse_nvidia_smi_pmon_output(stdout: str) -> Dict[int, Dict[str, int]]:
    """
    Parse `nvidia-smi pmon` output into: pid -> {"sm": int, "mem": int}.

    Example format:
        # gpu   pid   type    sm   mem   enc   dec   command
        # Idx     #   C/G     %     %     %     %   name
            0  12345     C     7     2     0     0   python
    """
    pid_to_metrics: Dict[int, Dict[str, int]] = {}
    line_regex = re.compile(
        r"^\s*(?P<gpu>\d+)\s+(?P<pid>\d+)\s+(?P<type>\S+)\s+(?P<sm>\S+)\s+(?P<mem>\S+)\s+"
    )

    for line in stdout.splitlines():
        match = line_regex.match(line)
        if not match:
            continue

        pid = int(match.group("pid"))
        sm_raw = match.group("sm")
        mem_raw = match.group("mem")

        def parse_percent(value: str) -> int:
            if value in {"-", "N/A"}:
                return 0
            try:
                return int(value)
            except ValueError:
                return 0

        pid_to_metrics[pid] = {
            "sm": parse_percent(sm_raw),
            "mem": parse_percent(mem_raw),
        }

    return pid_to_metrics


def _nvidia_smi_pmon_sample(device: int) -> Dict[int, Dict[str, int]]:
    cmd = ["nvidia-smi", "pmon", "-c", "1", "-s", "um", "-i", str(device)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return _parse_nvidia_smi_pmon_output(result.stdout)
    except Exception:
        # Some driver builds don't accept `-i` for `pmon`; retry without it.
        result = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],
            capture_output=True,
            text=True,
            check=True,
        )
        return _parse_nvidia_smi_pmon_output(result.stdout)


def _collect_pmon_sm_samples(
    device: int,
    pids: Iterable[int],
    sample_count: int = 5,
    warmup_seconds: float = 1.5,
) -> Dict[int, List[int]]:
    pid_set = set(pids)
    samples: Dict[int, List[int]] = {pid: [] for pid in pid_set}

    time.sleep(max(0.0, warmup_seconds))
    for _ in range(sample_count):
        metrics = _nvidia_smi_pmon_sample(device)
        for pid in pid_set:
            if pid in metrics:
                samples[pid].append(metrics[pid]["sm"])

    return samples


def _gpu_burn_worker(
    role: str,
    device: int,
    stream_priority: int,
    start_event,
    run_seconds: float,
    matrix_size: int,
    result_queue,
    use_nvertake_high_priority: bool,
) -> None:
    import os as _os

    import torch as _torch

    _torch.manual_seed(0 if role == "low" else 1)
    _torch.cuda.set_device(device)

    if use_nvertake_high_priority:
        scheduler = PriorityScheduler(device=device, nice_value=0)
        stream = scheduler.get_high_priority_stream()
        if stream is None:
            # Should not happen when CUDA is available; fallback to default stream.
            stream = _torch.cuda.default_stream(device)
    else:
        stream = _torch.cuda.Stream(device=device, priority=stream_priority)

    left = _torch.randn((matrix_size, matrix_size), device=f"cuda:{device}", dtype=_torch.float16)
    right = _torch.randn((matrix_size, matrix_size), device=f"cuda:{device}", dtype=_torch.float16)
    out = _torch.empty((matrix_size, matrix_size), device=f"cuda:{device}", dtype=_torch.float16)

    # Warm up
    for _ in range(3):
        _torch.mm(left, right, out=out)
    _torch.cuda.synchronize(device)

    start_event.wait()
    end_time = time.perf_counter() + run_seconds

    iterations = 0
    batch_iters = 5
    while time.perf_counter() < end_time:
        with _torch.cuda.stream(stream):
            for _ in range(batch_iters):
                _torch.mm(left, right, out=out)
        stream.synchronize()
        iterations += batch_iters

    result_queue.put((role, _os.getpid(), iterations))


class TestStreamPriorityWithinProcess(unittest.TestCase):
    device = _get_test_device()

    @unittest.skipUnless(
        _gpu_tests_enabled(device),
        "Requires CUDA and NVERTAKE_ENABLE_GPU_TESTS=1",
    )
    def test_stream_priority_sanity_under_contention(self):
        device = self.device
        torch.cuda.set_device(device)

        dtype = torch.float16
        matrix_size = _pick_matmul_size(device, dtype)
        workload = _calibrate_iters(device=device, matrix_size=matrix_size, dtype=dtype)

        background_stream = torch.cuda.Stream(device=device, priority=0)
        probe_stream_default = torch.cuda.Stream(device=device, priority=0)
        scheduler = PriorityScheduler(device=device, nice_value=0)
        probe_stream_high = scheduler.get_high_priority_stream()

        self.assertIsNotNone(probe_stream_high, "Expected a CUDA stream when CUDA is available")

        background_tensors = _allocate_mm_tensors(
            device=device,
            matrix_size=workload.matrix_size,
            dtype=workload.dtype,
            seed=123,
        )
        probe_tensors = _allocate_mm_tensors(
            device=device,
            matrix_size=workload.matrix_size,
            dtype=workload.dtype,
            seed=456,
        )

        # Baseline (no background contention).
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        with torch.cuda.stream(probe_stream_default):
            for _ in range(workload.probe_iters):
                torch.mm(probe_tensors[0], probe_tensors[1], out=probe_tensors[2])
        probe_stream_default.synchronize()
        baseline_seconds = time.perf_counter() - start_time

        contended_default_seconds = _measure_probe_latency_seconds(
            device=device,
            workload=workload,
            background_stream=background_stream,
            probe_stream=probe_stream_default,
            background_tensors=background_tensors,
            probe_tensors=probe_tensors,
        )

        contended_high_seconds = _measure_probe_latency_seconds(
            device=device,
            workload=workload,
            background_stream=background_stream,
            probe_stream=probe_stream_high,
            background_tensors=background_tensors,
            probe_tensors=probe_tensors,
        )

        # Sanity: our contention actually changes latency.
        self.assertGreater(
            contended_default_seconds,
            baseline_seconds * 1.05,
            f"Expected contention to increase latency; baseline={baseline_seconds:.4f}s, "
            f"contended={contended_default_seconds:.4f}s",
        )

        # Print metrics to help users validate on their hardware.
        print(
            "\n[nVertake GPU sanity] "
            f"device={device} size={workload.matrix_size} "
            f"bg_iters={workload.background_iters} probe_iters={workload.probe_iters} "
            f"baseline={baseline_seconds:.4f}s contended_default={contended_default_seconds:.4f}s "
            f"contended_high={contended_high_seconds:.4f}s"
        )

    @unittest.skipUnless(
        _gpu_tests_enabled(device) and _strict_priority_asserts_enabled(),
        "Requires CUDA and NVERTAKE_ENABLE_GPU_TESTS=1 and NVERTAKE_STRICT_GPU_PRIORITY=1",
    )
    def test_high_priority_stream_reduces_probe_latency(self):
        device = self.device
        torch.cuda.set_device(device)

        dtype = torch.float16
        matrix_size = _pick_matmul_size(device, dtype)
        workload = _calibrate_iters(device=device, matrix_size=matrix_size, dtype=dtype)

        scheduler = PriorityScheduler(device=device, nice_value=0)
        high_stream = scheduler.get_high_priority_stream()
        self.assertIsNotNone(high_stream)

        background_stream = torch.cuda.Stream(device=device, priority=0)
        low_probe_stream = torch.cuda.Stream(device=device, priority=0)

        background_tensors = _allocate_mm_tensors(
            device=device,
            matrix_size=workload.matrix_size,
            dtype=workload.dtype,
            seed=100,
        )
        probe_tensors = _allocate_mm_tensors(
            device=device,
            matrix_size=workload.matrix_size,
            dtype=workload.dtype,
            seed=200,
        )

        trial_count = 5
        low_latencies: List[float] = []
        high_latencies: List[float] = []

        for _ in range(trial_count):
            low_latencies.append(
                _measure_probe_latency_seconds(
                    device=device,
                    workload=workload,
                    background_stream=background_stream,
                    probe_stream=low_probe_stream,
                    background_tensors=background_tensors,
                    probe_tensors=probe_tensors,
                )
            )
            high_latencies.append(
                _measure_probe_latency_seconds(
                    device=device,
                    workload=workload,
                    background_stream=background_stream,
                    probe_stream=high_stream,
                    background_tensors=background_tensors,
                    probe_tensors=probe_tensors,
                )
            )

        low_median = statistics.median(low_latencies)
        high_median = statistics.median(high_latencies)
        win_count = sum(1 for low, high in zip(low_latencies, high_latencies) if high < low)

        print(
            "\n[nVertake GPU priority] "
            f"device={device} size={workload.matrix_size} "
            f"low_median={low_median:.4f}s high_median={high_median:.4f}s wins={win_count}/{trial_count}"
        )

        self.assertGreaterEqual(
            win_count,
            math.ceil(trial_count * 0.6),
            f"High-priority stream did not consistently improve latency: "
            f"low={low_latencies}, high={high_latencies}",
        )
        self.assertLess(
            high_median,
            low_median,
            "Expected high-priority stream to reduce probe latency under contention.",
        )

    @unittest.skipUnless(
        _gpu_tests_enabled(device) and _report_table_enabled(),
        "Requires CUDA, NVERTAKE_ENABLE_GPU_TESTS=1 and NVERTAKE_PRINT_MARKDOWN_TABLE=1",
    )
    def test_print_markdown_table(self):
        """
        Convenience reporting test: prints a small markdown table with concrete numbers.

        Run with:
          NVERTAKE_ENABLE_GPU_TESTS=1 NVERTAKE_PRINT_MARKDOWN_TABLE=1 pytest -q -s \
            test/test_gpu_priority.py::TestStreamPriorityWithinProcess::test_print_markdown_table
        """
        device = self.device
        torch.cuda.set_device(device)

        dtype = torch.float16
        matrix_size = _pick_matmul_size(device, dtype)
        workload = _calibrate_iters(device=device, matrix_size=matrix_size, dtype=dtype)

        background_stream = torch.cuda.Stream(device=device, priority=0)
        probe_stream_default = torch.cuda.Stream(device=device, priority=0)
        scheduler = PriorityScheduler(device=device, nice_value=0)
        probe_stream_high = scheduler.get_high_priority_stream()
        self.assertIsNotNone(probe_stream_high)

        background_tensors = _allocate_mm_tensors(
            device=device,
            matrix_size=workload.matrix_size,
            dtype=workload.dtype,
            seed=123,
        )
        probe_tensors = _allocate_mm_tensors(
            device=device,
            matrix_size=workload.matrix_size,
            dtype=workload.dtype,
            seed=456,
        )

        torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        with torch.cuda.stream(probe_stream_default):
            for _ in range(workload.probe_iters):
                torch.mm(probe_tensors[0], probe_tensors[1], out=probe_tensors[2])
        probe_stream_default.synchronize()
        baseline_seconds = time.perf_counter() - start_time

        contended_default_seconds = _measure_probe_latency_seconds(
            device=device,
            workload=workload,
            background_stream=background_stream,
            probe_stream=probe_stream_default,
            background_tensors=background_tensors,
            probe_tensors=probe_tensors,
        )
        contended_high_seconds = _measure_probe_latency_seconds(
            device=device,
            workload=workload,
            background_stream=background_stream,
            probe_stream=probe_stream_high,
            background_tensors=background_tensors,
            probe_tensors=probe_tensors,
        )

        speedup = contended_default_seconds / max(1e-9, contended_high_seconds)
        improvement = (1.0 - contended_high_seconds / max(1e-9, contended_default_seconds))

        rows = [
            ("device", str(device)),
            ("matrix_size", str(workload.matrix_size)),
            ("probe_iters", str(workload.probe_iters)),
            ("background_iters", str(workload.background_iters)),
            ("baseline_probe_seconds", _fmt_seconds(baseline_seconds)),
            ("contended_default_seconds", _fmt_seconds(contended_default_seconds)),
            ("contended_high_seconds", _fmt_seconds(contended_high_seconds)),
            ("speedup_default_over_high", _fmt_ratio(speedup)),
            ("latency_improvement", f"{_pct(improvement):.1f}%"),
        ]
        _print_markdown_table(rows)


class TestProcessSmShareWithPmon(unittest.TestCase):
    device = _get_test_device()

    @unittest.skipUnless(
        _pmon_tests_enabled(device),
        "Requires CUDA + nvidia-smi and NVERTAKE_ENABLE_GPU_TESTS=1 and NVERTAKE_ENABLE_PMON_TESTS=1",
    )
    def test_pmon_can_observe_two_gpu_processes(self):
        device = self.device
        torch.cuda.set_device(device)

        matrix_size = min(2048, _pick_matmul_size(device, torch.float16))
        run_seconds = 8.0

        ctx = get_context("spawn")
        start_event = ctx.Event()
        result_queue = ctx.Queue()

        low_process = ctx.Process(
            target=_gpu_burn_worker,
            kwargs={
                "role": "low",
                "device": device,
                "stream_priority": 0,
                "start_event": start_event,
                "run_seconds": run_seconds,
                "matrix_size": matrix_size,
                "result_queue": result_queue,
                "use_nvertake_high_priority": False,
            },
            daemon=True,
        )
        high_process = ctx.Process(
            target=_gpu_burn_worker,
            kwargs={
                "role": "high",
                "device": device,
                "stream_priority": -1,
                "start_event": start_event,
                "run_seconds": run_seconds,
                "matrix_size": matrix_size,
                "result_queue": result_queue,
                "use_nvertake_high_priority": True,
            },
            daemon=True,
        )

        low_process.start()
        high_process.start()
        start_event.set()

        try:
            pids = [low_process.pid, high_process.pid]
            self.assertTrue(all(pid is not None for pid in pids))

            samples = _collect_pmon_sm_samples(device=device, pids=pids, sample_count=5)
            low_samples = samples.get(low_process.pid or -1, [])
            high_samples = samples.get(high_process.pid or -1, [])

            print(
                "\n[nVertake pmon] "
                f"device={device} size={matrix_size} low_pid={low_process.pid} high_pid={high_process.pid} "
                f"low_sm_samples={low_samples} high_sm_samples={high_samples}"
            )

            self.assertGreater(len(low_samples), 0, "Expected pmon to report SM% for low process at least once")
            self.assertGreater(len(high_samples), 0, "Expected pmon to report SM% for high process at least once")
        finally:
            low_process.join(timeout=run_seconds + 5.0)
            high_process.join(timeout=run_seconds + 5.0)
            if low_process.is_alive():
                low_process.terminate()
            if high_process.is_alive():
                high_process.terminate()

    @unittest.skipUnless(
        _pmon_tests_enabled(device) and _strict_pmon_asserts_enabled(),
        "Requires NVERTAKE_ENABLE_PMON_TESTS=1 and NVERTAKE_STRICT_GPU_PMON=1",
    )
    def test_high_priority_process_has_higher_sm_share(self):
        device = self.device
        torch.cuda.set_device(device)

        matrix_size = min(2048, _pick_matmul_size(device, torch.float16))
        run_seconds = 10.0

        ctx = get_context("spawn")
        start_event = ctx.Event()
        result_queue = ctx.Queue()

        low_process = ctx.Process(
            target=_gpu_burn_worker,
            kwargs={
                "role": "low",
                "device": device,
                "stream_priority": 0,
                "start_event": start_event,
                "run_seconds": run_seconds,
                "matrix_size": matrix_size,
                "result_queue": result_queue,
                "use_nvertake_high_priority": False,
            },
            daemon=True,
        )
        high_process = ctx.Process(
            target=_gpu_burn_worker,
            kwargs={
                "role": "high",
                "device": device,
                "stream_priority": -1,
                "start_event": start_event,
                "run_seconds": run_seconds,
                "matrix_size": matrix_size,
                "result_queue": result_queue,
                "use_nvertake_high_priority": True,
            },
            daemon=True,
        )

        low_process.start()
        high_process.start()
        start_event.set()

        try:
            pids = [low_process.pid, high_process.pid]
            samples = _collect_pmon_sm_samples(device=device, pids=pids, sample_count=6, warmup_seconds=2.0)

            low_samples = samples.get(low_process.pid or -1, [])
            high_samples = samples.get(high_process.pid or -1, [])

            low_avg = statistics.mean(low_samples) if low_samples else 0.0
            high_avg = statistics.mean(high_samples) if high_samples else 0.0

            sm_count = torch.cuda.get_device_properties(device).multi_processor_count
            print(
                "\n[nVertake pmon strict] "
                f"device={device} size={matrix_size} SMs={sm_count} "
                f"low_pid={low_process.pid} high_pid={high_process.pid} "
                f"low_avg_sm%={low_avg:.1f} (est~{sm_count*low_avg/100:.1f} SMs) "
                f"high_avg_sm%={high_avg:.1f} (est~{sm_count*high_avg/100:.1f} SMs) "
                f"low_samples={low_samples} high_samples={high_samples}"
            )

            self.assertGreater(len(low_samples), 0)
            self.assertGreater(len(high_samples), 0)
            self.assertGreater(
                high_avg,
                low_avg,
                "Expected high-priority process to get higher SM utilization share.",
            )
        finally:
            low_process.join(timeout=run_seconds + 5.0)
            high_process.join(timeout=run_seconds + 5.0)
            if low_process.is_alive():
                low_process.terminate()
            if high_process.is_alive():
                high_process.terminate()
