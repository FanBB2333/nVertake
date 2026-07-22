#!/usr/bin/env python3
"""Compare equal and target-heavy CUDA Green Context SM partitions."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nvertake.green_context import (  # noqa: E402
    GreenContextExecutor,
    current_green_context_lane,
)


_MEASUREMENT_BARRIER: Optional[threading.Barrier] = None


def gemm_task(
    *,
    role: str,
    matrix_size: int = 4096,
    batch_iters: int = 4,
    warmup_iters: int = 4,
    duration_seconds: float = 4.0,
) -> Dict[str, Any]:
    """Run a saturated FP16 GEMM loop in the calling Green Context lane."""

    import torch

    lane = current_green_context_lane()
    if lane is None:
        raise RuntimeError("gemm_task must run inside GreenContextExecutor")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required")

    left = torch.randn((matrix_size, matrix_size), device="cuda", dtype=torch.float16)
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    stream = torch.cuda.current_stream()

    for _ in range(warmup_iters):
        torch.mm(left, right, out=output)
    stream.synchronize()

    barrier = _MEASUREMENT_BARRIER
    if barrier is not None:
        barrier.wait(timeout=60.0)

    start = time.perf_counter()
    iterations = 0
    while time.perf_counter() - start < duration_seconds:
        for _ in range(batch_iters):
            torch.mm(left, right, out=output)
        stream.synchronize()
        iterations += batch_iters
    elapsed = time.perf_counter() - start
    operations = 2.0 * matrix_size**3 * iterations
    checksum = float(output[0, 0].item())
    result = {
        "role": role,
        "lane": lane.index,
        "requested_share": lane.requested_share,
        "sm_count": lane.sm_count,
        "actual_sm_share": lane.actual_sm_share,
        "elapsed_seconds": elapsed,
        "iterations": iterations,
        "tflops_per_second": operations / elapsed / 1e12,
        "checksum": checksum,
        "device_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    del output, right, left
    return result


def _run_case(
    case_id: str,
    shares: Tuple[float, float],
    *,
    device: int,
    matrix_size: int,
    batch_iters: int,
    warmup_iters: int,
    duration_seconds: float,
) -> Dict[str, Any]:
    global _MEASUREMENT_BARRIER
    _MEASUREMENT_BARRIER = threading.Barrier(2)
    task_kwargs = (
        {
            "role": "background",
            "matrix_size": matrix_size,
            "batch_iters": batch_iters,
            "warmup_iters": warmup_iters,
            "duration_seconds": duration_seconds,
        },
        {
            "role": "target",
            "matrix_size": matrix_size,
            "batch_iters": batch_iters,
            "warmup_iters": warmup_iters,
            "duration_seconds": duration_seconds,
        },
    )
    try:
        with GreenContextExecutor(device=device, shares=shares) as executor:
            run = executor.run((gemm_task, gemm_task), task_kwargs=task_kwargs)
    finally:
        _MEASUREMENT_BARRIER = None

    background, target = run.results
    combined = background["tflops_per_second"] + target["tflops_per_second"]
    return {
        "case_id": case_id,
        "requested_shares": list(shares),
        "total_sm_count": run.total_sm_count,
        "compute_capability_major": run.compute_capability_major,
        "lanes": [lane.to_dict() for lane in run.lanes],
        "background": background,
        "target": target,
        "combined_tflops_per_second": combined,
        "target_throughput_share": target["tflops_per_second"] / combined,
    }


def run_experiment(
    *,
    device: int = 0,
    matrix_size: int = 4096,
    batch_iters: int = 4,
    warmup_iters: int = 4,
    duration_seconds: float = 4.0,
    cases: Sequence[Tuple[str, Tuple[float, float]]] = (
        ("equal_50_50", (50.0, 50.0)),
        ("target_75_background_25", (25.0, 75.0)),
    ),
) -> Dict[str, Any]:
    results = [
        _run_case(
            case_id,
            shares,
            device=device,
            matrix_size=matrix_size,
            batch_iters=batch_iters,
            warmup_iters=warmup_iters,
            duration_seconds=duration_seconds,
        )
        for case_id, shares in cases
    ]
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "device": device,
        "matrix_size": matrix_size,
        "batch_iters": batch_iters,
        "warmup_iters": warmup_iters,
        "duration_seconds": duration_seconds,
        "cases": results,
    }
    if len(results) == 2:
        payload["comparison"] = {
            "target_throughput_share_delta_percentage_points": 100.0
            * (
                results[1]["target_throughput_share"]
                - results[0]["target_throughput_share"]
            ),
            "target_tflops_ratio": (
                results[1]["target"]["tflops_per_second"]
                / results[0]["target"]["tflops_per_second"]
            ),
        }
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare 50/50 and 25/75 in-process CUDA Green Context partitions."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--batch-iters", type=int, default=4)
    parser.add_argument("--warmup-iters", type=int, default=4)
    parser.add_argument("--duration-seconds", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = run_experiment(
        device=args.device,
        matrix_size=args.matrix_size,
        batch_iters=args.batch_iters,
        warmup_iters=args.warmup_iters,
        duration_seconds=args.duration_seconds,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
