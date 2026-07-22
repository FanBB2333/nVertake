#!/usr/bin/env python3
"""Saturated PyTorch GEMM worker for the public green-procs launcher."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_worker(
    *,
    role: str,
    matrix_size: int,
    batch_iters: int,
    warmup_iters: int,
    duration_seconds: float,
) -> Dict[str, Any]:
    import torch

    from nvertake import current_green_context_lane

    lane = current_green_context_lane()
    if lane is None:
        raise RuntimeError("Worker is not running inside a Green process context")

    left = torch.randn((matrix_size, matrix_size), device="cuda", dtype=torch.float16)
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    stream = torch.cuda.current_stream()
    for _ in range(warmup_iters):
        torch.mm(left, right, out=output)
    stream.synchronize()

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
        "pid": os.getpid(),
        "lane": lane.index,
        "requested_share": lane.requested_share,
        "sm_count": lane.sm_count,
        "total_sm_count": lane.total_sm_count,
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--batch-iters", type=int, default=4)
    parser.add_argument("--warmup-iters", type=int, default=4)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run_worker(
        role=args.role,
        matrix_size=args.matrix_size,
        batch_iters=args.batch_iters,
        warmup_iters=args.warmup_iters,
        duration_seconds=args.duration_seconds,
    )
    encoded = json.dumps(result, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
