#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import torch

from nvertake import inject_priority


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {value} (supported: {', '.join(sorted(mapping))})")
    return mapping[normalized]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Invader workload: optional nVertake priority injection.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--matrix-size", type=int, default=4096, help="Square matmul size N for NxN (default: 4096)")
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Matmul dtype: float16/bfloat16/float32 (default: float16)",
    )
    parser.add_argument("--batch-iters", type=int, default=10, help="Matmuls per batch before sync (default: 10)")
    parser.add_argument("--duration-seconds", type=float, default=10.0, help="Run time (default: 10)")
    parser.add_argument("--use-nvertake", action="store_true", help="Enable nVertake priority injection")
    parser.add_argument(
        "--nice",
        type=int,
        default=0,
        help="CPU nice value passed to nVertake when --use-nvertake (default: 0)",
    )
    ns = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("error: torch.cuda.is_available() is False; a real CUDA GPU is required.", file=sys.stderr)
        return 2

    if ns.batch_iters <= 0:
        parser.error("--batch-iters must be > 0")
    if ns.duration_seconds <= 0:
        parser.error("--duration-seconds must be > 0")

    device = int(ns.device)
    matrix_size = int(ns.matrix_size)
    dtype = _parse_dtype(ns.dtype)
    duration_seconds = float(ns.duration_seconds)

    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    with torch.cuda.device(device):
        left = torch.randn((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)
        right = torch.randn((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)
        out = torch.empty((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)

    end_time = time.time() + duration_seconds

    def run_loop(stream: Optional[torch.cuda.Stream]) -> None:
        if stream is None:
            stream = torch.cuda.current_stream(device)
        while time.time() < end_time:
            with torch.cuda.stream(stream):
                for _ in range(int(ns.batch_iters)):
                    torch.mm(left, right, out=out)
            stream.synchronize()

    if ns.use_nvertake:
        @inject_priority(device=device, nice_value=int(ns.nice))
        def run_loop_with_nvertake() -> None:
            run_loop(None)

        run_loop_with_nvertake()
        return 0

    low_priority_stream = torch.cuda.Stream(device=device, priority=0)
    run_loop(low_priority_stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
