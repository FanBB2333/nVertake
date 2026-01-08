#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

import torch


_STOP = False


def _handle_signal(_sig: int, _frame) -> None:
    global _STOP
    _STOP = True


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


def _write_jsonl(handle: Optional[TextIO], payload: Dict[str, Any]) -> None:
    line = json.dumps(payload, sort_keys=True)
    if handle is None:
        print(line, flush=True)
        return
    handle.write(line + "\n")
    handle.flush()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Resident workload: stable GEMM loop + per-second throughput.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--matrix-size", type=int, default=4096, help="Square matmul size N for NxN (default: 4096)")
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Matmul dtype: float16/bfloat16/float32 (default: float16)",
    )
    parser.add_argument("--batch-iters", type=int, default=10, help="Matmuls per batch before sync (default: 10)")
    parser.add_argument("--report-interval", type=float, default=1.0, help="Seconds between throughput reports (default: 1.0)")
    parser.add_argument("--warmup-seconds", type=float, default=2.0, help="Warmup seconds before reporting (default: 2.0)")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Total run time; 0 means run until SIGINT/SIGTERM (default: 0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSONL reports to this path (default: stdout)",
    )
    ns = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("error: torch.cuda.is_available() is False; a real CUDA GPU is required.", file=sys.stderr)
        return 2

    if ns.batch_iters <= 0:
        parser.error("--batch-iters must be > 0")
    if ns.report_interval <= 0:
        parser.error("--report-interval must be > 0")
    if ns.warmup_seconds < 0:
        parser.error("--warmup-seconds must be >= 0")
    if ns.duration_seconds < 0:
        parser.error("--duration-seconds must be >= 0")

    dtype = _parse_dtype(ns.dtype)
    device = int(ns.device)
    matrix_size = int(ns.matrix_size)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    output_handle: Optional[TextIO] = None
    if ns.output:
        output_path = Path(ns.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")

    try:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        stream = torch.cuda.Stream(device=device, priority=0)

        with torch.cuda.device(device):
            left = torch.randn((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)
            right = torch.randn((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)
            out = torch.empty((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)

        device_name = torch.cuda.get_device_name(device)
        run_started = time.time()
        _write_jsonl(
            output_handle,
            {
                "event": "start",
                "t": run_started,
                "device": device,
                "device_name": device_name,
                "dtype": str(dtype).replace("torch.", ""),
                "matrix_size": matrix_size,
                "batch_iters": int(ns.batch_iters),
                "report_interval": float(ns.report_interval),
            },
        )

        # Warm up (avoid counting initialization / autotune overhead).
        warmup_end = time.time() + float(ns.warmup_seconds)
        while not _STOP and time.time() < warmup_end:
            with torch.cuda.stream(stream):
                for _ in range(int(ns.batch_iters)):
                    torch.mm(left, right, out=out)
            stream.synchronize()

        torch.cuda.synchronize(device)
        reporting_started = time.time()
        _write_jsonl(output_handle, {"event": "ready", "t": reporting_started})

        end_time = (reporting_started + float(ns.duration_seconds)) if ns.duration_seconds > 0 else None

        total_iters = 0
        last_report_time = time.time()
        last_report_iters = 0

        while not _STOP and (end_time is None or time.time() < end_time):
            with torch.cuda.stream(stream):
                for _ in range(int(ns.batch_iters)):
                    torch.mm(left, right, out=out)
            stream.synchronize()
            total_iters += int(ns.batch_iters)

            now = time.time()
            if now - last_report_time >= float(ns.report_interval):
                window_seconds = max(1e-6, now - last_report_time)
                delta_iters = total_iters - last_report_iters
                iters_per_sec = float(delta_iters) / window_seconds
                _write_jsonl(
                    output_handle,
                    {
                        "event": "throughput",
                        "t": now,
                        "iters_total": total_iters,
                        "iters_per_sec": iters_per_sec,
                        "window_seconds": window_seconds,
                        "device": device,
                        "device_name": device_name,
                        "matrix_size": matrix_size,
                        "dtype": str(dtype).replace("torch.", ""),
                    },
                )
                last_report_time = now
                last_report_iters = total_iters

        finished = time.time()
        _write_jsonl(output_handle, {"event": "stop", "t": finished, "iters_total": total_iters})
        return 0
    finally:
        if output_handle is not None:
            output_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())

