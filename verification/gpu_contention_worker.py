#!/usr/bin/env python3
"""
Sustained GPU worker used by contention experiments.

Each process runs repeated GEMMs and emits JSONL throughput samples. The runner
uses two workers to create a saturated resident-vs-invader contention window.
"""

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


def _handle_signal(_sig: int, _frame: Any) -> None:
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
        raise ValueError(f"Unsupported dtype: {value}")
    return mapping[normalized]


def _write_jsonl(handle: TextIO, payload: Dict[str, Any]) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()


def _stream_payload(device: int, stream: torch.cuda.Stream, mode: str) -> Dict[str, Any]:
    default = torch.cuda.default_stream(device)
    return {
        "stream_mode": mode,
        "stream_cuda_ptr": int(stream.cuda_stream),
        "default_stream_cuda_ptr": int(default.cuda_stream),
        "stream_is_default": bool(stream.cuda_stream == default.cuda_stream),
    }


def _make_stream(device: int, mode: str) -> torch.cuda.Stream:
    if mode == "priority0":
        return torch.cuda.Stream(device=device, priority=0)
    if mode == "nvertake":
        from nvertake.scheduler import PriorityScheduler

        scheduler = PriorityScheduler(device=device, nice_value=0)
        stream = scheduler.get_high_priority_stream()
        if stream is None:
            raise RuntimeError("nVertake high-priority stream creation failed")
        return stream
    raise ValueError(f"Unsupported stream mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("resident", "invader"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-iters", type=int, default=4)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--report-interval", type=float, default=1.0)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--stream-mode", choices=("priority0", "nvertake"), default="priority0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; run on a real CUDA host.", file=sys.stderr)
        return 2
    if args.batch_iters <= 0:
        parser.error("--batch-iters must be > 0")
    if args.report_interval <= 0:
        parser.error("--report-interval must be > 0")
    if args.duration_seconds < 0:
        parser.error("--duration-seconds must be >= 0")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    device = int(args.device)
    matrix_size = int(args.matrix_size)
    dtype = _parse_dtype(args.dtype)

    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        stream = _make_stream(device=device, mode=str(args.stream_mode))
        stream_info = _stream_payload(device=device, stream=stream, mode=str(args.stream_mode))

        with torch.cuda.device(device):
            left = torch.randn((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)
            right = torch.randn((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)
            out = torch.empty((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)

        started = time.time()
        _write_jsonl(
            handle,
            {
                "event": "start",
                "t": started,
                "role": args.role,
                "pid": os_getpid(),
                "device": device,
                "device_name": torch.cuda.get_device_name(device),
                "matrix_size": matrix_size,
                "dtype": str(dtype).replace("torch.", ""),
                "batch_iters": int(args.batch_iters),
                "report_interval": float(args.report_interval),
                **stream_info,
            },
        )

        warmup_end = time.time() + float(args.warmup_seconds)
        while not _STOP and time.time() < warmup_end:
            with torch.cuda.stream(stream):
                for _ in range(int(args.batch_iters)):
                    torch.mm(left, right, out=out)
            stream.synchronize()

        torch.cuda.synchronize(device)
        ready = time.time()
        _write_jsonl(handle, {"event": "ready", "t": ready, "role": args.role, "pid": os_getpid()})

        end_time = ready + float(args.duration_seconds) if args.duration_seconds > 0 else None
        total_iters = 0
        last_report_t = time.time()
        last_report_iters = 0
        flops_per_iter = 2.0 * (matrix_size**3)

        while not _STOP and (end_time is None or time.time() < end_time):
            with torch.cuda.stream(stream):
                for _ in range(int(args.batch_iters)):
                    torch.mm(left, right, out=out)
            stream.synchronize()
            total_iters += int(args.batch_iters)

            now = time.time()
            if now - last_report_t >= float(args.report_interval):
                window_seconds = max(1e-9, now - last_report_t)
                delta_iters = total_iters - last_report_iters
                iters_per_sec = float(delta_iters) / window_seconds
                _write_jsonl(
                    handle,
                    {
                        "event": "throughput",
                        "t": now,
                        "role": args.role,
                        "pid": os_getpid(),
                        "iters_total": total_iters,
                        "delta_iters": delta_iters,
                        "window_seconds": window_seconds,
                        "iters_per_sec": iters_per_sec,
                        "tflops_per_sec": iters_per_sec * flops_per_iter / 1e12,
                    },
                )
                last_report_t = now
                last_report_iters = total_iters

        torch.cuda.synchronize(device)
        _write_jsonl(
            handle,
            {
                "event": "stop",
                "t": time.time(),
                "role": args.role,
                "pid": os_getpid(),
                "iters_total": total_iters,
            },
        )

    return 0


def os_getpid() -> int:
    import os

    return os.getpid()


if __name__ == "__main__":
    raise SystemExit(main())
