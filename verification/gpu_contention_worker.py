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
from typing import Any, Dict, List, Optional, TextIO, Tuple

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


def _stream_payload(
    device: int,
    streams: List[torch.cuda.Stream],
    mode: str,
    launch_mode: str,
) -> Dict[str, Any]:
    default = torch.cuda.default_stream(device)
    return {
        "stream_mode": mode,
        "launch_mode": launch_mode,
        "num_streams": len(streams),
        "stream_cuda_ptrs": [int(stream.cuda_stream) for stream in streams],
        "default_stream_cuda_ptr": int(default.cuda_stream),
        "stream_is_default": any(stream.cuda_stream == default.cuda_stream for stream in streams),
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


def _make_streams(device: int, mode: str, num_streams: int) -> List[torch.cuda.Stream]:
    if num_streams <= 0:
        raise ValueError("num_streams must be > 0")
    streams = [_make_stream(device=device, mode=mode)]
    if mode == "nvertake":
        streams.extend(torch.cuda.Stream(device=device, priority=-1) for _ in range(num_streams - 1))
    else:
        streams.extend(_make_stream(device=device, mode=mode) for _ in range(num_streams - 1))
    return streams


def _allocate_workloads(
    *,
    device: int,
    matrix_size: int,
    dtype: torch.dtype,
    count: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    workloads: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.cuda.device(device):
        for _ in range(count):
            left = torch.randn((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)
            right = torch.randn((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)
            out = torch.empty((matrix_size, matrix_size), device=f"cuda:{device}", dtype=dtype)
            workloads.append((left, right, out))
    return workloads


def _run_eager_batch(
    *,
    streams: List[torch.cuda.Stream],
    workloads: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    batch_iters: int,
) -> int:
    for stream, (left, right, out) in zip(streams, workloads):
        with torch.cuda.stream(stream):
            for _ in range(batch_iters):
                torch.mm(left, right, out=out)
    for stream in streams:
        stream.synchronize()
    return batch_iters * len(streams)


def _make_cuda_graph_batch(
    *,
    device: int,
    stream: torch.cuda.Stream,
    workload: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_iters: int,
) -> torch.cuda.CUDAGraph:
    left, right, out = workload
    with torch.cuda.stream(stream):
        for _ in range(batch_iters):
            torch.mm(left, right, out=out)
    stream.synchronize()
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(batch_iters):
            torch.mm(left, right, out=out)
    torch.cuda.synchronize(device)
    return graph


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
    parser.add_argument("--num-streams", type=int, default=1)
    parser.add_argument("--launch-mode", choices=("eager", "cuda_graph"), default="eager")
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
    if args.num_streams <= 0:
        parser.error("--num-streams must be > 0")
    if args.launch_mode == "cuda_graph" and args.num_streams != 1:
        parser.error("--launch-mode cuda_graph requires --num-streams 1")

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
        streams = _make_streams(device=device, mode=str(args.stream_mode), num_streams=int(args.num_streams))
        stream_info = _stream_payload(
            device=device,
            streams=streams,
            mode=str(args.stream_mode),
            launch_mode=str(args.launch_mode),
        )
        workloads = _allocate_workloads(
            device=device,
            matrix_size=matrix_size,
            dtype=dtype,
            count=len(streams),
        )
        graph = None
        if args.launch_mode == "cuda_graph":
            graph = _make_cuda_graph_batch(
                device=device,
                stream=streams[0],
                workload=workloads[0],
                batch_iters=int(args.batch_iters),
            )

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
                "effective_batch_iters": int(args.batch_iters) * len(streams),
                "report_interval": float(args.report_interval),
                **stream_info,
            },
        )

        warmup_end = time.time() + float(args.warmup_seconds)
        while not _STOP and time.time() < warmup_end:
            if graph is not None:
                graph.replay()
                streams[0].synchronize()
            else:
                _run_eager_batch(
                    streams=streams,
                    workloads=workloads,
                    batch_iters=int(args.batch_iters),
                )

        torch.cuda.synchronize(device)
        ready = time.time()
        _write_jsonl(handle, {"event": "ready", "t": ready, "role": args.role, "pid": os_getpid()})

        end_time = ready + float(args.duration_seconds) if args.duration_seconds > 0 else None
        total_iters = 0
        last_report_t = time.time()
        last_report_iters = 0
        flops_per_iter = 2.0 * (matrix_size**3)

        while not _STOP and (end_time is None or time.time() < end_time):
            if graph is not None:
                graph.replay()
                streams[0].synchronize()
                completed_iters = int(args.batch_iters)
            else:
                completed_iters = _run_eager_batch(
                    streams=streams,
                    workloads=workloads,
                    batch_iters=int(args.batch_iters),
                )
            total_iters += completed_iters

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
