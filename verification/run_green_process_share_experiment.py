#!/usr/bin/env python3
"""Measure an arbitrary multi-process CUDA Green Context share plan."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "verification" / "green_process_gemm_worker.py"


def _parse_shares(value: str) -> Tuple[float, ...]:
    try:
        shares = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shares must be comma-separated numbers") from exc
    if len(shares) < 2 or any(
        not math.isfinite(share) or share <= 0 for share in shares
    ):
        raise argparse.ArgumentTypeError("shares must contain at least two positive numbers")
    return shares


def _launcher_command(
    *,
    shares: Sequence[float],
    output_paths: Sequence[Path],
    device: int,
    matrix_size: int,
    batch_iters: int,
    warmup_iters: int,
    duration_seconds: float,
    startup_timeout: float,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "nvertake.cli",
        "--device",
        str(device),
        "green-procs",
        "--shares",
        ",".join(str(share) for share in shares),
        "--startup-timeout",
        str(startup_timeout),
    ]
    for index, output_path in enumerate(output_paths):
        script_args = [
            "--role",
            f"lane-{index}",
            "--matrix-size",
            str(matrix_size),
            "--batch-iters",
            str(batch_iters),
            "--warmup-iters",
            str(warmup_iters),
            "--duration-seconds",
            str(duration_seconds),
            "--output",
            str(output_path),
        ]
        command.extend(["--script-args", json.dumps(script_args)])
    command.extend(str(WORKER) for _ in shares)
    return command


def run_experiment(
    *,
    shares: Sequence[float],
    device: int = 0,
    matrix_size: int = 4096,
    batch_iters: int = 4,
    warmup_iters: int = 4,
    duration_seconds: float = 3.0,
    startup_timeout: float = 60.0,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="nvertake-green-procs-experiment-") as root:
        output_paths = tuple(
            Path(root) / f"lane-{index}.json" for index in range(len(shares))
        )
        command = _launcher_command(
            shares=shares,
            output_paths=output_paths,
            device=device,
            matrix_size=matrix_size,
            batch_iters=batch_iters,
            warmup_iters=warmup_iters,
            duration_seconds=duration_seconds,
            startup_timeout=startup_timeout,
        )
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=startup_timeout + duration_seconds + 60.0,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"green-procs exited with status {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        workers = [
            json.loads(path.read_text(encoding="utf-8")) for path in output_paths
        ]

    workers.sort(key=lambda item: int(item["lane"]))
    combined_tflops = sum(float(item["tflops_per_second"]) for item in workers)
    for item in workers:
        item["throughput_share"] = (
            float(item["tflops_per_second"]) / combined_tflops
        )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "device": device,
        "requested_shares": list(shares),
        "actual_sm_counts": [int(item["sm_count"]) for item in workers],
        "combined_tflops_per_second": combined_tflops,
        "workers": workers,
        "launcher_stderr": completed.stderr.strip(),
        "configuration": {
            "matrix_size": matrix_size,
            "batch_iters": batch_iters,
            "warmup_iters": warmup_iters,
            "duration_seconds": duration_seconds,
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shares", type=_parse_shares, default=(20.0, 30.0, 50.0))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--batch-iters", type=int, default=4)
    parser.add_argument("--warmup-iters", type=int, default=4)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = run_experiment(
        shares=args.shares,
        device=args.device,
        matrix_size=args.matrix_size,
        batch_iters=args.batch_iters,
        warmup_iters=args.warmup_iters,
        duration_seconds=args.duration_seconds,
        startup_timeout=args.startup_timeout,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
