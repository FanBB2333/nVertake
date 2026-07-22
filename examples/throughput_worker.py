"""Small cooperative GEMM workload for nVertake YAML and calibration examples."""

import argparse
import json
import os
import time

import torch

from nvertake import report_throughput


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    seconds = float(os.environ.get("NVERTAKE_CALIBRATION_SECONDS", args.seconds))
    left = torch.randn((args.size, args.size), device="cuda", dtype=torch.float16)
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    for _ in range(3):
        torch.mm(left, right, out=output)
    torch.cuda.synchronize()

    started = time.monotonic()
    completed = 0
    next_report = started + 0.5
    while time.monotonic() - started < seconds:
        torch.mm(left, right, out=output)
        completed += 1
        now = time.monotonic()
        if now >= next_report:
            torch.cuda.synchronize()
            report_throughput(completed / (time.monotonic() - started), unit="GEMM/s")
            next_report = now + 0.5

    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    throughput = completed / elapsed
    report_throughput(throughput, unit="GEMM/s")
    print(
        json.dumps(
            {
                "job": os.environ.get("NVERTAKE_JOB_NAME"),
                "iterations": completed,
                "seconds": elapsed,
                "throughput": throughput,
                "memory_share": os.environ.get("NVERTAKE_MEMORY_SHARE"),
                "actual_memory_fraction": torch.cuda.get_per_process_memory_fraction(0),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
