#!/usr/bin/env python3
"""Measure nVertake MPS GPU shares with two saturated CUDA processes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "verification" / "gpu_contention_worker.py"


@dataclass(frozen=True)
class ShareCase:
    case_id: str
    resident_share: int
    target_share: int


CASES = (
    ShareCase("equal_50_50", resident_share=50, target_share=50),
    ShareCase("target_75_background_25", resident_share=25, target_share=75),
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tail_text(path: Path, line_count: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-line_count:])


def _wait_for_event(
    path: Path,
    event: str,
    timeout: float,
    *,
    process: Optional[subprocess.Popen[Any]] = None,
    stderr_path: Optional[Path] = None,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for record in _load_jsonl(path):
            if record.get("event") == event:
                return record
        if process is not None and process.poll() is not None:
            detail = ""
            if stderr_path is not None:
                stderr_tail = _tail_text(stderr_path)
                detail = f"\nstderr ({stderr_path}):"
                if stderr_tail:
                    detail += f"\n{stderr_tail}"
            raise RuntimeError(
                f"Process exited with status {process.returncode} before event={event!r}{detail}"
            )
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for event={event!r} in {path}")


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5.0)


def _mean_tflops(records: Iterable[Dict[str, Any]], start: float, end: float) -> float:
    values = [
        float(record["tflops_per_sec"])
        for record in records
        if record.get("event") == "throughput"
        and start <= float(record.get("t", 0.0)) < end
    ]
    return float(statistics.mean(values)) if values else 0.0


def _worker_command(
    *,
    device: int,
    share: int,
    role: str,
    output: Path,
    matrix_size: int,
    dtype: str,
    batch_iters: int,
    duration_seconds: float,
    warmup_seconds: float,
    report_interval: float,
    mps_priority: Optional[str],
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "nvertake.cli",
        "--device",
        str(device),
        "--nice",
        "0",
        "--no-torch-priority",
        "--gpu-share",
        str(share),
    ]
    if mps_priority:
        command.extend(["--mps-priority", mps_priority])
    command.extend(
        [
            "exec",
            sys.executable,
            str(WORKER),
            "--role",
            role,
            # The MPS daemon maps the selected physical GPU to client device 0.
            "--device",
            "0",
            "--matrix-size",
            str(matrix_size),
            "--dtype",
            dtype,
            "--batch-iters",
            str(batch_iters),
            "--warmup-seconds",
            str(warmup_seconds),
            "--duration-seconds",
            str(duration_seconds),
            "--report-interval",
            str(report_interval),
            "--stream-mode",
            "priority0",
            "--output",
            str(output),
        ]
    )
    return command


def _run_case(
    case: ShareCase,
    *,
    run_directory: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    resident_log = run_directory / f"{case.case_id}_resident.jsonl"
    target_log = run_directory / f"{case.case_id}_target.jsonl"
    resident_stderr = run_directory / f"{case.case_id}_resident.stderr.log"
    target_stderr = run_directory / f"{case.case_id}_target.stderr.log"

    resident_command = _worker_command(
        device=args.device,
        share=case.resident_share,
        role="resident",
        output=resident_log,
        matrix_size=args.matrix_size,
        dtype=args.dtype,
        batch_iters=args.batch_iters,
        duration_seconds=0.0,
        warmup_seconds=args.warmup_seconds,
        report_interval=args.report_interval,
        mps_priority=args.resident_priority,
    )
    target_command = _worker_command(
        device=args.device,
        share=case.target_share,
        role="target",
        output=target_log,
        matrix_size=args.matrix_size,
        dtype=args.dtype,
        batch_iters=args.batch_iters,
        duration_seconds=args.duration_seconds,
        warmup_seconds=args.warmup_seconds,
        report_interval=args.report_interval,
        mps_priority=args.target_priority,
    )

    resident: Optional[subprocess.Popen[Any]] = None
    target: Optional[subprocess.Popen[Any]] = None
    with resident_stderr.open("w", encoding="utf-8") as resident_error, target_stderr.open(
        "w", encoding="utf-8"
    ) as target_error:
        try:
            resident = subprocess.Popen(
                resident_command,
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=resident_error,
                text=True,
                start_new_session=True,
            )
            _wait_for_event(
                resident_log,
                "ready",
                timeout=args.startup_timeout,
                process=resident,
                stderr_path=resident_stderr,
            )

            target = subprocess.Popen(
                target_command,
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=target_error,
                text=True,
                start_new_session=True,
            )
            target_ready = _wait_for_event(
                target_log,
                "ready",
                timeout=args.startup_timeout,
                process=target,
                stderr_path=target_stderr,
            )
            target_returncode = target.wait(timeout=args.duration_seconds + args.startup_timeout)
            if target_returncode != 0:
                raise RuntimeError(
                    f"Target exited with status {target_returncode}; see {target_stderr}"
                )
            target_records = _load_jsonl(target_log)
            target_stop = next(
                record for record in reversed(target_records) if record.get("event") == "stop"
            )
        finally:
            if target is not None:
                _terminate(target)
            if resident is not None:
                _terminate(resident)

    resident_records = _load_jsonl(resident_log)
    target_records = _load_jsonl(target_log)
    window_start = float(target_ready["t"])
    window_end = float(target_stop["t"])
    resident_tflops = _mean_tflops(resident_records, window_start, window_end)
    target_tflops = _mean_tflops(target_records, window_start, window_end)
    combined = resident_tflops + target_tflops

    return {
        "case": asdict(case),
        "window": {"start": window_start, "end": window_end},
        "resident_tflops_per_sec": resident_tflops,
        "target_tflops_per_sec": target_tflops,
        "target_throughput_share": target_tflops / combined if combined else 0.0,
        "combined_tflops_per_sec": combined,
        "artifacts": {
            "resident_log": str(resident_log),
            "target_log": str(target_log),
            "resident_stderr": str(resident_stderr),
            "target_stderr": str(target_stderr),
        },
    }


def _stop_mps(device: int) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nvertake.cli",
            "--device",
            str(device),
            "mps",
            "stop",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15.0,
        check=False,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare 50/50 and 25/75 NVIDIA MPS active-thread limits."
    )
    parser.add_argument("--device", type=int, default=0, help="Physical GPU index")
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-iters", type=int, default=4)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--duration-seconds", type=float, default=12.0)
    parser.add_argument("--report-interval", type=float, default=0.5)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument(
        "--resident-priority",
        choices=("normal", "below-normal"),
        default="normal",
    )
    parser.add_argument(
        "--target-priority",
        choices=("normal", "below-normal"),
        default="normal",
    )
    parser.add_argument("--output", default=None, help="Summary JSON path")
    parser.add_argument(
        "--keep-mps",
        action="store_true",
        help="Leave the nVertake MPS daemon running after the experiment",
    )
    args = parser.parse_args(argv)

    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be > 0")

    output_path = Path(args.output) if args.output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_directory = output_path.parent / (output_path.stem + "_artifacts")
        run_directory.mkdir(parents=True, exist_ok=True)
        temporary_directory = None
    else:
        temporary_directory = tempfile.TemporaryDirectory(prefix="nvertake-mps-share-")
        run_directory = Path(temporary_directory.name)

    results: List[Dict[str, Any]] = []
    try:
        for case in CASES:
            print(
                f"[{case.case_id}] resident={case.resident_share}% "
                f"target={case.target_share}%",
                flush=True,
            )
            result = _run_case(case, run_directory=run_directory, args=args)
            results.append(result)
            print(
                "  target throughput share: "
                f"{result['target_throughput_share'] * 100.0:.1f}%",
                flush=True,
            )
    finally:
        if not args.keep_mps:
            _stop_mps(args.device)

    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "device": args.device,
        "workload": {
            "matrix_size": args.matrix_size,
            "dtype": args.dtype,
            "batch_iters": args.batch_iters,
            "duration_seconds": args.duration_seconds,
        },
        "cases": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        output_path.write_text(rendered, encoding="utf-8")
        print(f"Summary: {output_path}")
    else:
        print(rendered, end="")
        if temporary_directory is not None:
            temporary_directory.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
