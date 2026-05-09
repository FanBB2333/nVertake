#!/usr/bin/env python3
"""
Run sustained GPU contention experiments with utilization sampling.

The output JSON is designed for plotting:
- GPU utilization time series for each scenario
- resident/invader throughput time series
- per-window aggregate metrics
- cross-scenario comparison between baseline and nVertake invader
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "verification" / "gpu_contention_worker.py"


@dataclass(frozen=True)
class LoadConfig:
    load_id: str
    matrix_size: int
    dtype: str
    batch_iters: int
    pre_seconds: float
    invade_seconds: float
    post_seconds: float
    warmup_seconds: float
    report_interval: float
    sample_interval: float


@dataclass
class WindowMetrics:
    mean: Optional[float]
    median: Optional[float]
    min: Optional[float]
    max: Optional[float]
    samples: int


LOAD_CONFIGS: List[LoadConfig] = [
    LoadConfig(
        load_id="gemm4096_fp16_batch12",
        matrix_size=4096,
        dtype="float16",
        batch_iters=12,
        pre_seconds=6.0,
        invade_seconds=12.0,
        post_seconds=3.0,
        warmup_seconds=2.0,
        report_interval=1.0,
        sample_interval=0.5,
    ),
    LoadConfig(
        load_id="gemm8192_fp16_batch4",
        matrix_size=8192,
        dtype="float16",
        batch_iters=4,
        pre_seconds=6.0,
        invade_seconds=12.0,
        post_seconds=3.0,
        warmup_seconds=2.0,
        report_interval=1.0,
        sample_interval=0.5,
    ),
]


def _mean(values: List[float]) -> Optional[float]:
    return float(statistics.mean(values)) if values else None


def _median(values: List[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def _window_metrics(values: List[float]) -> WindowMetrics:
    return WindowMetrics(
        mean=_mean(values),
        median=_median(values),
        min=float(min(values)) if values else None,
        max=float(max(values)) if values else None,
        samples=len(values),
    )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _throughput(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [record for record in records if record.get("event") == "throughput"]


def _in_window(samples: Iterable[Dict[str, Any]], start: float, end: float, key: str) -> List[float]:
    return [
        float(sample[key])
        for sample in samples
        if start <= float(sample.get("t", 0.0)) < end and sample.get(key) is not None
    ]


def _pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if old is None or old == 0 or new is None:
        return None
    return (new - old) / old * 100.0


def _wait_for_event(path: Path, event: str, timeout_seconds: float = 30.0) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        for record in _load_jsonl(path):
            if record.get("event") == event:
                return record
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for event={event!r} in {path}")


class UtilMonitor:
    def __init__(self, device: int, sample_interval: float) -> None:
        self.device = device
        self.sample_interval = sample_interval
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="nvertake-util-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._sample())
            self._stop.wait(self.sample_interval)

    def _sample(self) -> Dict[str, Any]:
        now = time.time()
        cmd = [
            "nvidia-smi",
            f"--id={self.device}",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
            raw = completed.stdout.strip().splitlines()[0]
            parts = [part.strip() for part in raw.split(",")]
            return {
                "t": now,
                "gpu_util_percent": float(parts[0]),
                "memory_used_mib": float(parts[1]),
                "power_draw_w": float(parts[2]) if len(parts) > 2 and parts[2] != "[N/A]" else None,
                "error": None,
            }
        except Exception as exc:
            return {
                "t": now,
                "gpu_util_percent": None,
                "memory_used_mib": None,
                "power_draw_w": None,
                "error": str(exc),
            }


def _worker_cmd(
    *,
    role: str,
    device: int,
    config: LoadConfig,
    duration_seconds: float,
    stream_mode: str,
    output: Path,
) -> List[str]:
    return [
        sys.executable,
        str(WORKER),
        "--role",
        role,
        "--device",
        str(device),
        "--matrix-size",
        str(config.matrix_size),
        "--dtype",
        config.dtype,
        "--batch-iters",
        str(config.batch_iters),
        "--duration-seconds",
        str(duration_seconds),
        "--report-interval",
        str(config.report_interval),
        "--warmup-seconds",
        str(config.warmup_seconds),
        "--stream-mode",
        stream_mode,
        "--output",
        str(output),
    ]


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _summarize_scenario(
    *,
    label: str,
    config: LoadConfig,
    resident_records: List[Dict[str, Any]],
    invader_records: List[Dict[str, Any]],
    util_samples: List[Dict[str, Any]],
    windows: Dict[str, float],
    resident_returncode: Optional[int],
    invader_returncode: Optional[int],
) -> Dict[str, Any]:
    resident_series = _throughput(resident_records)
    invader_series = _throughput(invader_records)
    util_valid = [sample for sample in util_samples if sample.get("gpu_util_percent") is not None]

    pre_start = windows["pre_start"]
    invade_start = windows["invade_start"]
    invade_end = windows["invade_end"]
    post_end = windows["post_end"]

    resident_pre = _window_metrics(_in_window(resident_series, pre_start, invade_start, "iters_per_sec"))
    resident_during = _window_metrics(_in_window(resident_series, invade_start, invade_end, "iters_per_sec"))
    resident_post = _window_metrics(_in_window(resident_series, invade_end, post_end, "iters_per_sec"))
    invader_during = _window_metrics(_in_window(invader_series, invade_start, invade_end, "iters_per_sec"))

    util_pre_values = _in_window(util_valid, pre_start, invade_start, "gpu_util_percent")
    util_during_values = _in_window(util_valid, invade_start, invade_end, "gpu_util_percent")
    util_post_values = _in_window(util_valid, invade_end, post_end, "gpu_util_percent")
    util_during = _window_metrics(util_during_values)

    resident_mean = resident_during.mean
    invader_mean = invader_during.mean
    combined_mean = None
    resident_share = None
    invader_share = None
    if resident_mean is not None and invader_mean is not None:
        combined_mean = resident_mean + invader_mean
        if combined_mean > 0:
            resident_share = resident_mean / combined_mean
            invader_share = invader_mean / combined_mean

    util_samples_ge_90 = sum(1 for value in util_during_values if value >= 90.0)
    util_samples_ge_80 = sum(1 for value in util_during_values if value >= 80.0)

    return {
        "label": label,
        "config": asdict(config),
        "process_returncodes": {
            "resident": resident_returncode,
            "invader": invader_returncode,
        },
        "windows": windows,
        "resident": {
            "pre_iters_per_sec": asdict(resident_pre),
            "during_iters_per_sec": asdict(resident_during),
            "post_iters_per_sec": asdict(resident_post),
            "drop_during_vs_pre_percent": _pct_change(resident_during.mean, resident_pre.mean),
            "series": resident_series,
        },
        "invader": {
            "during_iters_per_sec": asdict(invader_during),
            "share_of_combined_during_iters": invader_share,
            "series": invader_series,
        },
        "resource_split": {
            "resident_share_of_combined_during_iters": resident_share,
            "invader_share_of_combined_during_iters": invader_share,
            "combined_during_iters_per_sec": combined_mean,
        },
        "gpu_util": {
            "pre_percent": asdict(_window_metrics(util_pre_values)),
            "during_percent": asdict(util_during),
            "post_percent": asdict(_window_metrics(util_post_values)),
            "during_samples_ge_80_percent": util_samples_ge_80,
            "during_samples_ge_90_percent": util_samples_ge_90,
            "during_fraction_ge_80_percent": (
                util_samples_ge_80 / len(util_during_values) if util_during_values else None
            ),
            "during_fraction_ge_90_percent": (
                util_samples_ge_90 / len(util_during_values) if util_during_values else None
            ),
            "samples": util_samples,
        },
    }


def _run_scenario(
    *,
    label: str,
    device: int,
    config: LoadConfig,
    invader_stream_mode: str,
    tmp_dir: Path,
) -> Dict[str, Any]:
    resident_log = tmp_dir / f"{config.load_id}_{label}_resident.jsonl"
    invader_log = tmp_dir / f"{config.load_id}_{label}_invader.jsonl"

    monitor = UtilMonitor(device=device, sample_interval=config.sample_interval)
    resident = subprocess.Popen(
        _worker_cmd(
            role="resident",
            device=device,
            config=config,
            duration_seconds=0.0,
            stream_mode="priority0",
            output=resident_log,
        ),
        cwd=str(REPO_ROOT),
    )
    monitor.start()

    invader: Optional[subprocess.Popen[Any]] = None
    try:
        _wait_for_event(resident_log, "ready")
        pre_start = time.time()
        time.sleep(config.pre_seconds)
        invade_start = time.time()
        invader = subprocess.Popen(
            _worker_cmd(
                role="invader",
                device=device,
                config=config,
                duration_seconds=config.invade_seconds,
                stream_mode=invader_stream_mode,
                output=invader_log,
            ),
            cwd=str(REPO_ROOT),
        )
        invader_returncode = invader.wait(timeout=config.invade_seconds + config.warmup_seconds + 30.0)
        invade_end = time.time()
        time.sleep(config.post_seconds)
        post_end = time.time()
    finally:
        _terminate_process(resident)
        monitor.stop()

    resident_returncode = resident.poll()
    if invader is None:
        invader_returncode = None
        invade_end = time.time()
        post_end = invade_end

    windows = {
        "pre_start": pre_start,
        "invade_start": invade_start,
        "invade_end": invade_end,
        "post_end": post_end,
    }
    return _summarize_scenario(
        label=label,
        config=config,
        resident_records=_load_jsonl(resident_log),
        invader_records=_load_jsonl(invader_log),
        util_samples=monitor.samples,
        windows=windows,
        resident_returncode=resident_returncode,
        invader_returncode=invader_returncode,
    )


def _compare(no_nvertake: Dict[str, Any], with_nvertake: Dict[str, Any]) -> Dict[str, Any]:
    no_invader = no_nvertake["invader"]["during_iters_per_sec"]["mean"]
    yes_invader = with_nvertake["invader"]["during_iters_per_sec"]["mean"]
    no_resident = no_nvertake["resident"]["during_iters_per_sec"]["mean"]
    yes_resident = with_nvertake["resident"]["during_iters_per_sec"]["mean"]
    no_invader_share = no_nvertake["resource_split"]["invader_share_of_combined_during_iters"]
    yes_invader_share = with_nvertake["resource_split"]["invader_share_of_combined_during_iters"]

    invader_share_change = (
        (yes_invader_share - no_invader_share) if yes_invader_share is not None and no_invader_share is not None else None
    )
    invader_iters_change = _pct_change(yes_invader, no_invader)
    resident_iters_change = _pct_change(yes_resident, no_resident)

    return {
        "invader_iters_per_sec_change_percent": _pct_change(yes_invader, no_invader),
        "resident_iters_per_sec_change_percent": resident_iters_change,
        "invader_share_change_points": invader_share_change,
        "nvertake_improved_invader_throughput": bool(invader_iters_change is not None and invader_iters_change > 0.0),
        "nvertake_improved_invader_share": bool(invader_share_change is not None and invader_share_change > 0.0),
        "no_nvertake_invader_share": no_invader_share,
        "with_nvertake_invader_share": yes_invader_share,
        "no_nvertake_gpu_util_during_mean": no_nvertake["gpu_util"]["during_percent"]["mean"],
        "with_nvertake_gpu_util_during_mean": with_nvertake["gpu_util"]["during_percent"]["mean"],
    }


def _top_level_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    loads: List[Dict[str, Any]] = []
    for result in results:
        comparison = result["comparison"]
        scenarios = {scenario["label"]: scenario for scenario in result["scenarios"]}
        no_nvertake = scenarios["invader_no_nvertake"]
        with_nvertake = scenarios["invader_with_nvertake"]
        no_util_mean = no_nvertake["gpu_util"]["during_percent"]["mean"]
        yes_util_mean = with_nvertake["gpu_util"]["during_percent"]["mean"]
        no_ge90 = no_nvertake["gpu_util"]["during_fraction_ge_90_percent"]
        yes_ge90 = with_nvertake["gpu_util"]["during_fraction_ge_90_percent"]
        invader_share_change = comparison["invader_share_change_points"]
        invader_iters_change = comparison["invader_iters_per_sec_change_percent"]
        improved_share = bool(invader_share_change is not None and invader_share_change > 0.0)
        improved_throughput = bool(invader_iters_change is not None and invader_iters_change > 0.0)
        saturated = bool(
            no_util_mean is not None
            and yes_util_mean is not None
            and no_util_mean >= 90.0
            and yes_util_mean >= 90.0
            and no_ge90 is not None
            and yes_ge90 is not None
            and no_ge90 >= 0.8
            and yes_ge90 >= 0.8
        )
        loads.append(
            {
                "load_id": result["load_id"],
                "saturated_competition": saturated,
                "no_nvertake_gpu_util_during_mean": no_util_mean,
                "with_nvertake_gpu_util_during_mean": yes_util_mean,
                "no_nvertake_gpu_util_ge90_fraction": no_ge90,
                "with_nvertake_gpu_util_ge90_fraction": yes_ge90,
                "no_nvertake_invader_share": comparison["no_nvertake_invader_share"],
                "with_nvertake_invader_share": comparison["with_nvertake_invader_share"],
                "invader_share_change_points": invader_share_change,
                "invader_iters_per_sec_change_percent": invader_iters_change,
                "resident_iters_per_sec_change_percent": comparison["resident_iters_per_sec_change_percent"],
                "nvertake_improved_invader_share": improved_share,
                "nvertake_improved_invader_throughput": improved_throughput,
            }
        )

    saturated_count = sum(1 for load in loads if load["saturated_competition"])
    improved_share_count = sum(1 for load in loads if load["nvertake_improved_invader_share"])
    improved_throughput_count = sum(1 for load in loads if load["nvertake_improved_invader_throughput"])
    return {
        "total_loads": len(loads),
        "saturated_loads": saturated_count,
        "nvertake_improved_invader_share_loads": improved_share_count,
        "nvertake_improved_invader_throughput_loads": improved_throughput_count,
        "loads": loads,
        "interpretation": (
            "This experiment verifies saturated two-process GPU contention. "
            "For these bulk GEMM workloads, nVertake did not increase the invader's "
            "throughput share, so the current stream-priority implementation does not "
            "prove stronger inter-process GPU resource allocation under full utilization."
            if loads and improved_share_count == 0
            else "nVertake improved invader share for at least one saturated load."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--output",
        default="verification/results/contention_util_latest.json",
        help="JSON summary path.",
    )
    args = parser.parse_args()

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    results: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="nvertake-contention-") as tmp:
        tmp_dir = Path(tmp)
        for config in LOAD_CONFIGS:
            print(f"[contention] {config.load_id}: invader_no_nvertake", flush=True)
            no_nvertake = _run_scenario(
                label="invader_no_nvertake",
                device=args.device,
                config=config,
                invader_stream_mode="priority0",
                tmp_dir=tmp_dir,
            )
            print(f"[contention] {config.load_id}: invader_with_nvertake", flush=True)
            with_nvertake = _run_scenario(
                label="invader_with_nvertake",
                device=args.device,
                config=config,
                invader_stream_mode="nvertake",
                tmp_dir=tmp_dir,
            )
            results.append(
                {
                    "load_id": config.load_id,
                    "config": asdict(config),
                    "scenarios": [no_nvertake, with_nvertake],
                    "comparison": _compare(no_nvertake, with_nvertake),
                }
            )

    payload = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": args.device,
        "summary": _top_level_summary(results),
        "metric_notes": {
            "gpu_util_percent": "Sampled from nvidia-smi at the configured sample interval.",
            "throughput": "Each iteration is one square GEMM. Throughput is per-process iters/sec and TFLOP/sec.",
            "resource_split": "Approximate compute share from resident/invader GEMM throughput during the contention window.",
        },
        "results": results,
    }

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[contention] wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
