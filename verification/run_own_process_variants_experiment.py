#!/usr/bin/env python3
"""
Evaluate own-process-only strategies under saturated GPU contention.

Resident is fixed to a long-running GEMM workload. Each variant changes only
the invader side:
- CUDA stream priority
- number of invader CUDA streams
- CUDA Graph replay
- smaller kernel granularity
- multiple invader process replicas

The JSON output is meant for plotting and for deciding whether a strategy
actually increases the invader's compute share under full utilization.
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
class ResidentConfig:
    load_id: str = "resident_gemm8192_fp16_batch4"
    matrix_size: int = 8192
    dtype: str = "float16"
    batch_iters: int = 4
    pre_seconds: float = 5.0
    invade_seconds: float = 10.0
    post_seconds: float = 2.0
    warmup_seconds: float = 2.0
    report_interval: float = 1.0
    sample_interval: float = 0.5


@dataclass(frozen=True)
class InvaderVariant:
    variant_id: str
    description: str
    matrix_size: int
    dtype: str
    batch_iters: int
    stream_mode: str
    num_streams: int = 1
    launch_mode: str = "eager"
    replicas: int = 1


@dataclass
class WindowMetrics:
    mean: Optional[float]
    median: Optional[float]
    min: Optional[float]
    max: Optional[float]
    samples: int


VARIANTS: List[InvaderVariant] = [
    InvaderVariant(
        variant_id="control_priority0_single_eager",
        description="single invader process, normal-priority CUDA stream",
        matrix_size=8192,
        dtype="float16",
        batch_iters=4,
        stream_mode="priority0",
    ),
    InvaderVariant(
        variant_id="nvertake_single_eager",
        description="single invader process, nVertake high-priority stream",
        matrix_size=8192,
        dtype="float16",
        batch_iters=4,
        stream_mode="nvertake",
    ),
    InvaderVariant(
        variant_id="priority0_multistream4_eager",
        description="single invader process, four normal-priority CUDA streams",
        matrix_size=8192,
        dtype="float16",
        batch_iters=2,
        stream_mode="priority0",
        num_streams=4,
    ),
    InvaderVariant(
        variant_id="nvertake_multistream4_eager",
        description="single invader process, four high-priority CUDA streams",
        matrix_size=8192,
        dtype="float16",
        batch_iters=2,
        stream_mode="nvertake",
        num_streams=4,
    ),
    InvaderVariant(
        variant_id="nvertake_cuda_graph",
        description="single invader process, nVertake stream with CUDA Graph replay",
        matrix_size=8192,
        dtype="float16",
        batch_iters=4,
        stream_mode="nvertake",
        launch_mode="cuda_graph",
    ),
    InvaderVariant(
        variant_id="nvertake_small_kernel4096",
        description="single invader process, smaller 4096 GEMM kernels on nVertake stream",
        matrix_size=4096,
        dtype="float16",
        batch_iters=12,
        stream_mode="nvertake",
    ),
    InvaderVariant(
        variant_id="priority0_replicas2",
        description="two invader process replicas, normal-priority CUDA streams",
        matrix_size=8192,
        dtype="float16",
        batch_iters=4,
        stream_mode="priority0",
        replicas=2,
    ),
    InvaderVariant(
        variant_id="nvertake_replicas2",
        description="two invader process replicas, nVertake high-priority streams",
        matrix_size=8192,
        dtype="float16",
        batch_iters=4,
        stream_mode="nvertake",
        replicas=2,
    ),
]


class UtilMonitor:
    def __init__(self, device: int, sample_interval: float) -> None:
        self.device = device
        self.sample_interval = sample_interval
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="nvertake-variant-util-monitor", daemon=True)
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


def _pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / old * 100.0


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _throughput(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [record for record in records if record.get("event") == "throughput"]


def _in_window(samples: Iterable[Dict[str, Any]], start: float, end: float, key: str) -> List[float]:
    return [
        float(sample[key])
        for sample in samples
        if start <= float(sample.get("t", 0.0)) < end and sample.get(key) is not None
    ]


def _wait_for_event(path: Path, event: str, timeout_seconds: float = 30.0) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        for record in _load_jsonl(path):
            if record.get("event") == event:
                return record
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for event={event!r} in {path}")


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _worker_cmd(
    *,
    role: str,
    device: int,
    matrix_size: int,
    dtype: str,
    batch_iters: int,
    duration_seconds: float,
    report_interval: float,
    warmup_seconds: float,
    stream_mode: str,
    num_streams: int,
    launch_mode: str,
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
        str(matrix_size),
        "--dtype",
        dtype,
        "--batch-iters",
        str(batch_iters),
        "--duration-seconds",
        str(duration_seconds),
        "--report-interval",
        str(report_interval),
        "--warmup-seconds",
        str(warmup_seconds),
        "--stream-mode",
        stream_mode,
        "--num-streams",
        str(num_streams),
        "--launch-mode",
        launch_mode,
        "--output",
        str(output),
    ]


def _aggregate_series(
    series_by_replica: List[List[Dict[str, Any]]],
    *,
    start: float,
    end: float,
    bucket_seconds: float,
) -> List[Dict[str, Any]]:
    buckets: Dict[int, Dict[str, float]] = {}
    for replica_index, series in enumerate(series_by_replica):
        for sample in series:
            t = float(sample.get("t", 0.0))
            if not (start <= t < end):
                continue
            bucket = int((t - start) // bucket_seconds)
            payload = buckets.setdefault(
                bucket,
                {
                    "t": start + bucket * bucket_seconds,
                    "iters_per_sec": 0.0,
                    "tflops_per_sec": 0.0,
                    "replica_samples": 0.0,
                },
            )
            payload["iters_per_sec"] += float(sample.get("iters_per_sec", 0.0))
            payload["tflops_per_sec"] += float(sample.get("tflops_per_sec", 0.0))
            payload["replica_samples"] += 1.0
            payload[f"replica_{replica_index}_tflops_per_sec"] = float(sample.get("tflops_per_sec", 0.0))
    return [buckets[key] for key in sorted(buckets)]


def _summarize_variant(
    *,
    config: ResidentConfig,
    variant: InvaderVariant,
    resident_records: List[Dict[str, Any]],
    invader_records_by_replica: List[List[Dict[str, Any]]],
    util_samples: List[Dict[str, Any]],
    windows: Dict[str, float],
    resident_returncode: Optional[int],
    invader_returncodes: List[Optional[int]],
) -> Dict[str, Any]:
    pre_start = windows["pre_start"]
    invade_start = windows["invade_start"]
    invade_end = windows["invade_end"]
    post_end = windows["post_end"]

    resident_series = _throughput(resident_records)
    invader_series_by_replica = [_throughput(records) for records in invader_records_by_replica]
    invader_aggregate_series = _aggregate_series(
        invader_series_by_replica,
        start=invade_start,
        end=invade_end,
        bucket_seconds=config.report_interval,
    )
    util_valid = [sample for sample in util_samples if sample.get("gpu_util_percent") is not None]

    resident_pre_tflops = _window_metrics(_in_window(resident_series, pre_start, invade_start, "tflops_per_sec"))
    resident_during_tflops = _window_metrics(_in_window(resident_series, invade_start, invade_end, "tflops_per_sec"))
    resident_post_tflops = _window_metrics(_in_window(resident_series, invade_end, post_end, "tflops_per_sec"))
    invader_during_tflops = _window_metrics(
        [float(sample["tflops_per_sec"]) for sample in invader_aggregate_series]
    )

    resident_during = resident_during_tflops.mean
    invader_during = invader_during_tflops.mean
    combined_tflops = None
    resident_share = None
    invader_share = None
    if resident_during is not None and invader_during is not None:
        combined_tflops = resident_during + invader_during
        if combined_tflops > 0:
            resident_share = resident_during / combined_tflops
            invader_share = invader_during / combined_tflops

    util_during_values = _in_window(util_valid, invade_start, invade_end, "gpu_util_percent")
    util_pre_values = _in_window(util_valid, pre_start, invade_start, "gpu_util_percent")
    util_post_values = _in_window(util_valid, invade_end, post_end, "gpu_util_percent")
    util_ge90 = sum(1 for value in util_during_values if value >= 90.0)

    return {
        "variant": asdict(variant),
        "resident_config": asdict(config),
        "process_returncodes": {
            "resident": resident_returncode,
            "invaders": invader_returncodes,
        },
        "windows": windows,
        "resident": {
            "pre_tflops_per_sec": asdict(resident_pre_tflops),
            "during_tflops_per_sec": asdict(resident_during_tflops),
            "post_tflops_per_sec": asdict(resident_post_tflops),
            "during_vs_pre_tflops_change_percent": _pct_change(
                resident_during_tflops.mean,
                resident_pre_tflops.mean,
            ),
            "series": resident_series,
        },
        "invader": {
            "during_tflops_per_sec": asdict(invader_during_tflops),
            "share_of_combined_during_tflops": invader_share,
            "aggregate_series": invader_aggregate_series,
            "replica_series": invader_series_by_replica,
        },
        "resource_split": {
            "resident_share_of_combined_during_tflops": resident_share,
            "invader_share_of_combined_during_tflops": invader_share,
            "combined_during_tflops_per_sec": combined_tflops,
        },
        "gpu_util": {
            "pre_percent": asdict(_window_metrics(util_pre_values)),
            "during_percent": asdict(_window_metrics(util_during_values)),
            "post_percent": asdict(_window_metrics(util_post_values)),
            "during_fraction_ge_90_percent": util_ge90 / len(util_during_values) if util_during_values else None,
            "samples": util_samples,
        },
    }


def _run_variant(
    *,
    device: int,
    config: ResidentConfig,
    variant: InvaderVariant,
    tmp_dir: Path,
) -> Dict[str, Any]:
    resident_log = tmp_dir / f"{variant.variant_id}_resident.jsonl"
    invader_logs = [tmp_dir / f"{variant.variant_id}_invader_{idx}.jsonl" for idx in range(variant.replicas)]

    resident = subprocess.Popen(
        _worker_cmd(
            role="resident",
            device=device,
            matrix_size=config.matrix_size,
            dtype=config.dtype,
            batch_iters=config.batch_iters,
            duration_seconds=0.0,
            report_interval=config.report_interval,
            warmup_seconds=config.warmup_seconds,
            stream_mode="priority0",
            num_streams=1,
            launch_mode="eager",
            output=resident_log,
        ),
        cwd=str(REPO_ROOT),
    )
    monitor = UtilMonitor(device=device, sample_interval=config.sample_interval)
    monitor.start()

    invaders: List[subprocess.Popen[Any]] = []
    try:
        _wait_for_event(resident_log, "ready")
        pre_start = time.time()
        time.sleep(config.pre_seconds)
        invade_start = time.time()

        for replica_index, invader_log in enumerate(invader_logs):
            invaders.append(
                subprocess.Popen(
                    _worker_cmd(
                        role="invader",
                        device=device,
                        matrix_size=variant.matrix_size,
                        dtype=variant.dtype,
                        batch_iters=variant.batch_iters,
                        duration_seconds=config.invade_seconds,
                        report_interval=config.report_interval,
                        warmup_seconds=config.warmup_seconds,
                        stream_mode=variant.stream_mode,
                        num_streams=variant.num_streams,
                        launch_mode=variant.launch_mode,
                        output=invader_log,
                    ),
                    cwd=str(REPO_ROOT),
                )
            )

        invader_returncodes = [
            process.wait(timeout=config.invade_seconds + config.warmup_seconds + 60.0)
            for process in invaders
        ]
        invade_end = time.time()
        time.sleep(config.post_seconds)
        post_end = time.time()
    finally:
        for process in invaders:
            _terminate(process)
        _terminate(resident)
        monitor.stop()

    windows = {
        "pre_start": pre_start,
        "invade_start": invade_start,
        "invade_end": invade_end,
        "post_end": post_end,
    }
    return _summarize_variant(
        config=config,
        variant=variant,
        resident_records=_load_jsonl(resident_log),
        invader_records_by_replica=[_load_jsonl(path) for path in invader_logs],
        util_samples=monitor.samples,
        windows=windows,
        resident_returncode=resident.poll(),
        invader_returncodes=invader_returncodes,
    )


def _compare_to_control(control: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    control_invader = control["invader"]["during_tflops_per_sec"]["mean"]
    variant_invader = variant["invader"]["during_tflops_per_sec"]["mean"]
    control_resident = control["resident"]["during_tflops_per_sec"]["mean"]
    variant_resident = variant["resident"]["during_tflops_per_sec"]["mean"]
    control_share = control["resource_split"]["invader_share_of_combined_during_tflops"]
    variant_share = variant["resource_split"]["invader_share_of_combined_during_tflops"]
    share_delta = None if control_share is None or variant_share is None else variant_share - control_share
    throughput_change = _pct_change(variant_invader, control_invader)

    return {
        "invader_tflops_change_percent": throughput_change,
        "resident_tflops_change_percent": _pct_change(variant_resident, control_resident),
        "invader_share_change_points": share_delta,
        "control_invader_share": control_share,
        "variant_invader_share": variant_share,
        "improved_invader_share": bool(share_delta is not None and share_delta > 0.0),
        "improved_invader_tflops": bool(throughput_change is not None and throughput_change > 0.0),
    }


def _summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    control = next(
        result for result in results if result["variant"]["variant_id"] == "control_priority0_single_eager"
    )
    variants: List[Dict[str, Any]] = []
    for result in results:
        comparison = _compare_to_control(control, result)
        util_mean = result["gpu_util"]["during_percent"]["mean"]
        util_ge90 = result["gpu_util"]["during_fraction_ge_90_percent"]
        variants.append(
            {
                "variant_id": result["variant"]["variant_id"],
                "description": result["variant"]["description"],
                "gpu_util_during_mean": util_mean,
                "gpu_util_ge90_fraction": util_ge90,
                "saturated_competition": bool(
                    util_mean is not None and util_mean >= 90.0 and util_ge90 is not None and util_ge90 >= 0.8
                ),
                "invader_tflops_per_sec": result["invader"]["during_tflops_per_sec"]["mean"],
                "resident_tflops_per_sec": result["resident"]["during_tflops_per_sec"]["mean"],
                "invader_share": result["resource_split"]["invader_share_of_combined_during_tflops"],
                **comparison,
            }
        )

    non_control = [variant for variant in variants if variant["variant_id"] != "control_priority0_single_eager"]
    best_share = max(variants, key=lambda item: item["invader_share"] or -1.0)
    best_tflops = max(variants, key=lambda item: item["invader_tflops_per_sec"] or -1.0)
    return {
        "total_variants": len(variants),
        "saturated_variants": sum(1 for variant in variants if variant["saturated_competition"]),
        "non_control_improved_share_variants": sum(1 for variant in non_control if variant["improved_invader_share"]),
        "non_control_improved_tflops_variants": sum(1 for variant in non_control if variant["improved_invader_tflops"]),
        "best_invader_share_variant": best_share,
        "best_invader_tflops_variant": best_tflops,
        "variants": variants,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--output",
        default="verification/results/own_process_variants_latest.json",
        help="JSON summary path.",
    )
    args = parser.parse_args()

    config = ResidentConfig()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    results: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="nvertake-own-variants-") as tmp:
        tmp_dir = Path(tmp)
        for variant in VARIANTS:
            print(f"[own-variants] running {variant.variant_id}", flush=True)
            results.append(_run_variant(device=args.device, config=config, variant=variant, tmp_dir=tmp_dir))

    payload = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": args.device,
        "resident_config": asdict(config),
        "metric_notes": {
            "gpu_util_percent": "Sampled from nvidia-smi.",
            "tflops_share": "Invader share is aggregate invader TFLOP/s divided by resident+invader TFLOP/s during contention.",
            "scope": "Only the invader implementation changes. Resident stays fixed.",
        },
        "summary": _summary(results),
        "results": results,
    }

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[own-variants] wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
