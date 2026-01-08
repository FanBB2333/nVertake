#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Summary:
    label: str
    device: Optional[int]
    device_name: Optional[str]
    dtype: Optional[str]
    matrix_size: Optional[int]
    pre_mean_iters_per_sec: float
    during_mean_iters_per_sec: float
    drop_percent: float
    pre_samples: int
    during_samples: int
    pre_window_start: float
    pre_window_end: float
    during_window_start: float
    during_window_end: float


def _load_throughput_samples(path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "throughput":
            samples.append(record)
    return samples


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze resident throughput before/during an invader window.")
    parser.add_argument("--label", required=True, help="Scenario label (e.g. invader_no_nvertake / invader_with_nvertake)")
    parser.add_argument("--native-log", required=True, help="Path to resident JSONL log")
    parser.add_argument("--pre-start", type=float, required=True, help="Epoch seconds start of pre window")
    parser.add_argument("--pre-end", type=float, required=True, help="Epoch seconds end of pre window")
    parser.add_argument("--during-start", type=float, required=True, help="Epoch seconds start of during window")
    parser.add_argument("--during-end", type=float, required=True, help="Epoch seconds end of during window")
    parser.add_argument("--json-out", default=None, help="Write summary JSON to this path")
    ns = parser.parse_args(argv)

    log_path = Path(ns.native_log)
    samples = _load_throughput_samples(log_path)

    device = None
    device_name = None
    dtype = None
    matrix_size = None
    if samples:
        device = samples[0].get("device")
        device_name = samples[0].get("device_name")
        dtype = samples[0].get("dtype")
        matrix_size = samples[0].get("matrix_size")

    pre_values = [
        float(s["iters_per_sec"])
        for s in samples
        if ns.pre_start <= float(s["t"]) < ns.pre_end
    ]
    during_values = [
        float(s["iters_per_sec"])
        for s in samples
        if ns.during_start <= float(s["t"]) < ns.during_end
    ]

    pre_mean = _mean(pre_values)
    during_mean = _mean(during_values)
    drop = 0.0
    if pre_mean > 0:
        drop = (during_mean - pre_mean) / pre_mean * 100.0

    summary = Summary(
        label=str(ns.label),
        device=device,
        device_name=device_name,
        dtype=dtype,
        matrix_size=matrix_size,
        pre_mean_iters_per_sec=pre_mean,
        during_mean_iters_per_sec=during_mean,
        drop_percent=drop,
        pre_samples=len(pre_values),
        during_samples=len(during_values),
        pre_window_start=float(ns.pre_start),
        pre_window_end=float(ns.pre_end),
        during_window_start=float(ns.during_start),
        during_window_end=float(ns.during_end),
    )

    payload = asdict(summary)
    print(json.dumps(payload, sort_keys=True))

    if ns.json_out:
        out_path = Path(ns.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

