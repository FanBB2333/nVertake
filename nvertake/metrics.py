"""Cooperative workload metrics used by monitoring and share calibration."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_METRICS_ENV = "NVERTAKE_METRICS_PATH"
_MIB = 1024 * 1024
_MAX_SAMPLES = 256


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _pytorch_memory_snapshot() -> Dict[str, Any]:
    """Read allocator memory only when the workload already initialized PyTorch CUDA."""

    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    try:
        if cuda is None or not cuda.is_initialized():
            return {}
        allocated = int(cuda.memory_allocated(0) // _MIB)
        reserved = int(cuda.memory_reserved(0) // _MIB)
    except (AttributeError, RuntimeError):
        return {}
    return {
        "pytorch_allocated_memory_mib": allocated,
        "pytorch_reserved_memory_mib": reserved,
        "gpu_memory_mib": max(allocated, reserved),
        "gpu_memory_source": "pytorch_allocator",
    }


def report_throughput(
    value: float,
    *,
    unit: str = "items/s",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Publish the latest workload throughput for ``nvertake monitor``.

    Returns ``False`` when the process was not launched by nVertake, allowing
    scripts to keep the call in place during ordinary execution.
    """

    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < 0:
        raise ValueError("throughput must be a finite non-negative number")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("throughput unit must be a non-empty string")
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("throughput metadata must be a mapping")

    raw_path = os.environ.get(_METRICS_ENV)
    if not raw_path:
        return False
    sample: Dict[str, Any] = {
        "throughput": numeric_value,
        "unit": unit.strip(),
        "pid": os.getpid(),
        "updated_at": _utc_now(),
        "monotonic_time": time.monotonic(),
    }
    sample.update(_pytorch_memory_snapshot())
    if metadata:
        sample["metadata"] = metadata
    previous = read_throughput_metric(Path(raw_path))
    samples = []
    if (
        previous is not None
        and previous.get("pid") == sample["pid"]
        and previous.get("unit") == sample["unit"]
    ):
        raw_samples = previous.get("samples")
        if isinstance(raw_samples, list):
            samples = [item for item in raw_samples if isinstance(item, dict)]
        else:
            samples = [
                {
                    key: previous[key]
                    for key in (
                        "throughput",
                        "unit",
                        "pid",
                        "updated_at",
                        "monotonic_time",
                        "gpu_memory_mib",
                        "gpu_memory_source",
                        "pytorch_allocated_memory_mib",
                        "pytorch_reserved_memory_mib",
                        "metadata",
                    )
                    if key in previous
                }
            ]
    samples.append(dict(sample))
    payload = dict(sample)
    payload["samples"] = samples[-_MAX_SAMPLES:]
    _atomic_write_json(Path(raw_path), payload)
    return True


def read_throughput_metric(path: Path) -> Optional[Dict[str, Any]]:
    """Read a complete metric update, tolerating an absent/stale file."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        value = float(payload["throughput"])
    except (KeyError, TypeError, ValueError):
        return None
    unit = payload.get("unit")
    if not math.isfinite(value) or value < 0 or not isinstance(unit, str):
        return None
    payload["throughput"] = value
    return payload


def read_throughput_samples(path: Path) -> Tuple[Dict[str, Any], ...]:
    """Return all valid retained samples, including legacy single-value files."""

    payload = read_throughput_metric(path)
    if payload is None:
        return ()
    raw_samples = payload.get("samples")
    candidates = raw_samples if isinstance(raw_samples, list) else [payload]
    samples = []
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        try:
            value = float(raw["throughput"])
        except (KeyError, TypeError, ValueError):
            continue
        unit = raw.get("unit")
        if not math.isfinite(value) or value < 0 or not isinstance(unit, str):
            continue
        sample = dict(raw)
        sample["throughput"] = value
        samples.append(sample)
    return tuple(samples)
