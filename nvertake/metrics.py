"""Cooperative workload metrics used by monitoring and share calibration."""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_METRICS_ENV = "NVERTAKE_METRICS_PATH"


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
    payload: Dict[str, Any] = {
        "throughput": numeric_value,
        "unit": unit.strip(),
        "pid": os.getpid(),
        "updated_at": _utc_now(),
        "monotonic_time": time.monotonic(),
    }
    if metadata:
        payload["metadata"] = metadata
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
