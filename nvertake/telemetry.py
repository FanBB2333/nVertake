"""Read-only NVIDIA GPU utilization and profiling telemetry."""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Tuple


_DCGM_FIELDS = (
    (1002, "sm_active"),
    (1003, "sm_occupancy"),
    (1004, "tensor_active"),
    (1005, "dram_active"),
    (1007, "fp32_active"),
    (1008, "fp16_active"),
)


def _number(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        return None


def query_process_utilization() -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Sample per-process SM and memory utilization through ``nvidia-smi pmon``."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return {}
    if result.returncode != 0:
        return {}

    samples: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5 or fields[1] == "-":
            continue
        try:
            device = int(fields[0])
            pid = int(fields[1])
        except ValueError:
            continue
        sm = _number(fields[3])
        memory = _number(fields[4])
        samples[(device, pid)] = {
            "device": device,
            "pid": pid,
            "process_type": fields[2],
            "sm_util_percent": sm,
            "memory_util_percent": memory,
            "source": "nvidia-smi pmon",
        }
    return samples


def query_device_utilization(
    devices: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    """Read device-wide utilization, clocks, power, and temperature."""

    requested = {int(device) for device in devices} if devices is not None else None
    fields = (
        "index",
        "utilization.gpu",
        "utilization.memory",
        "clocks.sm",
        "clocks.mem",
        "power.draw",
        "power.limit",
        "temperature.gpu",
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    if result.returncode != 0:
        return []

    records = []
    names = (
        "device",
        "gpu_util_percent",
        "memory_util_percent",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "power_draw_w",
        "power_limit_w",
        "temperature_c",
    )
    for raw_line in result.stdout.splitlines():
        values = [value.strip() for value in raw_line.split(",")]
        if len(values) != len(names):
            continue
        try:
            device = int(values[0])
        except ValueError:
            continue
        if requested is not None and device not in requested:
            continue
        record: Dict[str, Any] = {"device": device, "source": "nvidia-smi"}
        for name, value in zip(names[1:], values[1:]):
            record[name] = _number(value)
        records.append(record)
    return records


def query_dcgm_profile(device: int) -> Dict[str, Any]:
    """Collect one DCGM profiling sample, returning an unavailable reason safely."""

    binary = shutil.which("dcgmi")
    if binary is None:
        return {
            "available": False,
            "source": "dcgm",
            "detail": "dcgmi was not found",
        }
    field_ids = ",".join(str(field_id) for field_id, _name in _DCGM_FIELDS)
    try:
        result = subprocess.run(
            [binary, "dmon", "-i", str(int(device)), "-e", field_ids, "-c", "1"],
            check=False,
            capture_output=True,
            text=True,
            timeout=6.0,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return {
            "available": False,
            "source": "dcgm",
            "detail": str(exc),
        }
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return {
            "available": False,
            "source": "dcgm",
            "detail": detail or f"dcgmi exited with {result.returncode}",
        }

    for raw_line in reversed(result.stdout.splitlines()):
        tokens = raw_line.split()
        if len(tokens) < len(_DCGM_FIELDS) + 1 or raw_line.lstrip().startswith("#"):
            continue
        values = tokens[-len(_DCGM_FIELDS) :]
        parsed = [_number(value) for value in values]
        if any(value is None for value in parsed):
            continue
        record: Dict[str, Any] = {
            "available": True,
            "device": int(device),
            "source": "dcgm profiling fields",
        }
        for (_field_id, name), value in zip(_DCGM_FIELDS, parsed):
            assert value is not None
            record[name] = value
            record[f"{name}_percent"] = value * 100.0
        return record
    return {
        "available": False,
        "source": "dcgm",
        "detail": "dcgmi returned no parseable profiling sample",
    }


def inspect_telemetry_capabilities(device: int = 0) -> Dict[str, Any]:
    """Report installed telemetry paths without requiring a running workload."""

    nvidia_smi = shutil.which("nvidia-smi")
    dcgmi = shutil.which("dcgmi")
    pmon = False
    pmon_detail = "nvidia-smi was not found"
    if nvidia_smi is not None:
        try:
            result = subprocess.run(
                [nvidia_smi, "pmon", "-h"],
                check=False,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            pmon = result.returncode == 0
            pmon_detail = (
                "available"
                if pmon
                else (result.stderr or result.stdout).strip()[-300:]
            )
        except (subprocess.SubprocessError, OSError) as exc:
            pmon_detail = str(exc)
    if "microsoft" in platform.release().lower():
        pmon = False
        pmon_detail = (
            "nvidia-smi pmon does not expose per-process utilization inside WSL"
        )

    dcgm_profile = False
    dcgm_detail = "dcgmi was not found"
    if dcgmi is not None:
        try:
            result = subprocess.run(
                [dcgmi, "profile", "-l", "-i", str(int(device))],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            dcgm_profile = result.returncode == 0
            dcgm_detail = (
                "available"
                if dcgm_profile
                else (result.stderr or result.stdout).strip()[-500:]
            )
        except (subprocess.SubprocessError, OSError) as exc:
            dcgm_detail = str(exc)
    return {
        "nvidia_smi": {
            "available": nvidia_smi is not None,
            "path": nvidia_smi,
        },
        "process_utilization": {
            "available": pmon,
            "source": "nvidia-smi pmon",
            "detail": pmon_detail,
        },
        "dcgm_profiling": {
            "available": dcgm_profile,
            "path": dcgmi,
            "field_ids": [field_id for field_id, _name in _DCGM_FIELDS],
            "detail": dcgm_detail,
        },
    }
