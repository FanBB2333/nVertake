"""Persistent launch reports and local process monitoring."""

from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .metrics import read_throughput_metric


TERMINAL_RUN_STATES = frozenset(("completed", "failed", "cancelled"))
TERMINAL_JOB_STATES = frozenset(("completed", "failed", "cancelled", "timed_out"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def write_report_snapshot(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist a reconciled report snapshot atomically."""

    _atomic_write_json(path.expanduser().resolve(), payload)


def runtime_registry_directory() -> Path:
    configured = os.environ.get("NVERTAKE_RUNTIME_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    state_home = os.environ.get("XDG_STATE_HOME")
    if state_home:
        return Path(state_home).expanduser().resolve() / "nvertake" / "runs"
    return Path.home() / ".cache" / "nvertake" / "runs"


def _registry_path(run_id: str) -> Path:
    return runtime_registry_directory() / f"{run_id}.json"


def register_report(run_id: str, report_path: Path) -> None:
    payload = {
        "run_id": run_id,
        "report_path": str(report_path.resolve()),
        "registered_at": utc_now(),
    }
    _atomic_write_json(_registry_path(run_id), payload)


def resolve_report_path(identifier: Optional[str] = None) -> Path:
    """Resolve a report path, run id, or the latest registered local run."""

    if identifier:
        direct = Path(identifier).expanduser()
        if direct.is_file():
            return direct.resolve()
        registry = _registry_path(identifier)
        if not registry.is_file():
            raise FileNotFoundError(f"No nVertake report or run id found: {identifier}")
    else:
        registry_root = runtime_registry_directory()
        candidates = sorted(
            registry_root.glob("*.json") if registry_root.is_dir() else (),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("No registered nVertake runs were found")
        registry = candidates[0]

    try:
        payload = json.loads(registry.read_text(encoding="utf-8"))
        report_path = Path(payload["report_path"]).expanduser()
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Invalid nVertake run registry: {registry}") from exc
    if not report_path.is_file():
        raise FileNotFoundError(f"Registered report no longer exists: {report_path}")
    return report_path.resolve()


def load_report(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid nVertake JSON report: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
        raise RuntimeError(f"Invalid nVertake report schema: {path}")
    return payload


def _launcher_identity_is_alive(payload: Mapping[str, Any]) -> bool:
    metadata = dict(payload.get("metadata") or {})
    launcher_pid = metadata.get("launcher_pid")
    if not isinstance(launcher_pid, int) or launcher_pid <= 0:
        return False
    if not _pid_is_alive(launcher_pid):
        return False
    expected = metadata.get("launcher_create_time")
    if expected is None:
        return True
    actual = process_create_time(launcher_pid)
    return actual is None or abs(actual - float(expected)) < 0.01


def reconcile_report(path: Path) -> Dict[str, Any]:
    """Refresh a report and mark a local non-terminal orphan as failed."""

    resolved = path.expanduser().resolve()
    payload = load_report(resolved)
    if (payload.get("metadata") or {}).get("orchestrator") == "ssh":
        from .orchestration import refresh_distributed_snapshot

        snapshot = refresh_distributed_snapshot(payload)
    else:
        snapshot = enrich_report(payload)
    if (
        snapshot.get("status") not in TERMINAL_RUN_STATES
        and not _launcher_identity_is_alive(snapshot)
    ):
        ended_at = utc_now()
        snapshot["status"] = "failed"
        snapshot["error"] = "Launcher process is no longer running"
        snapshot["ended_at"] = ended_at
        for job in snapshot.get("jobs", []):
            if job.get("status") not in TERMINAL_JOB_STATES:
                job["status"] = "failed"
                job["observed_status"] = "exited"
                job["ended_at"] = ended_at
    snapshot["updated_at"] = utc_now()
    if snapshot.get("status") in TERMINAL_RUN_STATES and not snapshot.get("ended_at"):
        snapshot["ended_at"] = utc_now()
    write_report_snapshot(resolved, snapshot)
    return snapshot


def list_registered_runs(
    *,
    refresh: bool = False,
    prune: bool = False,
) -> List[Dict[str, Any]]:
    """Return newest-first summaries for locally registered reports."""

    root = runtime_registry_directory()
    candidates = sorted(
        root.glob("*.json") if root.is_dir() else (),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    summaries: List[Dict[str, Any]] = []
    for registry in candidates:
        try:
            entry = json.loads(registry.read_text(encoding="utf-8"))
            report_path = Path(entry["report_path"]).expanduser().resolve()
            report = reconcile_report(report_path) if refresh else load_report(report_path)
        except (KeyError, TypeError, json.JSONDecodeError, OSError, RuntimeError):
            if prune:
                try:
                    registry.unlink()
                except OSError:
                    pass
            continue
        summaries.append(
            {
                "run_id": report.get("run_id", entry.get("run_id")),
                "status": report.get("status", "unknown"),
                "started_at": report.get("started_at"),
                "updated_at": report.get("updated_at"),
                "ended_at": report.get("ended_at"),
                "jobs": len(report.get("jobs", [])),
                "hosts": sorted(
                    {
                        str(job.get("host"))
                        for job in report.get("jobs", [])
                        if job.get("host")
                    }
                ),
                "report_path": str(report_path),
            }
        )
    return summaries


def query_gpu_process_memory() -> Dict[int, int]:
    """Return per-PID framebuffer memory in MiB using nvidia-smi."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return {}

    memory: Dict[int, int] = {}
    for raw_line in result.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) < 2:
            continue
        try:
            pid = int(fields[0])
            used = int(float(fields[1]))
        except ValueError:
            continue
        memory[pid] = memory.get(pid, 0) + used
    return memory


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def process_create_time(pid: int) -> Optional[float]:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.Error, OSError, ValueError):
        return None


def stop_local_report(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Signal only PIDs recorded by a non-terminal local run report."""

    if payload.get("status") in TERMINAL_RUN_STATES:
        return {"already_terminal": True, "status": payload.get("status"), "pids": []}
    metadata = dict(payload.get("metadata") or {})
    launcher_pid = metadata.get("launcher_pid")
    expected_create_time = metadata.get("launcher_create_time")
    if (
        not isinstance(launcher_pid, int)
        or launcher_pid <= 0
        or expected_create_time is None
    ):
        raise RuntimeError("Run report has no verifiable launcher identity")
    actual_create_time = process_create_time(launcher_pid)
    if (
        actual_create_time is None
        or abs(actual_create_time - float(expected_create_time)) >= 0.01
    ):
        raise RuntimeError(
            "Recorded launcher is no longer running; refusing to signal stale job PIDs"
        )

    signalled = []
    for job in payload.get("jobs", []):
        pid = job.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            signalled.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            raise RuntimeError(f"Permission denied while signalling PID {pid}") from exc

    if launcher_pid != os.getpid():
        try:
            os.kill(launcher_pid, signal.SIGINT)
            signalled.append(launcher_pid)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise RuntimeError(
                f"Permission denied while signalling launcher PID {launcher_pid}"
            ) from exc
    return {"already_terminal": False, "status": "stopping", "pids": signalled}


def read_report_logs(
    payload: Mapping[str, Any],
    *,
    job_name: Optional[str] = None,
    lines: int = 100,
) -> List[Dict[str, Any]]:
    if lines <= 0:
        raise ValueError("--lines must be positive")
    logs = []
    for job in payload.get("jobs", []):
        if job_name is not None and job.get("name") != job_name:
            continue
        raw_path = job.get("log_path")
        if not isinstance(raw_path, str):
            continue
        path = Path(raw_path).expanduser().resolve()
        try:
            content_lines = path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError as exc:
            raise RuntimeError(f"Cannot read log {path}: {exc}") from exc
        selected = content_lines[-lines:]
        logs.append(
            {
                "name": job.get("name"),
                "host": job.get("host"),
                "path": str(path),
                "content": "\n".join(selected) + ("\n" if selected else ""),
            }
        )
    if job_name is not None and not logs:
        raise ValueError(f"No log found for job {job_name!r}")
    return logs


def enrich_report(
    payload: Mapping[str, Any],
    *,
    profile: bool = False,
) -> Dict[str, Any]:
    """Add live memory, cooperative throughput, and liveness to a snapshot."""

    snapshot = copy.deepcopy(dict(payload))
    memory = query_gpu_process_memory()
    process_utilization: Dict[Any, Dict[str, Any]] = {}
    device_telemetry: List[Dict[str, Any]] = []
    if profile:
        from .telemetry import (
            query_dcgm_profile,
            query_device_utilization,
            query_process_utilization,
        )

        process_utilization = query_process_utilization()
        devices = sorted(
            {
                int(job["device"])
                for job in snapshot.get("jobs", [])
                if isinstance(job.get("device"), int) and not job.get("remote")
            }
        )
        device_telemetry = query_device_utilization(devices)
        for record in device_telemetry:
            record["dcgm"] = query_dcgm_profile(int(record["device"]))
    for job in snapshot.get("jobs", []):
        if job.get("remote"):
            job["observed_status"] = job.get("observed_status") or job.get(
                "status", "unknown"
            )
            continue
        pid = job.get("pid")
        device = job.get("device")
        utilization = (
            process_utilization.get((device, pid))
            if isinstance(device, int) and isinstance(pid, int)
            else None
        )
        if utilization is not None:
            job["sm_util_percent"] = utilization.get("sm_util_percent")
            job["memory_util_percent"] = utilization.get("memory_util_percent")
            job["utilization_source"] = utilization.get("source")
        if isinstance(pid, int) and pid in memory:
            job["gpu_memory_mib"] = memory[pid]
            job["gpu_memory_source"] = "nvidia-smi"
            peak = int(job.get("peak_gpu_memory_mib") or 0)
            job["peak_gpu_memory_mib"] = max(peak, memory[pid])

        metrics_path = job.get("metrics_path")
        if isinstance(metrics_path, str):
            metric = read_throughput_metric(Path(metrics_path))
            if metric is not None:
                job["throughput"] = metric["throughput"]
                job["throughput_unit"] = metric["unit"]
                job["throughput_updated_at"] = metric.get("updated_at")
                if not (isinstance(pid, int) and pid in memory) and not (
                    job.get("gpu_memory_source") == "nvidia-smi"
                    and job.get("gpu_memory_mib") is not None
                ):
                    fallback_memory = metric.get("gpu_memory_mib")
                    if isinstance(fallback_memory, (int, float)):
                        job["gpu_memory_mib"] = int(fallback_memory)
                        job["gpu_memory_source"] = metric.get(
                            "gpu_memory_source", "cooperative_metric"
                        )
                        peak = int(job.get("peak_gpu_memory_mib") or 0)
                        job["peak_gpu_memory_mib"] = max(
                            peak, int(fallback_memory)
                        )

        if (
            job.get("status") in ("starting", "running")
            and isinstance(pid, int)
            and not _pid_is_alive(pid)
        ):
            job["observed_status"] = "exited"
        else:
            job["observed_status"] = job.get("status", "unknown")
    snapshot["observed_at"] = utc_now()
    if profile:
        snapshot["telemetry"] = {
            "profile": True,
            "devices": device_telemetry,
            "process_sample_count": len(process_utilization),
        }
    return snapshot


class RunReport:
    """Thread-safe JSON state shared by all local GPU launch groups."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        config_path: Optional[Path] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.run_id = run_id
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "preparing",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "ended_at": None,
            "config_path": str(config_path.resolve()) if config_path else None,
            "report_path": str(self.path),
            "calibration": None,
            "metadata": dict(metadata or {}),
            "jobs": [],
        }
        self._write_locked()
        register_report(run_id, self.path)

    def _write_locked(self) -> None:
        self._data["updated_at"] = utc_now()
        _atomic_write_json(self.path, self._data)

    def add_jobs(self, jobs: Iterable[Mapping[str, Any]]) -> None:
        with self._lock:
            existing = {job["name"] for job in self._data["jobs"]}
            for raw_job in jobs:
                job = dict(raw_job)
                name = job.get("name")
                if not isinstance(name, str) or not name or name in existing:
                    raise ValueError(f"Duplicate or invalid report job name: {name!r}")
                existing.add(name)
                job.setdefault("status", "pending")
                job.setdefault("pid", None)
                job.setdefault("sm_count", None)
                job.setdefault("gpu_memory_mib", None)
                job.setdefault("peak_gpu_memory_mib", 0)
                job.setdefault("throughput", None)
                job.setdefault("throughput_unit", None)
                job.setdefault("exit_code", None)
                self._data["jobs"].append(job)
            self._write_locked()

    def update_job(self, name: str, **changes: Any) -> None:
        with self._lock:
            for job in self._data["jobs"]:
                if job.get("name") == name:
                    job.update(changes)
                    self._write_locked()
                    return
        raise KeyError(f"Unknown nVertake job: {name}")

    def update_jobs(self, changes: Mapping[str, Mapping[str, Any]]) -> None:
        """Update several jobs with one atomic report write."""

        with self._lock:
            remaining = set(changes)
            for job in self._data["jobs"]:
                name = job.get("name")
                if name in changes:
                    job.update(changes[name])
                    remaining.discard(name)
            if remaining:
                raise KeyError(
                    "Unknown nVertake jobs: " + ", ".join(sorted(remaining))
                )
            self._write_locked()

    def set_status(self, status: str, **changes: Any) -> None:
        with self._lock:
            self._data["status"] = status
            self._data.update(changes)
            if status in TERMINAL_RUN_STATES:
                self._data["ended_at"] = utc_now()
            self._write_locked()

    def set_calibration(self, calibration: Mapping[str, Any]) -> None:
        with self._lock:
            self._data["calibration"] = dict(calibration)
            self._write_locked()

    def mark_jobs_cancelled(self) -> None:
        """Convert active or signal-failed jobs to a terminal cancelled state."""

        with self._lock:
            ended_at = utc_now()
            changed = False
            for job in self._data["jobs"]:
                status = job.get("status")
                exit_code = job.get("exit_code")
                signal_failed = (
                    status == "failed"
                    and isinstance(exit_code, int)
                    and exit_code < 0
                )
                if status not in TERMINAL_JOB_STATES or signal_failed:
                    job["status"] = "cancelled"
                    job["ended_at"] = job.get("ended_at") or ended_at
                    changed = True
            if changed:
                self._write_locked()

    def mark_unfinished_failed(self, error: str) -> None:
        """Mark jobs that never reached a terminal state after launcher failure."""

        with self._lock:
            ended_at = utc_now()
            changed = False
            for job in self._data["jobs"]:
                if job.get("status") not in TERMINAL_JOB_STATES:
                    job["status"] = "failed"
                    job["error"] = error
                    job["ended_at"] = ended_at
                    changed = True
            if changed:
                self._write_locked()

    def set_metadata(self, **changes: Any) -> None:
        with self._lock:
            self._data["metadata"].update(changes)
            self._write_locked()

    def refresh_live_stats(self) -> None:
        memory = query_gpu_process_memory()
        with self._lock:
            changed = False
            for job in self._data["jobs"]:
                pid = job.get("pid")
                if isinstance(pid, int) and pid in memory:
                    used = memory[pid]
                    job["gpu_memory_mib"] = used
                    job["gpu_memory_source"] = "nvidia-smi"
                    job["peak_gpu_memory_mib"] = max(
                        int(job.get("peak_gpu_memory_mib") or 0), used
                    )
                    changed = True
                metrics_path = job.get("metrics_path")
                if isinstance(metrics_path, str):
                    metric = read_throughput_metric(Path(metrics_path))
                    if metric is not None:
                        job["throughput"] = metric["throughput"]
                        job["throughput_unit"] = metric["unit"]
                        job["throughput_updated_at"] = metric.get("updated_at")
                        if not (isinstance(pid, int) and pid in memory) and not (
                            job.get("gpu_memory_source") == "nvidia-smi"
                            and job.get("gpu_memory_mib") is not None
                        ):
                            fallback_memory = metric.get("gpu_memory_mib")
                            if isinstance(fallback_memory, (int, float)):
                                used = int(fallback_memory)
                                job["gpu_memory_mib"] = used
                                job["gpu_memory_source"] = metric.get(
                                    "gpu_memory_source", "cooperative_metric"
                                )
                                job["peak_gpu_memory_mib"] = max(
                                    int(job.get("peak_gpu_memory_mib") or 0), used
                                )
                        changed = True
            if changed:
                self._write_locked()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)


def format_monitor_table(payload: Mapping[str, Any]) -> str:
    jobs: List[Mapping[str, Any]] = list(payload.get("jobs", []))
    show_host = any(job.get("host") for job in jobs)
    show_utilization = any(
        job.get("sm_util_percent") is not None
        or job.get("memory_util_percent") is not None
        for job in jobs
    )
    headers = ["NAME", "GPU", "PID", "SM"]
    if show_utilization:
        headers.extend(("SM%", "MEM%"))
    headers.extend(("VRAM MiB", "THROUGHPUT", "STATUS"))
    if show_host:
        headers.insert(0, "HOST")
    rows = []
    for job in jobs:
        throughput = "-"
        if job.get("throughput") is not None:
            throughput = f"{float(job['throughput']):.3g} {job.get('throughput_unit') or ''}".strip()
        values_list = [
            str(job.get("name", "-")),
            str(job.get("device", "-")),
            str(job.get("pid") if job.get("pid") is not None else "-"),
            str(job.get("sm_count") if job.get("sm_count") is not None else "-"),
        ]
        if show_utilization:
            values_list.extend(
                (
                    (
                        f"{float(job['sm_util_percent']):.1f}"
                        if job.get("sm_util_percent") is not None
                        else "-"
                    ),
                    (
                        f"{float(job['memory_util_percent']):.1f}"
                        if job.get("memory_util_percent") is not None
                        else "-"
                    ),
                )
            )
        values_list.extend(
            (
            str(
                job.get("gpu_memory_mib")
                if job.get("gpu_memory_mib") is not None
                else "-"
            ),
            throughput,
            str(job.get("observed_status") or job.get("status", "unknown")),
            )
        )
        values = tuple(values_list)
        if show_host:
            values = (str(job.get("host", "-")),) + values
        rows.append(values)
    header_values = tuple(headers)
    widths = [len(value) for value in header_values]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    line = "  ".join(
        value.ljust(widths[index]) for index, value in enumerate(header_values)
    )
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    title = f"run={payload.get('run_id', '-')} status={payload.get('status', 'unknown')}"
    return "\n".join([title, line, separator] + body)
