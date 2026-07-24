"""JSON-over-stdio worker used by nVertake's SSH orchestration."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from .diagnostics import inspect_green_device, inspect_scheduler_capabilities
from .jobs import (
    CalibrationSpec,
    JobConfig,
    JobSpec,
    _validate_memory_groups,
    launch_jobs,
    plan_job_config,
)
from .runtime import (
    TERMINAL_RUN_STATES,
    enrich_report,
    load_report,
    reconcile_report,
    utc_now,
)
from .signals import launch_signal_handlers


def _git_metadata(repo: Path) -> Dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo)] + list(arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        dirty_output = run("status", "--porcelain")
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError(f"Cannot inspect Git repository {repo}: {exc}") from exc
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty_output),
    }


def _gpu_inventory() -> Tuple[Dict[str, Any], ...]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError(f"Cannot query NVIDIA GPUs with nvidia-smi: {exc}") from exc

    devices = []
    for raw_line in result.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",", 4)]
        if len(fields) != 5:
            continue
        try:
            index = int(fields[0])
            total = int(float(fields[2]))
            free = int(float(fields[3]))
        except ValueError as exc:
            raise RuntimeError(f"Invalid nvidia-smi GPU row: {raw_line}") from exc
        diagnostics = inspect_green_device(index)
        device = {
            "index": index,
            "name": fields[1],
            "memory_total_mib": total,
            "memory_free_mib": free,
            "uuid": fields[4],
        }
        device.update(diagnostics.to_dict())
        device["scheduling_capabilities"] = inspect_scheduler_capabilities(
            diagnostics
        )
        devices.append(device)
    if not devices:
        raise RuntimeError("nvidia-smi did not report any NVIDIA GPUs")
    return tuple(devices)


def probe(repo_value: str) -> Dict[str, Any]:
    repo = Path(repo_value).expanduser().resolve()
    if not repo.is_dir():
        raise NotADirectoryError(f"Repository directory not found: {repo}")
    try:
        import torch

        torch_metadata: Dict[str, Any] = {
            "version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except (ImportError, OSError, RuntimeError) as exc:
        torch_metadata = {"error": str(exc)}
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "torch": torch_metadata,
        "repo_path": str(repo),
        "git": _git_metadata(repo),
        "gpus": list(_gpu_inventory()),
        "probed_at": utc_now(),
    }


def _resolve_from(base: Path, raw_value: str) -> Path:
    path = Path(raw_value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _calibration(payload: Mapping[str, Any]) -> CalibrationSpec:
    return CalibrationSpec(
        enabled=bool(payload.get("enabled", False)),
        rounds=int(payload.get("rounds", 2)),
        duration=float(payload.get("duration", 5.0)),
        tolerance=float(payload.get("tolerance", 0.05)),
        damping=float(payload.get("damping", 0.5)),
        warmup=float(payload.get("warmup", 0.0)),
        minimum_samples=int(payload.get("minimum_samples", 1)),
        sample_window=int(payload.get("sample_window", 20)),
        timeout_grace=float(payload.get("timeout_grace", 10.0)),
    )


def _materialize_config(payload: Mapping[str, Any]) -> JobConfig:
    repo = Path(str(payload["repo"])).expanduser().resolve()
    host_name = str(payload["host_name"])
    jobs = []
    for raw_job in payload["jobs"]:
        cwd = _resolve_from(repo, str(raw_job["cwd"]))
        script = _resolve_from(cwd, str(raw_job["script"]))
        if not cwd.is_dir():
            raise NotADirectoryError(f"Working directory not found: {cwd}")
        if not script.is_file():
            raise FileNotFoundError(f"Python script not found: {script}")
        raw_log = raw_job.get("log")
        jobs.append(
            JobSpec(
                name=str(raw_job["name"]),
                script=script,
                args=tuple(str(value) for value in raw_job.get("args", ())),
                calibration_args=tuple(
                    str(value) for value in raw_job.get("calibration_args", ())
                ),
                sm_share=float(raw_job["sm_share"]),
                target_share=float(raw_job["target_share"]),
                memory_share=(
                    float(raw_job["memory_share"])
                    if raw_job.get("memory_share") is not None
                    else None
                ),
                device=int(raw_job["device"]),
                env={
                    str(key): str(value)
                    for key, value in dict(raw_job.get("env", {})).items()
                },
                cwd=cwd,
                log=Path(str(raw_log)).expanduser() if raw_log is not None else None,
                host=host_name,
                work_queue_connections=(
                    int(raw_job["work_queue_connections"])
                    if raw_job.get("work_queue_connections") is not None
                    else None
                ),
            )
        )
    _validate_memory_groups(jobs)
    report_path = Path(str(payload["report_path"])).expanduser().resolve()
    return JobConfig(
        path=report_path.with_name("source-jobs.yaml"),
        jobs=tuple(jobs),
        logs_dir=repo / ".nvertake" / "runs",
        report=report_path,
        startup_timeout=float(payload["startup_timeout"]),
        calibration=_calibration(dict(payload.get("calibration", {}))),
        hosts={},
        git_check=True,
        lease_timeout=float(payload.get("lease_timeout", 0.0)),
        backend=str(payload.get("backend", "green")),
    )


def _assert_expected_git(payload: Mapping[str, Any], repo: Path) -> Dict[str, Any]:
    actual = _git_metadata(repo)
    expected = payload.get("expected_git_commit")
    if expected and actual["commit"] != expected:
        raise RuntimeError(
            f"Git commit mismatch: expected {expected}, found {actual['commit']}"
        )
    if payload.get("require_clean", False) and actual["dirty"]:
        raise RuntimeError(f"Remote repository has working-tree changes: {repo}")
    return actual


def _process_start_ticks(pid: int) -> Optional[str]:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        raw = stat_path.read_text(encoding="utf-8")
    except OSError:
        return None
    closing = raw.rfind(")")
    fields = raw[closing + 2 :].split() if closing >= 0 else []
    return fields[19] if len(fields) > 19 else None


def _report_path(payload: Mapping[str, Any]) -> Path:
    repo = Path(str(payload["repo"])).expanduser().resolve()
    report_path = Path(str(payload["report_path"])).expanduser().resolve()
    runtime_root = (repo / ".nvertake" / "runs").resolve()
    try:
        report_path.relative_to(runtime_root)
    except ValueError as exc:
        raise ValueError(
            f"Report path must be inside {runtime_root}: {report_path}"
        ) from exc
    return report_path


def _snapshot(payload: Mapping[str, Any]) -> Dict[str, Any]:
    report_path = _report_path(payload)
    return enrich_report(
        load_report(report_path),
        profile=bool(payload.get("profile", False)),
    )


def _launch_result_from_snapshot(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    status = str(snapshot.get("status", "failed"))
    exit_code = 0 if status == "completed" else 130 if status == "cancelled" else 1
    return {
        "exit_code": exit_code,
        "run_id": snapshot.get("run_id"),
        "report_path": snapshot.get("report_path"),
        "snapshot": dict(snapshot),
        "reattached": True,
    }


def _existing_launch(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Reattach a repeated launch request to its existing host-local run."""

    report_path = _report_path(payload)
    if not report_path.is_file():
        return None
    snapshot = load_report(report_path)
    if snapshot.get("run_id") != payload.get("run_id"):
        raise RuntimeError("Existing remote report has a different run id")

    while snapshot.get("status") not in TERMINAL_RUN_STATES:
        metadata = dict(snapshot.get("metadata") or {})
        launcher_pid = metadata.get("launcher_pid")
        expected_ticks = metadata.get("launcher_start_ticks")
        if (
            not isinstance(launcher_pid, int)
            or launcher_pid <= 0
            or expected_ticks is None
            or _process_start_ticks(launcher_pid) != str(expected_ticks)
        ):
            snapshot = reconcile_report(report_path)
            break
        time.sleep(0.25)
        snapshot = load_report(report_path)
    return _launch_result_from_snapshot(enrich_report(snapshot))


@contextmanager
def _launch_claim(payload: Mapping[str, Any]) -> Iterator[bool]:
    """Serialize duplicate SSH launch requests before the report exists."""

    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("Remote launch claims require POSIX file locking") from exc
    report_path = _report_path(payload)
    claim_path = report_path.with_name(f".{report_path.name}.launch.lock")
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    stream = claim_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            acquired = False
        yield acquired
    finally:
        if acquired:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _launch_fresh(payload: Mapping[str, Any]) -> Dict[str, Any]:
    config = _materialize_config(payload)
    repo = Path(str(payload["repo"])).expanduser().resolve()
    git = _assert_expected_git(payload, repo)
    with launch_signal_handlers():
        result = launch_jobs(
            config,
            quiet=True,
            run_id=str(payload["run_id"]),
            metadata={
                "host": str(payload["host_name"]),
                "launcher_start_ticks": _process_start_ticks(os.getpid()),
                "git": git,
                "config_sha256": payload.get("config_sha256"),
                "source_git_commit": payload.get("expected_git_commit"),
            },
        )
    return {
        "exit_code": result.exit_code,
        "run_id": result.run_id,
        "report_path": str(result.report_path),
        "snapshot": enrich_report(load_report(result.report_path)),
        "reattached": False,
    }


def _signal_if_alive(pid: Any, requested_signal: int) -> bool:
    if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, requested_signal)
        return True
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise RuntimeError(f"Permission denied while signalling PID {pid}") from exc


def _stop(payload: Mapping[str, Any]) -> Dict[str, Any]:
    snapshot = _snapshot(payload)
    if snapshot.get("run_id") != payload.get("run_id"):
        raise RuntimeError("Run id does not match the remote report")
    if snapshot.get("status") in TERMINAL_RUN_STATES:
        return {"already_terminal": True, "status": snapshot.get("status"), "pids": []}

    metadata = dict(snapshot.get("metadata") or {})
    launcher_pid = metadata.get("launcher_pid")
    expected_ticks = metadata.get("launcher_start_ticks")
    if (
        not isinstance(launcher_pid, int)
        or launcher_pid <= 0
        or expected_ticks is None
        or _process_start_ticks(launcher_pid) != str(expected_ticks)
    ):
        raise RuntimeError(
            "Recorded launcher is no longer running; refusing to signal stale job PIDs"
        )

    signalled = []
    for job in snapshot.get("jobs", []):
        pid = job.get("pid")
        if _signal_if_alive(pid, signal.SIGTERM):
            signalled.append(pid)

    time.sleep(0.2)
    if _signal_if_alive(launcher_pid, signal.SIGINT):
        signalled.append(launcher_pid)
    return {"already_terminal": False, "status": "stopping", "pids": signalled}


def _tail_lines(path: Path, count: int) -> str:
    if count <= 0:
        raise ValueError("lines must be positive")
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            lines = deque(stream, maxlen=count)
    except OSError as exc:
        raise RuntimeError(f"Cannot read log {path}: {exc}") from exc
    return "".join(lines)


def _logs(payload: Mapping[str, Any]) -> Dict[str, Any]:
    snapshot = _snapshot(payload)
    requested_name = payload.get("job")
    count = int(payload.get("lines", 100))
    logs = []
    for job in snapshot.get("jobs", []):
        if requested_name is not None and job.get("name") != requested_name:
            continue
        raw_path = job.get("log_path")
        if not isinstance(raw_path, str):
            continue
        path = Path(raw_path).expanduser().resolve()
        logs.append(
            {
                "name": job.get("name"),
                "host": job.get("host"),
                "path": str(path),
                "content": _tail_lines(path, count),
            }
        )
    if requested_name is not None and not logs:
        raise ValueError(f"No log found for job {requested_name!r}")
    return {"logs": logs}


def handle(action: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    if action == "probe":
        return probe(str(payload["repo"]))
    if action == "plan":
        config = _materialize_config(payload)
        return plan_job_config(
            config,
            reveal_secrets=bool(payload.get("reveal_secrets", False)),
        )
    if action == "launch":
        while True:
            existing = _existing_launch(payload)
            if existing is not None:
                return existing
            with _launch_claim(payload) as acquired:
                if acquired:
                    existing = _existing_launch(payload)
                    if existing is not None:
                        return existing
                    return _launch_fresh(payload)
            time.sleep(0.1)
    if action == "snapshot":
        return _snapshot(payload)
    if action == "stop":
        return _stop(payload)
    if action == "logs":
        return _logs(payload)
    raise ValueError(f"Unsupported remote agent action: {action}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m nvertake.remote_agent")
    parser.add_argument(
        "action", choices=("probe", "plan", "launch", "snapshot", "stop", "logs")
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw or "{}")
        if not isinstance(payload, dict):
            raise ValueError("Agent request must be a JSON object")
        result = handle(args.action, payload)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except BaseException as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
