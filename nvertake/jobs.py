"""YAML job files, multi-GPU launches, logs, and throughput calibration."""

from __future__ import annotations

import math
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from .diagnostics import inspect_green_device, plan_green_partitions
from .green_process import GreenProcessRunResult, launch_green_process_scripts
from .metrics import read_throughput_metric
from .runtime import RunReport, process_create_time, utc_now


@dataclass(frozen=True)
class JobSpec:
    name: str
    script: Path
    args: Tuple[str, ...]
    calibration_args: Tuple[str, ...]
    sm_share: float
    target_share: float
    memory_share: Optional[float]
    device: Optional[int]
    env: Mapping[str, str]
    cwd: Path
    log: Optional[Path]
    host: str = "local"


@dataclass(frozen=True)
class HostSpec:
    """Execution transport and repository settings for one YAML host."""

    name: str
    repo: str
    python: str
    ssh: Optional[str] = None
    ssh_options: Tuple[str, ...] = ()
    local: bool = False


@dataclass(frozen=True)
class CalibrationSpec:
    enabled: bool = False
    rounds: int = 2
    duration: float = 5.0
    tolerance: float = 0.05
    damping: float = 0.5


@dataclass(frozen=True)
class JobConfig:
    path: Path
    jobs: Tuple[JobSpec, ...]
    logs_dir: Path
    report: Optional[Path]
    startup_timeout: float
    calibration: CalibrationSpec
    hosts: Mapping[str, HostSpec]
    git_check: bool = True


@dataclass(frozen=True)
class JobLaunchResult:
    exit_code: int
    run_id: str
    report_path: Path


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    return value


def _as_string_tuple(value: Any, label: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a YAML list")
    if any(isinstance(item, (dict, list)) or item is None for item in value):
        raise ValueError(f"{label} values must be strings or scalars")
    return tuple(str(item) for item in value)


def _as_environment(value: Any, label: str) -> Dict[str, str]:
    mapping = _as_mapping(value, label)
    environment: Dict[str, str] = {}
    for key, item in mapping.items():
        if not isinstance(key, str) or isinstance(item, (dict, list)) or item is None:
            raise ValueError(f"{label} must map string names to scalar values")
        if isinstance(item, bool):
            environment[key] = "1" if item else "0"
        else:
            environment[key] = str(item)
    return environment


def _positive_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return number


def _memory_fraction(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().endswith("%"):
        number = _positive_number(value.strip()[:-1], label) / 100.0
    else:
        number = _positive_number(value, label)
        if number > 1.0:
            number /= 100.0
    if not (0.0 < number <= 1.0):
        raise ValueError(f"{label} must be in (0, 1], or use a percentage up to 100%")
    return number


def _resolve_path(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _as_path(value: Any, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise ValueError(f"{label} must be a non-empty path")
    return Path(str(value))


def _parse_device(value: Any, label: str) -> Optional[int]:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    try:
        device = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer or 'auto'") from exc
    if device < 0:
        raise ValueError(f"{label} must be non-negative or 'auto'")
    return device


def _parse_hosts(
    value: Any,
    *,
    base: Path,
) -> Dict[str, HostSpec]:
    if value is None:
        return {}
    mapping = _as_mapping(value, "hosts")
    if not mapping:
        raise ValueError("hosts must not be empty")

    hosts: Dict[str, HostSpec] = {}
    for raw_name, raw_settings in mapping.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("host names must be non-empty")
        if name in hosts:
            raise ValueError(f"Duplicate host name: {name}")
        settings = _as_mapping(raw_settings, f"hosts.{name}")
        raw_local = settings.get("local", False)
        if not isinstance(raw_local, bool):
            raise ValueError(f"hosts.{name}.local must be true or false")
        local = raw_local
        raw_ssh = settings.get("ssh")
        if raw_ssh is not None and (
            not isinstance(raw_ssh, str) or not raw_ssh.strip()
        ):
            raise ValueError(f"hosts.{name}.ssh must be a non-empty string")
        ssh = raw_ssh.strip() if isinstance(raw_ssh, str) else None
        if not local and ssh is None:
            raise ValueError(f"hosts.{name} requires ssh or local: true")

        raw_repo = settings.get("repo", ".")
        if not isinstance(raw_repo, (str, os.PathLike)) or not str(raw_repo):
            raise ValueError(f"hosts.{name}.repo must be a non-empty path")
        if local:
            repo = str(_resolve_path(base, raw_repo, f"hosts.{name}.repo"))
        else:
            repo = str(raw_repo)

        raw_python = settings.get("python", sys.executable if local else "python")
        if not isinstance(raw_python, str) or not raw_python.strip():
            raise ValueError(f"hosts.{name}.python must be a non-empty string")
        ssh_options = _as_string_tuple(
            settings.get("ssh_options"), f"hosts.{name}.ssh_options"
        )
        hosts[name] = HostSpec(
            name=name,
            repo=repo,
            python=raw_python.strip(),
            ssh=ssh,
            ssh_options=ssh_options,
            local=local,
        )
    return hosts


def _parse_calibration(value: Any) -> CalibrationSpec:
    if value in (None, False):
        return CalibrationSpec()
    if value is True:
        mapping: Mapping[str, Any] = {}
    else:
        mapping = _as_mapping(value, "calibration")
    enabled = bool(mapping.get("enabled", True))
    rounds = int(mapping.get("rounds", 2))
    duration = float(mapping.get("duration", 5.0))
    tolerance = float(mapping.get("tolerance", 0.05))
    damping = float(mapping.get("damping", 0.5))
    if rounds <= 0:
        raise ValueError("calibration.rounds must be positive")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("calibration.duration must be positive")
    if not math.isfinite(tolerance) or not (0 < tolerance < 1):
        raise ValueError("calibration.tolerance must be between 0 and 1")
    if not math.isfinite(damping) or not (0 < damping <= 1):
        raise ValueError("calibration.damping must be in (0, 1]")
    return CalibrationSpec(enabled, rounds, duration, tolerance, damping)


def load_job_config(path: str) -> JobConfig:
    """Load and validate a version-1 nVertake YAML job file."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"YAML job file not found: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML job file {config_path}: {exc}") from exc
    root = _as_mapping(payload, "job file")
    version = root.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported YAML job schema version: {version!r}")

    base = config_path.parent
    hosts = _parse_hosts(root.get("hosts"), base=base)
    defaults = _as_mapping(root.get("defaults"), "defaults")
    default_device = _parse_device(defaults.get("device", 0), "defaults.device")
    raw_default_cwd = defaults.get("cwd", ".")
    default_host_value = defaults.get("host")
    if default_host_value is not None:
        default_host = str(default_host_value).strip()
    elif not hosts:
        default_host = "local"
    elif len(hosts) == 1:
        default_host = next(iter(hosts))
    else:
        default_host = None
    if default_host is not None and hosts and default_host not in hosts:
        raise ValueError(f"Unknown defaults.host: {default_host}")
    if hosts:
        default_cwd = _as_path(raw_default_cwd, "defaults.cwd")
    else:
        default_cwd = _resolve_path(base, raw_default_cwd, "defaults.cwd")
        if not default_cwd.is_dir():
            raise NotADirectoryError(
                f"Default working directory not found: {default_cwd}"
            )
    default_env = _as_environment(defaults.get("env"), "defaults.env")
    default_memory = _memory_fraction(
        defaults.get("memory_share"), "defaults.memory_share"
    )

    raw_jobs = root.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("jobs must be a non-empty YAML list")

    jobs: List[JobSpec] = []
    names = set()
    for index, raw_job in enumerate(raw_jobs):
        item = _as_mapping(raw_job, f"jobs[{index}]")
        raw_host = item.get("host", default_host)
        if raw_host is None:
            raise ValueError(
                f"jobs[{index}] requires host because no defaults.host was set"
            )
        host = str(raw_host).strip()
        if not host:
            raise ValueError(f"jobs[{index}].host must be non-empty")
        if hosts and host not in hosts:
            raise ValueError(f"jobs[{index}] references unknown host: {host}")

        if hosts:
            cwd = _as_path(
                item.get("cwd", default_cwd), f"jobs[{index}].cwd"
            )
            script = _as_path(item.get("script"), f"jobs[{index}].script")
        else:
            cwd = _resolve_path(
                base, item.get("cwd", default_cwd), f"jobs[{index}].cwd"
            )
            if not cwd.is_dir():
                raise NotADirectoryError(f"Working directory not found: {cwd}")
            script = _resolve_path(cwd, item.get("script"), f"jobs[{index}].script")
            if not script.is_file():
                raise FileNotFoundError(f"Python script not found: {script}")

        name_value = item.get("name", script.stem)
        if not isinstance(name_value, str) or not name_value.strip():
            raise ValueError(f"jobs[{index}].name must be a non-empty string")
        name = name_value.strip()
        if name in names:
            raise ValueError(f"Duplicate job name: {name}")
        names.add(name)

        if "device" in item:
            device = _parse_device(item["device"], f"jobs[{index}].device")
        else:
            device = default_device
        raw_sm_share = item.get("sm_share", item.get("share"))
        if raw_sm_share is None:
            raise ValueError(f"jobs[{index}] requires sm_share")
        sm_share = _positive_number(raw_sm_share, f"jobs[{index}].sm_share")
        target_share = _positive_number(
            item.get("target_share", sm_share), f"jobs[{index}].target_share"
        )
        memory_share = _memory_fraction(
            item.get("memory_share", default_memory),
            f"jobs[{index}].memory_share",
        )
        environment = dict(default_env)
        environment.update(_as_environment(item.get("env"), f"jobs[{index}].env"))
        raw_log = item.get("log")
        log = None
        if raw_log is not None:
            if not isinstance(raw_log, str) or not raw_log:
                raise ValueError(f"jobs[{index}].log must be a non-empty path")
            log = Path(raw_log).expanduser()

        jobs.append(
            JobSpec(
                name=name,
                script=script,
                args=_as_string_tuple(item.get("args"), f"jobs[{index}].args"),
                calibration_args=_as_string_tuple(
                    item.get("calibration_args"),
                    f"jobs[{index}].calibration_args",
                ),
                sm_share=sm_share,
                target_share=target_share,
                memory_share=memory_share,
                device=device,
                env=environment,
                cwd=cwd,
                log=log,
                host=host,
            )
        )

    _validate_memory_groups(jobs, allow_auto=True)

    logs_dir = _resolve_path(base, root.get("logs_dir", "nvertake-logs"), "logs_dir")
    report = (
        _resolve_path(base, root["report"], "report")
        if root.get("report") is not None
        else None
    )
    startup_timeout = float(root.get("startup_timeout", 60.0))
    if not math.isfinite(startup_timeout) or startup_timeout <= 0:
        raise ValueError("startup_timeout must be positive")
    raw_git_check = root.get("git_check", True)
    if not isinstance(raw_git_check, bool):
        raise ValueError("git_check must be true or false")
    return JobConfig(
        path=config_path,
        jobs=tuple(jobs),
        logs_dir=logs_dir,
        report=report,
        startup_timeout=startup_timeout,
        calibration=_parse_calibration(root.get("calibration")),
        hosts=hosts,
        git_check=raw_git_check,
    )


def _group_jobs(jobs: Sequence[JobSpec]) -> Dict[int, Tuple[JobSpec, ...]]:
    groups: Dict[int, List[JobSpec]] = {}
    for job in jobs:
        if job.device is None:
            raise ValueError(
                f"Job {job.name!r} still has device: auto; device placement was not run"
            )
        groups.setdefault(job.device, []).append(job)
    return {device: tuple(group) for device, group in sorted(groups.items())}


def _validate_memory_groups(
    jobs: Sequence[JobSpec],
    *,
    allow_auto: bool = False,
) -> None:
    groups: Dict[Tuple[str, int], List[JobSpec]] = {}
    for job in jobs:
        if job.device is None:
            if allow_auto:
                continue
            raise ValueError(f"Job {job.name!r} has no resolved GPU device")
        groups.setdefault((job.host, job.device), []).append(job)

    for (host, device), group in groups.items():
        fractions = [job.memory_share for job in group]
        label = f"{host} GPU {device}" if host != "local" else f"GPU {device}"
        if any(value is not None for value in fractions):
            if any(value is None for value in fractions):
                raise ValueError(
                    f"{label}: memory_share must be set for every job or none"
                )
            total = sum(
                float(value) for value in fractions if value is not None
            )
            if total > 1.0 + 1e-9:
                raise ValueError(
                    f"{label}: memory_share values total more than 1.0"
                )


def assign_auto_devices(
    jobs: Sequence[JobSpec],
    inventories: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Tuple[JobSpec, ...]:
    """Resolve ``device: auto`` using free memory and assigned load."""

    resolved: List[JobSpec] = list(jobs)
    by_host: Dict[str, List[int]] = {}
    for index, job in enumerate(resolved):
        by_host.setdefault(job.host, []).append(index)

    for host, indices in by_host.items():
        inventory = list(inventories.get(host, ()))
        if not inventory:
            raise RuntimeError(f"Host {host!r} did not report any CUDA GPUs")
        devices = {}
        for raw_device in inventory:
            try:
                ordinal = int(raw_device["index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Host {host!r} returned an invalid GPU inventory"
                ) from exc
            total = float(raw_device.get("memory_total_mib") or 0)
            free = float(raw_device.get("memory_free_mib") or total or 0)
            base_pressure = 0.0 if total <= 0 else max(0.0, 1.0 - free / total)
            devices[ordinal] = {
                "base_pressure": base_pressure,
                "memory": 0.0,
                "count": 0,
                "mode": None,
                "max_processes": int(
                    raw_device.get("max_green_processes") or 2**31 - 1
                ),
            }

        for index in indices:
            job = resolved[index]
            if job.device is None:
                continue
            if job.device not in devices:
                raise ValueError(
                    f"Host {host!r} has no CUDA device {job.device}"
                )
            state = devices[job.device]
            mode = job.memory_share is not None
            if state["mode"] is not None and state["mode"] != mode:
                raise ValueError(
                    f"{host} GPU {job.device}: memory_share must be set for "
                    "every job or none"
                )
            state["mode"] = mode
            state["memory"] += float(job.memory_share or 0.0)
            state["count"] += 1
            if state["count"] > state["max_processes"]:
                raise ValueError(
                    f"Host {host!r} GPU {job.device} has "
                    f"{state['count']} jobs but supports at most "
                    f"{state['max_processes']} Green processes"
                )

        auto_indices = [index for index in indices if resolved[index].device is None]
        auto_indices.sort(
            key=lambda index: (
                float(resolved[index].memory_share or 0.0),
                resolved[index].sm_share,
                resolved[index].name,
            ),
            reverse=True,
        )
        for index in auto_indices:
            job = resolved[index]
            mode = job.memory_share is not None
            candidates = []
            for ordinal, state in devices.items():
                if state["count"] >= state["max_processes"]:
                    continue
                if state["mode"] is not None and state["mode"] != mode:
                    continue
                next_memory = state["memory"] + float(job.memory_share or 0.0)
                if mode and next_memory > 1.0 + 1e-9:
                    continue
                candidates.append(
                    (
                        state["base_pressure"] + state["memory"],
                        state["count"],
                        ordinal,
                    )
                )
            if not candidates:
                raise ValueError(
                    f"Host {host!r} has no GPU that can accept auto-placed "
                    f"job {job.name!r} without violating memory_share"
                )
            _pressure, _count, ordinal = min(candidates)
            resolved[index] = replace(job, device=ordinal)
            state = devices[ordinal]
            state["mode"] = mode
            state["memory"] += float(job.memory_share or 0.0)
            state["count"] += 1

    _validate_memory_groups(resolved)
    return tuple(resolved)


def _resolve_local_auto_devices(config: JobConfig) -> JobConfig:
    if not any(job.device is None for job in config.jobs):
        _validate_memory_groups(config.jobs)
        return config
    from .remote_agent import _gpu_inventory

    inventory = _gpu_inventory()
    jobs = assign_auto_devices(config.jobs, {"local": inventory})
    return replace(config, jobs=jobs)


def _normalized(values: Sequence[float]) -> Tuple[float, ...]:
    total = sum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("normalized values must have a positive finite total")
    return tuple(value / total for value in values)


def adjust_shares_for_throughput(
    current_shares: Sequence[float],
    target_shares: Sequence[float],
    observed_throughput: Sequence[float],
    *,
    damping: float = 0.5,
) -> Tuple[float, ...]:
    """Apply a damped multiplicative correction to SM allocation weights."""

    if not (
        len(current_shares) == len(target_shares) == len(observed_throughput)
        and current_shares
    ):
        raise ValueError("current, target, and observed values must have equal lengths")
    current = tuple(_positive_number(value, "current share") for value in current_shares)
    target = _normalized(
        tuple(_positive_number(value, "target share") for value in target_shares)
    )
    observed = _normalized(
        tuple(_positive_number(value, "observed throughput") for value in observed_throughput)
    )
    if not (0 < damping <= 1):
        raise ValueError("damping must be in (0, 1]")
    corrected = tuple(
        current[index] * (target[index] / observed[index]) ** damping
        for index in range(len(current))
    )
    normalized = _normalized(corrected)
    return tuple(value * 100.0 for value in normalized)


def plan_job_config(config: JobConfig) -> Dict[str, Any]:
    """Preview every GPU group without creating a context or process."""

    if config.hosts:
        from .orchestration import plan_distributed_jobs

        return plan_distributed_jobs(config)
    config = _resolve_local_auto_devices(config)

    devices = []
    for device, group in _group_jobs(config.jobs).items():
        if len(group) == 1:
            diagnostics = inspect_green_device(device)
            lanes = [
                {
                    "index": 0,
                    "requested_share": 100.0,
                    "sm_count": diagnostics.total_sm_count,
                    "total_sm_count": diagnostics.total_sm_count,
                    "actual_sm_share": 1.0,
                }
            ]
        else:
            plan = plan_green_partitions(
                tuple(job.sm_share for job in group), device=device
            )
            diagnostics = plan.diagnostics
            lanes = [lane.to_dict() for lane in plan.lanes]
        planned_jobs = []
        for job, lane in zip(group, lanes):
            planned_jobs.append(
                {
                    "name": job.name,
                    "script": str(job.script),
                    "args": list(job.args),
                    "cwd": str(job.cwd),
                    "env": dict(job.env),
                    "host": job.host,
                    "device": device,
                    "requested_sm_share": job.sm_share,
                    "memory_share": job.memory_share,
                    "sm_count": lane["sm_count"],
                    "actual_sm_share": lane["actual_sm_share"],
                }
            )
        devices.append(
            {
                "device": diagnostics.to_dict(),
                "jobs": planned_jobs,
            }
        )
    return {
        "dry_run": True,
        "creates_contexts": False,
        "starts_processes": False,
        "config_path": str(config.path),
        "devices": devices,
    }


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return cleaned or "job"


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _group_paths(
    group: Sequence[JobSpec],
    run_dir: Path,
    *,
    calibration_round: Optional[int] = None,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    log_paths = []
    metrics_paths = []
    for index, job in enumerate(group):
        safe = f"{index:02d}-{_safe_name(job.name)}"
        if calibration_round is None:
            if job.log is None:
                log_path = run_dir / f"{safe}.log"
            elif job.log.is_absolute():
                log_path = job.log
            else:
                log_path = run_dir / job.log
            metrics_path = run_dir / "metrics" / f"{safe}.json"
        else:
            root = run_dir / "calibration" / f"round-{calibration_round}"
            log_path = root / f"gpu-{job.device}-{safe}.log"
            metrics_path = root / "metrics" / f"gpu-{job.device}-{safe}.json"
        log_paths.append(str(log_path.resolve()))
        metrics_paths.append(str(metrics_path.resolve()))
    return tuple(log_paths), tuple(metrics_paths)


def _launch_group(
    group: Sequence[JobSpec],
    shares: Mapping[str, float],
    *,
    startup_timeout: float,
    log_paths: Sequence[str],
    metrics_paths: Sequence[str],
    quiet: bool,
    calibration: Optional[CalibrationSpec] = None,
    calibration_round: Optional[int] = None,
    on_ready: Any = None,
    on_exit: Any = None,
    cancel_event: Any = None,
) -> GreenProcessRunResult:
    environments = []
    arguments = []
    for job in group:
        environment = dict(job.env)
        if calibration is not None:
            environment.update(
                {
                    "NVERTAKE_CALIBRATION": "1",
                    "NVERTAKE_CALIBRATION_SECONDS": str(calibration.duration),
                    "NVERTAKE_CALIBRATION_ROUND": str(calibration_round or 1),
                }
            )
            arguments.append(job.args + job.calibration_args)
        else:
            arguments.append(job.args)
        environments.append(environment)
    group_shares = (
        (100.0,)
        if len(group) == 1
        else tuple(shares[job.name] for job in group)
    )
    return launch_green_process_scripts(
        [str(job.script) for job in group],
        shares=group_shares,
        script_args=arguments,
        device=group[0].device,
        startup_timeout=startup_timeout,
        quiet=quiet,
        memory_shares=[job.memory_share for job in group],
        environments=environments,
        working_directories=[str(job.cwd) for job in group],
        log_paths=log_paths,
        job_names=[job.name for job in group],
        metrics_paths=metrics_paths,
        run_timeout=(calibration.duration if calibration is not None else None),
        timeout_is_success=(calibration is not None),
        on_ready=on_ready,
        on_exit=on_exit,
        cancel_event=cancel_event,
    )


def _wait_futures(
    futures: Mapping[int, Future],
    *,
    report: Optional[RunReport] = None,
) -> Dict[int, GreenProcessRunResult]:
    pending = set(futures.values())
    while pending:
        pending = {future for future in pending if not future.done()}
        if report is not None:
            report.refresh_live_stats()
        if pending:
            time.sleep(0.2)
    return {device: future.result() for device, future in futures.items()}


def _calibrate(
    config: JobConfig,
    run_dir: Path,
    *,
    quiet: bool,
    cancel_event: Any = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    calibration = config.calibration
    groups = _group_jobs(config.jobs)
    effective = {job.name: job.sm_share for job in config.jobs}
    records: List[Dict[str, Any]] = []

    for round_index in range(1, calibration.rounds + 1):
        paths: Dict[int, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {}
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures = {}
            for device, group in groups.items():
                log_paths, metrics_paths = _group_paths(
                    group, run_dir, calibration_round=round_index
                )
                paths[device] = (log_paths, metrics_paths)
                futures[device] = executor.submit(
                    _launch_group,
                    group,
                    effective,
                    startup_timeout=config.startup_timeout,
                    log_paths=log_paths,
                    metrics_paths=metrics_paths,
                    quiet=quiet,
                    calibration=calibration,
                    calibration_round=round_index,
                    cancel_event=cancel_event,
                )
            try:
                results = _wait_futures(futures)
            except KeyboardInterrupt:
                if cancel_event is not None:
                    cancel_event.set()
                raise

        round_record: Dict[str, Any] = {"round": round_index, "devices": []}
        all_converged = True
        for device, group in groups.items():
            result = results[device]
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Calibration workload on GPU {device} exited with {result.exit_code}"
                )
            _logs, metrics_paths = paths[device]
            metrics = [read_throughput_metric(Path(path)) for path in metrics_paths]
            missing = [
                group[index].name
                for index, metric in enumerate(metrics)
                if metric is None
            ]
            if missing:
                raise RuntimeError(
                    "Calibration requires report_throughput() updates; no metric from: "
                    + ", ".join(missing)
                )
            units = {str(metric["unit"]) for metric in metrics if metric is not None}
            if len(units) != 1:
                raise RuntimeError(
                    f"GPU {device} calibration jobs reported incompatible units: {sorted(units)}"
                )
            observed = tuple(
                float(metric["throughput"]) for metric in metrics if metric is not None
            )
            observed_ratios = _normalized(observed)
            targets = tuple(job.target_share for job in group)
            target_ratios = _normalized(targets)
            error = max(
                abs(observed_ratios[index] - target_ratios[index])
                for index in range(len(group))
            )
            converged = error <= calibration.tolerance or len(group) == 1
            if not converged:
                adjusted = adjust_shares_for_throughput(
                    tuple(effective[job.name] for job in group),
                    targets,
                    observed,
                    damping=calibration.damping,
                )
                for job, value in zip(group, adjusted):
                    effective[job.name] = value
            all_converged = all_converged and converged
            round_record["devices"].append(
                {
                    "device": device,
                    "partition_sm_counts": list(result.partition_sm_counts),
                    "observed_throughput": list(observed),
                    "throughput_unit": next(iter(units)),
                    "observed_ratios": list(observed_ratios),
                    "target_ratios": list(target_ratios),
                    "maximum_ratio_error": error,
                    "converged": converged,
                    "next_sm_shares": [effective[job.name] for job in group],
                }
            )
        records.append(round_record)
        if all_converged:
            break

    details = {
        "enabled": True,
        "duration": calibration.duration,
        "requested_rounds": calibration.rounds,
        "completed_rounds": len(records),
        "tolerance": calibration.tolerance,
        "damping": calibration.damping,
        "rounds": records,
        "final_sm_shares": dict(effective),
    }
    return effective, details


def launch_jobs(
    config: JobConfig,
    *,
    quiet: bool = False,
    run_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> JobLaunchResult:
    """Launch a validated YAML job configuration."""

    if config.hosts:
        from .orchestration import launch_distributed_jobs

        return launch_distributed_jobs(config, quiet=quiet)
    config = _resolve_local_auto_devices(config)

    run_id = run_id or _new_run_id()
    run_dir = (config.logs_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.report.resolve() if config.report else run_dir / "report.json"
    launch_metadata = {
        "launcher_pid": os.getpid(),
        "launcher_create_time": process_create_time(os.getpid()),
        "python": sys.executable,
    }
    launch_metadata.update(dict(metadata or {}))
    report = RunReport(
        report_path,
        run_id=run_id,
        config_path=config.path,
        metadata=launch_metadata,
    )
    groups = _group_jobs(config.jobs)
    paths: Dict[int, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
        device: _group_paths(group, run_dir) for device, group in groups.items()
    }

    report_jobs = []
    for device, group in groups.items():
        log_paths, metrics_paths = paths[device]
        for job, log_path, metrics_path in zip(group, log_paths, metrics_paths):
            report_jobs.append(
                {
                    "name": job.name,
                    "host": job.host,
                    "device": device,
                    "script": str(job.script),
                    "args": list(job.args),
                    "cwd": str(job.cwd),
                    "requested_sm_share": job.sm_share,
                    "target_throughput_share": job.target_share,
                    "effective_sm_share": job.sm_share,
                    "memory_share": job.memory_share,
                    "log_path": log_path,
                    "metrics_path": metrics_path,
                }
            )
    report.add_jobs(report_jobs)
    if not quiet:
        print(f"nVertake run id: {run_id}", file=sys.stderr)
        print(f"nVertake JSON report: {report_path}", file=sys.stderr)

    effective = {job.name: job.sm_share for job in config.jobs}
    cancel_event = threading.Event()
    try:
        if config.calibration.enabled:
            report.set_status("calibrating")
            effective, calibration_details = _calibrate(
                config, run_dir, quiet=quiet, cancel_event=cancel_event
            )
            report.set_calibration(calibration_details)
            for job in config.jobs:
                report.update_job(
                    job.name,
                    effective_sm_share=effective[job.name],
                )

        report.set_status("running")
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures: Dict[int, Future] = {}
            for device, group in groups.items():
                log_paths, metrics_paths = paths[device]
                for job in group:
                    report.update_job(job.name, status="starting")

                def on_ready(
                    metadata: Tuple[Dict[str, Any], ...],
                    sm_counts: Tuple[int, ...],
                    *,
                    current_group: Tuple[JobSpec, ...] = group,
                ) -> None:
                    total_sms = sum(sm_counts)
                    for index, job in enumerate(current_group):
                        report.update_job(
                            job.name,
                            status="running",
                            pid=int(metadata[index]["pid"]),
                            sm_count=sm_counts[index],
                            actual_sm_share=sm_counts[index] / total_sms,
                            started_at=utc_now(),
                        )

                def on_exit(
                    lane_index: int,
                    return_code: Optional[int],
                    status: str,
                    *,
                    current_group: Tuple[JobSpec, ...] = group,
                ) -> None:
                    job = current_group[lane_index]
                    report.update_job(
                        job.name,
                        status=status,
                        exit_code=return_code,
                        ended_at=utc_now(),
                    )

                futures[device] = executor.submit(
                    _launch_group,
                    group,
                    effective,
                    startup_timeout=config.startup_timeout,
                    log_paths=log_paths,
                    metrics_paths=metrics_paths,
                    quiet=quiet,
                    on_ready=on_ready,
                    on_exit=on_exit,
                    cancel_event=cancel_event,
                )
            try:
                results = _wait_futures(futures, report=report)
            except KeyboardInterrupt:
                cancel_event.set()
                raise

        report.refresh_live_stats()
        exit_code = next(
            (
                result.exit_code
                for _device, result in sorted(results.items())
                if result.exit_code != 0
            ),
            0,
        )
        report.set_status("completed" if exit_code == 0 else "failed")
        return JobLaunchResult(exit_code, run_id, report_path)
    except KeyboardInterrupt:
        cancel_event.set()
        report.set_status("cancelled", error="Interrupted by user")
        raise
    except BaseException as exc:
        report.set_status("failed", error=str(exc))
        raise
