"""Cross-machine YAML orchestration over SSH."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .jobs import (
    HostSpec,
    JobConfig,
    JobLaunchResult,
    JobSpec,
    _new_run_id,
    assign_auto_devices,
)
from .runtime import (
    RunReport,
    TERMINAL_RUN_STATES,
    enrich_report,
    utc_now,
)


def _repository_git_metadata(repo: Path) -> Dict[str, Any]:
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
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError(f"Cannot inspect local Git repository {repo}: {exc}") from exc


def _source_metadata(config: JobConfig) -> Dict[str, Any]:
    source_repo = Path(__file__).resolve().parent.parent
    return {
        "git": _repository_git_metadata(source_repo),
        "config_sha256": hashlib.sha256(config.path.read_bytes()).hexdigest(),
        "coordinator": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": {
                "executable": sys.executable,
                "version": platform.python_version(),
            },
        },
    }


def _remote_repo_expression(value: str) -> str:
    if value == "~":
        return '"$HOME"'
    if value.startswith("~/"):
        return '"$HOME"/' + shlex.quote(value[2:])
    return shlex.quote(value)


def _host_command(host: HostSpec, action: str) -> Tuple[List[str], Optional[Path]]:
    if host.local:
        return (
            [host.python, "-m", "nvertake.remote_agent", action],
            Path(host.repo).expanduser().resolve(),
        )
    remote_command = (
        f"cd -- {_remote_repo_expression(host.repo)} && "
        f"exec {shlex.quote(host.python)} -m nvertake.remote_agent "
        f"{shlex.quote(action)}"
    )
    return (
        ["ssh"] + list(host.ssh_options) + [str(host.ssh), remote_command],
        None,
    )


def call_host(
    host: HostSpec,
    action: str,
    payload: Mapping[str, Any],
    *,
    timeout: Optional[float] = 30.0,
) -> Dict[str, Any]:
    command, cwd = _host_command(host, action)
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError(f"Host {host.name!r} {action} failed: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no error output"
        raise RuntimeError(
            f"Host {host.name!r} {action} exited with {result.returncode}: {detail}"
        )
    try:
        payload_result = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Host {host.name!r} {action} returned invalid JSON: "
            f"{result.stdout[-500:]}"
        ) from exc
    if not isinstance(payload_result, dict):
        raise RuntimeError(f"Host {host.name!r} {action} returned a non-object")
    return payload_result


def _probe_hosts(config: JobConfig) -> Dict[str, Dict[str, Any]]:
    used_names = tuple(
        name for name in config.hosts if any(job.host == name for job in config.jobs)
    )
    with ThreadPoolExecutor(max_workers=len(used_names)) as executor:
        futures = {
            name: executor.submit(
                call_host,
                host,
                "probe",
                {"repo": host.repo},
                timeout=30.0,
            )
            for name, host in config.hosts.items()
            if name in used_names
        }
        return {name: future.result() for name, future in futures.items()}


def _check_git(
    config: JobConfig,
    source: Mapping[str, Any],
    probes: Mapping[str, Mapping[str, Any]],
) -> None:
    if not config.git_check:
        return
    source_git = dict(source["git"])
    if source_git.get("dirty"):
        raise RuntimeError(
            "The coordinator repository has working-tree changes; commit and sync them "
            "before a remote launch, or set git_check: false"
        )
    expected = source_git.get("commit")
    mismatches = []
    for name, probe in probes.items():
        remote_git = dict(probe.get("git") or {})
        if remote_git.get("commit") != expected:
            mismatches.append(
                f"{name}: {remote_git.get('commit') or 'unknown'}"
            )
        elif remote_git.get("dirty"):
            mismatches.append(f"{name}: matching commit but working-tree changes exist")
    if mismatches:
        raise RuntimeError(
            f"Remote Git state must match {expected}; " + "; ".join(mismatches)
        )


def _serialize_job(job: JobSpec) -> Dict[str, Any]:
    if job.device is None:
        raise ValueError(f"Job {job.name!r} has unresolved device: auto")
    return {
        "name": job.name,
        "script": str(job.script),
        "args": list(job.args),
        "calibration_args": list(job.calibration_args),
        "sm_share": job.sm_share,
        "target_share": job.target_share,
        "memory_share": job.memory_share,
        "device": job.device,
        "env": dict(job.env),
        "cwd": str(job.cwd),
        "log": str(job.log) if job.log is not None else None,
    }


def _report_path(probe: Mapping[str, Any], run_id: str, host_name: str) -> str:
    safe_host = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in host_name
    ).strip(".-") or "host"
    return str(
        Path(str(probe["repo_path"]))
        / ".nvertake"
        / "runs"
        / run_id
        / f"report-{safe_host}.json"
    )


def _host_payload(
    config: JobConfig,
    host: HostSpec,
    jobs: Sequence[JobSpec],
    probe: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    run_id: str,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "host_name": host.name,
        "repo": str(probe["repo_path"]),
        "report_path": _report_path(probe, run_id, host.name),
        "startup_timeout": config.startup_timeout,
        "calibration": asdict(config.calibration),
        "jobs": [_serialize_job(job) for job in jobs],
        "expected_git_commit": source["git"].get("commit"),
        "require_clean": bool(config.git_check),
        "config_sha256": source["config_sha256"],
    }


def _group_by_host(jobs: Sequence[JobSpec]) -> Dict[str, Tuple[JobSpec, ...]]:
    groups: Dict[str, List[JobSpec]] = {}
    for job in jobs:
        groups.setdefault(job.host, []).append(job)
    return {name: tuple(group) for name, group in groups.items()}


def _prepare(
    config: JobConfig,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Tuple[JobSpec, ...]]:
    source = _source_metadata(config)
    probes = _probe_hosts(config)
    _check_git(config, source, probes)
    inventory = {
        name: tuple(probe.get("gpus", ())) for name, probe in probes.items()
    }
    jobs = assign_auto_devices(config.jobs, inventory)
    return source, probes, jobs


def plan_distributed_jobs(config: JobConfig) -> Dict[str, Any]:
    """Probe hosts and preview exact remote driver allocations."""

    source, probes, jobs = _prepare(config)
    groups = _group_by_host(jobs)
    run_id = "dry-run"
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = {
            name: executor.submit(
                call_host,
                config.hosts[name],
                "plan",
                _host_payload(
                    config,
                    config.hosts[name],
                    group,
                    probes[name],
                    source,
                    run_id=run_id,
                ),
                timeout=60.0,
            )
            for name, group in groups.items()
        }
        plans = {name: future.result() for name, future in futures.items()}
    return {
        "dry_run": True,
        "creates_contexts": False,
        "starts_processes": False,
        "config_path": str(config.path),
        "config_sha256": source["config_sha256"],
        "git": source["git"],
        "hosts": [
            {
                "name": name,
                "ssh": config.hosts[name].ssh,
                "probe": probes[name],
                "devices": plans[name]["devices"],
            }
            for name in groups
        ],
    }


def _transport_metadata(
    host: HostSpec,
    probe: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "name": host.name,
        "ssh": host.ssh,
        "ssh_options": list(host.ssh_options),
        "local": host.local,
        "repo": host.repo,
        "python": host.python,
        "report_path": payload["report_path"],
        "probe": dict(probe),
        "status": "preparing",
        "error": None,
    }


def _merge_host_snapshot(
    report: RunReport,
    host_name: str,
    snapshot: Mapping[str, Any],
) -> None:
    changes = {}
    for raw_job in snapshot.get("jobs", []):
        job = dict(raw_job)
        name = job.get("name")
        if not isinstance(name, str):
            continue
        job["host"] = host_name
        job["remote"] = True
        changes[name] = job
    if changes:
        report.update_jobs(changes)


def _host_spec_from_metadata(raw: Mapping[str, Any]) -> HostSpec:
    return HostSpec(
        name=str(raw["name"]),
        repo=str(raw["repo"]),
        python=str(raw["python"]),
        ssh=str(raw["ssh"]) if raw.get("ssh") is not None else None,
        ssh_options=tuple(str(value) for value in raw.get("ssh_options", ())),
        local=bool(raw.get("local", False)),
    )


def refresh_distributed_snapshot(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Fetch remote host reports for monitor/list without mutating the report."""

    snapshot = json.loads(json.dumps(payload))
    metadata = dict(snapshot.get("metadata") or {})
    if metadata.get("orchestrator") != "ssh":
        return enrich_report(snapshot)
    hosts = dict(metadata.get("hosts") or {})
    if not hosts:
        return snapshot

    results: Dict[str, Mapping[str, Any]] = {}
    errors: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        futures = {}
        for name, raw in hosts.items():
            host = _host_spec_from_metadata(raw)
            request = {
                "repo": raw["probe"]["repo_path"],
                "report_path": raw["report_path"],
                "run_id": snapshot.get("run_id"),
            }
            futures[name] = executor.submit(
                call_host, host, "snapshot", request, timeout=15.0
            )
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except RuntimeError as exc:
                errors[name] = str(exc)

    indexed_jobs = {
        str(job.get("name")): job for job in snapshot.get("jobs", [])
    }
    host_states = []
    for name, raw in hosts.items():
        host_snapshot = results.get(name)
        host_record = dict(raw)
        if host_snapshot is not None:
            host_record["status"] = host_snapshot.get("status", "unknown")
            host_record["error"] = host_snapshot.get("error")
            for child in host_snapshot.get("jobs", []):
                target = indexed_jobs.get(str(child.get("name")))
                if target is not None:
                    target.update(dict(child))
                    target["host"] = name
                    target["remote"] = True
        else:
            host_record["error"] = errors.get(name)
            for target in snapshot.get("jobs", []):
                if target.get("host") == name:
                    target["observed_status"] = "host-unreachable"
        host_states.append(host_record)

    metadata["hosts"] = {item["name"]: item for item in host_states}
    snapshot["metadata"] = metadata
    snapshot["observed_at"] = utc_now()
    if results and len(results) == len(hosts):
        states = [result.get("status") for result in results.values()]
        if all(state in TERMINAL_RUN_STATES for state in states):
            if any(state == "failed" for state in states):
                snapshot["status"] = "failed"
            elif any(state == "cancelled" for state in states):
                snapshot["status"] = "cancelled"
            else:
                snapshot["status"] = "completed"
        elif any(state in ("running", "calibrating") for state in states):
            snapshot["status"] = "running"
    return snapshot


def launch_distributed_jobs(
    config: JobConfig,
    *,
    quiet: bool = False,
) -> JobLaunchResult:
    """Launch one host-local scheduler per configured host and aggregate reports."""

    source, probes, jobs = _prepare(config)
    groups = _group_by_host(jobs)
    run_id = _new_run_id()
    run_dir = (config.logs_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.report.resolve() if config.report else run_dir / "report.json"
    payloads = {
        name: _host_payload(
            config,
            config.hosts[name],
            group,
            probes[name],
            source,
            run_id=run_id,
        )
        for name, group in groups.items()
    }
    host_metadata = {
        name: _transport_metadata(
            config.hosts[name], probes[name], payloads[name]
        )
        for name in groups
    }
    report = RunReport(
        report_path,
        run_id=run_id,
        config_path=config.path,
        metadata={
            "orchestrator": "ssh",
            "launcher_pid": os.getpid(),
            "git": source["git"],
            "config_sha256": source["config_sha256"],
            "coordinator": source["coordinator"],
            "hosts": host_metadata,
        },
    )
    report.add_jobs(
        {
            "name": job.name,
            "host": job.host,
            "remote": True,
            "device": job.device,
            "script": str(job.script),
            "args": list(job.args),
            "cwd": str(job.cwd),
            "requested_sm_share": job.sm_share,
            "target_throughput_share": job.target_share,
            "effective_sm_share": job.sm_share,
            "memory_share": job.memory_share,
        }
        for job in jobs
    )
    if not quiet:
        print(f"nVertake run id: {run_id}", file=sys.stderr)
        print(f"nVertake aggregate JSON report: {report_path}", file=sys.stderr)

    report.set_status("running")
    stopped_after_failure = False
    try:
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures: Dict[str, Future] = {
                name: executor.submit(
                    call_host,
                    config.hosts[name],
                    "launch",
                    payloads[name],
                    timeout=None,
                )
                for name in groups
            }
            pending = set(futures)
            last_poll = 0.0
            failures: Dict[str, str] = {}
            cancellations = set()
            try:
                while pending:
                    now = time.monotonic()
                    if now - last_poll >= 1.0:
                        for name in tuple(pending):
                            try:
                                snapshot = call_host(
                                    config.hosts[name],
                                    "snapshot",
                                    {
                                        "repo": probes[name]["repo_path"],
                                        "report_path": payloads[name]["report_path"],
                                        "run_id": run_id,
                                    },
                                    timeout=10.0,
                                )
                            except RuntimeError:
                                continue
                            _merge_host_snapshot(report, name, snapshot)
                        last_poll = now

                    for name in tuple(pending):
                        future = futures[name]
                        if not future.done():
                            continue
                        pending.remove(name)
                        try:
                            result = future.result()
                            if isinstance(result.get("snapshot"), dict):
                                _merge_host_snapshot(
                                    report, name, result["snapshot"]
                                )
                            if int(result.get("exit_code", 1)) != 0:
                                failures[name] = (
                                    f"host launcher exited with "
                                    f"{result.get('exit_code')}"
                                )
                        except RuntimeError as exc:
                            try:
                                final_snapshot = call_host(
                                    config.hosts[name],
                                    "snapshot",
                                    {
                                        "repo": probes[name]["repo_path"],
                                        "report_path": payloads[name]["report_path"],
                                        "run_id": run_id,
                                    },
                                    timeout=10.0,
                                )
                                _merge_host_snapshot(report, name, final_snapshot)
                            except RuntimeError:
                                final_snapshot = {}
                            if final_snapshot.get("status") == "cancelled":
                                cancellations.add(name)
                            else:
                                failures[name] = str(exc)

                    if (failures or cancellations) and pending and not stopped_after_failure:
                        for name in pending:
                            try:
                                call_host(
                                    config.hosts[name],
                                    "stop",
                                    {
                                        "repo": probes[name]["repo_path"],
                                        "report_path": payloads[name]["report_path"],
                                        "run_id": run_id,
                                    },
                                    timeout=10.0,
                                )
                            except RuntimeError:
                                pass
                        stopped_after_failure = True
                    if pending:
                        time.sleep(0.2)
            except KeyboardInterrupt:
                for name in groups:
                    try:
                        call_host(
                            config.hosts[name],
                            "stop",
                            {
                                "repo": probes[name]["repo_path"],
                                "report_path": payloads[name]["report_path"],
                                "run_id": run_id,
                            },
                            timeout=10.0,
                        )
                    except RuntimeError:
                        pass
                raise

        if failures:
            detail = "; ".join(f"{name}: {error}" for name, error in failures.items())
            report.set_status("failed", error=detail)
            return JobLaunchResult(1, run_id, report_path)
        if cancellations:
            report.set_status(
                "cancelled",
                error="Stopped on host(s): " + ", ".join(sorted(cancellations)),
            )
            return JobLaunchResult(130, run_id, report_path)
        report.set_status("completed")
        return JobLaunchResult(0, run_id, report_path)
    except KeyboardInterrupt:
        for name in groups:
            try:
                call_host(
                    config.hosts[name],
                    "stop",
                    {
                        "repo": probes[name]["repo_path"],
                        "report_path": payloads[name]["report_path"],
                        "run_id": run_id,
                    },
                    timeout=10.0,
                )
            except RuntimeError:
                pass
        report.set_status("cancelled", error="Interrupted by user")
        raise
    except BaseException as exc:
        report.set_status("failed", error=str(exc))
        raise


def host_actions_from_report(
    payload: Mapping[str, Any],
    action: str,
    *,
    job: Optional[str] = None,
    lines: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """Run a management action against every host in an aggregate report."""

    metadata = dict(payload.get("metadata") or {})
    hosts = dict(metadata.get("hosts") or {})
    if metadata.get("orchestrator") != "ssh" or not hosts:
        raise ValueError("Run is not a distributed SSH launch")
    results = {}
    selected_hosts = {
        str(item.get("host"))
        for item in payload.get("jobs", [])
        if job is not None and item.get("name") == job
    }
    if job is not None and not selected_hosts:
        raise ValueError(f"Unknown nVertake job: {job}")
    for name, raw in hosts.items():
        if selected_hosts and name not in selected_hosts:
            continue
        host = _host_spec_from_metadata(raw)
        request = {
            "repo": raw["probe"]["repo_path"],
            "report_path": raw["report_path"],
            "run_id": payload.get("run_id"),
        }
        if action == "logs":
            request.update({"job": job, "lines": lines})
        results[name] = call_host(host, action, request, timeout=15.0)
    return results
