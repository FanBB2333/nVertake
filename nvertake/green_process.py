"""Launch Python scripts in separate CUDA Green Context processes."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TextIO, Tuple

from .green_context import GreenProcessContext, _CudaDriver, _normalize_shares


class GreenProcessLaunchError(RuntimeError):
    """Raised when process lanes cannot be initialized consistently."""


@dataclass(frozen=True)
class GreenProcessRunResult:
    """Detailed outcome for one physical GPU launch group."""

    exit_code: int
    pids: Tuple[int, ...]
    partition_sm_counts: Tuple[int, ...]
    process_exit_codes: Tuple[Optional[int], ...]
    timed_out: bool = False


def _worker_command(
    *,
    lane_index: int,
    shares: Sequence[float],
    script: Path,
    script_args: Sequence[str],
    barrier_directory: Path,
    startup_timeout: float,
    memory_share: Optional[float] = None,
    job_name: Optional[str] = None,
    metrics_path: Optional[Path] = None,
    plain: bool = False,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "nvertake.green_process",
        "--worker",
        "--lane-index",
        str(lane_index),
        "--shares-json",
        json.dumps(list(shares), separators=(",", ":")),
        "--script",
        str(script),
        "--script-args-json",
        json.dumps(list(script_args), separators=(",", ":")),
        "--barrier-directory",
        str(barrier_directory),
        "--startup-timeout",
        str(startup_timeout),
        "--parent-pid",
        str(os.getpid()),
    ]
    if memory_share is not None:
        command.extend(("--memory-share", str(memory_share)))
    if job_name is not None:
        command.extend(("--job-name", job_name))
    if metrics_path is not None:
        command.extend(("--metrics-path", str(metrics_path)))
    if plain:
        command.append("--plain")
    return command


def _child_environment(device: int) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["NVERTAKE_GREEN_PROCESS"] = "1"
    env["NVERTAKE_GREEN_PHYSICAL_DEVICE"] = str(device)
    for name in (
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
        "CUDA_MPS_CLIENT_PRIORITY",
        "CUDA_MPS_PIPE_DIRECTORY",
        "CUDA_MPS_LOG_DIRECTORY",
        "NVERTAKE_AUTO_PRIORITY",
        "NVERTAKE_AUTO_PRIORITY_DEVICE",
        "NVERTAKE_AUTO_PRIORITY_PHYSICAL_DEVICE",
        "NVERTAKE_AUTO_PRIORITY_QUIET",
    ):
        env.pop(name, None)
    package_root = str(Path(__file__).resolve().parent.parent)
    existing_python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        package_root
        if not existing_python_path
        else package_root + os.pathsep + existing_python_path
    )
    return env


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    process.wait(timeout=5.0)


def _read_ready_metadata(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _wait_for_workers_ready(
    processes: Sequence[subprocess.Popen[Any]],
    ready_paths: Sequence[Path],
    *,
    timeout: float,
    cancel_event: Optional[Any] = None,
) -> Tuple[Dict[str, Any], ...]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            raise GreenProcessLaunchError("Green process launch was cancelled")
        metadata = tuple(_read_ready_metadata(path) for path in ready_paths)
        if all(item is not None for item in metadata):
            return tuple(item for item in metadata if item is not None)
        for lane_index, process in enumerate(processes):
            return_code = process.poll()
            if return_code is not None:
                raise GreenProcessLaunchError(
                    f"Green process lane {lane_index} exited with status {return_code} "
                    "before every lane was ready"
                )
        time.sleep(0.02)
    raise GreenProcessLaunchError(
        f"Timed out after {timeout:.1f}s while creating Green process contexts"
    )


def _validate_ready_metadata(
    metadata: Sequence[Dict[str, Any]], expected_count: int
) -> Tuple[int, ...]:
    if len(metadata) != expected_count:
        raise GreenProcessLaunchError("Incomplete Green process readiness metadata")
    partition_maps = []
    process_ids = set()
    for expected_lane, item in enumerate(metadata):
        if int(item.get("lane_index", -1)) != expected_lane:
            raise GreenProcessLaunchError("Green process lane metadata is inconsistent")
        process_id = int(item.get("pid", -1))
        if process_id <= 0 or process_id in process_ids:
            raise GreenProcessLaunchError("Green process PIDs are invalid or duplicated")
        process_ids.add(process_id)
        partitions = tuple(int(value) for value in item.get("partition_sm_counts", []))
        if len(partitions) != expected_count or any(value <= 0 for value in partitions):
            raise GreenProcessLaunchError("Green process partition metadata is incomplete")
        if int(item.get("sm_count", -1)) != partitions[expected_lane]:
            raise GreenProcessLaunchError("Green process lane SM metadata is inconsistent")
        partition_maps.append(partitions)
    if any(partitions != partition_maps[0] for partitions in partition_maps[1:]):
        raise GreenProcessLaunchError(
            "CUDA workers derived different SM partition maps"
        )
    return partition_maps[0]


def launch_green_process_scripts(
    scripts: Sequence[str],
    *,
    shares: Sequence[float],
    script_args: Optional[Sequence[Sequence[str]]] = None,
    device: int = 0,
    startup_timeout: float = 60.0,
    quiet: bool = False,
    memory_shares: Optional[Sequence[Optional[float]]] = None,
    environments: Optional[Sequence[Mapping[str, str]]] = None,
    working_directories: Optional[Sequence[Optional[str]]] = None,
    log_paths: Optional[Sequence[Optional[str]]] = None,
    job_names: Optional[Sequence[str]] = None,
    metrics_paths: Optional[Sequence[Optional[str]]] = None,
    run_timeout: Optional[float] = None,
    timeout_is_success: bool = False,
    on_ready: Optional[
        Callable[[Tuple[Dict[str, Any], ...], Tuple[int, ...]], None]
    ] = None,
    on_exit: Optional[Callable[[int, Optional[int], str], None]] = None,
    cancel_event: Optional[Any] = None,
) -> GreenProcessRunResult:
    """Run one physical GPU group and return its detailed process outcome."""

    normalized = _normalize_shares(shares, minimum_count=1)
    if len(scripts) != len(normalized):
        raise ValueError("The number of scripts must match the number of shares")
    if device < 0:
        raise ValueError("device must be non-negative")
    if startup_timeout <= 0:
        raise ValueError("startup_timeout must be positive")
    if run_timeout is not None and run_timeout <= 0:
        raise ValueError("run_timeout must be positive")

    resolved_scripts = tuple(Path(script).expanduser().resolve() for script in scripts)
    for script in resolved_scripts:
        if not script.is_file():
            raise FileNotFoundError(f"Python script not found: {script}")

    arguments: Sequence[Sequence[str]] = (
        tuple(() for _ in resolved_scripts) if script_args is None else script_args
    )
    if len(arguments) != len(resolved_scripts):
        raise ValueError("script_args must contain one argument list per script")
    if any(isinstance(item, (str, bytes)) for item in arguments) or any(
        any(not isinstance(value, str) for value in item) for item in arguments
    ):
        raise TypeError("Every script argument must be a string")

    lane_count = len(resolved_scripts)
    lane_memory_shares: Sequence[Optional[float]] = (
        tuple(None for _ in resolved_scripts)
        if memory_shares is None
        else memory_shares
    )
    if len(lane_memory_shares) != lane_count:
        raise ValueError("memory_shares must contain one value per script")
    for fraction in lane_memory_shares:
        if fraction is not None and not (0.0 < float(fraction) <= 1.0):
            raise ValueError("Every memory share must be in the range (0, 1]")

    lane_environments: Sequence[Mapping[str, str]] = (
        tuple({} for _ in resolved_scripts) if environments is None else environments
    )
    if len(lane_environments) != lane_count:
        raise ValueError("environments must contain one mapping per script")
    for environment in lane_environments:
        if not isinstance(environment, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        ):
            raise TypeError("Every process environment must map strings to strings")

    lane_working_directories: Sequence[Optional[str]] = (
        tuple(None for _ in resolved_scripts)
        if working_directories is None
        else working_directories
    )
    lane_log_paths: Sequence[Optional[str]] = (
        tuple(None for _ in resolved_scripts) if log_paths is None else log_paths
    )
    lane_metrics_paths: Sequence[Optional[str]] = (
        tuple(None for _ in resolved_scripts)
        if metrics_paths is None
        else metrics_paths
    )
    lane_job_names: Sequence[str] = (
        tuple(f"lane-{index}" for index in range(lane_count))
        if job_names is None
        else job_names
    )
    for values, label in (
        (lane_working_directories, "working_directories"),
        (lane_log_paths, "log_paths"),
        (lane_metrics_paths, "metrics_paths"),
        (lane_job_names, "job_names"),
    ):
        if len(values) != lane_count:
            raise ValueError(f"{label} must contain one value per script")

    resolved_working_directories: Tuple[Optional[Path], ...] = tuple(
        Path(value).expanduser().resolve() if value is not None else None
        for value in lane_working_directories
    )
    for directory in resolved_working_directories:
        if directory is not None and not directory.is_dir():
            raise NotADirectoryError(f"Working directory not found: {directory}")
    resolved_log_paths: Tuple[Optional[Path], ...] = tuple(
        Path(value).expanduser().resolve() if value is not None else None
        for value in lane_log_paths
    )
    resolved_metrics_paths: Tuple[Optional[Path], ...] = tuple(
        Path(value).expanduser().resolve() if value is not None else None
        for value in lane_metrics_paths
    )
    for path in resolved_metrics_paths:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    processes: List[subprocess.Popen[Any]] = []
    log_handles: List[TextIO] = []
    exit_notified = set()

    def notify_exit(lane_index: int, return_code: Optional[int], status: str) -> None:
        if lane_index in exit_notified:
            return
        exit_notified.add(lane_index)
        if on_exit is not None:
            on_exit(lane_index, return_code, status)

    with tempfile.TemporaryDirectory(prefix="nvertake-green-procs-") as temp_root:
        barrier_directory = Path(temp_root)
        ready_paths = tuple(
            barrier_directory / f"ready-{index}.json"
            for index in range(len(resolved_scripts))
        )
        start_path = barrier_directory / "start"
        try:
            for lane_index, (
                script,
                item_args,
                memory_share,
                custom_environment,
                working_directory,
                log_path,
                metrics_path,
                job_name,
            ) in enumerate(
                zip(
                    resolved_scripts,
                    arguments,
                    lane_memory_shares,
                    lane_environments,
                    resolved_working_directories,
                    resolved_log_paths,
                    resolved_metrics_paths,
                    lane_job_names,
                )
            ):
                env = _child_environment(device)
                env.update(custom_environment)
                env["CUDA_VISIBLE_DEVICES"] = str(device)
                env["NVERTAKE_GREEN_PROCESS"] = "1"
                env["NVERTAKE_GREEN_PHYSICAL_DEVICE"] = str(device)
                env["NVERTAKE_JOB_NAME"] = job_name
                if memory_share is not None:
                    env["NVERTAKE_MEMORY_SHARE"] = str(float(memory_share))
                if metrics_path is not None:
                    env["NVERTAKE_METRICS_PATH"] = str(metrics_path)

                output_handle: Optional[TextIO] = None
                if log_path is not None:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    output_handle = log_path.open("w", encoding="utf-8")
                    log_handles.append(output_handle)
                process = subprocess.Popen(
                    _worker_command(
                        lane_index=lane_index,
                        shares=normalized,
                        script=script,
                        script_args=item_args,
                        barrier_directory=barrier_directory,
                        startup_timeout=startup_timeout,
                        memory_share=(
                            float(memory_share) if memory_share is not None else None
                        ),
                        job_name=job_name,
                        metrics_path=metrics_path,
                        plain=(lane_count == 1),
                    ),
                    env=env,
                    cwd=(str(working_directory) if working_directory else None),
                    stdout=output_handle,
                    stderr=(subprocess.STDOUT if output_handle else None),
                    start_new_session=(os.name == "posix"),
                )
                processes.append(process)

            metadata = _wait_for_workers_ready(
                processes,
                ready_paths,
                timeout=startup_timeout,
                cancel_event=cancel_event,
            )
            partition_sm_counts = _validate_ready_metadata(
                metadata, len(resolved_scripts)
            )
            if on_ready is not None:
                on_ready(metadata, partition_sm_counts)
            if not quiet:
                assignments = ", ".join(
                    f"lane {index} (pid {metadata[index]['pid']})={count} SM "
                    f"({normalized[index]:.2f}% requested)"
                    for index, count in enumerate(partition_sm_counts)
                )
                print(f"nVertake Green processes ready: {assignments}", file=sys.stderr)

            start_path.write_text("start\n", encoding="utf-8")
            remaining = set(range(len(processes)))
            run_deadline = (
                time.monotonic() + run_timeout if run_timeout is not None else None
            )
            while remaining:
                if cancel_event is not None and cancel_event.is_set():
                    for lane_index in tuple(remaining):
                        _terminate_process(processes[lane_index])
                        notify_exit(
                            lane_index,
                            processes[lane_index].returncode,
                            "cancelled",
                        )
                    return GreenProcessRunResult(
                        exit_code=130,
                        pids=tuple(int(item["pid"]) for item in metadata),
                        partition_sm_counts=partition_sm_counts,
                        process_exit_codes=tuple(
                            process.poll() for process in processes
                        ),
                    )
                if run_deadline is not None and time.monotonic() >= run_deadline:
                    for lane_index in tuple(remaining):
                        _terminate_process(processes[lane_index])
                        notify_exit(
                            lane_index,
                            processes[lane_index].returncode,
                            "timed_out",
                        )
                    process_codes = tuple(process.poll() for process in processes)
                    return GreenProcessRunResult(
                        exit_code=0 if timeout_is_success else 124,
                        pids=tuple(int(item["pid"]) for item in metadata),
                        partition_sm_counts=partition_sm_counts,
                        process_exit_codes=process_codes,
                        timed_out=True,
                    )
                failed_lane: Optional[int] = None
                failed_code = 0
                for lane_index in tuple(remaining):
                    return_code = processes[lane_index].poll()
                    if return_code is None:
                        continue
                    remaining.remove(lane_index)
                    notify_exit(
                        lane_index,
                        return_code,
                        "completed" if return_code == 0 else "failed",
                    )
                    if return_code != 0 and failed_lane is None:
                        failed_lane = lane_index
                        failed_code = return_code
                if failed_lane is not None:
                    if not quiet:
                        print(
                            f"nVertake Green process lane {failed_lane} exited with "
                            f"status {failed_code}; terminating remaining lanes",
                            file=sys.stderr,
                        )
                    for lane_index in remaining:
                        _terminate_process(processes[lane_index])
                        notify_exit(
                            lane_index,
                            processes[lane_index].returncode,
                            "cancelled",
                        )
                    return GreenProcessRunResult(
                        exit_code=failed_code,
                        pids=tuple(int(item["pid"]) for item in metadata),
                        partition_sm_counts=partition_sm_counts,
                        process_exit_codes=tuple(
                            process.poll() for process in processes
                        ),
                    )
                if remaining:
                    time.sleep(0.05)
            return GreenProcessRunResult(
                exit_code=0,
                pids=tuple(int(item["pid"]) for item in metadata),
                partition_sm_counts=partition_sm_counts,
                process_exit_codes=tuple(process.poll() for process in processes),
            )
        finally:
            for lane_index, process in enumerate(processes):
                _terminate_process(process)
                if lane_index not in exit_notified:
                    status = "completed" if process.returncode == 0 else "cancelled"
                    notify_exit(lane_index, process.returncode, status)
            for handle in log_handles:
                handle.close()


def run_green_process_scripts(
    scripts: Sequence[str],
    *,
    shares: Sequence[float],
    script_args: Optional[Sequence[Sequence[str]]] = None,
    device: int = 0,
    startup_timeout: float = 60.0,
    quiet: bool = False,
    memory_shares: Optional[Sequence[Optional[float]]] = None,
) -> int:
    """Compatibility API returning the aggregate process exit code."""

    result = launch_green_process_scripts(
        scripts,
        shares=shares,
        script_args=script_args,
        device=device,
        startup_timeout=startup_timeout,
        quiet=quiet,
        memory_shares=memory_shares,
    )
    return result.exit_code


def _parent_is_alive(parent_pid: int) -> bool:
    if parent_pid <= 0 or os.name != "posix":
        return True
    try:
        os.kill(parent_pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_for_start(path: Path, *, timeout: float, parent_pid: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if not _parent_is_alive(parent_pid):
            raise GreenProcessLaunchError("Green process launcher exited before start")
        time.sleep(0.02)
    raise GreenProcessLaunchError("Timed out waiting for all Green process lanes")


def _execute_python_script(script: Path, script_args: Sequence[str]) -> int:
    previous_argv = sys.argv
    previous_path = list(sys.path)
    sys.argv = [str(script)] + list(script_args)
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
        return 0
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(exc.code, file=sys.stderr)
        return 1
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


def _write_ready_payload(
    barrier_directory: Path,
    lane_index: int,
    payload: Mapping[str, Any],
) -> None:
    ready_path = barrier_directory / f"ready-{lane_index}.json"
    temporary_path = barrier_directory / f"ready-{lane_index}.{os.getpid()}.tmp"
    temporary_path.write_text(json.dumps(dict(payload), sort_keys=True), encoding="utf-8")
    os.replace(str(temporary_path), str(ready_path))


def _write_ready_metadata(
    barrier_directory: Path,
    context: GreenProcessContext,
    *,
    memory_share: Optional[float] = None,
    job_name: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {
        "lane_index": context.lane_index,
        "pid": os.getpid(),
        "requested_share": context.lane.requested_share,
        "sm_count": context.lane.sm_count,
        "partition_sm_counts": [lane.sm_count for lane in context.lanes],
        "memory_share": memory_share,
        "job_name": job_name,
    }
    _write_ready_payload(
        barrier_directory,
        context.lane_index,
        payload,
    )


def _apply_torch_memory_share(memory_share: Optional[float]) -> None:
    if memory_share is None:
        return
    import torch

    torch.cuda.set_per_process_memory_fraction(float(memory_share), device=0)


def _plain_worker_metadata(
    *,
    lane_index: int,
    memory_share: Optional[float],
    job_name: Optional[str],
) -> Dict[str, Any]:
    driver = _CudaDriver()
    driver.initialize()
    cuda_device = driver.get_device(0)
    available = driver.device_sm_resource(cuda_device)
    sm_count = driver.sm_count(available)
    return {
        "lane_index": lane_index,
        "pid": os.getpid(),
        "requested_share": 100.0,
        "sm_count": sm_count,
        "partition_sm_counts": [sm_count],
        "memory_share": memory_share,
        "job_name": job_name,
        "green_context": False,
    }


def _worker_main(args: argparse.Namespace) -> int:
    shares = json.loads(args.shares_json)
    script_args = json.loads(args.script_args_json)
    if not isinstance(shares, list):
        raise ValueError("Internal shares payload must be a JSON array")
    if not isinstance(script_args, list) or any(
        not isinstance(value, str) for value in script_args
    ):
        raise ValueError("Internal script arguments must be a JSON string array")

    barrier_directory = Path(args.barrier_directory)
    if args.metrics_path:
        os.environ["NVERTAKE_METRICS_PATH"] = args.metrics_path
    if args.job_name:
        os.environ["NVERTAKE_JOB_NAME"] = args.job_name
    if args.memory_share is not None:
        os.environ["NVERTAKE_MEMORY_SHARE"] = str(args.memory_share)

    if args.plain:
        metadata = _plain_worker_metadata(
            lane_index=args.lane_index,
            memory_share=args.memory_share,
            job_name=args.job_name,
        )
        _write_ready_payload(
            barrier_directory,
            args.lane_index,
            metadata,
        )
        _wait_for_start(
            barrier_directory / "start",
            timeout=args.startup_timeout,
            parent_pid=args.parent_pid,
        )
        _apply_torch_memory_share(args.memory_share)
        return _execute_python_script(Path(args.script), script_args)

    with GreenProcessContext(
        device=0,
        shares=shares,
        lane_index=args.lane_index,
    ) as context:
        _write_ready_metadata(
            barrier_directory,
            context,
            memory_share=args.memory_share,
            job_name=args.job_name,
        )
        _wait_for_start(
            barrier_directory / "start",
            timeout=args.startup_timeout,
            parent_pid=args.parent_pid,
        )
        with context.bind():
            _apply_torch_memory_share(args.memory_share)
            return _execute_python_script(Path(args.script), script_args)


def create_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--lane-index", type=int, required=True)
    parser.add_argument("--shares-json", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--script-args-json", required=True)
    parser.add_argument("--barrier-directory", required=True)
    parser.add_argument("--startup-timeout", type=float, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--memory-share", type=float, default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--plain", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = create_worker_parser().parse_args(argv)
    try:
        return _worker_main(args)
    except KeyboardInterrupt:
        return 130
    except BaseException as exc:
        print(f"nVertake Green process worker failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
