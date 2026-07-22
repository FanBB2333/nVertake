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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .green_context import GreenProcessContext, _normalize_shares


class GreenProcessLaunchError(RuntimeError):
    """Raised when process lanes cannot be initialized consistently."""


def _worker_command(
    *,
    lane_index: int,
    shares: Sequence[float],
    script: Path,
    script_args: Sequence[str],
    barrier_directory: Path,
    startup_timeout: float,
) -> List[str]:
    return [
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
) -> Tuple[Dict[str, Any], ...]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
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


def run_green_process_scripts(
    scripts: Sequence[str],
    *,
    shares: Sequence[float],
    script_args: Optional[Sequence[Sequence[str]]] = None,
    device: int = 0,
    startup_timeout: float = 60.0,
    quiet: bool = False,
) -> int:
    """Run Python files concurrently in deterministic Green Context partitions."""

    normalized = _normalize_shares(shares)
    if len(scripts) != len(normalized):
        raise ValueError("The number of scripts must match the number of shares")
    if device < 0:
        raise ValueError("device must be non-negative")
    if startup_timeout <= 0:
        raise ValueError("startup_timeout must be positive")

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

    processes: List[subprocess.Popen[Any]] = []
    with tempfile.TemporaryDirectory(prefix="nvertake-green-procs-") as temp_root:
        barrier_directory = Path(temp_root)
        ready_paths = tuple(
            barrier_directory / f"ready-{index}.json"
            for index in range(len(resolved_scripts))
        )
        start_path = barrier_directory / "start"
        env = _child_environment(device)
        try:
            for lane_index, (script, item_args) in enumerate(
                zip(resolved_scripts, arguments)
            ):
                process = subprocess.Popen(
                    _worker_command(
                        lane_index=lane_index,
                        shares=normalized,
                        script=script,
                        script_args=item_args,
                        barrier_directory=barrier_directory,
                        startup_timeout=startup_timeout,
                    ),
                    env=env,
                    start_new_session=(os.name == "posix"),
                )
                processes.append(process)

            metadata = _wait_for_workers_ready(
                processes,
                ready_paths,
                timeout=startup_timeout,
            )
            partition_sm_counts = _validate_ready_metadata(
                metadata, len(resolved_scripts)
            )
            if not quiet:
                assignments = ", ".join(
                    f"lane {index} (pid {metadata[index]['pid']})={count} SM "
                    f"({normalized[index]:.2f}% requested)"
                    for index, count in enumerate(partition_sm_counts)
                )
                print(f"nVertake Green processes ready: {assignments}", file=sys.stderr)

            start_path.write_text("start\n", encoding="utf-8")
            remaining = set(range(len(processes)))
            while remaining:
                failed_lane: Optional[int] = None
                failed_code = 0
                for lane_index in tuple(remaining):
                    return_code = processes[lane_index].poll()
                    if return_code is None:
                        continue
                    remaining.remove(lane_index)
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
                    return failed_code
                if remaining:
                    time.sleep(0.05)
            return 0
        finally:
            for process in processes:
                _terminate_process(process)


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


def _write_ready_metadata(
    barrier_directory: Path, context: GreenProcessContext
) -> None:
    payload = {
        "lane_index": context.lane_index,
        "pid": os.getpid(),
        "requested_share": context.lane.requested_share,
        "sm_count": context.lane.sm_count,
        "partition_sm_counts": [lane.sm_count for lane in context.lanes],
    }
    ready_path = barrier_directory / f"ready-{context.lane_index}.json"
    temporary_path = barrier_directory / (
        f"ready-{context.lane_index}.{os.getpid()}.tmp"
    )
    temporary_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary_path), str(ready_path))


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
    with GreenProcessContext(
        device=0,
        shares=shares,
        lane_index=args.lane_index,
    ) as context:
        _write_ready_metadata(barrier_directory, context)
        _wait_for_start(
            barrier_directory / "start",
            timeout=args.startup_timeout,
            parent_pid=args.parent_pid,
        )
        with context.bind():
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
