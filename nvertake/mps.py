"""NVIDIA Multi-Process Service helpers for weighted GPU sharing."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


MPS_CONTROL_BINARY = "nvidia-cuda-mps-control"
MPS_PRIORITY_VALUES = {
    "normal": "0",
    "below-normal": "1",
}
_PROBE_PREFIX = "__NVERTAKE_MPS_PROBE__="


class MPSControlError(RuntimeError):
    """Raised when the MPS daemon or an MPS client cannot be configured."""


@dataclass(frozen=True)
class MPSPaths:
    """Filesystem locations shared by one MPS daemon and its clients."""

    pipe_directory: Path
    log_directory: Path


@dataclass(frozen=True)
class MPSStatus:
    """Current state of an nVertake-managed MPS daemon."""

    available: bool
    running: bool
    pipe_directory: Path
    log_directory: Path
    server_pids: Tuple[int, ...] = ()
    client_pids: Tuple[int, ...] = ()
    detail: str = ""


def validate_active_thread_percentage(value: int) -> int:
    """Validate an MPS active-thread percentage and return it as an int."""
    percentage = int(value)
    if not 1 <= percentage <= 100:
        raise ValueError("GPU share must be an integer between 1 and 100")
    return percentage


def validate_client_priority(value: str) -> str:
    """Validate a human-readable MPS client priority."""
    if value not in MPS_PRIORITY_VALUES:
        choices = ", ".join(sorted(MPS_PRIORITY_VALUES))
        raise ValueError(f"MPS client priority must be one of: {choices}")
    return value


def default_mps_paths(device: int) -> MPSPaths:
    """Return per-user, per-device MPS paths on the local filesystem."""
    uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
    temp_root = Path(tempfile.gettempdir())
    suffix = f"{uid}-gpu{int(device)}"
    return MPSPaths(
        pipe_directory=temp_root / f"nvertake-mps-{suffix}",
        log_directory=temp_root / f"nvertake-mps-log-{suffix}",
    )


def configure_mps_client_env(
    env: MutableMapping[str, str],
    *,
    paths: MPSPaths,
    active_thread_percentage: Optional[int] = None,
    client_priority: Optional[str] = None,
) -> MutableMapping[str, str]:
    """Configure a child process to connect to an nVertake MPS daemon.

    The daemon selects the physical GPU and exposes it to clients as logical
    device 0. Keeping a physical ordinal in ``CUDA_VISIBLE_DEVICES`` in the
    client would apply device remapping a second time, so it must be removed.
    """
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["CUDA_MPS_PIPE_DIRECTORY"] = str(paths.pipe_directory)
    env["CUDA_MPS_LOG_DIRECTORY"] = str(paths.log_directory)

    if active_thread_percentage is not None:
        percentage = validate_active_thread_percentage(active_thread_percentage)
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percentage)
    else:
        env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)

    if client_priority is not None:
        priority = validate_client_priority(client_priority)
        env["CUDA_MPS_CLIENT_PRIORITY"] = MPS_PRIORITY_VALUES[priority]
    else:
        env.pop("CUDA_MPS_CLIENT_PRIORITY", None)

    return env


def _numeric_lines(text: str) -> Tuple[int, ...]:
    values: List[int] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"[0-9]+", stripped):
            values.append(int(stripped))
    return tuple(values)


class MPSController:
    """Manage one per-user MPS daemon for one physical GPU."""

    def __init__(
        self,
        device: int = 0,
        *,
        pipe_directory: Optional[str] = None,
        log_directory: Optional[str] = None,
        control_binary: str = MPS_CONTROL_BINARY,
    ) -> None:
        self.device = int(device)
        defaults = default_mps_paths(self.device)
        self.paths = MPSPaths(
            pipe_directory=Path(pipe_directory) if pipe_directory else defaults.pipe_directory,
            log_directory=Path(log_directory) if log_directory else defaults.log_directory,
        )
        self.control_binary = control_binary

    def _platform_error(self) -> Optional[str]:
        if platform.system() != "Linux":
            return "NVIDIA MPS requires a native Linux or QNX host"
        if "microsoft" in platform.release().lower():
            return (
                "NVIDIA MPS is not exposed by the CUDA driver inside WSL; "
                "use a native Linux host"
            )
        return None

    def _resolved_control_binary(self) -> Optional[str]:
        return shutil.which(self.control_binary)

    def _ensure_supported(self) -> str:
        platform_error = self._platform_error()
        if platform_error:
            raise MPSControlError(platform_error)
        binary = self._resolved_control_binary()
        if binary is None:
            raise MPSControlError(
                f"{self.control_binary!r} was not found; install the NVIDIA compute utilities"
            )
        return binary

    def _control_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = str(self.paths.pipe_directory)
        env["CUDA_MPS_LOG_DIRECTORY"] = str(self.paths.log_directory)
        return env

    def _daemon_env(self, gpu_uuid: str) -> Dict[str, str]:
        env = self._control_env()
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        # Client settings inherited by the daemon become upper bounds/defaults.
        # Keep the daemon unrestricted and apply limits per client instead.
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "100"
        env["CUDA_MPS_CLIENT_PRIORITY"] = MPS_PRIORITY_VALUES["normal"]
        return env

    def _ensure_directories(self) -> None:
        for directory in (self.paths.pipe_directory, self.paths.log_directory):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                directory.chmod(0o700)
            except OSError:
                pass

    def _run_control(
        self,
        command: str,
        *,
        timeout: float = 5.0,
    ) -> subprocess.CompletedProcess[str]:
        binary = self._resolved_control_binary() or self.control_binary
        return subprocess.run(
            [binary],
            input=command.rstrip("\n") + "\n",
            capture_output=True,
            text=True,
            env=self._control_env(),
            timeout=timeout,
            check=False,
        )

    @staticmethod
    def _control_succeeded(completed: subprocess.CompletedProcess[str]) -> bool:
        combined = f"{completed.stdout}\n{completed.stderr}".lower()
        return completed.returncode == 0 and "cannot find mps control daemon" not in combined

    def is_running(self) -> bool:
        if self._resolved_control_binary() is None:
            return False
        try:
            completed = self._run_control("get_server_list", timeout=2.0)
        except (OSError, subprocess.SubprocessError):
            return False
        return self._control_succeeded(completed)

    def _gpu_uuid(self) -> str:
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device}",
                    "--query-gpu=uuid",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=5.0,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise MPSControlError(
                f"Unable to resolve UUID for physical GPU {self.device}: {exc}"
            ) from exc

        first_line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
        if not first_line.startswith("GPU-"):
            raise MPSControlError(
                f"nvidia-smi returned an invalid UUID for physical GPU {self.device}: {first_line!r}"
            )
        return first_line

    def start(self, *, timeout: float = 8.0) -> bool:
        """Start the daemon if needed; return True only when newly started."""
        binary = self._ensure_supported()
        self._ensure_directories()
        if self.is_running():
            return False

        gpu_uuid = self._gpu_uuid()
        try:
            completed = subprocess.run(
                [binary, "-d"],
                capture_output=True,
                text=True,
                env=self._daemon_env(gpu_uuid),
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise MPSControlError(f"Failed to start NVIDIA MPS: {exc}") from exc
        if completed.returncode != 0 and not self.is_running():
            detail = (completed.stderr or completed.stdout).strip()
            raise MPSControlError(f"Failed to start NVIDIA MPS: {detail or 'unknown error'}")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_running():
                return True
            time.sleep(0.1)
        raise MPSControlError(
            "NVIDIA MPS control daemon did not become ready" + self._log_diagnostics()
        )

    def ensure_started(self) -> bool:
        """Ensure the daemon is available; return whether this call started it."""
        return self.start()

    def _server_pids(self) -> Tuple[int, ...]:
        completed = self._run_control("get_server_list")
        if not self._control_succeeded(completed):
            return ()
        return _numeric_lines(completed.stdout)

    def _client_pids(self, server_pids: Iterable[int]) -> Tuple[int, ...]:
        clients: List[int] = []
        for server_pid in server_pids:
            completed = self._run_control(f"get_client_list {server_pid}")
            if self._control_succeeded(completed):
                clients.extend(_numeric_lines(completed.stdout))
        return tuple(sorted(set(clients)))

    def status(self) -> MPSStatus:
        platform_error = self._platform_error()
        if platform_error:
            return MPSStatus(
                available=False,
                running=False,
                pipe_directory=self.paths.pipe_directory,
                log_directory=self.paths.log_directory,
                detail=platform_error,
            )
        if self._resolved_control_binary() is None:
            return MPSStatus(
                available=False,
                running=False,
                pipe_directory=self.paths.pipe_directory,
                log_directory=self.paths.log_directory,
                detail=f"{self.control_binary!r} was not found",
            )
        if not self.is_running():
            return MPSStatus(
                available=True,
                running=False,
                pipe_directory=self.paths.pipe_directory,
                log_directory=self.paths.log_directory,
            )

        servers = self._server_pids()
        return MPSStatus(
            available=True,
            running=True,
            pipe_directory=self.paths.pipe_directory,
            log_directory=self.paths.log_directory,
            server_pids=servers,
            client_pids=self._client_pids(servers),
        )

    def stop(self, *, force: bool = False, timeout: float = 8.0) -> bool:
        """Stop an idle daemon, or stop an active daemon when force is True."""
        self._ensure_supported()
        status = self.status()
        if not status.running:
            return False
        if status.client_pids and not force:
            clients = ", ".join(str(pid) for pid in status.client_pids)
            raise MPSControlError(
                f"Refusing to stop MPS while client processes are active: {clients}; "
                "use --force only if interrupting them is intended"
            )

        try:
            completed = self._run_control("quit", timeout=timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            raise MPSControlError(f"Failed to stop NVIDIA MPS: {exc}") from exc
        if not self._control_succeeded(completed):
            detail = (completed.stderr or completed.stdout).strip()
            raise MPSControlError(f"Failed to stop NVIDIA MPS: {detail or 'unknown error'}")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_running():
                return True
            time.sleep(0.1)
        raise MPSControlError("NVIDIA MPS control daemon did not stop before timeout")

    def client_environment(
        self,
        env: MutableMapping[str, str],
        *,
        active_thread_percentage: Optional[int] = None,
        client_priority: Optional[str] = None,
    ) -> MutableMapping[str, str]:
        return configure_mps_client_env(
            env,
            paths=self.paths,
            active_thread_percentage=active_thread_percentage,
            client_priority=client_priority,
        )

    def probe_client(
        self,
        env: Mapping[str, str],
        *,
        python_executable: str = sys.executable,
        timeout: float = 20.0,
    ) -> Dict[str, Any]:
        """Initialize a short-lived CUDA client before launching user code."""
        probe_code = (
            "import json, torch; "
            "assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'; "
            "p=torch.cuda.get_device_properties(0); "
            f"print({_PROBE_PREFIX!r}+json.dumps({{"
            "'device_count': torch.cuda.device_count(), "
            "'device_name': p.name, "
            "'sm_count': p.multi_processor_count"
            "}}, sort_keys=True))"
        )
        try:
            completed = subprocess.run(
                [python_executable, "-c", probe_code],
                capture_output=True,
                text=True,
                env=dict(env),
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise MPSControlError(
                f"MPS CUDA client probe could not run: {exc}" + self._log_diagnostics()
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise MPSControlError(
                "MPS CUDA client probe failed before launching the task"
                + (f": {detail}" if detail else "")
                + self._log_diagnostics()
            )

        for line in completed.stdout.splitlines():
            if line.startswith(_PROBE_PREFIX):
                try:
                    payload = json.loads(line[len(_PROBE_PREFIX) :])
                except json.JSONDecodeError as exc:
                    raise MPSControlError(f"Invalid MPS probe output: {line!r}") from exc
                if int(payload.get("device_count", 0)) != 1:
                    raise MPSControlError(
                        "MPS client did not see exactly one remapped GPU; "
                        f"probe output was {payload!r}"
                    )
                return payload
        raise MPSControlError(
            f"MPS probe produced no result marker; stdout was {completed.stdout.strip()!r}"
        )

    def _log_diagnostics(self, *, line_count: int = 20) -> str:
        sections: List[str] = []
        for name in ("control.log", "server.log"):
            path = self.paths.log_directory / name
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            if lines:
                sections.append(f"\n{name} (last {min(line_count, len(lines))} lines):")
                sections.extend(lines[-line_count:])
        return "\n" + "\n".join(sections) if sections else ""
