"""Cross-process leases for nVertake-managed physical GPUs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

from .runtime import process_create_time, runtime_registry_directory, utc_now


class GpuLeaseError(RuntimeError):
    """Raised when another nVertake launcher already owns a GPU lease."""


def gpu_lease_directory() -> Path:
    configured = os.environ.get("NVERTAKE_LEASE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return runtime_registry_directory().parent / "leases"


def _read_metadata(stream: TextIO) -> Dict[str, Any]:
    try:
        stream.seek(0)
        payload = json.loads(stream.read() or "{}")
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


class GpuLease:
    """Hold an advisory lock for one physical GPU until the context exits."""

    def __init__(
        self,
        device: int,
        *,
        run_id: str,
        timeout: float = 0.0,
        poll_interval: float = 0.1,
    ) -> None:
        self.device = int(device)
        self.run_id = str(run_id)
        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)
        if self.device < 0:
            raise ValueError("GPU lease device must be non-negative")
        if self.timeout < 0:
            raise ValueError("GPU lease timeout must be non-negative")
        if self.poll_interval <= 0:
            raise ValueError("GPU lease poll interval must be positive")
        self.path = gpu_lease_directory() / f"gpu-{self.device}.lock"
        self._stream: Optional[TextIO] = None

    def acquire(self) -> "GpuLease":
        try:
            import fcntl
        except ImportError as exc:
            raise GpuLeaseError(
                "GPU leases require POSIX advisory file locking"
            ) from exc

        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                owner = _read_metadata(stream)
                if time.monotonic() >= deadline:
                    stream.close()
                    detail = (
                        f"run {owner.get('run_id')} (PID {owner.get('pid')})"
                        if owner
                        else "another launcher"
                    )
                    raise GpuLeaseError(
                        f"GPU {self.device} is leased by {detail}; "
                        "increase lease_timeout to wait"
                    )
                time.sleep(min(self.poll_interval, max(0.0, deadline - time.monotonic())))

        metadata = {
            "device": self.device,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "process_create_time": process_create_time(os.getpid()),
            "acquired_at": utc_now(),
        }
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps(metadata, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        self._stream = stream
        return self

    def release(self) -> None:
        stream = self._stream
        if stream is None:
            return
        self._stream = None
        try:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> "GpuLease":
        return self.acquire()

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.release()
