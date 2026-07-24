"""Temporary signal handling for cancellable command-line launches."""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from types import FrameType
from typing import Dict, Iterator, Optional, Union


SignalHandler = Union[
    int,
    signal.Handlers,
    None,
]


class LaunchSignalInterrupt(KeyboardInterrupt):
    """KeyboardInterrupt carrying the operating-system signal that cancelled a run."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        try:
            self.signal_name = signal.Signals(self.signum).name
        except ValueError:
            self.signal_name = f"signal {self.signum}"
        super().__init__(f"Received {self.signal_name}")


def _raise_launch_interrupt(
    signum: int,
    _frame: Optional[FrameType],
) -> None:
    raise LaunchSignalInterrupt(signum)


@contextmanager
def launch_signal_handlers() -> Iterator[None]:
    """Translate SIGTERM/SIGHUP into a cancellable launch interruption.

    Python only permits signal-handler changes on its main thread. Library
    callers that launch from a worker thread retain the process' existing
    signal behavior.
    """

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous: Dict[int, SignalHandler] = {}
    candidates = [
        getattr(signal, name)
        for name in ("SIGTERM", "SIGHUP")
        if hasattr(signal, name)
    ]
    try:
        for signum in candidates:
            current = signal.getsignal(signum)
            if current == signal.SIG_IGN:
                continue
            previous[int(signum)] = current
            signal.signal(signum, _raise_launch_interrupt)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
