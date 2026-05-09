"""
Low-intrusion PyTorch priority injection.

The CLI uses this module through a `sitecustomize` hook so a training script can
run under a high-priority CUDA stream without editing the original source.
"""

from __future__ import annotations

import builtins
import logging
import os
import sys
import threading
from typing import Any, Callable, Dict, Optional


logger = logging.getLogger("nvertake")

_LOCK = threading.RLock()
_STREAMS: Dict[int, Any] = {}
_ORIGINAL_IMPORT: Optional[Callable[..., Any]] = None
_IMPORT_HOOK_INSTALLED = False
_ENSURING = False


def _strict() -> bool:
    return os.environ.get("NVERTAKE_AUTO_PRIORITY_STRICT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _quiet() -> bool:
    return os.environ.get("NVERTAKE_AUTO_PRIORITY_QUIET", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _warn(message: str) -> None:
    if not _quiet():
        logger.warning(message)


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        _warn(f"Ignoring invalid {name}={raw!r}")
        return default


def _coerce_device_index(device: Any) -> Optional[int]:
    if device is None:
        return None
    index = getattr(device, "index", None)
    if index is not None:
        return int(index)
    if isinstance(device, str):
        if ":" in device:
            _, _, tail = device.partition(":")
            if tail:
                try:
                    return int(tail)
                except ValueError:
                    return None
        if device == "cuda":
            return None
    try:
        return int(device)
    except (TypeError, ValueError):
        return None


def _configured_device(default: Optional[int] = None) -> Optional[int]:
    return _env_int("NVERTAKE_AUTO_PRIORITY_DEVICE", default)


def _configured_priority(torch_module: Any) -> int:
    configured = _env_int("NVERTAKE_STREAM_PRIORITY", None)
    if configured is not None:
        return configured
    try:
        _least, greatest = torch_module.cuda.get_stream_priority_range()
        return int(greatest)
    except Exception:
        return -1


def _ensure_priority_stream(torch_module: Any, device: Any = None) -> Optional[Any]:
    global _ENSURING

    with _LOCK:
        if _ENSURING:
            return None

        _ENSURING = True
        try:
            cuda = torch_module.cuda
            try:
                if not bool(cuda.is_available()):
                    return None
            except Exception:
                return None

            device_index = _configured_device(_coerce_device_index(device))
            if device_index is None:
                try:
                    device_index = int(cuda.current_device())
                except Exception:
                    device_index = 0

            stream = _STREAMS.get(device_index)
            if stream is None:
                with cuda.device(device_index):
                    stream = cuda.Stream(
                        device=device_index,
                        priority=_configured_priority(torch_module),
                    )
                _STREAMS[device_index] = stream
                if not _quiet():
                    logger.info(
                        "nVertake auto priority enabled on CUDA device %s",
                        device_index,
                    )

            with cuda.device(device_index):
                cuda.set_stream(stream)
            return stream
        except Exception as exc:
            if _strict():
                raise
            _warn(f"Failed to enable nVertake torch priority stream: {exc}")
            return None
        finally:
            _ENSURING = False


def _patch_torch_cuda(
    torch_module: Any,
    device: Any = None,
    *,
    eager: bool = True,
) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return False
    if getattr(cuda, "_nvertake_auto_priority_patched", False):
        if eager:
            _ensure_priority_stream(torch_module, device)
        return True

    def wrap_after(name: str, device_from_args: bool = False) -> None:
        original = getattr(cuda, name, None)
        if original is None or getattr(original, "_nvertake_wrapped", False):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            target_device = device
            if device_from_args and args:
                target_device = args[0]
            _ensure_priority_stream(torch_module, target_device)
            return result

        wrapped._nvertake_wrapped = True  # type: ignore[attr-defined]
        wrapped._nvertake_original = original  # type: ignore[attr-defined]
        setattr(cuda, name, wrapped)

    wrap_after("_lazy_init")
    wrap_after("init")
    wrap_after("set_device", device_from_args=True)

    original_current_stream = getattr(cuda, "current_stream", None)
    if original_current_stream is not None and not getattr(
        original_current_stream,
        "_nvertake_wrapped",
        False,
    ):

        def current_stream_wrapped(device_arg: Any = None) -> Any:
            _ensure_priority_stream(torch_module, device_arg)
            return original_current_stream(device_arg)

        current_stream_wrapped._nvertake_wrapped = True  # type: ignore[attr-defined]
        current_stream_wrapped._nvertake_original = original_current_stream  # type: ignore[attr-defined]
        cuda.current_stream = current_stream_wrapped

    cuda._nvertake_auto_priority_patched = True
    if eager:
        _ensure_priority_stream(torch_module, device)
    return True


def enable_torch_priority(
    torch_module: Any = None,
    device: Any = None,
    *,
    eager: bool = True,
) -> bool:
    """
    Enable the current CUDA stream to be a high-priority stream.

    Use this as a one-line in-process fallback:

        from nvertake import enable_torch_priority
        enable_torch_priority()

    The CLI normally calls this lazily through `sitecustomize`, so user training
    scripts do not need to import nVertake directly.
    """
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception as exc:
            if _strict():
                raise
            _warn(f"PyTorch is not importable; auto priority disabled: {exc}")
            return False
    return _patch_torch_cuda(torch_module, device=device, eager=eager)


def install_torch_priority_import_hook(device: Any = None) -> None:
    """Patch PyTorch when it is imported, without importing torch eagerly."""
    global _IMPORT_HOOK_INSTALLED, _ORIGINAL_IMPORT

    if "torch" in sys.modules:
        enable_torch_priority(sys.modules["torch"], device=device, eager=False)
        return

    with _LOCK:
        if _IMPORT_HOOK_INSTALLED:
            return

        _ORIGINAL_IMPORT = builtins.__import__

        def import_wrapped(
            name: str,
            globals_: Any = None,
            locals_: Any = None,
            fromlist: Any = (),
            level: int = 0,
        ) -> Any:
            assert _ORIGINAL_IMPORT is not None
            module = _ORIGINAL_IMPORT(name, globals_, locals_, fromlist, level)
            torch_module = sys.modules.get("torch")
            if torch_module is not None:
                try:
                    enable_torch_priority(torch_module, device=device, eager=False)
                finally:
                    builtins.__import__ = _ORIGINAL_IMPORT
            return module

        builtins.__import__ = import_wrapped
        _IMPORT_HOOK_INSTALLED = True


def install_from_env() -> None:
    if os.environ.get("NVERTAKE_AUTO_PRIORITY", "") not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    install_torch_priority_import_hook(device=_configured_device(None))
