"""Experimental CUDA Green Context execution for two in-process tasks.

CUDA Green Contexts partition a device's streaming multiprocessors (SMs) into
separate CUDA contexts.  Unlike NVIDIA MPS, this API works without an MPS
daemon, including on WSL, but all participating work must live in one process.
"""

from __future__ import annotations

import ctypes
import importlib
import math
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union


_CU_SUCCESS = 0
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEV_RESOURCE_TYPE_SM = 1
_CU_GREEN_CTX_DEFAULT_STREAM = 0x1


class GreenContextError(RuntimeError):
    """Raised when Green Context setup or execution fails."""


class GreenContextUnavailableError(GreenContextError):
    """Raised when the installed CUDA driver does not expose Green Contexts."""


class _SmResource(ctypes.Structure):
    _fields_ = [("sm_count", ctypes.c_uint)]


class _ResourceData(ctypes.Union):
    _fields_ = [
        ("sm", _SmResource),
        ("reserved", ctypes.c_ubyte * 48),
    ]


class _DeviceResource(ctypes.Structure):
    # This layout is part of the CUDA Driver API ABI introduced in CUDA 12.4.
    _anonymous_ = ("data",)
    _fields_ = [
        ("resource_type", ctypes.c_int),
        ("internal", ctypes.c_ubyte * 92),
        ("data", _ResourceData),
    ]


@dataclass(frozen=True)
class GreenContextLane:
    """Public metadata for one Green Context task lane."""

    index: int
    requested_share: float
    sm_count: int
    total_sm_count: int

    @property
    def actual_sm_share(self) -> float:
        return self.sm_count / self.total_sm_count

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["actual_sm_share"] = self.actual_sm_share
        return payload


@dataclass(frozen=True)
class GreenRunResult:
    """Results and the actual driver-selected SM allocation for a run."""

    device: int
    compute_capability_major: int
    total_sm_count: int
    lanes: Tuple[GreenContextLane, GreenContextLane]
    results: Tuple[Any, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "compute_capability_major": self.compute_capability_major,
            "total_sm_count": self.total_sm_count,
            "lanes": [lane.to_dict() for lane in self.lanes],
            "results": list(self.results),
        }


@dataclass
class _ContextHandle:
    green_context: int
    cuda_context: int
    descriptor: Any
    resources: Any


class _CudaDriver:
    """Small ctypes wrapper around the CUDA 12.4 Green Context Driver API."""

    def __init__(self, library: Optional[Any] = None) -> None:
        if library is None:
            library_name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
            try:
                loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
                library = loader(library_name)
            except OSError as exc:
                raise GreenContextUnavailableError(
                    f"Unable to load the NVIDIA CUDA driver ({library_name}): {exc}"
                ) from exc
        self._library = library
        self._configure()

    def _function(self, name: str, argtypes: Sequence[Any]) -> Any:
        try:
            function = getattr(self._library, name)
        except AttributeError as exc:
            raise GreenContextUnavailableError(
                "CUDA Green Contexts require an NVIDIA driver exposing the CUDA 12.4 "
                f"Driver API; missing symbol: {name}"
            ) from exc
        function.argtypes = list(argtypes)
        function.restype = ctypes.c_int
        return function

    def _configure(self) -> None:
        resource_pointer = ctypes.POINTER(_DeviceResource)
        self._cu_init = self._function("cuInit", [ctypes.c_uint])
        self._cu_device_get = self._function(
            "cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        )
        self._cu_device_get_attribute = self._function(
            "cuDeviceGetAttribute",
            [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
        )
        self._cu_device_get_resource = self._function(
            "cuDeviceGetDevResource", [ctypes.c_int, resource_pointer, ctypes.c_int]
        )
        self._cu_split_sm_resource = self._function(
            "cuDevSmResourceSplitByCount",
            [
                resource_pointer,
                ctypes.POINTER(ctypes.c_uint),
                resource_pointer,
                resource_pointer,
                ctypes.c_uint,
                ctypes.c_uint,
            ],
        )
        self._cu_generate_descriptor = self._function(
            "cuDevResourceGenerateDesc",
            [ctypes.POINTER(ctypes.c_void_p), resource_pointer, ctypes.c_uint],
        )
        self._cu_green_context_create = self._function(
            "cuGreenCtxCreate",
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int, ctypes.c_uint],
        )
        self._cu_context_from_green = self._function(
            "cuCtxFromGreenCtx", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        )
        self._cu_context_get_current = self._function(
            "cuCtxGetCurrent", [ctypes.POINTER(ctypes.c_void_p)]
        )
        self._cu_context_set_current = self._function(
            "cuCtxSetCurrent", [ctypes.c_void_p]
        )
        self._cu_context_synchronize = self._function("cuCtxSynchronize", [])
        self._cu_context_get_resource = self._function(
            "cuCtxGetDevResource", [ctypes.c_void_p, resource_pointer, ctypes.c_int]
        )
        self._cu_green_context_destroy = self._function(
            "cuGreenCtxDestroy", [ctypes.c_void_p]
        )

        # Error-name helpers predate Green Contexts.  Treat them as optional so
        # an unusual loader can still report numeric CUDA errors.
        try:
            self._cu_get_error_name = self._function(
                "cuGetErrorName", [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            )
            self._cu_get_error_string = self._function(
                "cuGetErrorString", [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            )
        except GreenContextUnavailableError:
            self._cu_get_error_name = None
            self._cu_get_error_string = None

    def _error_detail(self, result: int) -> str:
        details: List[str] = []
        for function in (self._cu_get_error_name, self._cu_get_error_string):
            if function is None:
                continue
            value = ctypes.c_char_p()
            if int(function(result, ctypes.byref(value))) == _CU_SUCCESS and value.value:
                details.append(value.value.decode("utf-8", errors="replace"))
        return ": ".join(details)

    def _check(self, operation: str, function: Any, *args: Any) -> None:
        result = int(function(*args))
        if result == _CU_SUCCESS:
            return
        detail = self._error_detail(result)
        suffix = f" ({detail})" if detail else ""
        raise GreenContextError(f"{operation} failed with CUDA error {result}{suffix}")

    def initialize(self) -> None:
        self._check("cuInit", self._cu_init, 0)

    def get_device(self, ordinal: int) -> int:
        device = ctypes.c_int()
        self._check("cuDeviceGet", self._cu_device_get, ctypes.byref(device), ordinal)
        return int(device.value)

    def compute_capability_major(self, device: int) -> int:
        major = ctypes.c_int()
        self._check(
            "cuDeviceGetAttribute",
            self._cu_device_get_attribute,
            ctypes.byref(major),
            _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        )
        return int(major.value)

    def device_sm_resource(self, device: int) -> _DeviceResource:
        resource = _DeviceResource()
        self._check(
            "cuDeviceGetDevResource",
            self._cu_device_get_resource,
            device,
            ctypes.byref(resource),
            _CU_DEV_RESOURCE_TYPE_SM,
        )
        return resource

    @staticmethod
    def sm_count(resource: _DeviceResource) -> int:
        return int(resource.sm.sm_count)

    def split_sm_resource(
        self, resource: _DeviceResource, requested_sm_count: int
    ) -> Tuple[_DeviceResource, _DeviceResource]:
        partition = _DeviceResource()
        remainder = _DeviceResource()
        group_count = ctypes.c_uint(1)
        self._check(
            "cuDevSmResourceSplitByCount",
            self._cu_split_sm_resource,
            ctypes.byref(partition),
            ctypes.byref(group_count),
            ctypes.byref(resource),
            ctypes.byref(remainder),
            0,
            requested_sm_count,
        )
        if group_count.value != 1:
            raise GreenContextError(
                "CUDA driver did not produce the requested single SM resource group"
            )
        return partition, remainder

    def create_green_context(
        self, device: int, resources: Sequence[_DeviceResource]
    ) -> _ContextHandle:
        resource_array_type = _DeviceResource * len(resources)
        resource_array = resource_array_type(*resources)
        descriptor = ctypes.c_void_p()
        self._check(
            "cuDevResourceGenerateDesc",
            self._cu_generate_descriptor,
            ctypes.byref(descriptor),
            resource_array,
            len(resources),
        )
        green = ctypes.c_void_p()
        self._check(
            "cuGreenCtxCreate",
            self._cu_green_context_create,
            ctypes.byref(green),
            descriptor,
            device,
            _CU_GREEN_CTX_DEFAULT_STREAM,
        )
        context = ctypes.c_void_p()
        try:
            self._check(
                "cuCtxFromGreenCtx",
                self._cu_context_from_green,
                ctypes.byref(context),
                green,
            )
        except BaseException:
            self._cu_green_context_destroy(green)
            raise
        return _ContextHandle(
            green_context=int(green.value or 0),
            cuda_context=int(context.value or 0),
            descriptor=descriptor,
            resources=resource_array,
        )

    def context_sm_count(self, context: int) -> int:
        resource = _DeviceResource()
        self._check(
            "cuCtxGetDevResource",
            self._cu_context_get_resource,
            ctypes.c_void_p(context),
            ctypes.byref(resource),
            _CU_DEV_RESOURCE_TYPE_SM,
        )
        return self.sm_count(resource)

    def current_context(self) -> Optional[int]:
        context = ctypes.c_void_p()
        self._check(
            "cuCtxGetCurrent", self._cu_context_get_current, ctypes.byref(context)
        )
        return int(context.value) if context.value else None

    def set_current_context(self, context: Optional[int]) -> None:
        self._check(
            "cuCtxSetCurrent",
            self._cu_context_set_current,
            ctypes.c_void_p(context) if context is not None else ctypes.c_void_p(),
        )

    def synchronize_current_context(self) -> None:
        self._check("cuCtxSynchronize", self._cu_context_synchronize)

    def destroy_green_context(self, handle: _ContextHandle) -> None:
        self._check(
            "cuGreenCtxDestroy",
            self._cu_green_context_destroy,
            ctypes.c_void_p(handle.green_context),
        )


_Task = Union[str, Callable[..., Any]]
_CURRENT_LANE = threading.local()


def current_green_context_lane() -> Optional[GreenContextLane]:
    """Return Green Context lane metadata inside a running task thread."""

    return getattr(_CURRENT_LANE, "lane", None)


def _normalized_shares(shares: Sequence[float]) -> Tuple[float, float]:
    if len(shares) != 2:
        raise ValueError("Green Context execution currently requires exactly two shares")
    values = tuple(float(value) for value in shares)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Green Context shares must be finite positive numbers")
    total = sum(values)
    return values[0] * 100.0 / total, values[1] * 100.0 / total


def _requested_partition_sm_count(
    total_sm_count: int,
    smaller_share: float,
    compute_capability_major: int,
) -> int:
    if total_sm_count <= 0:
        raise GreenContextError("CUDA driver reported no SM resources")
    if compute_capability_major >= 9:
        minimum, alignment = 8, 8
    elif compute_capability_major == 8:
        minimum, alignment = 4, 2
    elif compute_capability_major == 7:
        minimum, alignment = 2, 2
    else:
        minimum, alignment = 1, 1
    if total_sm_count < minimum * 2:
        raise GreenContextUnavailableError(
            f"Device has only {total_sm_count} SMs; two Green Context lanes need at least "
            f"{minimum * 2} on this architecture"
        )
    ideal = total_sm_count * smaller_share / 100.0
    # Select the closest architecture-aligned count. Exact half-way cases use
    # the smaller value, avoiding accidental over-allocation of the small lane.
    units = math.floor(ideal / alignment + 0.5 - 1e-12)
    requested = max(minimum, units * alignment)
    return min(requested, total_sm_count - minimum)


def _resolve_task(task: _Task) -> Tuple[Callable[..., Any], str]:
    if callable(task):
        module = getattr(task, "__module__", "<callable>")
        name = getattr(task, "__qualname__", getattr(task, "__name__", repr(task)))
        return task, f"{module}:{name}"
    if not isinstance(task, str) or ":" not in task:
        raise ValueError("Task must be callable or use the 'module:callable' form")
    module_name, attribute_path = task.split(":", 1)
    if not module_name or not attribute_path:
        raise ValueError("Task must use the 'module:callable' form")
    value: Any = importlib.import_module(module_name)
    for component in attribute_path.split("."):
        value = getattr(value, component)
    if not callable(value):
        raise TypeError(f"Green Context task is not callable: {task}")
    return value, task


class GreenContextExecutor:
    """Run exactly two cooperating tasks in separately partitioned CUDA contexts.

    Each lane is bound to one worker thread. Tasks should create and release
    their CUDA tensors inside the task function, must not exchange CUDA tensors
    across lanes, and should return CPU/JSON-safe values.
    """

    def __init__(
        self,
        *,
        device: int = 0,
        shares: Sequence[float] = (25.0, 75.0),
        _driver: Optional[Any] = None,
    ) -> None:
        if device < 0:
            raise ValueError("device must be non-negative")
        normalized = _normalized_shares(shares)
        driver = _driver if _driver is not None else _CudaDriver()
        self._driver = driver
        self.device = int(device)
        self._handles: List[Optional[_ContextHandle]] = [None, None]
        self._closed = False
        self._run_lock = threading.Lock()

        driver.initialize()
        cuda_device = driver.get_device(self.device)
        self.compute_capability_major = driver.compute_capability_major(cuda_device)
        available = driver.device_sm_resource(cuda_device)
        self.total_sm_count = driver.sm_count(available)

        smaller_index = 0 if normalized[0] <= normalized[1] else 1
        requested_sms = _requested_partition_sm_count(
            self.total_sm_count,
            normalized[smaller_index],
            self.compute_capability_major,
        )
        smaller_resource, larger_resource = driver.split_sm_resource(
            available, requested_sms
        )
        resources = [larger_resource, larger_resource]
        resources[smaller_index] = smaller_resource
        resources[1 - smaller_index] = larger_resource

        try:
            for index, resource in enumerate(resources):
                self._handles[index] = driver.create_green_context(cuda_device, [resource])

            sm_counts = tuple(
                driver.context_sm_count(handle.cuda_context)
                for handle in self._required_handles()
            )
            if any(count <= 0 for count in sm_counts):
                raise GreenContextError("CUDA driver created a Green Context with no SMs")
            if sum(sm_counts) != self.total_sm_count:
                raise GreenContextError(
                    "CUDA driver did not partition all SM resources: "
                    f"device={self.total_sm_count}, lanes={sm_counts}"
                )
            self.lanes = (
                GreenContextLane(0, normalized[0], sm_counts[0], self.total_sm_count),
                GreenContextLane(1, normalized[1], sm_counts[1], self.total_sm_count),
            )
        except BaseException:
            try:
                self.close()
            except GreenContextError:
                pass
            raise

    def _required_handles(self) -> Tuple[_ContextHandle, _ContextHandle]:
        if self._handles[0] is None or self._handles[1] is None:
            raise GreenContextError("Green Context executor is not fully initialized")
        return self._handles[0], self._handles[1]

    @contextmanager
    def bind(self, lane_index: int) -> Iterator[GreenContextLane]:
        """Make one lane's CUDA context current on the calling thread."""

        if self._closed:
            raise GreenContextError("Green Context executor is closed")
        if lane_index not in (0, 1):
            raise ValueError("lane_index must be 0 or 1")
        handle = self._required_handles()[lane_index]
        previous = self._driver.current_context()
        self._driver.set_current_context(handle.cuda_context)
        lane = self.lanes[lane_index]
        _CURRENT_LANE.lane = lane
        try:
            yield lane
        finally:
            try:
                self._driver.synchronize_current_context()
            finally:
                try:
                    self._driver.set_current_context(previous)
                finally:
                    if hasattr(_CURRENT_LANE, "lane"):
                        del _CURRENT_LANE.lane

    def run(
        self,
        tasks: Sequence[_Task],
        task_kwargs: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> GreenRunResult:
        """Run two task callables concurrently and return results in lane order."""

        if self._closed:
            raise GreenContextError("Green Context executor is closed")
        if len(tasks) != 2:
            raise ValueError("Green Context execution currently requires exactly two tasks")
        kwargs_items: Sequence[Mapping[str, Any]] = task_kwargs or ({}, {})
        if len(kwargs_items) != 2:
            raise ValueError("task_kwargs must contain exactly two mappings")
        if any(not isinstance(item, Mapping) for item in kwargs_items):
            raise TypeError("Each task_kwargs item must be a mapping")
        if not self._run_lock.acquire(blocking=False):
            raise GreenContextError("This Green Context executor is already running tasks")

        barrier = threading.Barrier(2)
        results: List[Any] = [None, None]
        errors: List[Tuple[int, BaseException]] = []
        error_lock = threading.Lock()

        def worker(index: int) -> None:
            try:
                with self.bind(index):
                    task, _label = _resolve_task(tasks[index])
                    barrier.wait()
                    results[index] = task(**dict(kwargs_items[index]))
            except BaseException as exc:
                try:
                    barrier.abort()
                except BaseException:
                    pass
                with error_lock:
                    errors.append((index, exc))

        threads = [
            threading.Thread(target=worker, args=(index,), name=f"nvertake-green-{index}")
            for index in range(2)
        ]
        interrupted: Optional[KeyboardInterrupt] = None
        try:
            started_threads: List[threading.Thread] = []
            try:
                for thread in threads:
                    thread.start()
                    started_threads.append(thread)
            except BaseException:
                barrier.abort()
                for thread in started_threads:
                    thread.join()
                raise
            for thread in threads:
                while thread.is_alive():
                    try:
                        thread.join(timeout=0.25)
                    except KeyboardInterrupt as exc:
                        # Python cannot safely cancel an arbitrary task thread.
                        # Keep its CUDA context alive until both tasks return,
                        # then propagate the interrupt to the caller.
                        interrupted = exc
        finally:
            self._run_lock.release()

        if interrupted is not None:
            raise interrupted

        if errors:
            errors.sort(key=lambda item: item[0])
            detail = "; ".join(
                f"lane {index}: {type(exc).__name__}: {exc}" for index, exc in errors
            )
            raise GreenContextError(f"Green Context task execution failed: {detail}") from errors[0][1]

        return GreenRunResult(
            device=self.device,
            compute_capability_major=self.compute_capability_major,
            total_sm_count=self.total_sm_count,
            lanes=self.lanes,
            results=(results[0], results[1]),
        )

    def close(self) -> None:
        """Destroy both contexts after all task-owned CUDA objects are released."""

        if self._closed:
            return
        if self._run_lock.locked():
            raise GreenContextError("Cannot close Green Contexts while tasks are running")
        self._closed = True
        errors: List[BaseException] = []
        for index in (1, 0):
            handle = self._handles[index]
            if handle is None:
                continue
            try:
                self._driver.destroy_green_context(handle)
            except BaseException as exc:
                errors.append(exc)
            finally:
                self._handles[index] = None
        if errors:
            raise GreenContextError(
                "Failed to destroy one or more CUDA Green Contexts: "
                + "; ".join(str(error) for error in errors)
            ) from errors[0]

    def __enter__(self) -> "GreenContextExecutor":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            self.close()
        except GreenContextError:
            if exc is None:
                raise
        return False


def run_green_tasks(
    tasks: Sequence[_Task],
    *,
    device: int = 0,
    shares: Sequence[float] = (25.0, 75.0),
    task_kwargs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> GreenRunResult:
    """Convenience wrapper for a single two-task Green Context run."""

    with GreenContextExecutor(device=device, shares=shares) as executor:
        return executor.run(tasks, task_kwargs=task_kwargs)


__all__ = [
    "GreenContextError",
    "GreenContextExecutor",
    "GreenContextLane",
    "GreenContextUnavailableError",
    "GreenRunResult",
    "current_green_context_lane",
    "run_green_tasks",
]
