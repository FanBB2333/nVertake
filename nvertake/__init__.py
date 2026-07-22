"""
nVertake - Preemptive scheduling for NVIDIA GPUs

A Python package that enables priority scheduling and memory reservation
on NVIDIA GPUs.
"""

__version__ = "0.1.0"
__author__ = "nVertake Authors"

__all__ = [
    "__version__",
    "PriorityScheduler",
    "inject_priority",
    "enable_torch_priority",
    "install_torch_priority_import_hook",
    "MPSController",
    "MPSControlError",
    "MPSStatus",
    "GreenContextExecutor",
    "GreenContextError",
    "GreenContextUnavailableError",
    "GreenContextLane",
    "GreenProcessContext",
    "GreenRunResult",
    "current_green_context_lane",
    "run_green_tasks",
    "GreenProcessLaunchError",
    "run_green_process_scripts",
    "MemoryManager",
    "fill_gpu_memory",
]


def __getattr__(name):
    """Load torch-dependent exports lazily."""
    if name in {"PriorityScheduler", "inject_priority"}:
        from .scheduler import PriorityScheduler, inject_priority

        return {
            "PriorityScheduler": PriorityScheduler,
            "inject_priority": inject_priority,
        }[name]

    if name in {"MemoryManager", "fill_gpu_memory"}:
        from .memory import MemoryManager, fill_gpu_memory

        return {
            "MemoryManager": MemoryManager,
            "fill_gpu_memory": fill_gpu_memory,
        }[name]

    if name in {"enable_torch_priority", "install_torch_priority_import_hook"}:
        from .auto_priority import (
            enable_torch_priority,
            install_torch_priority_import_hook,
        )

        return {
            "enable_torch_priority": enable_torch_priority,
            "install_torch_priority_import_hook": install_torch_priority_import_hook,
        }[name]

    if name in {"MPSController", "MPSControlError", "MPSStatus"}:
        from .mps import MPSController, MPSControlError, MPSStatus

        return {
            "MPSController": MPSController,
            "MPSControlError": MPSControlError,
            "MPSStatus": MPSStatus,
        }[name]

    if name in {
        "GreenContextExecutor",
        "GreenContextError",
        "GreenContextUnavailableError",
        "GreenContextLane",
        "GreenProcessContext",
        "GreenRunResult",
        "current_green_context_lane",
        "run_green_tasks",
    }:
        from .green_context import (
            GreenContextError,
            GreenContextExecutor,
            GreenContextLane,
            GreenContextUnavailableError,
            GreenProcessContext,
            GreenRunResult,
            current_green_context_lane,
            run_green_tasks,
        )

        return {
            "GreenContextExecutor": GreenContextExecutor,
            "GreenContextError": GreenContextError,
            "GreenContextUnavailableError": GreenContextUnavailableError,
            "GreenContextLane": GreenContextLane,
            "GreenProcessContext": GreenProcessContext,
            "GreenRunResult": GreenRunResult,
            "current_green_context_lane": current_green_context_lane,
            "run_green_tasks": run_green_tasks,
        }[name]

    if name in {"GreenProcessLaunchError", "run_green_process_scripts"}:
        from .green_process import GreenProcessLaunchError, run_green_process_scripts

        return {
            "GreenProcessLaunchError": GreenProcessLaunchError,
            "run_green_process_scripts": run_green_process_scripts,
        }[name]

    raise AttributeError(f"module 'nvertake' has no attribute {name!r}")
