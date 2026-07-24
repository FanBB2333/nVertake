"""Read-only CUDA Green Context capability checks and SM allocation plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from .green_context import (
    GreenContextError,
    GreenContextLane,
    GreenContextUnavailableError,
    _CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING,
    _CudaDriver,
    _normalize_shares,
    _partition_requirements,
    _partition_resources_for_shares,
)


@dataclass(frozen=True)
class GreenDeviceDiagnostics:
    """Driver capabilities relevant to nVertake process partitions."""

    device: int
    device_name: str
    device_count: int
    driver_version: int
    driver_version_text: str
    compute_capability_major: int
    total_sm_count: int
    minimum_partition_sm_count: int
    sm_coscheduled_alignment: int
    fine_grained_partitioning: bool
    max_green_processes: int
    max_processes_basis: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GreenPartitionPlan:
    """A driver-selected SM plan that does not create contexts or processes."""

    diagnostics: GreenDeviceDiagnostics
    lanes: Tuple[GreenContextLane, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dry_run": True,
            "creates_contexts": False,
            "starts_processes": False,
            "device": self.diagnostics.to_dict(),
            "lanes": [lane.to_dict() for lane in self.lanes],
        }


def _format_driver_version(version: int) -> str:
    if version <= 0:
        return "unknown"
    major = version // 1000
    minor = (version % 1000) // 10
    return f"{major}.{minor}"


def _resource_constraint(resource: Any, name: str, fallback: int) -> int:
    sm = getattr(resource, "sm", None)
    value = int(getattr(sm, name, 0) or 0)
    return value if value > 0 else fallback


def inspect_green_device(
    device: int = 0,
    *,
    _driver: Optional[Any] = None,
) -> GreenDeviceDiagnostics:
    """Inspect one CUDA device without provisioning a Green Context."""

    if device < 0:
        raise ValueError("device must be non-negative")

    driver = _driver if _driver is not None else _CudaDriver()
    driver.initialize()
    device_count = driver.device_count()
    if device >= device_count:
        raise ValueError(
            f"CUDA device {device} does not exist; driver reports {device_count} device(s)"
        )

    cuda_device = driver.get_device(device)
    compute_capability_major = driver.compute_capability_major(cuda_device)
    resource = driver.device_sm_resource(cuda_device)
    total_sm_count = driver.sm_count(resource)
    fallback_minimum, fallback_alignment = _partition_requirements(
        compute_capability_major
    )
    minimum = _resource_constraint(
        resource, "min_sm_partition_size", fallback_minimum
    )
    alignment = _resource_constraint(
        resource, "sm_coscheduled_alignment", fallback_alignment
    )

    fine_grained = False
    max_processes_basis = "architecture-aligned SM groups"
    try:
        max_processes = driver.sm_resource_group_count(
            resource,
            requested_sm_count=1,
            flags=_CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING,
        )
        fine_grained = max_processes > 0
        if fine_grained:
            max_processes_basis = "fine-grained SM resource groups"
    except GreenContextError:
        max_processes = driver.sm_resource_group_count(
            resource,
            requested_sm_count=minimum,
        )

    if max_processes <= 0:
        raise GreenContextUnavailableError(
            "CUDA driver did not report any usable Green Context SM groups"
        )

    driver_version = driver.driver_version()
    return GreenDeviceDiagnostics(
        device=device,
        device_name=driver.device_name(cuda_device),
        device_count=device_count,
        driver_version=driver_version,
        driver_version_text=_format_driver_version(driver_version),
        compute_capability_major=compute_capability_major,
        total_sm_count=total_sm_count,
        minimum_partition_sm_count=minimum,
        sm_coscheduled_alignment=alignment,
        fine_grained_partitioning=fine_grained,
        max_green_processes=max_processes,
        max_processes_basis=max_processes_basis,
    )


def plan_green_partitions(
    shares: Sequence[float],
    *,
    device: int = 0,
    _driver: Optional[Any] = None,
) -> GreenPartitionPlan:
    """Ask the driver for an exact SM split without creating contexts or workers."""

    normalized = _normalize_shares(shares)
    driver = _driver if _driver is not None else _CudaDriver()
    diagnostics = inspect_green_device(device, _driver=driver)
    cuda_device = driver.get_device(device)
    available = driver.device_sm_resource(cuda_device)
    _resource_sets, lanes = _partition_resources_for_shares(
        driver,
        available,
        normalized,
        diagnostics.compute_capability_major,
    )
    return GreenPartitionPlan(diagnostics=diagnostics, lanes=lanes)


def format_doctor_report(diagnostics: GreenDeviceDiagnostics) -> str:
    """Format diagnostics for terminal output."""

    fine = "yes" if diagnostics.fine_grained_partitioning else "no"
    return "\n".join(
        (
            f"GPU {diagnostics.device}: {diagnostics.device_name}",
            f"CUDA driver capability: {diagnostics.driver_version_text}",
            f"Compute capability major: {diagnostics.compute_capability_major}",
            f"Allocatable SMs: {diagnostics.total_sm_count}",
            f"Minimum SM partition: {diagnostics.minimum_partition_sm_count}",
            f"SM co-scheduling alignment: {diagnostics.sm_coscheduled_alignment}",
            f"Fine-grained partitioning: {fine}",
            "Maximum Green processes: "
            f"{diagnostics.max_green_processes} "
            f"({diagnostics.max_processes_basis})",
        )
    )


def inspect_scheduler_capabilities(
    diagnostics: GreenDeviceDiagnostics,
) -> Dict[str, Any]:
    """Inspect optional scheduling and telemetry backends for one device."""

    from .mps import inspect_static_mps_capability
    from .telemetry import inspect_telemetry_capabilities

    return {
        "mps_static": inspect_static_mps_capability(
            diagnostics.device,
            compute_capability_major=diagnostics.compute_capability_major,
            driver_version=diagnostics.driver_version,
        ).to_dict(),
        "work_queue_connections": {
            "available": True,
            "mode": "CUDA_DEVICE_MAX_CONNECTIONS hint",
            "minimum": 1,
            "maximum": 32,
            "hard_partition": False,
            "detail": (
                "Controls per-process CUDA connection count; it does not reserve "
                "independent hardware work queues"
            ),
        },
        "telemetry": inspect_telemetry_capabilities(diagnostics.device),
    }


def format_scheduler_capabilities(capabilities: Dict[str, Any]) -> str:
    """Format optional backend capability results for ``doctor``."""

    static = dict(capabilities.get("mps_static") or {})
    telemetry = dict(capabilities.get("telemetry") or {})
    pmon = dict(telemetry.get("process_utilization") or {})
    dcgm = dict(telemetry.get("dcgm_profiling") or {})
    return "\n".join(
        (
            "Optional capabilities:",
            "  Static MPS partitions: "
            + ("yes" if static.get("available") else "no")
            + f" ({static.get('detail') or 'unknown'})",
            "  Per-process SM telemetry: "
            + ("yes" if pmon.get("available") else "no")
            + f" ({pmon.get('source') or 'nvidia-smi pmon'})",
            "  DCGM profiling fields: "
            + ("yes" if dcgm.get("available") else "no")
            + f" ({dcgm.get('detail') or 'unknown'})",
            "  Work-queue setting: CUDA_DEVICE_MAX_CONNECTIONS hint (1-32; "
            "not a hard queue partition)",
        )
    )
