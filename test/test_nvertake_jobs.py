"""Tests for diagnostics, YAML launches, reports, metrics, and calibration."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nvertake.cli import main
from nvertake.diagnostics import inspect_green_device, plan_green_partitions
from nvertake.green_process import (
    GreenProcessRunResult,
    _apply_torch_memory_share,
)
from nvertake.jobs import (
    adjust_shares_for_throughput,
    launch_jobs,
    load_job_config,
)
from nvertake.metrics import read_throughput_metric, report_throughput
from nvertake.runtime import RunReport, enrich_report, format_monitor_table


@dataclass
class _FakeSm:
    sm_count: int
    min_sm_partition_size: int = 2
    sm_coscheduled_alignment: int = 2


@dataclass
class _FakeResource:
    sm_count: int

    @property
    def sm(self):
        return _FakeSm(self.sm_count)


class _DiagnosticDriver:
    def initialize(self):
        return None

    def device_count(self):
        return 2

    def get_device(self, ordinal):
        return ordinal

    def device_name(self, device):
        return f"Fake GPU {device}"

    def driver_version(self):
        return 12080

    def compute_capability_major(self, device):
        return 8

    def device_sm_resource(self, device):
        return _FakeResource(12)

    @staticmethod
    def sm_count(resource):
        return resource.sm_count

    def sm_resource_group_count(self, resource, *, requested_sm_count, flags=0):
        return resource.sm_count // 2

    def split_sm_resource(self, resource, requested_sm_count):
        return (
            _FakeResource(requested_sm_count),
            _FakeResource(resource.sm_count - requested_sm_count),
        )

    def split_sm_resources(
        self, resource, *, group_count, requested_sm_count, flags=0
    ):
        groups = tuple(_FakeResource(2) for _ in range(group_count))
        return groups, _FakeResource(resource.sm_count - 2 * group_count)


def _write_worker(path: Path) -> None:
    path.write_text("print('worker')\n", encoding="utf-8")


class TestDiagnostics(unittest.TestCase):
    def test_doctor_reports_driver_limits_without_context_creation(self):
        driver = _DiagnosticDriver()
        diagnostics = inspect_green_device(1, _driver=driver)
        self.assertEqual(diagnostics.driver_version_text, "12.8")
        self.assertEqual(diagnostics.total_sm_count, 12)
        self.assertEqual(diagnostics.max_green_processes, 6)
        self.assertTrue(diagnostics.fine_grained_partitioning)

    def test_dry_plan_uses_every_sm(self):
        plan = plan_green_partitions((20, 30, 50), _driver=_DiagnosticDriver())
        self.assertEqual([lane.sm_count for lane in plan.lanes], [2, 4, 6])
        self.assertEqual(sum(lane.sm_count for lane in plan.lanes), 12)
        self.assertFalse(plan.to_dict()["starts_processes"])

    def test_green_procs_dry_run_never_calls_launcher(self):
        fake_plan = plan_green_partitions((50, 50), _driver=_DiagnosticDriver())
        output = StringIO()
        with patch(
            "nvertake.diagnostics.plan_green_partitions", return_value=fake_plan
        ), patch("nvertake.green_process.run_green_process_scripts") as launcher:
            with redirect_stdout(output):
                code = main(
                    [
                        "green-procs",
                        "--shares",
                        "50,50",
                        "--dry-run",
                        "first.py",
                        "second.py",
                    ]
                )
        self.assertEqual(code, 0)
        self.assertFalse(launcher.called)
        self.assertTrue(json.loads(output.getvalue())["dry_run"])


class TestYamlConfig(unittest.TestCase):
    def test_loads_scripts_arguments_environment_and_two_gpus(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            _write_worker(root_path / "a.py")
            _write_worker(root_path / "b.py")
            config_path = root_path / "jobs.yaml"
            config_path.write_text(
                """version: 1
logs_dir: logs
defaults:
  cwd: .
  env:
    COMMON: "yes"
jobs:
  - name: first
    script: a.py
    args: [--rank, 0]
    sm_share: 30
    memory_share: 25%
    device: 0
  - name: second
    script: b.py
    sm_share: 70
    memory_share: 0.75
    device: 0
  - name: other-gpu
    script: a.py
    sm_share: 1
    memory_share: 1
    device: 1
""",
                encoding="utf-8",
            )
            config = load_job_config(str(config_path))

        self.assertEqual(len(config.jobs), 3)
        self.assertEqual(config.jobs[0].args, ("--rank", "0"))
        self.assertEqual(config.jobs[0].env["COMMON"], "yes")
        self.assertEqual(config.jobs[0].memory_share, 0.25)
        self.assertEqual(config.jobs[2].device, 1)

    def test_requires_complete_memory_limits_on_each_device(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            _write_worker(root_path / "worker.py")
            config_path = root_path / "jobs.yaml"
            config_path.write_text(
                """jobs:
  - {name: a, script: worker.py, sm_share: 1, memory_share: 0.5}
  - {name: b, script: worker.py, sm_share: 1}
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "every job"):
                load_job_config(str(config_path))


class TestMemoryAndMetrics(unittest.TestCase):
    def test_memory_fraction_is_applied_to_logical_cuda_zero(self):
        fake_torch = SimpleNamespace(cuda=MagicMock())
        with patch.dict(sys.modules, {"torch": fake_torch}):
            _apply_torch_memory_share(0.25)
        fake_torch.cuda.set_per_process_memory_fraction.assert_called_once_with(
            0.25, device=0
        )

    def test_throughput_metric_is_atomic_and_optional(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(report_throughput(12.5, unit="samples/s"))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "metric.json"
            with patch.dict(
                os.environ, {"NVERTAKE_METRICS_PATH": str(path)}, clear=False
            ):
                self.assertTrue(report_throughput(12.5, unit="samples/s"))
            metric = read_throughput_metric(path)
        self.assertEqual(metric["throughput"], 12.5)
        self.assertEqual(metric["unit"], "samples/s")

    def test_throughput_metric_includes_pytorch_memory_fallback(self):
        fake_cuda = MagicMock()
        fake_cuda.is_initialized.return_value = True
        fake_cuda.memory_allocated.return_value = 8 * 1024 * 1024
        fake_cuda.memory_reserved.return_value = 16 * 1024 * 1024
        fake_torch = SimpleNamespace(cuda=fake_cuda)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "metric.json"
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.dict(
                os.environ, {"NVERTAKE_METRICS_PATH": str(path)}, clear=False
            ):
                report_throughput(1.0)
            metric = read_throughput_metric(path)
        self.assertEqual(metric["gpu_memory_mib"], 16)
        self.assertEqual(metric["gpu_memory_source"], "pytorch_allocator")


class TestCalibration(unittest.TestCase):
    def test_adjusts_toward_target_throughput_ratio(self):
        adjusted = adjust_shares_for_throughput(
            (50, 50), (25, 75), (50, 50), damping=0.5
        )
        self.assertLess(adjusted[0], 50)
        self.assertGreater(adjusted[1], 50)
        self.assertAlmostEqual(sum(adjusted), 100.0)


class TestReportsAndMultiGpuLaunch(unittest.TestCase):
    def test_report_enrichment_and_table_include_requested_columns(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            metric_path = root_path / "metric.json"
            report_path = root_path / "report.json"
            with patch.dict(
                os.environ,
                {
                    "NVERTAKE_RUNTIME_DIR": str(root_path / "registry"),
                    "NVERTAKE_METRICS_PATH": str(metric_path),
                },
                clear=False,
            ):
                report_throughput(42, unit="items/s")
                report = RunReport(report_path, run_id="test-run")
                report.add_jobs(
                    [
                        {
                            "name": "worker",
                            "device": 0,
                            "pid": 123,
                            "sm_count": 12,
                            "metrics_path": str(metric_path),
                            "status": "running",
                        }
                    ]
                )
                with patch(
                    "nvertake.runtime.query_gpu_process_memory",
                    return_value={123: 456},
                ), patch("nvertake.runtime._pid_is_alive", return_value=True):
                    snapshot = enrich_report(report.snapshot())
            table = format_monitor_table(snapshot)
        self.assertIn("PID", table)
        self.assertIn("SM", table)
        self.assertIn("VRAM MiB", table)
        self.assertIn("42 items/s", table)

    def test_launch_submits_one_group_per_gpu_and_writes_report(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            for name in ("a.py", "b.py", "c.py", "d.py"):
                _write_worker(root_path / name)
            config_path = root_path / "jobs.yaml"
            config_path.write_text(
                """logs_dir: logs
jobs:
  - {name: a, script: a.py, sm_share: 1, device: 0}
  - {name: b, script: b.py, sm_share: 1, device: 0}
  - {name: c, script: c.py, sm_share: 1, device: 1}
  - {name: d, script: d.py, sm_share: 1, device: 1}
""",
                encoding="utf-8",
            )
            config = load_job_config(str(config_path))

            def fake_launch(group, shares, **kwargs):
                metadata = tuple(
                    {"pid": 1000 + index} for index, _job in enumerate(group)
                )
                counts = tuple(10 for _job in group)
                kwargs["on_ready"](metadata, counts)
                for index, _job in enumerate(group):
                    kwargs["on_exit"](index, 0, "completed")
                return GreenProcessRunResult(
                    0,
                    tuple(item["pid"] for item in metadata),
                    counts,
                    tuple(0 for _job in group),
                )

            with patch.dict(
                os.environ,
                {"NVERTAKE_RUNTIME_DIR": str(root_path / "registry")},
                clear=False,
            ), patch("nvertake.jobs._launch_group", side_effect=fake_launch) as launch, patch(
                "nvertake.runtime.query_gpu_process_memory", return_value={}
            ):
                result = launch_jobs(config, quiet=True)
            report_payload = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(launch.call_count, 2)
        self.assertEqual(report_payload["status"], "completed")
        self.assertEqual(len({job["log_path"] for job in report_payload["jobs"]}), 4)


if __name__ == "__main__":
    unittest.main()
