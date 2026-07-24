"""Tests for diagnostics, YAML launches, reports, metrics, and calibration."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
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
    CalibrationSpec,
    _calibrate,
    assign_auto_devices,
    adjust_shares_for_throughput,
    launch_jobs,
    load_job_config,
)
from nvertake.lease import GpuLease, GpuLeaseError
from nvertake.metrics import (
    read_throughput_metric,
    read_throughput_samples,
    report_throughput,
)
from nvertake.orchestration import call_host
from nvertake.runtime import (
    RunReport,
    enrich_report,
    format_monitor_table,
    list_registered_runs,
    read_report_logs,
    stop_local_report,
)


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
lease_timeout: 12
backend: auto
defaults:
  cwd: .
  work_queue_connections: 3
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
        self.assertEqual(config.jobs[0].work_queue_connections, 3)
        self.assertEqual(config.lease_timeout, 12.0)
        self.assertEqual(config.backend, "auto")

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

    def test_loads_remote_hosts_without_requiring_remote_files_locally(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = Path(root) / "remote.yaml"
            config_path.write_text(
                """hosts:
  406:
    ssh: 406-tailscale
    repo: ~/repos/nVertake
    python: /opt/torch/bin/python
defaults:
  host: 406
  cwd: examples
  device: auto
jobs:
  - name: remote-worker
    script: throughput_worker.py
    sm_share: 1
""",
                encoding="utf-8",
            )
            config = load_job_config(str(config_path))

        self.assertIn("406", config.hosts)
        self.assertEqual(config.hosts["406"].ssh, "406-tailscale")
        self.assertEqual(config.jobs[0].host, "406")
        self.assertIsNone(config.jobs[0].device)
        self.assertEqual(config.jobs[0].cwd, Path("examples"))
        self.assertEqual(config.jobs[0].script, Path("throughput_worker.py"))

    def test_remote_job_rejects_unknown_host(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = Path(root) / "remote.yaml"
            config_path.write_text(
                """hosts:
  known: {ssh: known, repo: /repo}
jobs:
  - {host: missing, script: worker.py, sm_share: 1}
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown host"):
                load_job_config(str(config_path))

    def test_auto_device_placement_balances_memory_limits(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = Path(root) / "remote.yaml"
            config_path.write_text(
                """hosts:
  gpu-box: {ssh: gpu-box, repo: /repo}
defaults: {host: gpu-box, device: auto}
jobs:
  - {name: first, script: a.py, sm_share: 1, memory_share: 0.6}
  - {name: second, script: b.py, sm_share: 1, memory_share: 0.6}
""",
                encoding="utf-8",
            )
            config = load_job_config(str(config_path))
        placed = assign_auto_devices(
            config.jobs,
            {
                "gpu-box": (
                    {"index": 0, "memory_total_mib": 100, "memory_free_mib": 90},
                    {"index": 1, "memory_total_mib": 100, "memory_free_mib": 80},
                )
            },
        )
        self.assertEqual({job.device for job in placed}, {0, 1})

    def test_auto_device_placement_respects_driver_process_limit(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = Path(root) / "remote.yaml"
            config_path.write_text(
                """hosts:
  gpu-box: {ssh: gpu-box, repo: /repo}
defaults: {host: gpu-box, device: auto}
jobs:
  - {name: first, script: a.py, sm_share: 1}
  - {name: second, script: b.py, sm_share: 1}
""",
                encoding="utf-8",
            )
            config = load_job_config(str(config_path))
        with self.assertRaisesRegex(ValueError, "no GPU"):
            assign_auto_devices(
                config.jobs,
                {
                    "gpu-box": (
                        {
                            "index": 0,
                            "memory_total_mib": 100,
                            "memory_free_mib": 100,
                            "max_green_processes": 1,
                        },
                    )
                },
            )


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

    def test_throughput_metric_retains_samples_for_robust_calibration(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "metric.json"
            with patch.dict(
                os.environ, {"NVERTAKE_METRICS_PATH": str(path)}, clear=False
            ):
                for value in (10.0, 1000.0, 11.0):
                    report_throughput(value, unit="steps/s")
            samples = read_throughput_samples(path)
        self.assertEqual([item["throughput"] for item in samples], [10.0, 1000.0, 11.0])


class TestCalibration(unittest.TestCase):
    def test_adjusts_toward_target_throughput_ratio(self):
        adjusted = adjust_shares_for_throughput(
            (50, 50), (25, 75), (50, 50), damping=0.5
        )
        self.assertLess(adjusted[0], 50)
        self.assertGreater(adjusted[1], 50)
        self.assertAlmostEqual(sum(adjusted), 100.0)

    def test_calibration_uses_sample_median_instead_of_latest_outlier(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            _write_worker(root_path / "a.py")
            _write_worker(root_path / "b.py")
            config_path = root_path / "jobs.yaml"
            config_path.write_text(
                """jobs:
  - {name: a, script: a.py, sm_share: 50, target_share: 50}
  - {name: b, script: b.py, sm_share: 50, target_share: 50}
""",
                encoding="utf-8",
            )
            config = replace(
                load_job_config(str(config_path)),
                calibration=CalibrationSpec(
                    enabled=True,
                    rounds=1,
                    duration=1.0,
                    tolerance=0.05,
                    damping=0.5,
                    warmup=0.5,
                    minimum_samples=3,
                    sample_window=10,
                ),
            )

            def fake_launch(group, _shares, **kwargs):
                for index, path in enumerate(kwargs["metrics_paths"]):
                    values = (1.0, 1000.0, 2.0) if index == 0 else (2.0, 4.0, 3.0)
                    with patch.dict(
                        os.environ,
                        {"NVERTAKE_METRICS_PATH": str(path)},
                        clear=False,
                    ):
                        for value in values:
                            report_throughput(value, unit="steps/s")
                return GreenProcessRunResult(
                    0,
                    tuple(100 + index for index, _job in enumerate(group)),
                    tuple(10 for _job in group),
                    tuple(0 for _job in group),
                )

            with patch("nvertake.jobs._launch_group", side_effect=fake_launch):
                _effective, details = _calibrate(
                    config,
                    root_path / "run",
                    quiet=True,
                    backends={0: "green"},
                )

        device = details["rounds"][0]["devices"][0]
        self.assertEqual(device["observed_throughput"], [2.0, 3.0])
        self.assertEqual(device["throughput_samples"][0]["count"], 3)
        self.assertTrue(device["throughput_samples"][0]["warmup_fallback"])


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

    def test_completed_nvidia_memory_sample_is_not_replaced_by_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            metric_path = Path(root) / "metric.json"
            metric_path.write_text(
                json.dumps(
                    {
                        "throughput": 1,
                        "unit": "items/s",
                        "gpu_memory_mib": 16,
                        "gpu_memory_source": "pytorch_allocator",
                    }
                ),
                encoding="utf-8",
            )
            payload = {
                "run_id": "run",
                "status": "completed",
                "jobs": [
                    {
                        "name": "worker",
                        "status": "completed",
                        "gpu_memory_mib": 456,
                        "gpu_memory_source": "nvidia-smi",
                        "metrics_path": str(metric_path),
                    }
                ],
            }
            with patch(
                "nvertake.runtime.query_gpu_process_memory", return_value={}
            ):
                snapshot = enrich_report(payload)
        self.assertEqual(snapshot["jobs"][0]["gpu_memory_mib"], 456)
        self.assertEqual(snapshot["jobs"][0]["gpu_memory_source"], "nvidia-smi")

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

    def test_local_log_reader_and_safe_stop_use_reported_processes(self):
        with tempfile.TemporaryDirectory() as root:
            log_path = Path(root) / "worker.log"
            log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
            payload = {
                "run_id": "run",
                "status": "running",
                "metadata": {
                    "launcher_pid": 999,
                    "launcher_create_time": 12.5,
                },
                "jobs": [
                    {
                        "name": "worker",
                        "pid": 123,
                        "log_path": str(log_path),
                    }
                ],
            }
            logs = read_report_logs(payload, job_name="worker", lines=2)
            with patch(
                "nvertake.runtime.process_create_time", return_value=12.5
            ), patch("nvertake.runtime.os.kill") as kill:
                stopped = stop_local_report(payload)

        self.assertEqual(logs[0]["content"], "two\nthree\n")
        self.assertEqual(stopped["pids"], [123, 999])
        self.assertEqual(kill.call_count, 2)

    def test_interrupted_signal_failure_is_recorded_as_cancelled(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            with patch.dict(
                os.environ,
                {"NVERTAKE_RUNTIME_DIR": str(root_path / "registry")},
                clear=False,
            ):
                report = RunReport(root_path / "report.json", run_id="cancel-run")
                report.add_jobs([{"name": "worker", "status": "running"}])
                report.update_job("worker", status="failed", exit_code=-15)
                report.mark_jobs_cancelled()
                snapshot = report.snapshot()
        self.assertEqual(snapshot["jobs"][0]["status"], "cancelled")
        self.assertEqual(snapshot["jobs"][0]["exit_code"], -15)

    def test_gpu_lease_rejects_a_second_launcher_and_releases_cleanly(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {"NVERTAKE_LEASE_DIR": root},
                clear=False,
            ):
                first = GpuLease(0, run_id="first").acquire()
                try:
                    with self.assertRaisesRegex(GpuLeaseError, "run first"):
                        GpuLease(0, run_id="second").acquire()
                finally:
                    first.release()
                with GpuLease(0, run_id="third"):
                    pass

    def test_list_refresh_marks_a_disappeared_launcher_and_prunes_missing_report(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            registry = root_path / "registry"
            with patch.dict(
                os.environ,
                {"NVERTAKE_RUNTIME_DIR": str(registry)},
                clear=False,
            ):
                report = RunReport(
                    root_path / "report.json",
                    run_id="orphan",
                    metadata={
                        "launcher_pid": 99999999,
                        "launcher_create_time": 1.0,
                    },
                )
                report.add_jobs([{"name": "worker", "status": "running"}])
                report.set_status("running")
                refreshed = list_registered_runs(refresh=True)
                (root_path / "report.json").unlink()
                self.assertEqual(list_registered_runs(prune=True), [])
                registry_exists = (registry / "orphan.json").exists()
        self.assertEqual(refreshed[0]["status"], "failed")
        self.assertFalse(registry_exists)


class TestRemoteOrchestration(unittest.TestCase):
    def _remote_config(self, root: Path):
        config_path = root / "remote.yaml"
        config_path.write_text(
            """hosts:
  first:
    ssh: first.example
    repo: /srv/nVertake
    python: /opt/torch/bin/python
defaults:
  host: first
  cwd: examples
  device: auto
jobs:
  - {name: a, script: throughput_worker.py, sm_share: 30}
  - {name: b, script: throughput_worker.py, sm_share: 70}
""",
            encoding="utf-8",
        )
        return load_job_config(str(config_path))

    def test_distributed_plan_checks_git_and_sends_resolved_devices(self):
        from nvertake.orchestration import plan_distributed_jobs

        with tempfile.TemporaryDirectory() as root:
            config = self._remote_config(Path(root))
            source = {
                "git": {"commit": "abc", "branch": "main", "dirty": False},
                "config_sha256": "hash",
                "coordinator": {},
            }
            probe = {
                "hostname": "first",
                "repo_path": "/srv/nVertake",
                "git": {"commit": "abc", "branch": "main", "dirty": False},
                "gpus": [{"index": 0, "memory_total_mib": 10, "memory_free_mib": 10}],
            }

            def fake_call(_host, action, payload, **_kwargs):
                self.assertEqual(action, "plan")
                self.assertEqual(
                    [job["device"] for job in payload["jobs"]], [0, 0]
                )
                return {"devices": [{"device": {"device": 0}, "jobs": []}]}

            with patch(
                "nvertake.orchestration._source_metadata", return_value=source
            ), patch(
                "nvertake.orchestration._probe_hosts",
                return_value={"first": probe},
            ), patch(
                "nvertake.orchestration.call_host", side_effect=fake_call
            ):
                plan = plan_distributed_jobs(config)

        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["hosts"][0]["name"], "first")

    def test_distributed_plan_rejects_mismatched_remote_commit(self):
        from nvertake.orchestration import plan_distributed_jobs

        with tempfile.TemporaryDirectory() as root:
            config = self._remote_config(Path(root))
            source = {
                "git": {"commit": "abc", "branch": "main", "dirty": False},
                "config_sha256": "hash",
                "coordinator": {},
            }
            probe = {
                "repo_path": "/srv/nVertake",
                "git": {"commit": "other", "branch": "main", "dirty": False},
                "gpus": [{"index": 0}],
            }
            with patch(
                "nvertake.orchestration._source_metadata", return_value=source
            ), patch(
                "nvertake.orchestration._probe_hosts",
                return_value={"first": probe},
            ):
                with self.assertRaisesRegex(RuntimeError, "Remote Git state"):
                    plan_distributed_jobs(config)

    def test_distributed_launch_writes_aggregate_report(self):
        from nvertake.orchestration import launch_distributed_jobs

        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            config = self._remote_config(root_path)
            source = {
                "git": {"commit": "abc", "branch": "main", "dirty": False},
                "config_sha256": "hash",
                "coordinator": {"hostname": "coordinator"},
            }
            probe = {
                "hostname": "first",
                "repo_path": "/srv/nVertake",
                "git": {"commit": "abc", "branch": "main", "dirty": False},
                "gpus": [{"index": 0}],
            }
            jobs = assign_auto_devices(config.jobs, {"first": probe["gpus"]})

            def snapshot(status):
                return {
                    "run_id": "remote",
                    "status": status,
                    "jobs": [
                        {
                            "name": job.name,
                            "status": (
                                "completed" if status == "completed" else "running"
                            ),
                            "device": 0,
                            "pid": 100 + index,
                            "sm_count": 10,
                            "log_path": f"/tmp/{job.name}.log",
                        }
                        for index, job in enumerate(jobs)
                    ],
                }

            def fake_call(_host, action, _payload, **_kwargs):
                if action == "launch":
                    return {
                        "exit_code": 0,
                        "snapshot": snapshot("completed"),
                    }
                if action == "snapshot":
                    return snapshot("running")
                raise AssertionError(action)

            with patch.dict(
                os.environ,
                {"NVERTAKE_RUNTIME_DIR": str(root_path / "registry")},
                clear=False,
            ), patch(
                "nvertake.orchestration._prepare",
                return_value=(source, {"first": probe}, jobs),
            ), patch(
                "nvertake.orchestration.call_host", side_effect=fake_call
            ):
                result = launch_distributed_jobs(config, quiet=True)
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["metadata"]["config_sha256"], "hash")
        self.assertTrue(all(job["remote"] for job in report["jobs"]))
        self.assertTrue(all(job["status"] == "completed" for job in report["jobs"]))

    def test_remote_agent_materializes_paths_under_host_repo(self):
        from nvertake.remote_agent import _materialize_config

        with tempfile.TemporaryDirectory() as root:
            repo = Path(root)
            examples = repo / "examples"
            examples.mkdir()
            _write_worker(examples / "worker.py")
            report_path = repo / ".nvertake" / "runs" / "id" / "report.json"
            config = _materialize_config(
                {
                    "repo": str(repo),
                    "host_name": "host",
                    "report_path": str(report_path),
                    "startup_timeout": 10,
                    "calibration": {},
                    "jobs": [
                        {
                            "name": "worker",
                            "cwd": "examples",
                            "script": "worker.py",
                            "args": [],
                            "calibration_args": [],
                            "sm_share": 1,
                            "target_share": 1,
                            "memory_share": None,
                            "device": 0,
                            "env": {},
                            "log": None,
                        }
                    ],
                }
            )

        self.assertEqual(config.jobs[0].script, (examples / "worker.py").resolve())
        self.assertEqual(config.jobs[0].host, "host")

    def test_ssh_transport_status_255_is_retried(self):
        from nvertake.jobs import HostSpec

        host = HostSpec(
            name="remote",
            repo="/repo",
            python="python",
            ssh="remote.example",
        )
        failed = SimpleNamespace(
            returncode=255,
            stdout="",
            stderr="connection reset",
        )
        succeeded = SimpleNamespace(
            returncode=0,
            stdout='{"ok": true}',
            stderr="",
        )
        with patch(
            "nvertake.orchestration.subprocess.run",
            side_effect=[failed, succeeded],
        ) as run, patch("nvertake.orchestration.time.sleep"):
            result = call_host(host, "probe", {}, attempts=3)
        self.assertTrue(result["ok"])
        self.assertEqual(run.call_count, 2)

    def test_remote_launch_request_reattaches_to_terminal_report(self):
        from nvertake.remote_agent import _existing_launch

        with tempfile.TemporaryDirectory() as root:
            repo = Path(root)
            report_path = repo / ".nvertake" / "runs" / "run" / "report.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "run",
                        "report_path": str(report_path),
                        "status": "completed",
                        "jobs": [],
                    }
                ),
                encoding="utf-8",
            )
            result = _existing_launch(
                {
                    "repo": str(repo),
                    "report_path": str(report_path),
                    "run_id": "run",
                }
            )
        self.assertTrue(result["reattached"])
        self.assertEqual(result["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
