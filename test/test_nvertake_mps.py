"""Unit tests for NVIDIA MPS configuration and CLI wiring."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from nvertake.cli import _build_child_env, create_parser
from nvertake.mps import (
    MPSControlError,
    MPSController,
    MPSPaths,
    MPSStatus,
    StaticMPSCapability,
    configure_mps_client_env,
    configure_static_mps_client_env,
    default_mps_paths,
    inspect_static_mps_capability,
    plan_static_mps_chunks,
    validate_active_thread_percentage,
)
from verification.run_mps_share_experiment import _wait_for_event


def _mock_controller(*, started: bool) -> MagicMock:
    controller = MagicMock()
    controller.ensure_started.return_value = started
    return controller


class TestMPSConfiguration(unittest.TestCase):
    def test_validate_active_thread_percentage(self):
        self.assertEqual(validate_active_thread_percentage(75), 75)
        for invalid in (0, 101, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_active_thread_percentage(invalid)

    def test_client_env_sets_share_and_removes_device_remapping(self):
        paths = MPSPaths(Path("/tmp/test-pipe"), Path("/tmp/test-log"))
        env = {
            "CUDA_VISIBLE_DEVICES": "3",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "10",
            "CUDA_MPS_CLIENT_PRIORITY": "1",
        }

        configured = configure_mps_client_env(
            env,
            paths=paths,
            active_thread_percentage=75,
            client_priority="normal",
        )

        self.assertNotIn("CUDA_VISIBLE_DEVICES", configured)
        self.assertEqual(configured["CUDA_MPS_PIPE_DIRECTORY"], "/tmp/test-pipe")
        self.assertEqual(configured["CUDA_MPS_LOG_DIRECTORY"], "/tmp/test-log")
        self.assertEqual(configured["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"], "75")
        self.assertEqual(configured["CUDA_MPS_CLIENT_PRIORITY"], "0")

    def test_client_env_clears_inherited_client_limits_when_unspecified(self):
        paths = MPSPaths(Path("/tmp/test-pipe"), Path("/tmp/test-log"))
        env = {
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "10",
            "CUDA_MPS_CLIENT_PRIORITY": "1",
        }
        configure_mps_client_env(env, paths=paths)
        self.assertNotIn("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", env)
        self.assertNotIn("CUDA_MPS_CLIENT_PRIORITY", env)

    def test_default_paths_are_per_device(self):
        first = default_mps_paths(0)
        second = default_mps_paths(1)
        self.assertNotEqual(first.pipe_directory, second.pipe_directory)
        self.assertNotEqual(first.log_directory, second.log_directory)

    def test_static_chunk_plan_uses_every_complete_chunk(self):
        self.assertEqual(
            plan_static_mps_chunks(
                (30, 70),
                total_sm_count=78,
                chunk_sm_count=8,
            ),
            (3, 6),
        )

    def test_static_client_env_selects_partition_and_clears_dynamic_limit(self):
        paths = MPSPaths(Path("/tmp/static-pipe"), Path("/tmp/static-log"))
        env = {
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "50",
        }
        configure_static_mps_client_env(
            env,
            paths=paths,
            partition_id="GPU-test/Dpartition",
        )
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
        self.assertNotIn("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", env)
        self.assertEqual(env["CUDA_MPS_SM_PARTITION"], "GPU-test/Dpartition")

    def test_static_capability_detects_new_control_option(self):
        completed = subprocess.CompletedProcess(
            ["nvidia-cuda-mps-control", "--help"],
            0,
            stdout="usage: nvidia-cuda-mps-control -d -S --static-partitioning",
            stderr="",
        )
        with patch.object(
            MPSController, "_platform_error", return_value=None
        ), patch.object(
            MPSController,
            "_resolved_control_binary",
            return_value="/usr/bin/nvidia-cuda-mps-control",
        ), patch(
            "nvertake.mps.subprocess.run",
            return_value=completed,
        ):
            capability = inspect_static_mps_capability(
                0, compute_capability_major=9
            )
        self.assertTrue(capability.available)
        self.assertEqual(capability.chunk_sm_count, 8)

    def test_cli_parses_gpu_share_and_mps_priority(self):
        args = create_parser().parse_args(
            [
                "--gpu-share",
                "75",
                "--mps-priority",
                "normal",
                "run",
                "train.py",
            ]
        )
        self.assertEqual(args.gpu_share, 75)
        self.assertEqual(args.mps_priority, "normal")

    def test_child_env_stops_new_daemon_when_probe_fails(self):
        args = create_parser().parse_args(["--gpu-share", "75", "run", "train.py"])
        controller = _mock_controller(started=True)
        controller.probe_client.side_effect = MPSControlError("probe failed")

        with patch("nvertake.cli.MPSController", return_value=controller):
            with self.assertRaisesRegex(MPSControlError, "probe failed"):
                _build_child_env(args, device=0)

        controller.stop.assert_called_once_with(force=True)

    def test_child_env_preserves_probe_error_when_cleanup_fails(self):
        args = create_parser().parse_args(["--gpu-share", "75", "run", "train.py"])
        controller = _mock_controller(started=True)
        controller.probe_client.side_effect = MPSControlError("probe failed")
        controller.stop.side_effect = MPSControlError("cleanup failed")

        with patch("nvertake.cli.MPSController", return_value=controller):
            with self.assertLogs("nvertake", level="WARNING"):
                with self.assertRaisesRegex(MPSControlError, "probe failed"):
                    _build_child_env(args, device=0)

    def test_child_env_keeps_preexisting_daemon_when_probe_fails(self):
        args = create_parser().parse_args(["--gpu-share", "75", "run", "train.py"])
        controller = _mock_controller(started=False)
        controller.probe_client.side_effect = MPSControlError("probe failed")

        with patch("nvertake.cli.MPSController", return_value=controller):
            with self.assertRaisesRegex(MPSControlError, "probe failed"):
                _build_child_env(args, device=0)

        controller.stop.assert_not_called()


class TestMPSController(unittest.TestCase):
    def _controller(self, root: str) -> MPSController:
        return MPSController(
            device=2,
            pipe_directory=str(Path(root) / "pipe"),
            log_directory=str(Path(root) / "log"),
        )

    def test_start_selects_gpu_by_uuid_and_keeps_daemon_unrestricted(self):
        with tempfile.TemporaryDirectory() as root:
            controller = self._controller(root)
            completed = subprocess.CompletedProcess(
                ["nvidia-cuda-mps-control", "-d"],
                0,
                stdout="",
                stderr="",
            )
            with ExitStack() as stack:
                stack.enter_context(patch.object(controller, "_platform_error", return_value=None))
                stack.enter_context(
                    patch.object(
                        controller,
                        "_resolved_control_binary",
                        return_value="/usr/bin/nvidia-cuda-mps-control",
                    )
                )
                stack.enter_context(
                    patch.object(controller, "is_running", side_effect=[False, True])
                )
                stack.enter_context(
                    patch.object(controller, "_gpu_uuid", return_value="GPU-test-uuid")
                )
                run = stack.enter_context(
                    patch("nvertake.mps.subprocess.run", return_value=completed)
                )
                self.assertTrue(controller.start())

            daemon_call = run.call_args
            self.assertEqual(
                daemon_call.args[0],
                ["/usr/bin/nvidia-cuda-mps-control", "-d"],
            )
            daemon_env = daemon_call.kwargs["env"]
            self.assertEqual(daemon_env["CUDA_VISIBLE_DEVICES"], "GPU-test-uuid")
            self.assertEqual(daemon_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"], "100")
            self.assertEqual(daemon_env["CUDA_MPS_CLIENT_PRIORITY"], "0")
            self.assertEqual(os.stat(controller.paths.pipe_directory).st_mode & 0o777, 0o700)

    def test_probe_client_returns_device_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            controller = self._controller(root)
            payload = {"device_count": 1, "device_name": "Test GPU", "sm_count": 42}
            completed = subprocess.CompletedProcess(
                ["python", "-c", "probe"],
                0,
                stdout="__NVERTAKE_MPS_PROBE__=" + json.dumps(payload) + "\n",
                stderr="",
            )
            with patch("nvertake.mps.subprocess.run", return_value=completed) as run:
                self.assertEqual(controller.probe_client({"A": "B"}), payload)
            probe_code = run.call_args.args[0][2]
            compile(probe_code, "<nvertake-mps-probe>", "exec")

    def test_start_static_uses_static_daemon_flag(self):
        with tempfile.TemporaryDirectory() as root:
            controller = self._controller(root)
            capability = StaticMPSCapability(
                True,
                2,
                "/usr/bin/nvidia-cuda-mps-control",
                9,
                8,
                "available",
            )
            completed = subprocess.CompletedProcess(
                ["nvidia-cuda-mps-control", "-d", "-S"],
                0,
                stdout="",
                stderr="",
            )
            lspart = subprocess.CompletedProcess(
                ["nvidia-cuda-mps-control"],
                0,
                stdout="GPU Partition",
                stderr="",
            )
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        controller,
                        "_ensure_supported",
                        return_value="/usr/bin/nvidia-cuda-mps-control",
                    )
                )
                stack.enter_context(
                    patch(
                        "nvertake.mps.inspect_static_mps_capability",
                        return_value=capability,
                    )
                )
                stack.enter_context(
                    patch.object(controller, "is_running", side_effect=[False, True])
                )
                stack.enter_context(
                    patch.object(controller, "_gpu_uuid", return_value="GPU-test")
                )
                stack.enter_context(
                    patch.object(controller, "_run_control", return_value=lspart)
                )
                run = stack.enter_context(
                    patch("nvertake.mps.subprocess.run", return_value=completed)
                )
                self.assertTrue(controller.start_static())
            self.assertEqual(
                run.call_args.args[0],
                ["/usr/bin/nvidia-cuda-mps-control", "-d", "-S"],
            )

    def test_probe_failure_includes_server_log(self):
        with tempfile.TemporaryDirectory() as root:
            controller = self._controller(root)
            controller.paths.log_directory.mkdir(parents=True)
            (controller.paths.log_directory / "server.log").write_text(
                "Device with matching device ID not found.\n",
                encoding="utf-8",
            )
            completed = subprocess.CompletedProcess(
                ["python", "-c", "probe"],
                1,
                stdout="",
                stderr="CUDA error 805",
            )
            with patch("nvertake.mps.subprocess.run", return_value=completed):
                with self.assertRaisesRegex(MPSControlError, "Device with matching"):
                    controller.probe_client({})

    def test_stop_refuses_active_clients_without_force(self):
        with tempfile.TemporaryDirectory() as root:
            controller = self._controller(root)
            status = MPSStatus(
                available=True,
                running=True,
                pipe_directory=controller.paths.pipe_directory,
                log_directory=controller.paths.log_directory,
                server_pids=(100,),
                client_pids=(200, 201),
            )
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        controller,
                        "_ensure_supported",
                        return_value="/usr/bin/nvidia-cuda-mps-control",
                    )
                )
                stack.enter_context(patch.object(controller, "status", return_value=status))
                with self.assertRaisesRegex(MPSControlError, "200, 201"):
                    controller.stop()

    def test_force_stop_bypasses_client_query_and_terminates_wedged_daemon(self):
        with tempfile.TemporaryDirectory() as root:
            controller = self._controller(root)
            managed = (
                (100, "nvidia-cuda-mps-server"),
                (101, "nvidia-cuda-mps-control"),
            )
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        controller,
                        "_ensure_supported",
                        return_value="/usr/bin/nvidia-cuda-mps-control",
                    )
                )
                stack.enter_context(patch.object(controller, "is_running", return_value=True))
                stack.enter_context(
                    patch.object(controller, "_managed_processes", return_value=managed)
                )
                stack.enter_context(
                    patch.object(
                        controller,
                        "_run_control",
                        side_effect=MPSControlError("control timed out"),
                    )
                )
                terminate = stack.enter_context(
                    patch.object(
                        controller,
                        "_force_terminate_managed_processes",
                        return_value=True,
                    )
                )
                status = stack.enter_context(patch.object(controller, "status"))

                self.assertTrue(controller.stop(force=True))

            status.assert_not_called()
            terminate.assert_called_once_with(managed)

    def test_status_reports_platform_limitation(self):
        controller = MPSController()
        with patch.object(controller, "_platform_error", return_value="unsupported platform"):
            status = controller.status()
        self.assertFalse(status.available)
        self.assertFalse(status.running)
        self.assertEqual(status.detail, "unsupported platform")


class TestMPSShareExperiment(unittest.TestCase):
    def test_worker_exit_includes_stderr_before_temporary_artifacts_are_removed(self):
        with tempfile.TemporaryDirectory() as root:
            event_path = Path(root) / "worker.jsonl"
            stderr_path = Path(root) / "worker.stderr.log"
            stderr_path.write_text("CUDA error 805\nserver diagnostic\n", encoding="utf-8")
            process = MagicMock()
            process.poll.return_value = 1
            process.returncode = 1

            with self.assertRaisesRegex(RuntimeError, "CUDA error 805"):
                _wait_for_event(
                    event_path,
                    "ready",
                    timeout=1.0,
                    process=process,
                    stderr_path=stderr_path,
                )


if __name__ == "__main__":
    unittest.main()
