"""Unit tests for CUDA Green Context partitioning and task execution."""

from __future__ import annotations

import json
import os
import threading
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from nvertake.cli import create_parser, main
from nvertake.green_context import (
    GreenContextError,
    GreenContextExecutor,
    GreenProcessContext,
    GreenRunResult,
    _DeviceResource,
    _requested_partition_sm_count,
    current_green_context_lane,
)
from nvertake.green_process import (
    GreenProcessLaunchError,
    _child_environment,
    _execute_python_script,
    _validate_ready_metadata,
    run_green_process_scripts,
)


@dataclass
class _FakeResource:
    sm_count: int


@dataclass
class _FakeHandle:
    green_context: int
    cuda_context: int
    descriptor: object = None
    resources: object = None


class _FakeDriver:
    def __init__(
        self,
        total_sms=110,
        compute_capability_major=12,
        fine_group_sm_count=2,
    ):
        self.total_sms = total_sms
        self.major = compute_capability_major
        self.fine_group_sm_count = fine_group_sm_count
        self.requested_sm_count = None
        self.created = []
        self.destroyed = []
        self.synchronize_calls = 0
        self._thread_context = threading.local()

    def initialize(self):
        return None

    def get_device(self, ordinal):
        return ordinal + 100

    def compute_capability_major(self, device):
        return self.major

    def device_sm_resource(self, device):
        return _FakeResource(self.total_sms)

    @staticmethod
    def sm_count(resource):
        return resource.sm_count

    def split_sm_resource(self, resource, requested_sm_count):
        self.requested_sm_count = requested_sm_count
        return (
            _FakeResource(requested_sm_count),
            _FakeResource(resource.sm_count - requested_sm_count),
        )

    def split_sm_resources(
        self, resource, *, group_count, requested_sm_count, flags=0
    ):
        actual_sm_count = (
            self.fine_group_sm_count
            if flags and self.major >= 7 and requested_sm_count < 2
            else requested_sm_count
        )
        actual_group_count = min(group_count, resource.sm_count // actual_sm_count)
        groups = tuple(
            _FakeResource(actual_sm_count) for _ in range(actual_group_count)
        )
        remainder = _FakeResource(
            resource.sm_count - actual_group_count * actual_sm_count
        )
        return groups, remainder

    def sm_resource_group_count(self, resource, *, requested_sm_count, flags=0):
        actual_sm_count = (
            self.fine_group_sm_count
            if flags and self.major >= 7 and requested_sm_count < 2
            else requested_sm_count
        )
        return resource.sm_count // actual_sm_count

    def create_green_context(self, device, resources):
        index = len(self.created)
        handle = _FakeHandle(index + 1, 1000 + index)
        handle.resources = tuple(resources)
        self.created.append(handle)
        return handle

    def context_sm_count(self, context):
        handle = next(item for item in self.created if item.cuda_context == context)
        return sum(resource.sm_count for resource in handle.resources)

    def current_context(self):
        return getattr(self._thread_context, "value", None)

    def set_current_context(self, context):
        self._thread_context.value = context

    def synchronize_current_context(self):
        self.synchronize_calls += 1

    def destroy_green_context(self, handle):
        self.destroyed.append(handle.green_context)


class TestGreenContextPartitioning(unittest.TestCase):
    def test_driver_resource_abi_size(self):
        import ctypes

        self.assertEqual(ctypes.sizeof(_DeviceResource), 144)

    def test_partition_rounding_matches_tested_blackwell_split(self):
        self.assertEqual(_requested_partition_sm_count(110, 25.0, 12), 24)
        self.assertEqual(_requested_partition_sm_count(110, 50.0, 12), 56)

    def test_partition_rounding_matches_tested_ampere_split(self):
        self.assertEqual(_requested_partition_sm_count(84, 25.0, 8), 20)
        self.assertEqual(_requested_partition_sm_count(84, 50.0, 8), 42)
        self.assertEqual(_requested_partition_sm_count(84, 1.0, 8), 4)

    def test_executor_preserves_share_order_and_uses_all_sms(self):
        driver = _FakeDriver()
        with GreenContextExecutor(shares=(75, 25), _driver=driver) as executor:
            self.assertEqual(driver.requested_sm_count, 24)
            self.assertEqual([lane.sm_count for lane in executor.lanes], [86, 24])
            self.assertAlmostEqual(executor.lanes[0].requested_share, 75.0)
            self.assertEqual(sum(lane.sm_count for lane in executor.lanes), 110)

        self.assertEqual(driver.destroyed, [2, 1])


class TestGreenContextExecution(unittest.TestCase):
    def test_two_tasks_run_in_bound_contexts_and_return_in_lane_order(self):
        driver = _FakeDriver()
        executor = GreenContextExecutor(shares=(25, 75), _driver=driver)

        def task(label):
            lane = current_green_context_lane()
            return {
                "label": label,
                "lane": lane.index,
                "sm_count": lane.sm_count,
                "context": driver.current_context(),
            }

        result = executor.run(
            (task, task),
            task_kwargs=({"label": "background"}, {"label": "target"}),
        )
        executor.close()

        self.assertEqual(result.results[0]["label"], "background")
        self.assertEqual(result.results[0]["context"], 1000)
        self.assertEqual(result.results[1]["context"], 1001)
        self.assertEqual([item["sm_count"] for item in result.results], [24, 86])
        self.assertEqual(driver.synchronize_calls, 2)

    def test_task_error_is_reported_with_lane(self):
        driver = _FakeDriver()
        executor = GreenContextExecutor(_driver=driver)

        def fail():
            raise RuntimeError("intentional failure")

        with self.assertRaisesRegex(GreenContextError, "lane 0.*intentional failure"):
            executor.run((fail, lambda: "ok"))
        executor.close()

    def test_result_serializes_actual_sm_share(self):
        driver = _FakeDriver(total_sms=84, compute_capability_major=8)
        with GreenContextExecutor(_driver=driver) as executor:
            result = executor.run((lambda: 1, lambda: 2))

        payload = result.to_dict()
        self.assertEqual(payload["results"], [1, 2])
        self.assertAlmostEqual(payload["lanes"][1]["actual_sm_share"], 64 / 84)


class TestGreenProcessContext(unittest.TestCase):
    def test_three_process_partition_uses_all_sms(self):
        driver = _FakeDriver(total_sms=110, compute_capability_major=12)
        with GreenProcessContext(
            shares=(20, 30, 50), lane_index=2, _driver=driver
        ) as context:
            self.assertEqual([lane.sm_count for lane in context.lanes], [22, 34, 54])
            self.assertEqual(context.lane.sm_count, 54)
            with context.bind() as lane:
                self.assertEqual(lane.index, 2)
                self.assertEqual(driver.current_context(), 1000)
                self.assertEqual(current_green_context_lane(), lane)

        self.assertEqual(driver.destroyed, [1])

    def test_four_equal_processes_report_architecture_rounding(self):
        driver = _FakeDriver(total_sms=110, compute_capability_major=12)
        with GreenProcessContext(
            shares=(1, 1, 1, 1), lane_index=0, _driver=driver
        ) as context:
            self.assertEqual(
                [lane.sm_count for lane in context.lanes], [28, 28, 28, 26]
            )

    def test_process_plan_queries_driver_fine_group_count(self):
        driver = _FakeDriver(
            total_sms=15,
            compute_capability_major=12,
            fine_group_sm_count=1,
        )
        with GreenProcessContext(
            shares=(20, 30, 50), lane_index=1, _driver=driver
        ) as context:
            self.assertEqual([lane.sm_count for lane in context.lanes], [3, 5, 7])

    def test_process_count_is_limited_by_minimum_partition_size(self):
        driver = _FakeDriver(total_sms=110, compute_capability_major=12)
        with self.assertRaisesRegex(GreenContextError, "cannot create 56"):
            GreenProcessContext(shares=(1,) * 56, lane_index=0, _driver=driver)


class TestGreenProcessLauncherHelpers(unittest.TestCase):
    def test_child_environment_selects_device_and_clears_inherited_limits(self):
        with patch.dict(
            os.environ,
            {
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "25",
                "CUDA_MPS_PIPE_DIRECTORY": "/tmp/inherited-mps",
                "NVERTAKE_AUTO_PRIORITY": "1",
            },
            clear=False,
        ):
            env = _child_environment(3)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(env["NVERTAKE_GREEN_PROCESS"], "1")
        self.assertNotIn("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", env)
        self.assertNotIn("CUDA_MPS_PIPE_DIRECTORY", env)
        self.assertNotIn("NVERTAKE_AUTO_PRIORITY", env)

    def test_custom_environment_cannot_override_green_process_isolation(self):
        env = _child_environment(
            2,
            {
                "CUDA_VISIBLE_DEVICES": "7",
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "50",
                "NVERTAKE_AUTO_PRIORITY": "1",
                "USER_SETTING": "kept",
                "PYTHONPATH": "/custom",
            },
        )
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(env["USER_SETTING"], "kept")
        self.assertNotIn("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", env)
        self.assertNotIn("NVERTAKE_AUTO_PRIORITY", env)
        self.assertTrue(env["PYTHONPATH"].endswith(os.pathsep + "/custom"))

    def test_ready_metadata_requires_identical_partition_maps(self):
        metadata = (
            {
                "lane_index": 0,
                "pid": 100,
                "sm_count": 24,
                "partition_sm_counts": [24, 32, 54],
            },
            {
                "lane_index": 1,
                "pid": 101,
                "sm_count": 32,
                "partition_sm_counts": [24, 32, 54],
            },
            {
                "lane_index": 2,
                "pid": 102,
                "sm_count": 55,
                "partition_sm_counts": [24, 31, 55],
            },
        )
        with self.assertRaisesRegex(GreenProcessLaunchError, "different"):
            _validate_ready_metadata(metadata, 3)

    def test_python_script_receives_its_own_arguments(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = os.path.abspath(root)
            script = os.path.join(root_path, "worker.py")
            output = os.path.join(root_path, "argv.json")
            with open(script, "w", encoding="utf-8") as handle:
                handle.write(
                    "import json, sys\nfrom pathlib import Path\n"
                    f"Path({output!r}).write_text(json.dumps(sys.argv))\n"
                )

            self.assertEqual(
                _execute_python_script(Path(script), ("--value", "42")),
                0,
            )
            with open(output, encoding="utf-8") as handle:
                argv = json.load(handle)
            self.assertEqual(argv, [script, "--value", "42"])

    def test_script_argument_lists_must_match_process_count(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "worker.py"
            script.write_text("pass\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one argument list"):
                run_green_process_scripts(
                    [str(script), str(script)],
                    shares=(50, 50),
                    script_args=(),
                )
            with self.assertRaisesRegex(TypeError, "argument must be a string"):
                run_green_process_scripts(
                    [str(script), str(script)],
                    shares=(50, 50),
                    script_args=("ab", "cd"),
                )


class TestGreenContextCLI(unittest.TestCase):
    def test_parser_accepts_two_green_tasks(self):
        args = create_parser().parse_args(
            [
                "--device",
                "1",
                "green-run",
                "--shares",
                "25,75",
                "--task",
                "jobs:background",
                "--task",
                "jobs:target",
            ]
        )
        self.assertEqual(args.command, "green-run")
        self.assertEqual(args.green_tasks, ["jobs:background", "jobs:target"])

    def test_main_prints_green_run_json(self):
        fake_result = MagicMock(spec=GreenRunResult)
        fake_result.to_dict.return_value = {"total_sm_count": 110, "results": [1, 2]}
        output = StringIO()
        with patch("nvertake.green_context.run_green_tasks", return_value=fake_result) as run:
            with redirect_stdout(output):
                return_code = main(
                    [
                        "green-run",
                        "--shares",
                        "25,75",
                        "--task",
                        "jobs:background",
                        "--task",
                        "jobs:target",
                        "--task-kwargs",
                        '{"seconds": 1}',
                        "--task-kwargs",
                        '{"seconds": 2}',
                    ]
                )

        self.assertEqual(return_code, 0)
        self.assertEqual(json.loads(output.getvalue())["results"], [1, 2])
        self.assertEqual(run.call_args.kwargs["shares"], (25.0, 75.0))
        self.assertEqual(run.call_args.kwargs["task_kwargs"][1]["seconds"], 2)

    def test_parser_accepts_three_green_process_scripts(self):
        args = create_parser().parse_args(
            [
                "green-procs",
                "--shares",
                "20,30,50",
                "first.py",
                "second.py",
                "third.py",
            ]
        )
        self.assertEqual(args.command, "green-procs")
        self.assertEqual(args.scripts, ["first.py", "second.py", "third.py"])

    def test_main_launches_three_green_process_scripts(self):
        with patch(
            "nvertake.green_process.run_green_process_scripts", return_value=0
        ) as launch:
            return_code = main(
                [
                    "green-procs",
                    "--shares",
                    "20,30,50",
                    "--script-args",
                    '["--rank", "0"]',
                    "--script-args",
                    '["--rank", "1"]',
                    "--script-args",
                    '["--rank", "2"]',
                    "first.py",
                    "second.py",
                    "third.py",
                ]
            )

        self.assertEqual(return_code, 0)
        self.assertEqual(launch.call_args.kwargs["shares"], (20.0, 30.0, 50.0))
        self.assertEqual(launch.call_args.kwargs["script_args"][2], ("--rank", "2"))


if __name__ == "__main__":
    unittest.main()
