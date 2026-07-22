"""Unit tests for CUDA Green Context partitioning and task execution."""

from __future__ import annotations

import json
import threading
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from unittest.mock import MagicMock, patch

from nvertake.cli import create_parser, main
from nvertake.green_context import (
    GreenContextError,
    GreenContextExecutor,
    GreenRunResult,
    _DeviceResource,
    _requested_partition_sm_count,
    current_green_context_lane,
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
    def __init__(self, total_sms=110, compute_capability_major=12):
        self.total_sms = total_sms
        self.major = compute_capability_major
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


if __name__ == "__main__":
    unittest.main()
