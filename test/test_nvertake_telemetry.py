"""Tests for optional nvidia-smi and DCGM telemetry parsing."""

from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from nvertake.telemetry import (
    inspect_telemetry_capabilities,
    query_dcgm_profile,
    query_device_utilization,
    query_process_utilization,
)


class TestTelemetry(unittest.TestCase):
    def test_parses_per_process_pmon_sample(self):
        completed = subprocess.CompletedProcess(
            ["nvidia-smi", "pmon"],
            0,
            stdout=(
                "# gpu pid type sm mem enc dec jpg ofa command\n"
                "0 123 C 87 12 - - - - python\n"
            ),
            stderr="",
        )
        with patch(
            "nvertake.telemetry.subprocess.run",
            return_value=completed,
        ):
            samples = query_process_utilization()
        self.assertEqual(samples[(0, 123)]["sm_util_percent"], 87.0)
        self.assertEqual(samples[(0, 123)]["memory_util_percent"], 12.0)

    def test_parses_device_utilization(self):
        completed = subprocess.CompletedProcess(
            ["nvidia-smi"],
            0,
            stdout="0, 99, 40, 1800, 9000, 250, 450, 67\n",
            stderr="",
        )
        with patch(
            "nvertake.telemetry.subprocess.run",
            return_value=completed,
        ):
            records = query_device_utilization([0])
        self.assertEqual(records[0]["gpu_util_percent"], 99.0)
        self.assertEqual(records[0]["power_draw_w"], 250.0)

    def test_dcgm_unavailable_reason_is_preserved(self):
        completed = subprocess.CompletedProcess(
            ["dcgmi"],
            1,
            stdout="",
            stderr="Profiling module is not loaded",
        )
        with patch(
            "nvertake.telemetry.shutil.which",
            return_value="/usr/bin/dcgmi",
        ), patch(
            "nvertake.telemetry.subprocess.run",
            return_value=completed,
        ):
            record = query_dcgm_profile(0)
        self.assertFalse(record["available"])
        self.assertIn("not loaded", record["detail"])

    def test_wsl_capability_does_not_claim_per_process_pmon(self):
        completed = subprocess.CompletedProcess(
            ["nvidia-smi", "pmon", "-h"],
            0,
            stdout="pmon help",
            stderr="",
        )
        with patch(
            "nvertake.telemetry.shutil.which",
            side_effect=(
                lambda name: (
                    "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
                )
            ),
        ), patch(
            "nvertake.telemetry.subprocess.run",
            return_value=completed,
        ), patch(
            "nvertake.telemetry.platform.release",
            return_value="6.1.0-microsoft-standard-WSL2",
        ):
            capabilities = inspect_telemetry_capabilities(0)
        process = capabilities["process_utilization"]
        self.assertFalse(process["available"])
        self.assertIn("WSL", process["detail"])


if __name__ == "__main__":
    unittest.main()
