"""CPU-only checks for the pre-Isaac exclusive GPU/RAM gate."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gpu_capacity


class CapacityTest(unittest.TestCase):
    @staticmethod
    def completed(output):
        return SimpleNamespace(stdout=output)

    def test_records_exclusive_process_state_and_conservative_reserves(self):
        def run(command, **kwargs):
            if any("query-compute-apps" in item for item in command):
                return self.completed("")
            return self.completed("GPU, 24564, 1000, 23564, 0, 35\n")
        with patch.object(gpu_capacity.subprocess, "run", side_effect=run):
            record = gpu_capacity.check_capacity(
                "evaluate", 2328, "cuda:0", False)
        self.assertTrue(record["exclusive_compute_device"])
        self.assertEqual(record["compute_processes"], [])
        self.assertGreaterEqual(record["required_free_mib"], 6700)
        self.assertGreaterEqual(record["required_system_mib"], 12000)

    def test_rejects_any_existing_compute_process_without_stopping_it(self):
        def run(command, **kwargs):
            if any("query-compute-apps" in item for item in command):
                return self.completed("999999, other_job, 4000\n")
            return self.completed("GPU, 24564, 4000, 20564, 5, 40\n")
        with patch.object(gpu_capacity.subprocess, "run", side_effect=run):
            with self.assertRaisesRegex(RuntimeError, "active compute"):
                gpu_capacity.check_capacity("train", 4096, "cuda:0", False)


if __name__ == "__main__":
    unittest.main()
