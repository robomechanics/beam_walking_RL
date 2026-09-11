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

    def test_rejects_non_rustdesk_compute_process_without_stopping_it(self):
        def run(command, **kwargs):
            if any("query-compute-apps" in item for item in command):
                return self.completed("999999, other_job, 4000\n")
            return self.completed("GPU, 24564, 4000, 20564, 5, 40\n")
        with patch.object(gpu_capacity.subprocess, "run", side_effect=run):
            with self.assertRaisesRegex(RuntimeError, "outside the narrow RustDesk exception"):
                gpu_capacity.check_capacity("train", 4096, "cuda:0", False)


    def test_evaluate_mode_keeps_strict_exclusivity(self):
        def run(command, **kwargs):
            if any("query-compute-apps" in item for item in command):
                return self.completed("12345, /usr/share/rustdesk/rustdesk, 279\n")
            return self.completed("GPU, 24564, 1000, 23564, 0, 35\n")

        status = "Uid:\t" + str(gpu_capacity.os.getuid()) + "\t0\t0\t0\n"
        with patch.object(gpu_capacity.subprocess, "run", side_effect=run), \
             patch.object(gpu_capacity.Path, "read_text", autospec=True) as read_text, \
             patch.object(gpu_capacity.Path, "read_bytes", autospec=True) as read_bytes:
            def fake_text(path):
                return status if str(path).endswith("/status") else "MemAvailable: 20000000 kB\n"
            read_text.side_effect = fake_text
            read_bytes.return_value = b"/usr/share/rustdesk/rustdesk\0--server\0"
            with self.assertRaisesRegex(
                    RuntimeError, "outside the narrow RustDesk exception"):
                gpu_capacity.check_capacity("evaluate", 2328, "cuda:0", False)

    def test_allows_only_bounded_current_user_rustdesk(self):
        owner = gpu_capacity.pwd.getpwuid(gpu_capacity.os.getuid()).pw_name

        def run(command, **kwargs):
            if any("query-compute-apps" in item for item in command):
                return self.completed("12345, /usr/share/rustdesk/rustdesk, 279\n")
            return self.completed("GPU, 24564, 1000, 23564, 0, 35\n")

        status = "Uid:\t" + str(gpu_capacity.os.getuid()) + "\t0\t0\t0\n"
        with patch.object(gpu_capacity.subprocess, "run", side_effect=run), \
             patch.object(gpu_capacity.Path, "read_text", autospec=True) as read_text, \
             patch.object(gpu_capacity.Path, "read_bytes", autospec=True) as read_bytes:
            def fake_text(path):
                return status if str(path).endswith("/status") else "MemAvailable: 20000000 kB\n"
            read_text.side_effect = fake_text
            read_bytes.return_value = b"/usr/share/rustdesk/rustdesk\0--server\0"
            record = gpu_capacity.check_capacity("train", 4096, "cuda:0", False)
        self.assertEqual(record["allowed_rustdesk_processes"][0]["owner"], owner)
        self.assertEqual(record["blocked_compute_processes"], [])
        self.assertTrue(record["rustdesk_exception_applied"])



if __name__ == "__main__":
    unittest.main()
