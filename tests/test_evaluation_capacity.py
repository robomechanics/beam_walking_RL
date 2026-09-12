import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluation_capacity import classify_compute_processes, check_evaluation_capacity


def process(name="/usr/share/rustdesk/rustdesk", owner="rml2", memory=279,
            command="/usr/share/rustdesk/rustdesk --server"):
    return {"pid": 1, "process_name": name, "owner": owner,
            "used_gpu_memory_mib": memory, "command": command}


class EvaluationCapacityTest(unittest.TestCase):
    def test_allows_only_small_current_user_rustdesk(self):
        allowed, blocked = classify_compute_processes([process()], "rml2")
        self.assertEqual(len(allowed), 1)
        self.assertEqual(blocked, [])

    def test_rejects_other_process_owner_or_large_context(self):
        cases = [
            process(name="python", command="python train.py"),
            process(owner="someone_else"),
            process(memory=513),
        ]
        for candidate in cases:
            with self.subTest(candidate=candidate):
                allowed, blocked = classify_compute_processes([candidate], "rml2")
                self.assertEqual(allowed, [])
                self.assertEqual(blocked, [candidate])

    def test_rejects_excess_aggregate_rustdesk_memory(self):
        processes = [process(memory=300), process(memory=300)]
        allowed, blocked = classify_compute_processes(processes, "rml2")
        self.assertEqual(allowed, [])
        self.assertEqual(blocked, processes)

    def test_shared_gpu_accepts_ros_process_and_preserves_memory_limits(self):
        driver = process(name="robot_driver_node", command="robot_driver_node", memory=242)
        with patch("evaluation_capacity._compute_processes", return_value=[driver]), patch(
                "evaluation_capacity.Path.read_text", return_value="MemAvailable: 16777216 kB\n"), patch(
                "evaluation_capacity.subprocess.run") as query:
            query.return_value = SimpleNamespace(stdout="GPU, 16384, 1024, 15360, 20, 50")
            record = check_evaluation_capacity(64)
            self.assertTrue(record["shared_gpu_authorized"])
            self.assertEqual(record["blocked_compute_processes"], [])
            self.assertEqual(record["other_compute_processes"], [driver])
            query.return_value = SimpleNamespace(stdout="GPU, 16384, 15360, 1024, 20, 50")
            with self.assertRaisesRegex(RuntimeError, "Insufficient free memory"):
                check_evaluation_capacity(64)
            query.return_value = SimpleNamespace(stdout="GPU, 16384, 1024, 15360, 20, 50")
            with patch("evaluation_capacity.Path.read_text", return_value="MemAvailable: 1024 kB\n"):
                with self.assertRaisesRegex(RuntimeError, "Insufficient free memory"):
                    check_evaluation_capacity(64)


if __name__ == "__main__":
    unittest.main()
