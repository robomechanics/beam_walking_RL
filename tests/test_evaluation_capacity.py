import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluation_capacity import classify_compute_processes


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


if __name__ == "__main__":
    unittest.main()
