"""CPU checks for the deferred simulator launch prerequisites."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from subprocess import CompletedProcess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "deployment_queue", ROOT / "scripts/queue_deployment_smoke.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class DeploymentQueueTest(unittest.TestCase):
    def test_requires_final_checkpoint_and_successful_training_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.assertFalse(queue.adaptive_completed(run))
            (run / "model_1799.pt").touch()
            self.assertFalse(queue.adaptive_completed(run))
            report = {"mode": "train", "updates": 1800, "num_envs": 3072}
            (run / "throughput.json").write_text(json.dumps(report))
            self.assertTrue(queue.adaptive_completed(run))
            report["updates"] = 100
            (run / "throughput.json").write_text(json.dumps(report))
            self.assertFalse(queue.adaptive_completed(run))

    def test_process_identity_is_stable_and_missing_pid_is_exited(self):
        self.assertEqual(queue.process_token(os.getpid()),
                         queue.process_token(os.getpid()))
        self.assertIsNotNone(queue.process_token(os.getpid()))
        self.assertIsNone(queue.process_token(999999999))

    def test_zero_exit_without_smoke_artifact_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "results/adaptive"
            run.mkdir(parents=True)
            (run / "watcher_process.json").write_text(
                json.dumps({"training_pid": 999999999}))
            destination = root / "results/queue"
            argv = ["queue", "--adaptive-run", str(run),
                    "--queue-dir", str(destination), "--smoke-output",
                    str(root / "results/smoke")]
            with (patch.object(queue, "ROOT", root),
                  patch.object(sys, "argv", argv),
                  patch.object(queue, "source_hashes", return_value={"x": "y"}),
                  patch.object(queue, "adaptive_completed", return_value=True),
                  patch.object(queue, "check_capacity", return_value={}),
                  patch.object(queue.subprocess, "run",
                               return_value=CompletedProcess([], 0)) as launch):
                self.assertEqual(queue.main(), 1)
                launch.assert_called_once()
            status = json.loads((destination / "status.json").read_text())
            self.assertEqual(status["state"], "smoke_failed_review_required")
            self.assertFalse(status["training_queued"])


if __name__ == "__main__":
    unittest.main()
