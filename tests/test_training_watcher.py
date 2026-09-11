"""Synthetic CPU-only watcher tests; no training results are generated."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("watch_training", Path(__file__).resolve().parents[1] / "scripts/watch_training.py")
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


class TrainingWatcherTests(unittest.TestCase):
    def test_tensorboard_events_merge_latest_resumed_step(self):
        try:
            from tensorboard.compat.proto.event_pb2 import Event
            from tensorboard.compat.proto.summary_pb2 import Summary
            from tensorboard.summary.writer.event_file_writer import EventFileWriter
        except ImportError:
            self.skipTest("TensorBoard is optional for the JSONL reader")
        with tempfile.TemporaryDirectory() as directory:
            for suffix, timestamp, values in [(".first", 100., [(0, 1.), (1, 2.)]),
                                               (".resume", 200., [(1, 3.), (2, 4.)])]:
                writer = EventFileWriter(directory, filename_suffix=suffix)
                for step, value in values:
                    writer.add_event(Event(wall_time=timestamp, step=step,
                                           summary=Summary(value=[Summary.Value(tag="Train/mean_reward", simple_value=value)])))
                writer.flush()
                writer.close()
            series, warnings = watcher.load_tensorboard(directory)
            self.assertEqual(series["Train/mean_reward"], [(0, 1.), (1, 3.), (2, 4.)])
            self.assertEqual(warnings, [])

    def test_live_json_tail_and_resumed_duplicate_step(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            path.write_text('{"iteration": 0, "scalars": {"Train/mean_reward": 1}}\n'
                            '{"iteration": 0, "scalars": {"Train/mean_reward": 2}}\n'
                            '{"iteration": 1, "scalars":')
            series, warnings = watcher.load_json_progress(path)
            self.assertEqual(series["Train/mean_reward"], [(0, 2.)])
            self.assertIn("unfinished", warnings[0])
            path.write_text('{bad}\n')
            with self.assertRaisesRegex(ValueError, "Malformed"):
                watcher.load_json_progress(path)

    def test_nonfinite_values_are_reported_not_averaged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            path.write_text('{"iteration": 3, "scalars": {"Train/mean_reward": NaN}}\n')
            series, warnings = watcher.load_json_progress(path)
            self.assertFalse(series)
            self.assertIn("Non-finite", warnings[0])

    def test_plateau_requires_two_windows_and_rejects_improvement_or_volatility(self):
        constant = [(i, 10.) for i in range(40)]
        self.assertTrue(watcher.summarize_series(constant)["plateau_indicated"])
        self.assertIsNone(watcher.summarize_series(constant[:39])["plateau_indicated"])
        improving = [(i, float(i)) for i in range(40)]
        self.assertFalse(watcher.summarize_series(improving)["plateau_indicated"])
        oscillating = [(i, 10. + (-1) ** i * 5.) for i in range(40)]
        self.assertFalse(watcher.summarize_series(oscillating)["plateau_indicated"])

    def test_real_rslrl_tag_groups_and_time_axis(self):
        points = [(i, 10.) for i in range(40)]
        series = {"Train/mean_reward": points, "Train/mean_reward/time": [(100000, 10.)],
                  "Train/mean_episode_length": points,
                  "Episode_Termination/time_out": points,
                  "Episode_Termination/support_failure": points}
        report = watcher.build_report(series, exploratory=True)
        self.assertEqual(report["latest_logged_iteration"], 39)
        self.assertTrue(report["reward_plateau_indicated"])
        self.assertEqual(report["unavailable_metric_groups"], ["crossing", "gait"])
        self.assertEqual(report["classification"], "exploratory_only")
        self.assertIn("cannot establish convergence", report["limitation"])
        json.dumps(report, allow_nan=False)

    def test_gait_diagnostics_with_rslrl_episode_prefix(self):
        names = ["contact_accuracy", "stance_recall", "swing_recall", "foot_lateral_rmse",
                 "forward_speed", "speed_mae", "lateral_rmse"]
        tags = [f"{'Episode/' if i % 2 else ''}Gait/{name}" for i, name in enumerate(names)]
        report = watcher.build_report({tag: [(0, .3), (1, .4)] for tag in tags})
        self.assertEqual(set(report["metrics"]["gait"]), set(tags))
        self.assertIsNone(report["reward_plateau_indicated"])
        self.assertIn("not held-out", report["gait_metric_note"])

    def test_hourly_schedule_and_final_override(self):
        schedule = watcher.ReportSchedule(3600.)
        self.assertEqual(schedule.next_kind(100.), "initial")
        self.assertIsNone(schedule.next_kind(3699.))
        self.assertEqual(schedule.next_kind(3700.), "periodic")
        self.assertIsNone(schedule.next_kind(3701.))
        self.assertEqual(schedule.next_kind(10900.), "periodic")
        self.assertIsNone(schedule.next_kind(10901.))
        self.assertEqual(schedule.next_kind(10902., process_state="exited_zombie"), "final")
        self.assertIsNone(schedule.next_kind(20000., stopping=True))
        immediate_exit = watcher.ReportSchedule()
        self.assertEqual(immediate_exit.next_kind(0., stopping=True), "initial")
        self.assertEqual(immediate_exit.next_kind(0., stopping=True), "final")

    def test_watch_process_exit_saves_initial_and_final_archives(self):
        with tempfile.TemporaryDirectory(dir=watcher.ROOT) as directory:
            path = Path(directory) / "progress.jsonl"
            output = Path(directory) / "latest.json"
            path.write_text('{"iteration": 8, "scalars": {"Train/mean_reward": 2}}\n')
            states = [{"process_state": "present"}, {"process_state": "exited_or_missing"}]
            with mock.patch.object(watcher, "inspect_process", side_effect=states), \
                 mock.patch.object(watcher.threading.Event, "wait", return_value=False) as wait, \
                 mock.patch("builtins.print"):
                watcher.main(["--progress-json", str(path), "--output", str(output),
                              "--watch", "--pid", "12345", "--report_interval", "3600"])
            report = json.loads(output.read_text())
            self.assertEqual(report["report_kind"], "final")
            self.assertEqual(report["final_reason"], "exited_or_missing")
            self.assertEqual(report["latest_logged_iteration"], 8)
            archives = [json.loads(p.read_text()) for p in Path(directory).glob("latest_*.json")]
            self.assertCountEqual([r["report_kind"] for r in archives], ["initial", "final"])
            wait.assert_called_once_with(20.)

    def test_sigterm_wakes_poll_and_writes_final(self):
        with tempfile.TemporaryDirectory(dir=watcher.ROOT) as directory:
            output = Path(directory) / "latest.json"
            def terminate(_seconds):
                handler = watcher.signal.getsignal(watcher.signal.SIGTERM)
                handler(watcher.signal.SIGTERM, None)
                return True
            with mock.patch.object(watcher, "inspect_process", return_value={"process_state": "present"}), \
                 mock.patch.object(watcher.threading.Event, "wait", side_effect=terminate), \
                 mock.patch("builtins.print"):
                watcher.main(["--progress-json", str(Path(directory) / "not_yet_written.jsonl"),
                              "--output", str(output), "--watch"])
            report = json.loads(output.read_text())
            self.assertEqual(report["report_kind"], "final")
            self.assertEqual(report["final_reason"], "SIGTERM")
            self.assertIn("Progress file does not exist yet", report["warnings"])

    def test_log_errors_do_not_imply_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text("Iteration 20\nRuntimeError: CUDA out of memory\n")
            result = watcher.inspect_process(log_file=path)
            self.assertEqual(len(result["error_lines"]), 1)
            self.assertIsNone(result["exit_code"])
            self.assertEqual(result["process_state"], "not_checked")

    def test_report_cannot_overwrite_arbitrary_file_or_escape_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "within this repository"):
                watcher.save_report({}, Path(directory) / "report.json")
        with tempfile.TemporaryDirectory(dir=watcher.ROOT) as directory:
            path = Path(directory) / "existing.json"
            path.write_text('{"important": true}')
            with self.assertRaisesRegex(ValueError, "non-watcher"):
                watcher.save_report({}, path)
            self.assertEqual(json.loads(path.read_text()), {"important": True})
            path.unlink()
            report = watcher.build_report({})
            watcher.save_report(report, path)
            watcher.save_report(report, path)
            self.assertEqual(json.loads(path.read_text())["schema"], watcher.SCHEMA)


if __name__ == "__main__":
    unittest.main()
