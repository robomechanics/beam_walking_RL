"""Offline checks for the independent high-duty walk experiment."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))

from beam_walking.experiment.high_duty_walk import (  # noqa: E402
    HIGH_DUTY_WALK_LEVELS,
    HIGH_DUTY_WALK_MIN_SWING_STEPS,
    HIGH_DUTY_WALK_PERIOD_TICKS,
    sample_high_duty_walk_commands,
)
from beam_walking.experiment.high_duty_walk_selector import (  # noqa: E402
    HighDutyWalkSelector,
    fit_selector,
    load_selector,
    predict_rows,
    predict_supported_rows,
    selector_runtime_sha256,
    selector_state_sha256,
    validate_completed_grid,
)
from beam_walking.experiment.protocol import (  # noqa: E402
    CORE_CONTROL_STEPS,
    discrete_stance_fraction,
    FOUNDATION_CONTROL_STEPS,
)


class HighDutyWalkCommandTest(unittest.TestCase):

    def test_foundation_balances_every_registered_duty_factor(self):
        torch.manual_seed(4)
        values, ticks, stage = sample_high_duty_walk_commands(1200, 0)
        counts = Counter(round(float(value), 3) for value in values[:, 1])
        self.assertEqual(set(counts), set(HIGH_DUTY_WALK_LEVELS))
        self.assertTrue(all(value == 300 for value in counts.values()))
        self.assertTrue(torch.all(values[:, 4] == 1))
        self.assertTrue(torch.all(ticks == HIGH_DUTY_WALK_PERIOD_TICKS))
        self.assertEqual(stage, 0)

    def test_core_balances_every_registered_duty_factor(self):
        values, _, stage = sample_high_duty_walk_commands(
            1200, FOUNDATION_CONTROL_STEPS)
        counts = Counter(round(float(value), 3) for value in values[:, 1])
        self.assertEqual(set(counts), set(HIGH_DUTY_WALK_LEVELS))
        self.assertTrue(all(value == 300 for value in counts.values()))
        self.assertEqual(stage, 1)

    def test_continuous_stage_stays_in_registered_domain(self):
        torch.manual_seed(9)
        values, ticks, stage = sample_high_duty_walk_commands(
            6000, CORE_CONTROL_STEPS)
        self.assertEqual(stage, 2)
        self.assertTrue(torch.all(values[:, 4] == 1))
        self.assertTrue(torch.all(values[:, 1] >= HIGH_DUTY_WALK_LEVELS[0]))
        self.assertTrue(torch.all(values[:, 1] <= HIGH_DUTY_WALK_LEVELS[-1]))
        self.assertTrue(torch.all(ticks == HIGH_DUTY_WALK_PERIOD_TICKS))
        self.assertTrue(torch.all(
            (1 - values[:, 1]) * ticks >= HIGH_DUTY_WALK_MIN_SWING_STEPS))

    def test_registered_commands_have_distinct_discrete_schedules(self):
        duty = torch.tensor(HIGH_DUTY_WALK_LEVELS)
        ticks = torch.full((len(duty),), HIGH_DUTY_WALK_PERIOD_TICKS)
        gait = torch.ones(len(duty), dtype=torch.long)
        scheduled = discrete_stance_fraction(duty, ticks, gait)
        expected = torch.tensor([.75, 20 / 24, 21 / 24, 22 / 24])
        torch.testing.assert_close(scheduled[:, 0], expected)
        torch.testing.assert_close(
            scheduled, expected[:, None].expand(-1, 4))


class HighDutyWalkSelectorTest(unittest.TestCase):

    @staticmethod
    def targets():
        return [
            {"step_width": width, "speed": speed, "period": .48,
             "gait": "walk", "target_df": duty}
            for speed in (.25, .30, .35, .40)
            for width, duty in ((.10, .75), (.20, .80), (.30, .80),
                                (.40, .85), (.50, .90))
        ]

    def test_output_is_bounded_to_registered_walk_range(self):
        model = HighDutyWalkSelector()
        context = torch.tensor([[.10, .25], [.30, .325], [.50, .40]])
        values = model(context)
        self.assertTrue(torch.all(values >= .75))
        self.assertTrue(torch.all(values <= .90))

    def test_network_fits_width_speed_labels(self):
        model, history = fit_selector(self.targets(), seed=4, epochs=1200)
        prediction = predict_rows(model, self.targets())
        self.assertLess(history[-1], history[0])
        self.assertEqual(
            [row["selected_df"] for row in prediction],
            [row["target_df"] for row in self.targets()])
        selected = [row["selected_df"] for row in prediction]
        duty = torch.tensor(selected)
        ticks = torch.full((len(duty),), HIGH_DUTY_WALK_PERIOD_TICKS)
        gait = torch.ones(len(duty), dtype=torch.long)
        scheduled = discrete_stance_fraction(duty, ticks, gait)[:, 0]
        expected = torch.tensor([
            {.75: .75, .80: 20 / 24, .85: 21 / 24, .90: 22 / 24}[value]
            for value in selected
        ])
        torch.testing.assert_close(scheduled, expected)

    def test_invalid_context_is_rejected(self):
        model = HighDutyWalkSelector()
        for context in ([[.09, .30]], [[.30, .41]], [[float("nan"), .30]]):
            with self.assertRaises(ValueError):
                model(torch.tensor(context))

    def test_runtime_abstains_outside_exact_supported_contexts(self):
        model = HighDutyWalkSelector()
        payload = {"deployment_ready": True, "supported_contexts": [
            {"step_width": .30, "speed": .30}]}
        with self.assertRaisesRegex(ValueError, "abstains"):
            predict_supported_rows(model, payload, [{
                "step_width": .31, "speed": .30,
                "period": .48, "gait": "walk"}])

    def test_nonvalidated_checkpoint_loads_only_with_exact_runtime_bundle(self):
        model = HighDutyWalkSelector()
        source = (ROOT / "source/beam_walking/beam_walking/experiment/"
                  "high_duty_walk_selector.py")
        payload = {
            "schema": "high_duty_walk_selector_v1",
            "state_dict": model.state_dict(), "hidden_dims": [32, 32],
            "input_fields": ["step_width", "speed"],
            "input_ranges": {"step_width": [.10, .50],
                             "speed": [.25, .40]},
            "fixed_gait": "walk", "fixed_period_s": .48,
            "output_range": [.75, .90],
            "supported_contexts": [{"step_width": .30, "speed": .30}],
            "deployment_ready": False,
            "selector_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "selector_runtime_sha256": selector_runtime_sha256(ROOT),
            "selector_state_sha256": selector_state_sha256(model.state_dict()),
            "grid_task_sha256": "a" * 64,
            "grid_checkpoint_sha256": "b" * 64,
            "grid_collector_sha256": "c" * 64,
            "trial_csv_sha256": "d" * 64,
            "grid_manifest_sha256": "e" * 64,
            "grid_complete_sha256": "f" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.pt"
            torch.save(payload, path)
            loaded, metadata = load_selector(path)
            self.assertEqual(metadata["fixed_gait"], "walk")
            result = predict_rows(loaded, [{
                "step_width": .30, "speed": .30,
                "period": .48, "gait": "walk"}])[0]
            self.assertIn(result["selected_df"], HIGH_DUTY_WALK_LEVELS)
            payload["selector_runtime_sha256"] = "0" * 64
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "runtime source bundle"):
                load_selector(path)

    def test_completed_grid_binds_rows_archives_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            filename = "high_duty_walk_grid_walk_v0.300_p0.48_w0.300_d0.750.npz"
            task_hash, checkpoint_hash, collector_hash = (
                "a" * 64, "b" * 64, "c" * 64)
            seeds = np.arange(100, 108)
            np.savez_compressed(
                directory / filename,
                command=np.asarray([.30, .75, .30, .48, 1.]),
                seeds=seeds, reset_plan=np.zeros((8, 2)),
                stance_start=np.zeros(8, dtype=bool),
                initial_root=np.zeros((8, 13)),
                initial_joints=np.zeros((8, 12)),
                task_sha256=task_hash, checkpoint_sha256=checkpoint_hash,
                collector_sha256=collector_hash,
            )
            trial_path = directory / "duty_grid_trials.csv"
            fields = ["seed", "step_width", "speed", "period", "gait",
                      "command_df", "compliant", "positive_mechanical_cot"]
            with trial_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for seed in seeds:
                    writer.writerow({
                        "seed": int(seed), "step_width": .30, "speed": .30,
                        "period": .48, "gait": "walk", "command_df": .75,
                        "compliant": True, "positive_mechanical_cot": 1.2,
                    })
            manifest_path = directory / "grid_manifest.json"
            manifest_path.write_text(json.dumps({
                "schema": "high_duty_walk_grid_exploratory_v1",
                "terrain": "flat_ground", "terrain_width_input": False,
                "trials_per_candidate": 8, "evaluation_seed_start": 100,
                "task_sha256": task_hash,
                "checkpoint_sha256": checkpoint_hash,
                "collector_sha256": collector_hash,
                "conditions": [{
                    "gait": "walk", "speed": .30, "period": .48,
                    "step_width": .30, "command_df": .75,
                    "filename": filename,
                }],
            }))
            completion_path = directory / "GRID_COMPLETE"
            completion_path.write_text(json.dumps({
                "schema": "high_duty_walk_grid_exploratory_complete_v1",
                "manifest_file": manifest_path.name,
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()).hexdigest(),
                "trial_csv_file": trial_path.name,
                "trial_csv_sha256": hashlib.sha256(
                    trial_path.read_bytes()).hexdigest(),
                "archive_sha256": {filename: hashlib.sha256(
                    (directory / filename).read_bytes()).hexdigest()},
            }))
            rows, _, _ = validate_completed_grid(
                trial_path, require_primary=False)
            self.assertEqual(len(rows), 8)
            np.savez_compressed(directory / "high_duty_walk_grid_extra.npz", x=[1])
            with self.assertRaisesRegex(ValueError, "archive set"):
                validate_completed_grid(trial_path, require_primary=False)


if __name__ == "__main__":
    unittest.main()
