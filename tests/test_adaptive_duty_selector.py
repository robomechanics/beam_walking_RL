"""Offline tests for the separate adaptive duty-factor experiment."""

from collections import Counter
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))

from beam_walking.experiment.adaptive_duty import (  # noqa: E402
    ADAPTIVE_COMMAND_STRATA,
    feasible_duty_upper,
    sample_adaptive_training_commands,
)
from beam_walking.experiment.duty_selector import (  # noqa: E402
    DutyFactorSelector,
    aggregate_candidates,
    fit_selector,
    load_selector,
    predict_rows,
    read_trial_rows,
    select_targets,
    selector_state_sha256,
    validate_completed_grid,
)
from beam_walking.experiment.duty_grid import summarize_condition  # noqa: E402
from beam_walking.experiment.protocol import (  # noqa: E402
    CORE_CONTROL_STEPS,
    FOUNDATION_CONTROL_STEPS,
    GAITS,
    MIN_SWING_STEPS,
)


class AdaptiveCommandTest(unittest.TestCase):

    def test_foundation_balances_both_gaits_and_all_duty_anchors(self):
        torch.manual_seed(3)
        values, ticks, stage = sample_adaptive_training_commands(1200, 0)
        counts = Counter((int(gait), round(float(duty), 3)) for gait, duty in
                         zip(values[:, 4], values[:, 1]))
        expected = {(gait, round(duty, 3))
                    for gait, duty in ADAPTIVE_COMMAND_STRATA}
        self.assertEqual(set(counts), expected)
        self.assertTrue(all(value == 100 for value in counts.values()))
        self.assertTrue(torch.all(ticks == 24))
        self.assertEqual(stage, 0)

    def test_continuous_commands_respect_ranges_and_minimum_swing(self):
        torch.manual_seed(9)
        values, ticks, stage = sample_adaptive_training_commands(
            6000, CORE_CONTROL_STEPS)
        self.assertEqual(stage, 2)
        self.assertTrue(torch.all((values[:, 4] == 0) | (values[:, 4] == 1)))
        self.assertTrue(torch.all(values[:, 1] >= .5))
        self.assertTrue(torch.all(values[:, 1] <= feasible_duty_upper(ticks) + 1e-7))
        self.assertTrue(torch.all((1 - values[:, 1]) * ticks >= MIN_SWING_STEPS - 1e-5))
        # Both gaits must receive low, middle, and high DF experience.
        for gait in range(len(GAITS)):
            selected = values[values[:, 4] == gait, 1]
            self.assertLess(float(selected.min()), .54)
            self.assertTrue(bool(((selected > .60) & (selected < .66)).any()))
            self.assertGreater(float(selected.max()), .70)

    def test_core_stage_keeps_exact_anchor_balance(self):
        values, _, stage = sample_adaptive_training_commands(
            1200, FOUNDATION_CONTROL_STEPS)
        counts = Counter((int(gait), round(float(duty), 3)) for gait, duty in
                         zip(values[:, 4], values[:, 1]))
        expected = {(gait, round(duty, 3))
                    for gait, duty in ADAPTIVE_COMMAND_STRATA}
        self.assertEqual(set(counts), expected)
        self.assertTrue(all(value == 100 for value in counts.values()))
        self.assertEqual(stage, 1)


class DutySelectorTest(unittest.TestCase):

    @staticmethod
    def rows():
        rows = []
        for width, winning_df in ((.10, .70), (.30, .60), (.50, .55)):
            for duty, cot, passes in (
                    (.55, 1.2 + duty_offset(duty=.55, target=winning_df), 10),
                    (.60, 1.2 + duty_offset(duty=.60, target=winning_df), 10),
                    (.70, 1.2 + duty_offset(duty=.70, target=winning_df), 10)):
                for trial in range(10):
                    rows.append({
                        "seed": trial,
                        "step_width": width, "speed": .30, "period": .48,
                        "gait": "trot", "command_df": duty,
                        "compliant": trial < passes,
                        "positive_mechanical_cot": cot + trial * 1e-4,
                    })
        return rows

    def test_target_is_lowest_energy_compliant_candidate(self):
        rows = self.rows()
        candidates = aggregate_candidates(rows)
        self.assertEqual(len(candidates), 9)
        targets, rejected = select_targets(rows, min_trials=8)
        self.assertFalse(rejected)
        self.assertEqual(
            [row["target_df"] for row in targets], [.70, .60, .55])

    def test_noncompliant_low_energy_candidate_is_excluded(self):
        rows = self.rows()
        for row in rows:
            if row["step_width"] == .30 and row["command_df"] == .55:
                row["positive_mechanical_cot"] = .01
                row["compliant"] = False
        targets, _ = select_targets(rows, min_trials=8)
        middle = next(row for row in targets if row["step_width"] == .30)
        self.assertEqual(middle["target_df"], .60)

    def test_selector_outputs_physical_bounds_and_fits_labels(self):
        targets, _ = select_targets(self.rows(), min_trials=8)
        model, history = fit_selector(targets, seed=4, epochs=800)
        predictions = predict_rows(model, targets)
        self.assertLess(history[-1], history[0])
        for target, prediction in zip(targets, predictions):
            self.assertGreaterEqual(prediction["selected_df"], .5)
            self.assertLessEqual(prediction["selected_df"], .75)
            self.assertAlmostEqual(
                prediction["selected_df"], target["target_df"], delta=.025)

    def test_short_period_output_respects_swing_constraint(self):
        model = DutyFactorSelector()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(10.)
        context = torch.tensor([[.3, .3, .36, 0.]])
        self.assertLessEqual(
            float(model(context)[0]), feasible_duty_upper(18) + 1e-7)

    def test_saved_selector_can_be_loaded_for_inference(self):
        model = DutyFactorSelector()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.pt"
            torch.save({
                "schema": "flat_duty_selector_v1",
                "state_dict": model.state_dict(),
                "hidden_dims": [32, 32],
                "input_fields": ["step_width", "speed", "period", "gait_id"],
                "input_ranges": {
                    "step_width": [.1, .5], "speed": [.25, .4],
                    "period": [.36, .54]},
                "gait_id_mapping": {"trot": 0, "walk": 1},
                "control_dt_s": .02,
                "supported_contexts": [{
                    "step_width": .2, "speed": .3,
                    "period": .48, "gait": "walk"}],
                "deployment_ready": False,
                "selector_source_sha256": __import__("hashlib").sha256(
                    (ROOT / "source/beam_walking/beam_walking/experiment/"
                     "duty_selector.py").read_bytes()).hexdigest(),
                "trial_csv_sha256": "a", "grid_manifest_sha256": "b",
                "grid_complete_sha256": "c", "grid_task_sha256": "d",
                "grid_checkpoint_sha256": "e", "grid_collector_sha256": "f",
                "selector_state_sha256": selector_state_sha256(
                    model.state_dict()),
            }, path)
            loaded, metadata = load_selector(path)
            context = torch.tensor([[.2, .3, .48, 1.]])
            torch.testing.assert_close(loaded(context), model(context))
            self.assertEqual(metadata["schema"], "flat_duty_selector_v1")

    def test_selector_rejects_invalid_contexts(self):
        model = DutyFactorSelector()
        for values in (
                [[.3, .3, .10, 0.]], [[.3, .3, .48, .9]],
                [[.09, .3, .48, 0.]], [[.3, .41, .48, 0.]],
                [[float("nan"), .3, .48, 0.]]):
            with self.assertRaises(ValueError):
                model(torch.tensor(values))

    def test_energy_invalid_trials_cannot_make_candidate_eligible(self):
        rows = self.rows()
        for row in rows:
            if row["step_width"] == .30 and row["command_df"] == .55:
                row["compliant"] = False
                row["positive_mechanical_cot"] = float("nan")
        targets, _ = select_targets(rows, min_trials=8)
        middle = next(row for row in targets if row["step_width"] == .30)
        self.assertEqual(middle["target_df"], .60)

    def test_completed_grid_rejects_missing_manifest_row(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rows = self.rows()
            fields = list(rows[0])
            with (directory / "trials.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            conditions = [{
                "step_width": row["step_width"], "speed": row["speed"],
                "period": row["period"], "gait": row["gait"],
                "command_df": row["command_df"],
            } for row in rows if row["seed"] == 0]
            (directory / "grid_manifest.json").write_text(json.dumps({
                "schema": "adaptive_duty_grid_v1", "terrain": "flat_ground",
                "terrain_width_input": False, "trials_per_candidate": 10,
                "evaluation_seed_start": 0, "conditions": conditions,
                "task_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
                "collector_sha256": "c" * 64,
            }))
            rows.pop()
            with (directory / "trials.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            import hashlib
            manifest_path = directory / "grid_manifest.json"
            trial_path = directory / "trials.csv"
            (directory / "GRID_COMPLETE").write_text(json.dumps({
                "schema": "adaptive_duty_grid_complete_v1",
                "manifest_file": "grid_manifest.json",
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()).hexdigest(),
                "trial_csv_file": "trials.csv",
                "trial_csv_sha256": hashlib.sha256(
                    trial_path.read_bytes()).hexdigest(),
                "archive_sha256": {},
            }))
            with self.assertRaisesRegex(ValueError, "exactly match"):
                validate_completed_grid(
                    directory / "trials.csv", require_primary=False)

    def test_trial_csv_rejects_malformed_values_and_duplicates(self):
        base = {
            "seed": "1", "step_width": ".30", "speed": ".30",
            "period": ".48", "gait": "trot", "command_df": ".60",
            "compliant": "true", "positive_mechanical_cot": "1.2",
        }
        mutations = (
            {"compliant": "maybe"}, {"step_width": ".9"},
            {"speed": ".8"}, {"period": ".10"},
            {"command_df": ".9"}, {"positive_mechanical_cot": "-1"},
            {"positive_mechanical_cot": "nan"},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trials.csv"
                row = dict(base, **mutation)
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(base))
                    writer.writeheader()
                    writer.writerow(row)
                with self.assertRaises(ValueError):
                    read_trial_rows(path)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(base))
                writer.writeheader()
                writer.writerows((base, base))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_trial_rows(path)


class DutyGridSummaryTest(unittest.TestCase):

    def test_ideal_trace_is_compliant_and_has_finite_energy(self):
        import numpy as np

        trials, cycles, steps = 2, 2, 24
        offsets = np.asarray([0., .5, .5, 0.])
        desired_cycle = (
            (np.arange(steps)[:, None] / steps + offsets) % 1 < .625)
        desired = np.broadcast_to(
            desired_cycle, (trials, cycles, steps, 4)).copy()
        signs = np.asarray([1., -1., 1., -1.])
        feet = np.zeros((trials, cycles, steps, 4, 3))
        feet[..., 1] = .15 * signs
        samples = steps * 4
        payload = {
            "contacts": desired.copy(), "desired": desired,
            "feet_body": feet,
            "forward_velocity": np.full((trials, cycles, steps), .30),
            "lateral_position": np.zeros((trials, cycles, steps)),
            "heading": np.zeros((trials, cycles, steps)),
            "world_lateral_velocity": np.zeros((trials, cycles, steps)),
            "body_yaw_rate": np.zeros((trials, cycles, steps)),
            "failure": np.zeros((trials, cycles, steps), dtype=bool),
            "done": np.zeros((trials, cycles, steps), dtype=bool),
            "substep_contacts": np.repeat(desired[..., None, :], 4, axis=3),
            "initial_contacts": np.broadcast_to(
                desired_cycle[-1], (trials, cycles, 4)).copy(),
            "initial_desired": np.broadcast_to(
                desired_cycle[-1], (trials, cycles, 4)).copy(),
            "applied_torque": np.ones((trials, cycles, samples, 12)),
            "joint_velocity": np.full((trials, cycles, samples, 12), .1),
            "x_boundaries": np.broadcast_to(
                np.asarray([0., .144]), (trials, cycles, 2)).copy(),
            "robot_mass_kg": 15., "gravity_mps2": 9.81,
            "sample_dt": .005,
        }
        rows = summarize_condition(
            payload, step_width=.30, speed=.30, period=.48,
            gait="trot", command_df=.625, seed_start=100)
        self.assertEqual(len(rows), trials)
        self.assertTrue(all(row["compliant"] for row in rows))
        self.assertTrue(all(row["topology_fraction"] == 1 for row in rows))
        self.assertTrue(all(row["positive_mechanical_cot"] > 0 for row in rows))

        payload["any_failure"] = np.asarray([True, False])
        failed_rows = summarize_condition(
            payload, step_width=.30, speed=.30, period=.48,
            gait="trot", command_df=.625, seed_start=100)
        self.assertFalse(failed_rows[0]["compliant"])
        self.assertTrue(failed_rows[1]["compliant"])

        malformed = dict(payload)
        malformed["desired"] = payload["desired"].copy()
        malformed["desired"][0, 0] = True
        malformed_rows = summarize_condition(
            malformed, step_width=.30, speed=.30, period=.48,
            gait="trot", command_df=.625, seed_start=100)
        self.assertFalse(malformed_rows[0]["compliant"])
        self.assertTrue(np.isnan(malformed_rows[0]["topology_fraction"]))


def duty_offset(duty, target):
    return abs(duty - target)


if __name__ == "__main__":
    unittest.main()
