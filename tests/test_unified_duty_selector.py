"""Offline checks for the unified trot/walk duty-factor selector."""

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))

from beam_walking.experiment.unified_duty_selector import (  # noqa: E402
    GAIT_LEVELS,
    LEVELS,
    UnifiedDutySelector,
    feasible_levels,
    fit_selector,
    level_index,
    predict_rows,
    predict_supported_rows,
    select_targets,
)


def targets():
    rows = []
    trot = {.10: .50, .20: .625, .30: .625, .40: .75, .50: .75}
    walk = {.10: .75, .20: .80, .30: .80, .40: .85, .50: .90}
    for period in (.40, .48, .54):
        for speed in (.25, .30, .35, .40):
            for width in (.10, .20, .30, .40, .50):
                rows.append({"step_width": width, "speed": speed, "period": period,
                             "gait": "trot", "target_df": trot[width]})
                rows.append({"step_width": width, "speed": speed, "period": period,
                             "gait": "walk", "target_df": walk[width]})
    for speed in (.25, .30, .35, .40):
        for width in (.10, .20, .30, .40, .50):
            rows.append({"step_width": width, "speed": speed, "period": .36,
                         "gait": "trot", "target_df": min(trot[width], .625)})
    return rows


class UnifiedSelectorTest(unittest.TestCase):

    def test_levels_cover_both_gaits(self):
        self.assertEqual(len(LEVELS), 7)
        self.assertEqual(GAIT_LEVELS["trot"], (.50, .625, .75))
        self.assertEqual(GAIT_LEVELS["walk"], (.75, .80, .85, .90))
        self.assertEqual(level_index("trot", .75), 2)
        self.assertEqual(level_index("walk", .75), 3)
        with self.assertRaises(ValueError):
            level_index("walk", .625)
        with self.assertRaises(ValueError):
            level_index("trot", .80)
        with self.assertRaises(ValueError):
            level_index("trot", .75, .36)
        self.assertEqual(feasible_levels("trot", .36), (.50, .625))
        self.assertEqual(feasible_levels("walk", .40), (.75, .80, .85, .90))

    def test_output_is_masked_to_the_gait_and_period(self):
        model = UnifiedDutySelector()
        context = torch.tensor([[.10, .25, .48, 0.], [.30, .325, .40, 0.], [.50, .40, .54, 0.],
                                [.10, .25, .48, 1.], [.30, .325, .40, 1.], [.50, .40, .54, 1.]])
        values = model(context)
        self.assertTrue(torch.all(values[:3] >= .50))
        self.assertTrue(torch.all(values[:3] <= .75))
        self.assertTrue(torch.all(values[3:] >= .75))
        self.assertTrue(torch.all(values[3:] <= .90))
        for value in values[:3].tolist():
            self.assertIn(round(value, 3), [round(v, 3) for v in GAIT_LEVELS["trot"]])
        for value in values[3:].tolist():
            self.assertIn(round(value, 3), [round(v, 3) for v in GAIT_LEVELS["walk"]])
        # At 0.36 s trot may never select 0.75 (only 4.5 swing ticks), whatever
        # the network weights say.
        torch.manual_seed(0)
        for _ in range(20):
            model = UnifiedDutySelector()
            short = model(torch.tensor([[w, .30, .36, 0.] for w in (.1, .3, .5)]))
            self.assertTrue(torch.all(short <= .625 + 1e-6))
        with self.assertRaises(ValueError):
            model(torch.tensor([[.30, .30, .36, 1.]]))  # walk floor is 0.40 s

    def test_network_fits_gait_specific_labels(self):
        model, history = fit_selector(targets(), seed=0, epochs=1500)
        prediction = predict_rows(model, targets())
        self.assertLess(history[-1], history[0])
        self.assertEqual([row["selected_df"] for row in prediction],
                         [row["target_df"] for row in targets()])

    def test_invalid_context_is_rejected(self):
        model = UnifiedDutySelector()
        for context in ([[.09, .30, .48, 0.]], [[.30, .41, .48, 1.]], [[.30, .30, .48, 2.]],
                        [[.30, .30, .48, .5]], [[float("nan"), .30, .48, 0.]],
                        [[.30, .30, .35, 0.]], [[.30, .30, .47, 0.]]):
            with self.assertRaises(ValueError):
                model(torch.tensor(context))

    def test_select_targets_keeps_gaits_separate(self):
        rows = []
        for gait, duties in GAIT_LEVELS.items():
            for duty in duties:
                for seed in range(32):
                    rows.append({"step_width": .30, "speed": .30, "period": .48,
                                 "gait": gait, "command_df": duty, "seed": seed,
                                 "compliant": True,
                                 "positive_mechanical_cot": 1. + abs(duty - .625)
                                 if gait == "trot" else 1. + abs(duty - .85)})
        chosen, rejected = select_targets(rows, .90, 32)
        self.assertEqual(rejected, [])
        by_gait = {row["gait"]: row["target_df"] for row in chosen}
        self.assertEqual(by_gait, {"trot": .625, "walk": .85})

    def test_runtime_abstains_outside_supported_contexts(self):
        model = UnifiedDutySelector()
        payload = {"deployment_ready": True, "supported_contexts": [
            {"step_width": .30, "speed": .30, "period": .48, "gait": "trot"}]}
        with self.assertRaisesRegex(ValueError, "abstains"):
            predict_supported_rows(model, payload, [{
                "step_width": .30, "speed": .30, "period": .48, "gait": "walk"}])
        rows = predict_supported_rows(model, payload, [{
            "step_width": .30, "speed": .30, "period": .48, "gait": "trot"}])
        self.assertIn(rows[0]["selected_df"], GAIT_LEVELS["trot"])


if __name__ == "__main__":
    unittest.main()
