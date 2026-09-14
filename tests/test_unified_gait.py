"""Offline checks for the unified trot/walk command distribution."""

from collections import Counter
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))

from beam_walking.experiment.protocol import (  # noqa: E402
    CONTROL_DT,
    CORE_CONTROL_STEPS,
    FOUNDATION_CONTROL_STEPS,
    PERIOD_TICKS,
    SPEED_ANCHORS,
    STEP_WIDTH_ANCHORS,
    discrete_stance_fraction,
    normalized_gait_command,
)
from beam_walking.experiment.unified_gait import (  # noqa: E402
    TROT_DUTY_LEVELS,
    TROT_MIN_SWING_STEPS,
    TROT_PERIOD_TICKS,
    UNIFIED_STRATA,
    WALK_DUTY_LEVELS,
    WALK_MIN_SWING_STEPS,
    WALK_PERIOD_TICKS,
    max_duty_for_period,
    sample_unified_commands,
)


def strata_counts(values):
    return Counter(
        (int(gait), round(float(duty), 3))
        for gait, duty in zip(values[:, 4], values[:, 1]))


class UnifiedCommandTest(unittest.TestCase):

    def test_foundation_balances_gaits_then_duty_levels(self):
        torch.manual_seed(5)
        values, ticks, stage = sample_unified_commands(2400, 0)
        self.assertEqual(stage, 0)
        counts = strata_counts(values)
        self.assertEqual(set(counts), set(UNIFIED_STRATA))
        gaits = Counter(int(gait) for gait in values[:, 4])
        self.assertEqual(gaits[0], 1200)
        self.assertEqual(gaits[1], 1200)
        for duty in TROT_DUTY_LEVELS:
            self.assertEqual(counts[(0, round(duty, 3))], 400)
        for duty in WALK_DUTY_LEVELS:
            self.assertEqual(counts[(1, round(duty, 3))], 300)
        self.assertTrue(torch.all(ticks == 24))
        self.assertTrue(torch.all(values[:, 0] == .30))
        self.assertTrue(torch.all(values[:, 2] == .30))

    def test_core_covers_every_speed_width_anchor_per_stratum(self):
        torch.manual_seed(6)
        values, ticks, stage = sample_unified_commands(
            2400, FOUNDATION_CONTROL_STEPS)
        self.assertEqual(stage, 1)
        self.assertTrue(torch.all(ticks == 24))
        counts = strata_counts(values)
        self.assertEqual(set(counts), set(UNIFIED_STRATA))
        for gait, duty in UNIFIED_STRATA:
            member = (values[:, 4] == gait) & (
                torch.abs(values[:, 1] - duty) < 1e-6)
            combos = Counter(
                (round(float(s), 3), round(float(w), 3))
                for s, w in zip(values[member, 0], values[member, 2]))
            self.assertEqual(
                set(combos),
                {(s, w) for s in SPEED_ANCHORS for w in STEP_WIDTH_ANCHORS})

    def test_continuous_stage_respects_gait_specific_domains(self):
        torch.manual_seed(7)
        values, ticks, stage = sample_unified_commands(
            12000, CORE_CONTROL_STEPS)
        self.assertEqual(stage, 2)
        trot = values[:, 4] == 0
        walk = values[:, 4] == 1
        self.assertTrue(torch.all(values[trot, 1] >= TROT_DUTY_LEVELS[0]))
        self.assertTrue(torch.all(values[trot, 1] <= TROT_DUTY_LEVELS[-1]))
        self.assertTrue(torch.all(values[walk, 1] >= WALK_DUTY_LEVELS[0]))
        self.assertTrue(torch.all(values[walk, 1] <= WALK_DUTY_LEVELS[-1]))
        self.assertTrue(torch.all(ticks[trot] >= TROT_PERIOD_TICKS[0]))
        self.assertTrue(torch.all(ticks[walk] >= WALK_PERIOD_TICKS[0]))
        self.assertTrue(torch.all(ticks <= max(PERIOD_TICKS)))
        torch.testing.assert_close(values[:, 3], ticks * CONTROL_DT)
        self.assertTrue(torch.all(
            (1 - values[trot, 1]) * ticks[trot] >= TROT_MIN_SWING_STEPS - 1e-5))
        self.assertTrue(torch.all(
            (1 - values[walk, 1]) * ticks[walk] >= WALK_MIN_SWING_STEPS - 1e-5))
        # Both gaits see every period in their range and the anchor period
        # keeps roughly three quarters of the samples.
        self.assertEqual(
            set(ticks[trot].tolist()),
            set(range(TROT_PERIOD_TICKS[0], TROT_PERIOD_TICKS[1] + 1)))
        self.assertEqual(
            set(ticks[walk].tolist()),
            set(range(WALK_PERIOD_TICKS[0], WALK_PERIOD_TICKS[1] + 1)))
        anchored = float((ticks == 24).float().mean())
        self.assertGreater(anchored, .75)
        # Gait balance holds in the continuous stage too.
        self.assertEqual(int(trot.sum()), int(walk.sum()))
        # Walk DF 0.90 is actually reached at the shortest walk period.
        reached = walk & (ticks == WALK_PERIOD_TICKS[0]) & (values[:, 1] > .89)
        self.assertTrue(bool(reached.any()))

    def test_max_duty_bound_per_gait(self):
        ticks = torch.tensor([18, 20, 24, 27])
        trot = max_duty_for_period(torch.zeros(4), ticks)
        walk = max_duty_for_period(torch.ones(4), ticks)
        torch.testing.assert_close(
            trot, torch.tensor([1 - 5 / 18, .75, .75, .75]))
        torch.testing.assert_close(
            walk, torch.tensor([1 - 2 / 18, .90, .90, .90]))

    def test_walk_anchors_have_distinct_schedules_at_every_walk_period(self):
        for ticks_value in range(WALK_PERIOD_TICKS[0], WALK_PERIOD_TICKS[1] + 1):
            duty = torch.tensor(WALK_DUTY_LEVELS)
            ticks = torch.full((len(duty),), ticks_value)
            gait = torch.ones(len(duty), dtype=torch.long)
            scheduled = discrete_stance_fraction(duty, ticks, gait)
            swing = ((1 - scheduled) * ticks_value).round()
            self.assertTrue(torch.all(swing >= WALK_MIN_SWING_STEPS), ticks_value)
            for leg in range(4):
                self.assertEqual(
                    len(set(swing[:, leg].tolist())), len(duty), (ticks_value, leg))

    def test_trot_anchors_have_distinct_schedules_at_every_trot_period(self):
        for ticks_value in range(TROT_PERIOD_TICKS[0], TROT_PERIOD_TICKS[1] + 1):
            duty = torch.tensor(TROT_DUTY_LEVELS)
            ticks = torch.full((len(duty),), ticks_value)
            scheduled = discrete_stance_fraction(
                duty, ticks, torch.zeros(len(duty), dtype=torch.long))
            for leg in range(4):
                self.assertEqual(
                    len(set(scheduled[:, leg].tolist())), len(duty), (ticks_value, leg))

    def test_commands_normalize_without_error(self):
        torch.manual_seed(8)
        values, _, _ = sample_unified_commands(512, CORE_CONTROL_STEPS)
        normalized = normalized_gait_command(values)
        self.assertTrue(torch.isfinite(normalized).all())
        # Walk duty above 0.75 sits above the +1 normalization edge, as the
        # high-duty study already did; the interface width is unchanged.
        self.assertTrue(bool((normalized[:, 1] > 1).any()))

    def test_empty_and_negative_counts(self):
        values, ticks, _ = sample_unified_commands(0, CORE_CONTROL_STEPS)
        self.assertEqual(values.shape, (0, 5))
        self.assertEqual(ticks.shape, (0,))
        with self.assertRaises(ValueError):
            sample_unified_commands(-1, 0)


if __name__ == "__main__":
    unittest.main()
