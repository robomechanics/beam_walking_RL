"""Offline checks for the single-gait specialist sampler."""

from collections import Counter
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))

from beam_walking.experiment.protocol import (  # noqa: E402
    CORE_CONTROL_STEPS, FOUNDATION_CONTROL_STEPS)
from beam_walking.experiment.unified_gait import (  # noqa: E402
    TROT_DUTY_LEVELS, TROT_MIN_SWING_STEPS, TROT_PERIOD_TICKS,
    WALK_DUTY_LEVELS, WALK_MIN_SWING_STEPS, WALK_PERIOD_TICKS)
from beam_walking.experiment.unified_specialist import (  # noqa: E402
    sample_specialist_commands, specialist_lineage_id)


class SpecialistSamplerTest(unittest.TestCase):

    def test_single_gait_with_balanced_duty_anchors(self):
        torch.manual_seed(1)
        for gait, gait_id, levels in (("trot", 0, TROT_DUTY_LEVELS),
                                      ("walk", 1, WALK_DUTY_LEVELS)):
            for step in (0, FOUNDATION_CONTROL_STEPS):
                values, ticks, _ = sample_specialist_commands(gait, 1200, step)
                self.assertEqual(values.shape, (1200, 5))
                self.assertTrue(torch.all(values[:, 4] == gait_id))
                self.assertTrue(torch.all(ticks == 24))
                counts = Counter(round(float(v), 3) for v in values[:, 1])
                self.assertEqual(set(counts), set(levels))
                # Truncating the gait subset of a balanced double draw can leave a
                # handful of samples of imbalance; per-reset call sizes are small.
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 8)

    def test_continuous_stage_respects_the_gait_domain(self):
        torch.manual_seed(2)
        values, ticks, stage = sample_specialist_commands("walk", 6000, CORE_CONTROL_STEPS)
        self.assertEqual(stage, 2)
        self.assertTrue(torch.all(values[:, 4] == 1))
        self.assertTrue(torch.all(values[:, 1] >= WALK_DUTY_LEVELS[0]))
        self.assertTrue(torch.all(values[:, 1] <= WALK_DUTY_LEVELS[-1]))
        self.assertEqual(set(ticks.tolist()), set(range(WALK_PERIOD_TICKS[0], WALK_PERIOD_TICKS[1] + 1)))
        self.assertTrue(torch.all((1 - values[:, 1]) * ticks >= WALK_MIN_SWING_STEPS - 1e-5))
        values, ticks, _ = sample_specialist_commands("trot", 6000, CORE_CONTROL_STEPS)
        self.assertTrue(torch.all(values[:, 4] == 0))
        self.assertTrue(torch.all(values[:, 1] <= TROT_DUTY_LEVELS[-1]))
        self.assertEqual(set(ticks.tolist()), set(range(TROT_PERIOD_TICKS[0], TROT_PERIOD_TICKS[1] + 1)))
        self.assertTrue(torch.all((1 - values[:, 1]) * ticks >= TROT_MIN_SWING_STEPS - 1e-5))
        self.assertGreater(float((ticks == 24).float().mean()), .75)

    def test_empty_invalid_and_lineage(self):
        values, ticks, _ = sample_specialist_commands("trot", 0, 0)
        self.assertEqual(values.shape, (0, 5))
        self.assertEqual(ticks.shape, (0,))
        with self.assertRaises(ValueError):
            sample_specialist_commands("bound", 4, 0)
        with self.assertRaises(ValueError):
            sample_specialist_commands("trot", -1, 0)
        self.assertNotEqual(specialist_lineage_id("x", 5, "trot"),
                            specialist_lineage_id("x", 5, "walk"))


if __name__ == "__main__":
    unittest.main()
