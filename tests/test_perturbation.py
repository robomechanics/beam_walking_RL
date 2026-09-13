"""Ramped base perturbation: envelope schedule, sampling bounds, flags, summary."""
import argparse
import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/beam_walking"))
from beam_walking.experiment.perturbation import (
    PerturbationCfg, RampedPerturbation, add_perturbation_arguments,
    perturbation_from_args, ramp_envelope, summarize_outcomes, PERTURBATION_SCHEMA,
)


class EnvelopeTest(unittest.TestCase):

    def test_envelope_starts_small_and_saturates(self):
        env = ramp_envelope(torch.tensor([-1., 0., .5, 1., 3.]), .1)
        torch.testing.assert_close(env, torch.tensor([.1, .1, .55, 1., 1.]))

    def test_envelope_is_monotone_in_progress(self):
        progress = torch.linspace(0, 1.5, 200)
        env = ramp_envelope(progress, .05)
        self.assertTrue(torch.all(env[1:] >= env[:-1]))


class SamplingTest(unittest.TestCase):

    def setUp(self):
        self.cfg = PerturbationCfg(max_force_n=20., max_torque_nm=4., start_fraction=.1,
                                   ramp_time_s=2., hold_min_s=.1, hold_max_s=.3)
        self.n = 64

    def test_magnitude_bounded_by_ramped_envelope(self):
        pert = RampedPerturbation(self.cfg, self.n, "cpu", seed=3)
        dt = .02
        peak_by_time = []
        for step in range(150):
            time = torch.full((self.n,), step * dt)
            force, torque, envelope = pert.step(time, time / self.cfg.ramp_time_s,
                                                fresh=(time == 0))
            self.assertEqual(force.shape, (self.n, 3))
            bound_f = envelope * self.cfg.max_force_n + 1e-6
            bound_t = envelope * self.cfg.max_torque_nm + 1e-6
            self.assertTrue(torch.all(force.norm(dim=-1) <= bound_f))
            self.assertTrue(torch.all(torque.norm(dim=-1) <= bound_t))
            peak_by_time.append(float(force.norm(dim=-1).max()))
        # Early in the episode the strongest push is a fraction of the peak;
        # after the ramp it can approach the configured maximum.
        self.assertLessEqual(max(peak_by_time[:5]), .15 * self.cfg.max_force_n)
        self.assertGreater(max(peak_by_time[100:]), .8 * self.cfg.max_force_n)

    def test_hold_durations_within_range_and_resample_on_fresh(self):
        pert = RampedPerturbation(self.cfg, 8, "cpu", seed=0)
        dt = .02
        previous = None
        changes = [[] for _ in range(8)]
        for step in range(200):
            time = torch.full((8,), step * dt)
            force, _, _ = pert.step(time, time / 2., fresh=(time == 0))
            direction = force / force.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            if previous is not None:
                changed = (direction - previous).norm(dim=-1) > 1e-5
                for i in changed.nonzero().flatten().tolist():
                    changes[i].append(step * dt)
            previous = direction
        for times in changes:
            gaps = np.diff([0.] + times)
            self.assertTrue(len(gaps) >= 8)
            self.assertTrue(np.all(gaps >= self.cfg.hold_min_s - 1e-6))
            self.assertTrue(np.all(gaps <= self.cfg.hold_max_s + dt + 1e-6))
        # A fresh episode resamples immediately even inside a hold.
        before = pert.force_direction.clone()
        fresh = torch.zeros(8, dtype=torch.bool)
        fresh[2] = True
        time = torch.full((8,), 0.)
        pert.step(time, time, fresh=fresh)
        self.assertFalse(torch.allclose(before[2], pert.force_direction[2]))
        self.assertTrue(torch.allclose(before[:2], pert.force_direction[:2]))

    def test_reseed_reproduces_sequence(self):
        a = RampedPerturbation(self.cfg, 4, "cpu", seed=11)
        b = RampedPerturbation(self.cfg, 4, "cpu", seed=99)
        b.reseed(11)
        for step in range(30):
            time = torch.full((4,), step * .02)
            fa = a.step(time, time / 2., fresh=(time == 0))[0]
            fb = b.step(time, time / 2., fresh=(time == 0))[0]
            torch.testing.assert_close(fa, fb)

    def test_vertical_component_bounded(self):
        cfg = PerturbationCfg(vertical_force_fraction=.25)
        pert = RampedPerturbation(cfg, 256, "cpu", seed=1)
        pert.step(torch.zeros(256), torch.zeros(256), fresh=torch.ones(256, dtype=torch.bool))
        horizontal = pert.force_direction[:, :2].norm(dim=-1)
        ratio = pert.force_direction[:, 2].abs() / horizontal
        self.assertTrue(torch.all(ratio <= .25 + 1e-5))


class FlagsTest(unittest.TestCase):

    def parse(self, argv):
        parser = argparse.ArgumentParser()
        add_perturbation_arguments(parser)
        return perturbation_from_args(parser.parse_args(argv))

    def test_disabled_by_default(self):
        self.assertIsNone(self.parse([]))

    def test_round_trip_and_profile(self):
        cfg = self.parse(["--perturbation", "--perturbation_max_force", "30",
                          "--perturbation_ramp_mode", "distance",
                          "--perturbation_ramp_distance", "2.5",
                          "--perturbation_hold", "0.2", "0.5"])
        self.assertEqual(cfg.max_force_n, 30.)
        self.assertEqual(cfg.ramp_mode, "distance")
        profile = cfg.profile()
        self.assertEqual(profile["schema"], PERTURBATION_SCHEMA)
        self.assertEqual(profile["progress"], "forward_travel_m / ramp_distance_m")
        self.assertEqual(profile["hold_max_s"], .5)

    def test_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            self.parse(["--perturbation", "--perturbation_hold", "0.5", "0.2"])
        with self.assertRaises(ValueError):
            self.parse(["--perturbation", "--perturbation_max_force", "0",
                        "--perturbation_max_torque", "0"])
        with self.assertRaises(ValueError):
            self.parse(["--perturbation", "--perturbation_start_fraction", "1.5"])
        with self.assertRaises(ValueError):
            self.parse(["--perturbation", "--perturbation_ramp_time", "0"])


class SummaryTest(unittest.TestCase):

    def test_outcome_counts_and_envelope_statistics(self):
        steps, trials = 5, 3
        valid = np.ones((steps, trials), dtype=bool)
        valid[3:, 0] = False  # trial 0 ends at step 2
        failure = np.zeros((steps, trials), dtype=bool)
        failure[2, 0] = True
        success = np.zeros((steps, trials), dtype=bool)
        success[4, 1] = True
        envelope = np.linspace(.1, 1., steps)[:, None].repeat(trials, axis=1)
        force = np.zeros((steps, trials, 3))
        force[..., 0] = envelope * 10
        traces = {"valid": list(valid), "failure": list(failure), "success": list(success),
                  "perturbation_force_w": list(force),
                  "perturbation_torque_w": list(np.zeros_like(force)),
                  "perturbation_envelope": list(envelope)}
        summary = summarize_outcomes(traces)
        self.assertEqual((summary["failure"], summary["success"], summary["timeout"]),
                         (1, 1, 1))
        self.assertAlmostEqual(summary["envelope_at_start"], .1)
        self.assertAlmostEqual(summary["failure_step_envelope_mean"], envelope[2, 0])
        self.assertAlmostEqual(summary["applied_force_n_max"], 10.)


if __name__ == "__main__":
    unittest.main()
