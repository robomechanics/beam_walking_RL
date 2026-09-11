"""Numerical and chronology regressions for old-policy surface evaluation."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from policy_surface_data import summarize, validate_measurement, GRID_SEED, WIDTHS
from make_policy_surfaces import summarize_cells, summarize_factors, measured_mesh
import matplotlib.pyplot as plt


def ideal_payload():
    trials, cycles, steps = 32, 4, 24
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

    payload["force_norm_200hz"] = payload["substep_contacts"].astype(float) * 20
    payload["phase_ticks"] = np.broadcast_to(np.arange(24), (32, 4, 24)).copy()
    states = np.zeros((32, 5, 49))
    states[..., 3] = 1.
    states[..., 0] = np.arange(5)[None] * .144
    payload["phase_zero_states"] = states
    payload["any_failure"] = np.zeros(32, dtype=bool)
    return payload


class SurfaceMeasurementsTest(unittest.TestCase):
    condition = ("trot", .30, .48, .30, .625)

    def test_pure_forward_translation_is_periodic_and_energy_valid(self):
        payload = ideal_payload()
        validate_measurement(payload, self.condition)
        rows = summarize(payload, self.condition, GRID_SEED)
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(row["energy_trial_valid"] for row in rows))
        self.assertTrue(all(row["cycle_rms"] == 0 for row in rows))

    def test_action_memory_drift_and_early_failure_invalidate_endpoint(self):
        payload = ideal_payload()
        payload["phase_zero_states"][0, -1, 37] = 1.
        payload["any_failure"][1] = True
        rows = summarize(payload, self.condition, GRID_SEED)
        self.assertFalse(rows[0]["periodic_orbit_gate_pass"])
        self.assertFalse(rows[1]["energy_trial_valid"])
        self.assertTrue(rows[2]["energy_trial_valid"])

    def test_force_history_order_and_command_mismatch_are_rejected(self):
        payload = ideal_payload()
        payload["force_norm_200hz"][0, 0, 0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "chronology"):
            validate_measurement(payload, self.condition)
        payload = ideal_payload()
        payload["desired"][0, 0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, "Desired"):
            validate_measurement(payload, self.condition)

    def test_failed_width_cannot_be_hidden_by_aggregate(self):
        base = summarize(ideal_payload(), self.condition, GRID_SEED)
        rows = []
        for width in WIDTHS:
            for index, row in enumerate(base):
                row = dict(row, step_width=width)
                if width == .10 and index < 4:
                    row["energy_trial_valid"] = 0
                    row["periodic_orbit_gate_pass"] = 0
                rows.append(row)
        cells = summarize_cells(pd.DataFrame(rows))
        factors = summarize_factors(cells)
        self.assertEqual(factors.iloc[0].valid_widths, 4)
        self.assertTrue(np.isnan(factors.iloc[0].positive_mechanical_cot_median))
        self.assertAlmostEqual(factors.iloc[0].periodic_rate, 156 / 160)

    def test_mesh_omits_faces_adjacent_to_missing_vertex(self):
        rows = [dict(speed=speed, command_df=duty, z=1.)
                for speed in (.25, .30, .35, .40) for duty in (.50, .625, .75)]
        rows[4]["z"] = np.nan
        data = pd.DataFrame(rows)
        fig = plt.figure()
        axis = fig.add_subplot(projection="3d")
        with patch("make_policy_surfaces.Poly3DCollection", wraps=__import__(
                "mpl_toolkits.mplot3d.art3d", fromlist=["Poly3DCollection"]
                ).Poly3DCollection) as polygons:
            missing = measured_mesh(axis, data, "z", "blue")
            self.assertEqual(missing, 1)
            self.assertEqual(len(polygons.call_args.args[0]), 2)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
