"""CPU-only tests for the paper-aligned closed-loop return-map metric."""
import json
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/beam_walking"))
from beam_walking.experiment.stability import (
    ACTION_DIM, AUGMENTED_DIM, PHYSICAL_DIM, RAW_STATE_DIM, STATE_SCALES,
    apply_scaled_perturbation, analyze_stability, estimate_maps,
    quaternion_to_rotation_vector, rotation_vector_to_quaternion, state_delta,
)


def base_state():
    state = np.zeros(RAW_STATE_DIM)
    state[3] = 1.0
    return state


def linear_stencil(matrix, h=.05):
    """Construct exact initial/final central pairs for a normalized linear map."""
    initial = np.repeat(base_state()[None], 1 + 2 * AUGMENTED_DIM, axis=0)
    final = initial.copy()
    for j in range(AUGMENTED_DIM):
        initial[1 + 2*j] = apply_scaled_perturbation(initial[0], j, h)
        initial[2 + 2*j] = apply_scaled_perturbation(initial[0], j, -h)
        # Diagonal fixtures avoid confusing finite SO(3) composition with vector addition.
        amount = matrix[j, j] * h
        final[1 + 2*j] = apply_scaled_perturbation(final[0], j, amount)
        final[2 + 2*j] = apply_scaled_perturbation(final[0], j, -amount)
    return initial, final


class StabilityMetricTest(unittest.TestCase):
    def test_so3_round_trip_and_shortest_sign(self):
        vectors = np.asarray([[0., 0., 0.], [.1, -.2, .3], [2.8, 0., 0.]])
        recovered = quaternion_to_rotation_vector(rotation_vector_to_quaternion(vectors))
        np.testing.assert_allclose(recovered, vectors, atol=1e-10)
        q = rotation_vector_to_quaternion(vectors[1])
        np.testing.assert_allclose(quaternion_to_rotation_vector(-q), vectors[1], atol=1e-10)

    def test_scaled_perturbations_recover_all_tangent_coordinates(self):
        base = base_state()
        for j in range(AUGMENTED_DIM):
            changed = apply_scaled_perturbation(base, j, .05)
            delta = state_delta(base, changed) / STATE_SCALES
            expected = np.zeros(AUGMENTED_DIM); expected[j] = .05
            np.testing.assert_allclose(delta, expected, atol=1e-10)

    def test_known_linear_map_and_translation_removal(self):
        diagonal = np.linspace(.2, .9, AUGMENTED_DIM)
        diagonal[0], diagonal[1] = 1.2, 1.1
        initial, final = linear_stencil(np.diag(diagonal))
        maps = estimate_maps(initial, final)
        self.assertAlmostEqual(maps["augmented"]["chi"], 1.2, places=9)
        self.assertAlmostEqual(maps["physical_conditional"]["chi"], 1.2, places=9)
        self.assertAlmostEqual(maps["orbital_physical_conditional"]["chi"], diagonal[PHYSICAL_DIM-1], places=9)
        self.assertEqual(maps["augmented"]["rank"], AUGMENTED_DIM)

    def _nominal_payload(self, initial, final, h):
        ticks = 24
        desired = np.zeros((1, ticks, 4), dtype=bool)
        desired[:, :12, :] = True
        feet = np.zeros((1, ticks, 4, 3))
        feet[..., 1] = np.asarray([.15, -.15, .15, -.15])
        return dict(
            initial_states=initial[None], final_states=final[None],
            command=np.asarray([.3, .5, .3, .48, 0]), perturbation_h=h,
            cycle_rms=np.asarray([.01]), done=np.zeros((1, 97), dtype=bool),
            source_sha256="source", checkpoint_sha256="checkpoint",
            nominal_contacts=desired.copy(), nominal_desired=desired,
            nominal_feet_body=feet,
            nominal_forward_velocity=np.full((1, ticks), .3),
            nominal_failure=np.zeros((1, ticks), dtype=bool),
        )

    def test_manifest_rejects_force_fields_then_accepts_gated_stencil(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            initial, final = linear_stencil(np.eye(AUGMENTED_DIM))
            name = "stability_cell_h0.050.npz"
            payload = self._nominal_payload(initial, final, .05)
            manifest = {
                "external_pushes": False, "expected_files": [name],
                "source_sha256": "source", "checkpoint_sha256": "checkpoint",
            }
            np.savez_compressed(directory/name, **payload, force=np.zeros(1))
            (directory/"stability_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "push-free"):
                analyze_stability(directory)

            names = []
            for h in (.025, .05, .10):
                filename = f"stability_cell_h{h:.3f}.npz"
                names.append(filename)
                np.savez_compressed(
                    directory/filename, **self._nominal_payload(initial, final, h))
            manifest["expected_files"] = names
            (directory/"stability_manifest.json").write_text(json.dumps(manifest))
            summary = analyze_stability(directory)
            self.assertEqual(len(summary), 3)
            self.assertTrue(all(row["valid_references"] == 1 for row in summary))
            self.assertTrue(all(row["finite_difference_gate_rate"] == 1 for row in summary))
            self.assertAlmostEqual(summary[1]["chi_orbital_augmented_median"], 1.)
            claims = json.loads((directory/"paper_claims.json").read_text())
            self.assertEqual(claims["valid_conditions"], 1)

    def test_command_fidelity_failure_excludes_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            initial, final = linear_stencil(np.eye(AUGMENTED_DIM))
            names = []
            for h in (.025, .05, .10):
                name = f"stability_bad_h{h:.3f}.npz"
                names.append(name)
                payload = self._nominal_payload(initial, final, h)
                payload["nominal_forward_velocity"][:] = .1
                np.savez_compressed(directory/name, **payload)
            (directory/"stability_manifest.json").write_text(json.dumps({
                "external_pushes": False, "expected_files": names,
                "source_sha256": "source", "checkpoint_sha256": "checkpoint",
            }))
            summary = analyze_stability(directory)
            self.assertTrue(all(row["command_gate_rate"] == 0 for row in summary))
            self.assertTrue(all(row["valid_references"] == 0 for row in summary))



if __name__ == "__main__":
    unittest.main()
