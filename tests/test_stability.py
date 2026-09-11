"""CPU-only tests for the paper-aligned closed-loop return-map metric."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/beam_walking"))
from beam_walking.experiment.stability import (
    AUGMENTED_DIM, FROZEN_GATE_LIMITS, PHYSICAL_DIM, PRIMARY_METRIC,
    RAW_STATE_DIM, STATE_SCALES, confirmatory_protocol,
    analyze_policy_ensemble, analyze_stability, apply_scaled_perturbation,
    canonical_condition_keys, canonical_reference_states, command_fidelity,
    construct_master_stencil, contact_event_signature, estimate_maps,
    hybrid_topology_gate, paper_claim_summary, translation_symmetry_fidelity,
    master_stencil_layout, quaternion_to_rotation_vector,
    rotation_vector_to_quaternion, state_delta,
)


def base_state():
    state = np.zeros(RAW_STATE_DIM)
    state[3] = 1.
    return state


def state_from_normalized(vector):
    vector = np.asarray(vector) * STATE_SCALES
    state = base_state()
    state[:3] = vector[:3]
    state[3:7] = rotation_vector_to_quaternion(vector[3:6])
    state[7:19] = vector[6:18]
    state[19:37] = vector[18:36]
    state[37:49] = vector[36:48]
    return state


def linear_stencil(matrix, h=.05):
    """Construct exact normalized central pairs for a dense linear map."""
    initial = np.repeat(base_state()[None], 1 + 2 * AUGMENTED_DIM, axis=0)
    final = initial.copy()
    for coordinate in range(AUGMENTED_DIM):
        direction = np.zeros(AUGMENTED_DIM)
        direction[coordinate] = h
        initial[1 + 2 * coordinate] = state_from_normalized(direction)
        initial[2 + 2 * coordinate] = state_from_normalized(-direction)
        output = matrix @ direction
        final[1 + 2 * coordinate] = state_from_normalized(output)
        final[2 + 2 * coordinate] = state_from_normalized(-output)
    return initial, final


def desired_trot_cycle():
    desired = np.zeros((24, 4), dtype=bool)
    desired[:12, [0, 3]] = True
    desired[12:, [1, 2]] = True
    return desired


TEST_TRAINING_SOURCE = "training-source"
TEST_TRAINING_SEED = 0
TEST_TRAINING_LINEAGE = hashlib.sha256(
    f"{TEST_TRAINING_SOURCE}:{TEST_TRAINING_SEED}:fresh_v1".encode()
).hexdigest()
TRAINING_PROVENANCE = {
    "mode": "train",
    "task_sha256": "task",
    "seed": TEST_TRAINING_SEED,
    "fresh_training": True,
    "training_lineage_id": TEST_TRAINING_LINEAGE,
    "training_source_sha256": TEST_TRAINING_SOURCE,
    "training_num_envs": 4096,
    "training_iterations_requested": 1800,
    "checkpoint_selection_rule": "final_requested_iteration",
}
TRAINING_PROVENANCE_BYTES = json.dumps(
    TRAINING_PROVENANCE, sort_keys=True).encode()


def manifest_for(names, conditions):
    return {
        "schema": "beam_stability_v2",
        "external_pushes": False,
        "paper_metric_primary":
            "chi_orb=sigma_max(Phi_orbital_augmented_46D)",
        "primary_metric_scope":
            "translation-reduced learned-controller analogue",
        "unquotiented_full_48d_is_diagnostic": True,
        "translation_symmetry_residual_limit": .02,
        "gate_limits": FROZEN_GATE_LIMITS,
        "evaluation_reference_seed": 10000,
        "state_scales": STATE_SCALES.tolist(),
        "references": 1,
        "zero_clones_per_reference": 2,
        "master_stencil_size": 291,
        "settle_cycles": 12,
        "reference_initial_offset_half_width_normalized": .02,
        "gait_regime": {
            "trot_df": [.50, .75],
            "walk_df": [.75, .75],
            "minimum_swing_s": .10,
        },
        "confirmatory_grid": False,
        "initial_stencil_max_error_limit": 1e-3,
        "initial_condition_number_limit": 1.05,
        "zero_clone_noise_fraction_of_h_limit": .05,
        "settle_group_rms_limit": 1e-3,
        "task_sha256": "task",
        "evaluation_sha256": "evaluation",
        "checkpoint_sha256": "checkpoint",
        "training_seed": TEST_TRAINING_SEED,
        "fresh_training": True,
        "training_lineage_id": TEST_TRAINING_LINEAGE,
        "training_provenance_file": "training_provenance.json",
        "training_provenance_sha256":
            hashlib.sha256(TRAINING_PROVENANCE_BYTES).hexdigest(),
        "reference_initialization":
            "canonical_default_plus_matched_offsets_v1",
        "training_source_sha256": TEST_TRAINING_SOURCE,
        "training_num_envs": 4096,
        "training_iterations_requested": 1800,
        "checkpoint_selection_rule": "final_requested_iteration",
        "checkpoint_iteration": 1799,
        "joint_order": [f"joint_{index}" for index in range(12)],
        "action_order": [f"joint_{index}" for index in range(12)],
        "expected_files": names,
        "expected_conditions": conditions,
    }


def condition(name, h):
    return {
        "filename": name, "speed": .3, "duty_factor": .5,
        "step_width": .3, "period": .48, "gait": "trot",
        "perturbation_h": h,
    }


def nominal_payload(matrix, h):
    initial, final = linear_stencil(matrix, h)
    desired = desired_trot_cycle()
    feet = np.zeros((1, 24, 4, 3))
    feet[..., 1] = np.asarray([.15, -.15, .15, -.15])
    substeps = np.repeat(desired, 4, axis=0)
    initial_contacts = desired[-1]
    return {
        "initial_states": initial[None],
        "final_states": final[None],
        "command": np.asarray([.3, .5, .3, .48, 0]),
        "perturbation_h": h,
        "cycle_rms": np.asarray([.01]),
        "done": np.zeros((1, 97), dtype=bool),
        "task_sha256": "task",
        "evaluation_sha256": "evaluation",
        "checkpoint_sha256": "checkpoint",
        "state_scales": STATE_SCALES,
        "nominal_contacts": desired[None],
        "nominal_desired": desired[None],
        "nominal_feet_body": feet,
        "nominal_forward_velocity": np.full((1, 24), .3),
        "nominal_lateral_position": np.zeros((1, 24)),
        "nominal_heading": np.zeros((1, 24)),
        "nominal_world_lateral_velocity": np.zeros((1, 24)),
        "nominal_body_yaw_rate": np.zeros((1, 24)),
        "nominal_failure": np.zeros((1, 24), dtype=bool),
        "settle_group_rms": np.asarray([0.]),
        "settle_done": np.asarray([False]),
        "zero_initial_states": np.repeat(initial[None, :1], 2, axis=1),
        "zero_final_states": np.repeat(final[None, :1], 2, axis=1),
        "zero_initial_contacts": np.repeat(
            initial_contacts[None, None], 2, axis=1),
        "zero_substep_contacts": np.repeat(
            substeps[None, None], 2, axis=1),
        "stencil_initial_contacts": np.repeat(
            initial_contacts[None, None], 97, axis=1),
        "stencil_substep_contacts": np.repeat(
            substeps[None, None], 97, axis=1),
        "desired_initial_contacts": initial_contacts[None],
        "desired_substep_contacts": substeps[None],
    }


class StabilityMetricTest(unittest.TestCase):
    def test_confirmatory_protocol_requires_all_frozen_resources(self):
        protocol = {
            "condition_keys": canonical_condition_keys(),
            "references": 8, "settle_cycles": 12,
            "perturbation_sizes": [.025, .05, .10],
            "zero_clones": 2, "master_stencil_size": 291,
            "evaluation_reference_seed": 10000,
            "training_iterations_requested": 1800,
            "checkpoint_iteration": 1799,
            "fresh_training": True,
        }
        self.assertTrue(confirmatory_protocol(**protocol))
        for field, value in (
            ("condition_keys", list(canonical_condition_keys())[:-1]),
            ("references", 1),
            ("settle_cycles", 4),
            ("perturbation_sizes", [.025, .05, .10, .15]),
            ("evaluation_reference_seed", 10001),
            ("training_iterations_requested", 100),
            ("checkpoint_iteration", 1700),
            ("fresh_training", False),
        ):
            changed = {**protocol, field: value}
            self.assertFalse(confirmatory_protocol(**changed), field)

    def test_canonical_references_repeat_exactly_across_conditions(self):
        joints = np.linspace(-.8, .8, 12)
        first = canonical_reference_states(joints, 8, seed=10000)
        second = canonical_reference_states(joints, 8, seed=10000)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(
            first[:, :2], np.tile([-.65, 0.], (8, 1)))
        self.assertFalse(np.array_equal(
            first, canonical_reference_states(joints, 8, seed=10001)))

    def test_so3_round_trip_and_shortest_sign(self):
        vectors = np.asarray([[0., 0., 0.], [.1, -.2, .3], [2.8, 0., 0.]])
        recovered = quaternion_to_rotation_vector(
            rotation_vector_to_quaternion(vectors))
        np.testing.assert_allclose(recovered, vectors, atol=1e-10)
        quaternion = rotation_vector_to_quaternion(vectors[1])
        np.testing.assert_allclose(
            quaternion_to_rotation_vector(-quaternion), vectors[1], atol=1e-10)

    def test_scaled_perturbations_recover_all_tangent_coordinates(self):
        base = base_state()
        for coordinate in range(AUGMENTED_DIM):
            changed = apply_scaled_perturbation(base, coordinate, .05)
            delta = state_delta(base, changed) / STATE_SCALES
            expected = np.zeros(AUGMENTED_DIM)
            expected[coordinate] = .05
            np.testing.assert_allclose(delta, expected, atol=1e-10)

    def test_master_stencil_preserves_x_y_columns_and_zero_clones(self):
        layout = master_stencil_layout((.025, .05, .10), 2)
        settled = np.repeat(
            base_state()[None, None], layout["size"], axis=1)
        settled[..., 0] = 9.
        settled[..., 1] = -4.
        master, layout = construct_master_stencil(
            settled, (.025, .05, .10), 2)
        for h, indices in layout["h_indices"].items():
            x0, _ = linear_stencil(np.eye(AUGMENTED_DIM), h)
            measured = master[0, indices]
            expected_columns, _ = __import__(
                "beam_walking.experiment.stability",
                fromlist=["central_columns"]).central_columns(measured, measured)
            np.testing.assert_allclose(
                expected_columns, h * np.eye(AUGMENTED_DIM), atol=1e-10)
            self.assertNotEqual(measured[1, 0], measured[2, 0])
            self.assertNotEqual(measured[3, 1], measured[4, 1])
        np.testing.assert_allclose(master[0, 0], master[0, 1])

    def test_dense_nondiagonal_map_recovery(self):
        rng = np.random.default_rng(4)
        matrix = .03 * rng.normal(size=(AUGMENTED_DIM, AUGMENTED_DIM))
        matrix += np.diag(np.linspace(.2, .7, AUGMENTED_DIM))
        initial, final = linear_stencil(matrix)
        maps = estimate_maps(initial, final)
        np.testing.assert_allclose(maps["augmented"]["phi"], matrix, atol=1e-9)
        self.assertAlmostEqual(
            maps["augmented"]["chi"],
            np.linalg.svd(matrix, compute_uv=False)[0], places=9)

    def test_neutral_translations_floor_full_chi_but_not_orbital_chi(self):
        diagonal = np.linspace(.2, .6, AUGMENTED_DIM)
        diagonal[0], diagonal[1] = 1., 1.
        matrix = np.diag(diagonal)
        initial, final = linear_stencil(matrix)
        maps = estimate_maps(initial, final)
        self.assertAlmostEqual(maps["augmented"]["chi"], 1., places=9)
        self.assertAlmostEqual(
            maps["orbital_augmented"]["chi"], .6, places=9)
        symmetry = translation_symmetry_fidelity(matrix)
        self.assertEqual(symmetry["translation_symmetry_gate_pass"], 1)
        self.assertEqual(maps["augmented"]["rank"], AUGMENTED_DIM)

    def test_corrupted_translation_column_fails_symmetry_gate(self):
        matrix = np.eye(AUGMENTED_DIM)
        matrix[2, 0] = .03
        symmetry = translation_symmetry_fidelity(matrix)
        self.assertEqual(symmetry["translation_symmetry_gate_pass"], 0)
        self.assertGreater(symmetry["translation_cross_coupling_max"], .02)

    def test_hybrid_signature_rejects_swapped_leg_event_order(self):
        initial = np.zeros(4, dtype=bool)
        first = np.zeros((3, 4), dtype=bool)
        first[0:, 0] = True
        first[1:, 1] = True
        swapped = np.zeros((3, 4), dtype=bool)
        swapped[0:, 1] = True
        swapped[1:, 0] = True
        self.assertNotEqual(
            contact_event_signature(initial, first),
            contact_event_signature(initial, swapped))
        simultaneous = np.zeros((2, 4), dtype=bool)
        simultaneous[0, [0, 1]] = True
        self.assertEqual(
            contact_event_signature(initial, simultaneous)[0],
            ((0, 1), (1, 1)))

    def test_clone_consistent_wrong_contact_topology_fails(self):
        payload = nominal_payload(np.eye(AUGMENTED_DIM), .05)
        wrong = payload["desired_substep_contacts"][:, :, [1, 0, 3, 2]]
        payload["stencil_initial_contacts"][:] = payload[
            "desired_initial_contacts"][:, None]
        payload["zero_initial_contacts"][:] = payload[
            "desired_initial_contacts"][:, None]
        payload["stencil_substep_contacts"][:] = wrong[:, None]
        payload["zero_substep_contacts"][:] = wrong[:, None]
        self.assertFalse(hybrid_topology_gate(payload, 0))

    def test_stance_width_ignores_swing_excursion(self):
        payload = nominal_payload(np.eye(AUGMENTED_DIM), .05)
        contacts = payload["nominal_contacts"][0]
        feet = payload["nominal_feet_body"][0]
        for leg in range(4):
            feet[~contacts[:, leg], leg, 1] = 2. * (-1 if leg % 2 else 1)
        metrics = command_fidelity(payload, 0, .3, .5, .3)
        self.assertAlmostEqual(metrics["achieved_stance_width"], .3)
        self.assertEqual(metrics["command_gate_pass"], 1)
        self.assertGreater(metrics["all_phase_foot_lateral_mae"], .5)

    def test_turning_or_weaving_fails_stability_command_gate(self):
        payload = nominal_payload(np.eye(AUGMENTED_DIM), .05)
        payload["nominal_heading"][:] = .11
        metrics = command_fidelity(payload, 0, .3, .5, .3)
        self.assertEqual(metrics["command_gate_pass"], 0)
        self.assertGreater(metrics["heading_rmse_rad"], .10)

    def _write_grid(self, directory, matrix=None, mutate=None):
        matrix = np.eye(AUGMENTED_DIM) if matrix is None else matrix
        names, conditions = [], []
        for h in (.025, .05, .10):
            name = f"stability_cell_h{h:.3f}.npz"
            names.append(name)
            conditions.append(condition(name, h))
            payload = nominal_payload(matrix, h)
            if mutate:
                mutate(payload, h)
            np.savez_compressed(directory / name, **payload)
        (directory / "training_provenance.json").write_bytes(
            TRAINING_PROVENANCE_BYTES)
        (directory / "stability_manifest.json").write_text(json.dumps(
            manifest_for(names, conditions)))
        return names

    def test_force_rejected_then_complete_gated_grid_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            names = self._write_grid(directory)
            path = directory / names[1]
            payload = nominal_payload(np.eye(AUGMENTED_DIM), .05)
            np.savez_compressed(path, **payload, force=np.zeros(1))
            with self.assertRaisesRegex(ValueError, "push-free"):
                analyze_stability(directory)
            np.savez_compressed(path, **payload)
            summary = analyze_stability(directory)
            self.assertEqual(len(summary), 3)
            self.assertTrue(all(row["valid_references"] == 1 for row in summary))
            archive = np.load(directory / "return_maps.npz")
            self.assertEqual(len(archive.files), 3 * 4 * 3)
            claims = json.loads((directory / "paper_claims.json").read_text())
            self.assertFalse(claims["claim_values_released"])
            self.assertEqual(
                claims["status"], "suppressed_failed_validity_or_coverage_gate")

    def test_tampered_confirmatory_manifest_flag_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_grid(directory)
            manifest_path = directory / "stability_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["confirmatory_grid"] = True
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "confirmatory flag"):
                analyze_stability(directory)

    def test_rank_defect_and_bad_speed_exclude_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)

            def mutate(payload, h):
                payload["nominal_forward_velocity"][:] = .1
                coordinate = AUGMENTED_DIM - 1
                payload["initial_states"][0, 1 + 2 * coordinate] = base_state()
                payload["initial_states"][0, 2 + 2 * coordinate] = base_state()

            self._write_grid(directory, mutate=mutate)
            summary = analyze_stability(directory)
            self.assertTrue(all(row["valid_references"] == 0 for row in summary))
            table = (directory / "stability_references.csv").read_text()
            self.assertIn(",47,", table)

    def test_scale_and_command_identity_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)

            def bad_scale(payload, h):
                payload["state_scales"] = STATE_SCALES * 2

            self._write_grid(directory, mutate=bad_scale)
            with self.assertRaisesRegex(ValueError, "scales"):
                analyze_stability(directory)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)

            def bad_command(payload, h):
                payload["command"][0] = .35

            self._write_grid(directory, mutate=bad_command)
            with self.assertRaisesRegex(ValueError, "command"):
                analyze_stability(directory)

    def test_paper_effects_are_frozen_to_orbital_metric(self):
        rows = []
        for gait in ("trot", "walk"):
            for duty in (.50, .75):
                for width in (.10, .30):
                    for speed in (.25, .40):
                        orbital = (
                            1.0 - (.2 if duty == .75 else 0.)
                            + (.3 if width == .10 else 0.)
                            + (.02 if speed == .40 else 0.)
                            + (.01 if gait == "walk" else 0.))
                        rows.append({
                            "speed": speed, "command_df": duty,
                            "step_width": width, "period": .48, "gait": gait,
                            "condition_valid": 1,
                            "chi_orbital_augmented_median": orbital,
                            "chi_augmented_median": 10. - orbital,
                        })
        report = paper_claim_summary(rows, confirmatory_grid=True)
        self.assertEqual(PRIMARY_METRIC, "chi_orbital_augmented_median")
        self.assertEqual(
            report["primary_metric_scope"],
            "46D augmented orbital return map with neutral global x/y removed")
        self.assertAlmostEqual(
            report["duty_factor_matched_effect"]["median_high_minus_low"],
            -.2)

    def test_corrupted_translation_invalidates_analyzed_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            matrix = np.eye(AUGMENTED_DIM)
            matrix[2, 0] = .03
            self._write_grid(directory, matrix=matrix)
            summary = analyze_stability(directory)
            self.assertTrue(all(row["valid_references"] == 0 for row in summary))
            self.assertTrue(all(
                row["translation_symmetry_gate_pass_rate"] == 0
                for row in summary))

    def test_ensemble_aggregates_five_distinct_seed_matched_effects(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for gait, duties in (("trot", (.50, .625, .75)),
                                 ("walk", (.75,))):
                for duty in duties:
                    for width in (.10, .20, .30, .40, .50):
                        for speed in (.25, .30, .35, .40):
                            chi = (
                                1.0 - (.2 if duty == .75 else 0.)
                                + (.3 if width == .10 else 0.))
                            rows.append({
                                "speed": speed, "command_df": duty,
                                "step_width": width, "period": .48,
                                "gait": gait, "condition_valid": 1,
                                "chi_orbital_augmented_median": chi,
                            })
            directories = []
            expected = [{"row": index} for index in range(len(rows))]
            for policy in range(5):
                directory = root / f"policy_{policy}"
                directory.mkdir()
                directories.append(directory)
                manifest = {
                    "schema": "beam_stability_v2",
                    "task_sha256": "task",
                    "evaluation_sha256": "evaluation",
                    "state_scales": STATE_SCALES.tolist(),
                    "expected_conditions": expected,
                    "joint_order": [f"joint_{i}" for i in range(12)],
                    "action_order": [f"joint_{i}" for i in range(12)],
                    "paper_metric_primary":
                        "chi_orb=sigma_max(Phi_orbital_augmented_46D)",
                    "primary_metric_scope":
                        "translation-reduced learned-controller analogue",
                    "translation_symmetry_residual_limit": .02,
                    "gate_limits": FROZEN_GATE_LIMITS,
                    "confirmatory_grid": True,
                    "references": 8,
                    "zero_clones_per_reference": 2,
                    "master_stencil_size": 291,
                    "settle_cycles": 12,
                    "reference_initial_offset_half_width_normalized": .02,
                    "reference_initialization":
                        "canonical_default_plus_matched_offsets_v1",
                    "evaluation_reference_seed": 10000,
                    "training_source_sha256": "training-source",
                    "training_num_envs": 4096,
                    "training_iterations_requested": 1800,
                    "checkpoint_selection_rule":
                        "final_requested_iteration",
                    "checkpoint_iteration": 1799,
                    "gait_regime": {
                        "trot_df": [.50, .75],
                        "walk_df": [.75, .75],
                        "minimum_swing_s": .10,
                    },
                    "checkpoint_sha256": f"checkpoint_{policy}",
                    "training_seed": policy,
                    "fresh_training": True,
                    "training_lineage_id": f"{policy + 1:064x}",
                    "training_provenance_sha256": f"{policy + 101:064x}",
                }
                (directory / "stability_manifest.json").write_text(
                    json.dumps(manifest))
                pd.DataFrame(rows).to_csv(
                    directory / "paper_conditions.csv", index=False)
                (directory / "paper_claims.json").write_text(json.dumps({
                    "claim_values_released": True}))
            output = root / "ensemble"
            with patch(
                "beam_walking.experiment.stability.analyze_stability",
                return_value=[],
            ):
                report = analyze_policy_ensemble(directories, output)
            self.assertTrue(report["qualitative_alignment_pass"])
            self.assertNotIn("paper_claim_confirmed", report)
            self.assertIn("not supplied by the paper",
                          report["equivalence_margin_source"])
            self.assertEqual(report["primary_metric"], PRIMARY_METRIC)
            self.assertTrue((output / "policy_effects.csv").is_file())
            self.assertTrue((output / "ensemble_claims.json").is_file())

            last_path = directories[-1] / "stability_manifest.json"
            changed = json.loads(last_path.read_text())
            changed["settle_cycles"] = 4
            last_path.write_text(json.dumps(changed))
            with patch(
                "beam_walking.experiment.stability.analyze_stability",
                return_value=[],
            ), self.assertRaisesRegex(ValueError, "differ"):
                analyze_policy_ensemble(directories, root / "mixed_protocol")
            changed["settle_cycles"] = 12
            changed["fresh_training"] = False
            last_path.write_text(json.dumps(changed))
            with patch(
                "beam_walking.experiment.stability.analyze_stability",
                return_value=[],
            ), self.assertRaisesRegex(ValueError, "parent-free"):
                analyze_policy_ensemble(directories, root / "nonfresh")

    def test_ensemble_requires_five_policies(self):
        with self.assertRaisesRegex(ValueError, "at least five"):
            analyze_policy_ensemble(["a", "b", "c", "d"], "/tmp/not_written")


if __name__ == "__main__":
    unittest.main()
