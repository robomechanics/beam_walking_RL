"""CPU-only tests for the paper-aligned closed-loop return-map metric."""
import ast
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
    AUGMENTED_DIM, CONFIRMATORY_REFERENCE_SEED, FROZEN_GATE_LIMITS,
    EVALUATION_SCRIPT_FILENAMES, EVALUATION_SOURCE_PATHS,
    HYBRID_EVENT_TOLERANCE_S, PHYSICAL_DIM, PRIMARY_METRIC,
    RAW_STATE_DIM, STATE_SCALES, confirmatory_protocol,
    analyze_policy_ensemble, analyze_stability, apply_scaled_perturbation,
    canonical_condition_keys, canonical_physical_condition_keys,
    canonical_reference_states, command_fidelity, mechanical_cot,
    construct_master_stencil, contact_event_signature, estimate_maps,
    hybrid_topology_gate, paper_claim_summary, translation_symmetry_fidelity,
    master_stencil_layout, numerical_fidelity, quaternion_to_rotation_vector,
    STABILITY_SCHEMA, V4_STABILITY_SCHEMA, V5_ACTIVE_NUMERICAL_GATES,
    V5_RETIRED_V4_GATE,
    rotation_vector_to_quaternion, state_delta,
    zero_reproducibility_metrics, _energy_topology_gate,
)


class EvaluationSourceSnapshotTest(unittest.TestCase):
    def test_every_hashed_evaluation_script_is_snapshotted(self):
        hashed_scripts = {
            Path(path).name for path in EVALUATION_SOURCE_PATHS
            if Path(path).parts[0] == "scripts"
        }
        self.assertEqual(hashed_scripts, set(EVALUATION_SCRIPT_FILENAMES))
        self.assertIn("evaluation_capacity.py", EVALUATION_SCRIPT_FILENAMES)

    def test_stencil_cache_broadcasts_preserve_foot_axis(self):
        source = (
            Path(__file__).resolve().parents[1] / "scripts/stability_experiment.py"
        ).read_text()
        tree = ast.parse(source)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "repeat_interleave"
        ]
        # Settling, the exact finite-difference fork, and the zero-only
        # reproducibility fork each broadcast both contact caches.
        self.assertEqual(len(calls), 6)
        for call in calls:
            dimensions = [
                keyword.value for keyword in call.keywords if keyword.arg == "dim"
            ]
            self.assertEqual(len(dimensions), 1)
            self.assertIsInstance(dimensions[0], ast.Constant)
            self.assertEqual(dimensions[0].value, 0)

    def test_collector_does_not_assume_pre_step_failure_cache_exists(self):
        source = (
            Path(__file__).resolve().parents[1] / "scripts/stability_experiment.py"
        ).read_text()
        self.assertNotIn("env.substep_failure", source)

    def test_zero_diagnostic_is_fail_closed_before_app_launch(self):
        source = (
            Path(__file__).resolve().parents[1] / "scripts/stability_experiment.py"
        ).read_text()
        pre_app = source[:source.index("app = AppLauncher(args).app")]
        self.assertIn("args.zero_reproducibility_clones == 64", pre_app)
        self.assertIn("args.enhanced_determinism", pre_app)
        self.assertIn("args.seed == DEVELOPMENT_REFERENCE_SEED", pre_app)
        self.assertIn(
            "cfg.sim.physx.enable_enhanced_determinism = args.enhanced_determinism",
            source)
        self.assertIn("Identical fork produced unequal first actions", source)


class ZeroReproducibilityTest(unittest.TestCase):
    def payload(self):
        desired = desired_trot_cycle()
        states = np.broadcast_to(
            base_state(), (1, 25, 3, RAW_STATE_DIM)).copy()
        return dict(
            states=states,
            actions=np.zeros((1, 24, 3, 12)),
            initial_contacts=np.broadcast_to(desired[-1], (1, 3, 4)).copy(),
            substep_contacts=np.broadcast_to(
                np.repeat(desired, 4, axis=0), (1, 3, 96, 4)).copy(),
            desired_initial_contacts=desired[-1][None],
            desired_substep_contacts=np.repeat(desired, 4, axis=0)[None],
            settle_done=np.zeros(1, dtype=bool),
            done=np.zeros((1, 24, 3), dtype=bool),
            gait="trot", period=.48)

    def test_identical_boundary_and_noise_failure(self):
        values = self.payload()
        result = zero_reproducibility_metrics(**values)
        self.assertTrue(result["zero_reproducibility_pass"])
        values["states"][0, -1, 1, 2] = .000125
        self.assertTrue(zero_reproducibility_metrics(
            **values)["zero_reproducibility_pass"])
        values["states"][0, -1, 1, 2] = .000126
        self.assertFalse(zero_reproducibility_metrics(
            **values)["zero_reproducibility_pass"])

    def test_termination_and_topology_mismatch_fail(self):
        values = self.payload()
        values["done"][0, 2, 1] = True
        self.assertFalse(zero_reproducibility_metrics(
            **values)["zero_reproducibility_pass"])
        values = self.payload()
        values["substep_contacts"][0, 1, 10, 0] ^= True
        self.assertFalse(zero_reproducibility_metrics(
            **values)["zero_reproducibility_pass"])

    def test_malformed_and_nonfinite_reject(self):
        values = self.payload()
        values["states"][0, 0, 0, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            zero_reproducibility_metrics(**values)


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
    energy_name = "energy_s0.300_d0.500_v0.300_trot_p0.48.npz"
    return {
        "schema": STABILITY_SCHEMA,
        "hybrid_topology_definition":
            "schedule_centered_unique_circular_events_v1",
        "hybrid_event_tolerance_s": HYBRID_EVENT_TOLERANCE_S,
        "hybrid_contact_threshold_n": 5.,
        "stencil_fork_definition": "single_reference_exposed_state_v1",
        "pre_restore_group_rms_is_diagnostic": True,
        "immediate_restore_readback_required": True,
        "active_v5_numerical_gates": list(V5_ACTIVE_NUMERICAL_GATES),
        "retired_v4_numerical_gate": V5_RETIRED_V4_GATE,
        "v5_claim_protocol_frozen": False,
        "external_pushes": False,
        "paper_metric_primary":
            "chi_orb=sigma_max(Phi_orbital_augmented_46D)",
        "primary_metric_scope":
            "translation-reduced learned-controller analogue",
        "unquotiented_full_48d_is_diagnostic": True,
        "translation_symmetry_residual_limit": .02,
        "gate_limits": FROZEN_GATE_LIMITS,
        "evaluation_reference_seed": CONFIRMATORY_REFERENCE_SEED,
        "state_scales": STATE_SCALES.tolist(),
        "references": 1,
        "zero_clones_per_reference": 2,
        "master_stencil_size": 291,
        "settle_cycles": 12,
        "reference_initial_offset_half_width_normalized": .02,
        "gait_regime": {
            "trot_df": [.50, .625, .75],
            "walk_df": [.75],
            "minimum_swing_s": .10,
        },
        "confirmatory_grid": False,
        "candidate_confirmatory_grid": False,
        "initial_stencil_max_error_limit": 1e-3,
        "initial_condition_number_limit": 1.05,
        "zero_clone_noise_fraction_of_h_limit": .05,
        "retired_v4_settle_group_rms_limit": 1e-3,
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
        "energy_schema": "positive_mechanical_cot_v1",
        "energy_metric_primary": "positive_mechanical_cot",
        "energy_metric_diagnostic": "absolute_mechanical_cot",
        "electrical_loss_included": False,
        "energy_formula":
            "sum(max(applied_torque*left_endpoint_joint_velocity,0))*dt/(mass*g*delta_x)",
        "energy_measurement_cycles": 4,
        "energy_sample_dt": .005,
        "energy_samples_per_control": 4,
        "energy_torque_source": "robot.data.applied_torque_after_clipping",
        "energy_velocity_timestamp": "left_endpoint",
        "energy_integration_rule": "left_rectangle",
        "robot_mass_kg": 10.,
        "gravity_mps2": 9.81,
        "joint_effort_limits_nm": [23.5] * 12,
        "expected_energy_files": [energy_name],
        "expected_energy_conditions": [{
            "filename": energy_name, "speed": .3, "duty_factor": .5,
            "step_width": .3, "period": .48, "gait": "trot",
        }],
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
        "requested_initial_states": initial[None].copy(),
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
        "pre_restore_group_rms": np.asarray([0.]),
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


def energy_payload():
    references, cycles, steps, substeps = 1, 4, 24, 4
    desired = desired_trot_cycle()
    contacts = np.broadcast_to(
        desired, (references, cycles, steps, 4)).copy()
    substep_contacts = np.repeat(contacts[:, :, :, None, :], substeps, axis=3)
    feet = np.zeros((references, cycles, steps, 4, 3))
    feet[..., 1] = np.asarray([.15, -.15, .15, -.15])
    torque = np.ones((references, cycles, steps * substeps, 12))
    velocity = np.ones_like(torque)
    boundaries = np.zeros((references, cycles, 2))
    boundaries[..., 0] = np.arange(cycles) * .144
    boundaries[..., 1] = boundaries[..., 0] + .144
    values = mechanical_cot(torque, velocity, boundaries, 10., 9.81, .005)
    return {
        "schema": "positive_mechanical_cot_v1",
        "command": np.asarray([.3, .5, .3, .48, 0]),
        "applied_torque": torque, "joint_velocity": velocity,
        "x_boundaries": boundaries,
        "initial_contacts": np.broadcast_to(
            desired[-1], (references, cycles, 4)).copy(),
        "initial_desired": np.broadcast_to(
            desired[-1], (references, cycles, 4)).copy(),
        "phase_ticks": np.broadcast_to(
            np.arange(steps), (references, cycles, steps)).copy(),
        "contacts": contacts, "substep_contacts": substep_contacts,
        "desired": contacts.copy(), "feet_body": feet,
        "forward_velocity": np.full((references, cycles, steps), .3),
        "lateral_position": np.zeros((references, cycles, steps)),
        "heading": np.zeros((references, cycles, steps)),
        "world_lateral_velocity": np.zeros((references, cycles, steps)),
        "body_yaw_rate": np.zeros((references, cycles, steps)),
        "failure": np.zeros((references, cycles, steps), dtype=bool),
        "done": np.zeros((references, cycles, steps), dtype=bool),
        "cycle_rms": np.asarray([.01]),
        "settle_done": np.asarray([False]),
        **values,
        "sample_dt": .005, "measurement_cycles": 4,
        "robot_mass_kg": 10., "gravity_mps2": 9.81,
        "joint_effort_limits_nm": np.full(12, 23.5),
        "joint_order": np.asarray([f"joint_{index}" for index in range(12)]),
        "torque_source": "robot.data.applied_torque_after_clipping",
        "velocity_timestamp": "left_endpoint",
        "integration_rule": "left_rectangle",
        "electrical_loss_included": False,
        "task_sha256": "task", "evaluation_sha256": "evaluation",
        "checkpoint_sha256": "checkpoint",
    }


class StabilityMetricTest(unittest.TestCase):

    def test_canonical_physical_grid_has_exactly_eighty_cells(self):
        keys = canonical_physical_condition_keys()
        self.assertEqual(len(keys), 80)
        self.assertEqual(
            {key + (h,) for key in keys for h in (.025, .05, .10)},
            canonical_condition_keys())

    def test_positive_mechanical_cot_uses_left_rule_and_excludes_negative_power(self):
        torque = np.zeros((1, 1, 2, 12))
        velocity = np.zeros_like(torque)
        torque[0, 0, :, 0] = [2., 3.]
        velocity[0, 0, :, 0] = [4., -5.]
        result = mechanical_cot(
            torque, velocity, np.asarray([[[1., 1.5]]]),
            mass=2., gravity=10., sample_dt=.1)
        self.assertAlmostEqual(result["positive_work_j"].item(), .8)
        self.assertAlmostEqual(result["absolute_work_j"].item(), 2.3)
        self.assertAlmostEqual(result["forward_displacement_m"].item(), .5)
        self.assertAlmostEqual(result["positive_mechanical_cot"].item(), .08)
        self.assertAlmostEqual(result["absolute_mechanical_cot"].item(), .23)

    def test_mechanical_cot_rejects_bad_shapes_and_nonpositive_progress(self):
        torque = np.ones((2, 4, 8, 12))
        velocity = np.ones_like(torque)
        boundaries = np.zeros((2, 4, 2))
        boundaries[..., 1] = 1.
        with self.assertRaisesRegex(ValueError, "Torque"):
            mechanical_cot(torque[..., :11], velocity[..., :11], boundaries, 1., 9.81)
        with self.assertRaisesRegex(ValueError, "match"):
            mechanical_cot(torque, velocity[..., :11], boundaries, 1., 9.81)
        backwards = boundaries.copy()
        backwards[0, 0] = [1., 0.]
        with self.assertRaisesRegex(ValueError, "positive forward"):
            mechanical_cot(torque, velocity, backwards, 1., 9.81)
    def test_confirmatory_protocol_requires_all_frozen_resources(self):
        protocol = {
            "condition_keys": canonical_condition_keys(),
            "references": 8, "settle_cycles": 12,
            "perturbation_sizes": [.025, .05, .10],
            "zero_clones": 2, "master_stencil_size": 291,
            "evaluation_reference_seed": CONFIRMATORY_REFERENCE_SEED,
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
            ("evaluation_reference_seed", CONFIRMATORY_REFERENCE_SEED + 1),
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
        settled[0, 1:, 2] = np.linspace(
            -.4, .4, layout["size"] - 1)
        settled[0, 1:, 37] = np.linspace(
            -.8, .8, layout["size"] - 1)
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
        np.testing.assert_allclose(master[0, 0], master[0, 2])

    def test_master_stencil_keeps_reference_groups_separate(self):
        layout = master_stencil_layout((.025, .05, .10), 2)
        settled = np.zeros((2, layout["size"], RAW_STATE_DIM))
        settled[..., 3] = 1.
        settled[0, 0, 2] = .31
        settled[1, 0, 2] = .43
        settled[0, 1:, 2] = 8.
        settled[1, 1:, 2] = 9.
        master, result_layout = construct_master_stencil(
            settled, (.025, .05, .10), 2)
        np.testing.assert_allclose(master[0, result_layout["zero_clones"], 2], .31)
        np.testing.assert_allclose(master[1, result_layout["zero_clones"], 2], .43)
        self.assertAlmostEqual(master[0, 0, 2], .31)
        self.assertAlmostEqual(master[1, 0, 2], .43)

    def test_v5_pre_restore_spread_is_diagnostic_after_exact_fork(self):
        payload = nominal_payload(np.eye(AUGMENTED_DIM), .05)
        payload["settle_group_rms"][:] = 9.
        payload["pre_restore_group_rms"][:] = 9.
        metrics = numerical_fidelity(
            payload, 0, .05, {"schema": STABILITY_SCHEMA})
        self.assertEqual(metrics["numerical_gate_pass"], 1)
        self.assertEqual(metrics["pre_restore_group_rms"], 9.)

    def test_v5_readback_mismatch_fails_numerical_gate(self):
        payload = nominal_payload(np.eye(AUGMENTED_DIM), .05)
        payload["requested_initial_states"][0, 0, 2] += .001
        metrics = numerical_fidelity(
            payload, 0, .05, {"schema": STABILITY_SCHEMA})
        self.assertEqual(metrics["numerical_gate_pass"], 0)
        self.assertGreater(metrics["restore_readback_max_error"], .001)

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

    def test_stability_topology_accepts_circular_boundary_jitter(self):
        payload = nominal_payload(np.eye(AUGMENTED_DIM), .05)
        shifted = payload["desired_substep_contacts"][0].copy()
        shifted[:, 0] = np.roll(shifted[:, 0], -4)
        payload["stencil_substep_contacts"][:] = shifted
        payload["zero_substep_contacts"][:] = shifted
        payload["stencil_initial_contacts"][:] = shifted[-1]
        payload["zero_initial_contacts"][:] = shifted[-1]
        self.assertTrue(hybrid_topology_gate(payload, 0))
        self.assertFalse(hybrid_topology_gate(payload, 0, legacy=True))
        shifted[:, 0] = np.roll(
            payload["desired_substep_contacts"][0, :, 0], -5)
        payload["stencil_substep_contacts"][:] = shifted
        payload["zero_substep_contacts"][:] = shifted
        payload["stencil_initial_contacts"][:] = shifted[-1]
        payload["zero_initial_contacts"][:] = shifted[-1]
        self.assertFalse(hybrid_topology_gate(payload, 0))

    def test_stability_topology_requires_one_initial_hybrid_mode(self):
        payload = nominal_payload(np.eye(AUGMENTED_DIM), .05)
        shifted = payload["desired_substep_contacts"][0].copy()
        shifted[:, 0] = np.roll(shifted[:, 0], -4)
        # The shifted clone is individually schedule-compliant, but it begins
        # across the contact boundary from the baseline clone.
        payload["stencil_substep_contacts"][0, 1] = shifted
        payload["stencil_initial_contacts"][0, 1] = shifted[-1]
        self.assertFalse(hybrid_topology_gate(payload, 0))

    def test_energy_topology_uses_same_circular_matcher(self):
        payload = energy_payload()
        shifted = payload["substep_contacts"][0, 0].reshape(-1, 4)
        shifted[:, 0] = np.roll(shifted[:, 0], -4)
        payload["substep_contacts"][0, 0] = shifted.reshape(24, 4, 4)
        payload["initial_contacts"][0, 0] = shifted[-1]
        self.assertTrue(_energy_topology_gate(payload, 0))
        payload["substep_contacts"][0, 0, 10, 0, 0] ^= True
        self.assertFalse(_energy_topology_gate(payload, 0))

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

    def _write_grid(self, directory, matrix=None, mutate=None,
                    perturbation_sizes=(.025, .05, .10)):
        matrix = np.eye(AUGMENTED_DIM) if matrix is None else matrix
        names, conditions = [], []
        for h in perturbation_sizes:
            name = f"stability_cell_h{h:.3f}.npz"
            names.append(name)
            conditions.append(condition(name, h))
            payload = nominal_payload(matrix, h)
            if mutate:
                mutate(payload, h)
            np.savez_compressed(directory / name, **payload)
        (directory / "training_provenance.json").write_bytes(
            TRAINING_PROVENANCE_BYTES)
        np.savez_compressed(
            directory / "energy_s0.300_d0.500_v0.300_trot_p0.48.npz",
            **energy_payload())
        manifest = manifest_for(names, conditions)
        manifest["master_stencil_size"] = (
            1 + 2 + len(perturbation_sizes) * 2 * AUGMENTED_DIM)
        (directory / "stability_manifest.json").write_text(json.dumps(manifest))
        return names

    def test_five_h_pilot_reports_all_adjacent_pairs_without_false_gates(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_grid(
                directory,
                perturbation_sizes=(.005, .01, .025, .05, .10))
            analyze_stability(directory)
            pairs = pd.read_csv(directory / "finite_difference_pairs.csv")
            self.assertEqual(
                list(zip(pairs.h_low, pairs.h_high)),
                [(.005, .01), (.01, .025), (.025, .05), (.05, .10)])
            references = pd.read_csv(directory / "stability_references.csv")
            evaluated = dict(zip(
                references.perturbation_h,
                references.finite_difference_gate_evaluated))
            self.assertEqual(evaluated[.005], 0)
            self.assertEqual(evaluated[.01], 0)
            self.assertEqual(evaluated[.025], 1)
            self.assertEqual(evaluated[.05], 1)
            self.assertEqual(evaluated[.10], 1)

    def test_requested_pair_shared_offset_cannot_cancel_in_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)

            def shared_offset(payload, _h):
                payload["requested_initial_states"][0, 1, 2] += .001
                payload["requested_initial_states"][0, 2, 2] += .001

            self._write_grid(directory, mutate=shared_offset)
            with self.assertRaisesRegex(ValueError, "requested stencil"):
                analyze_stability(directory)

    def test_pre_restore_spread_alias_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)

            def mismatch(payload, _h):
                payload["pre_restore_group_rms"][:] = .5

            self._write_grid(directory, mutate=mismatch)
            with self.assertRaisesRegex(ValueError, "spread fields"):
                analyze_stability(directory)

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

    def test_v4_archives_are_ineligible_for_paper_claim_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_grid(directory)
            manifest_path = directory / "stability_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schema"] = V4_STABILITY_SCHEMA
            manifest.pop("stencil_fork_definition")
            manifest.pop("pre_restore_group_rms_is_diagnostic")
            manifest.pop("immediate_restore_readback_required")
            manifest.pop("active_v5_numerical_gates")
            manifest.pop("retired_v4_numerical_gate")
            manifest.pop("v5_claim_protocol_frozen")
            manifest.pop("candidate_confirmatory_grid")
            manifest["settle_group_rms_limit"] = manifest.pop(
                "retired_v4_settle_group_rms_limit")
            manifest_path.write_text(json.dumps(manifest))
            analyze_stability(directory)
            claims = json.loads((directory / "paper_claims.json").read_text())
            self.assertFalse(claims["schema_eligible_for_claims"])
            self.assertFalse(claims["claim_values_released"])

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

    def test_energy_rejects_nonperiodic_reference_even_when_other_gates_pass(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            def nonperiodic(payload, _h):
                payload["cycle_rms"] = np.asarray([.021])

            self._write_grid(directory, mutate=nonperiodic)
            path = directory / "energy_s0.300_d0.500_v0.300_trot_p0.48.npz"
            payload = energy_payload()
            payload["cycle_rms"] = np.asarray([.021])
            np.savez_compressed(path, **payload)
            analyze_stability(directory)
            table = pd.read_csv(directory / "paper_energy_conditions.csv")
            self.assertEqual(table.energy_condition_valid.tolist(), [0])
            references = pd.read_csv(directory / "energy_references.csv")
            self.assertEqual(references.energy_periodic_orbit_gate_pass.tolist(), [0])
            self.assertEqual(references.command_gate_pass.tolist(), [1])
            self.assertEqual(references.energy_topology_gate_pass.tolist(), [1])

    def test_energy_periodicity_must_match_paired_stability_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)

            def nonperiodic(payload, _h):
                payload["cycle_rms"] = np.asarray([.021])

            self._write_grid(directory, mutate=nonperiodic)
            energy_path = (
                directory / "energy_s0.300_d0.500_v0.300_trot_p0.48.npz")
            payload = energy_payload()
            payload["cycle_rms"] = np.asarray([.001])
            np.savez_compressed(energy_path, **payload)
            with self.assertRaisesRegex(ValueError, "paired stability"):
                analyze_stability(directory)

    def test_all_h_archives_must_share_settling_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            names = self._write_grid(directory)
            payload = nominal_payload(np.eye(AUGMENTED_DIM), .05)
            payload["cycle_rms"] = np.asarray([.011])
            np.savez_compressed(directory / names[1], **payload)
            with self.assertRaisesRegex(ValueError, "perturbation sizes"):
                analyze_stability(directory)

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
            energy_rows = []
            for row in rows:
                ratio = (
                    .8 + (row["speed"] - .25) * (.1 / .15)
                    if row["command_df"] == .75 else
                    (.9 if row["command_df"] == .625 else 1.))
                energy_rows.append({
                    **{key: row[key] for key in (
                        "speed", "command_df", "step_width", "period", "gait")},
                    "energy_condition_valid": 1,
                    "positive_mechanical_cot_median": (1 + row["speed"]) * ratio,
                })
            for policy in range(5):
                directory = root / f"policy_{policy}"
                directory.mkdir()
                directories.append(directory)
                manifest = {
                    "schema": STABILITY_SCHEMA,
                    "hybrid_topology_definition":
                        "schedule_centered_unique_circular_events_v1",
                    "hybrid_event_tolerance_s": HYBRID_EVENT_TOLERANCE_S,
                    "hybrid_contact_threshold_n": 5.,
                    "task_sha256": "task",
                    "evaluation_sha256": "evaluation",
                    "state_scales": STATE_SCALES.tolist(),
                    "expected_conditions": expected,
                    "expected_energy_conditions": [{"row": index} for index in range(80)],
                    "expected_energy_files": [f"energy_{index}" for index in range(80)],
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
                    "evaluation_reference_seed": CONFIRMATORY_REFERENCE_SEED,
                    "training_source_sha256": "training-source",
                    "training_num_envs": 4096,
                    "training_iterations_requested": 1800,
                    "checkpoint_selection_rule":
                        "final_requested_iteration",
                    "checkpoint_iteration": 1799,
                    "gait_regime": {
                        "trot_df": [.50, .625, .75],
                        "walk_df": [.75],
                        "minimum_swing_s": .10,
                    },
                    "energy_schema": "positive_mechanical_cot_v1",
                    "energy_metric_primary": "positive_mechanical_cot",
                    "energy_metric_diagnostic": "absolute_mechanical_cot",
                    "electrical_loss_included": False,
                    "energy_formula": "formula",
                    "energy_measurement_cycles": 4,
                    "energy_sample_dt": .005,
                    "energy_samples_per_control": 4,
                    "energy_torque_source": "robot.data.applied_torque_after_clipping",
                    "energy_velocity_timestamp": "left_endpoint",
                    "energy_integration_rule": "left_rectangle",
                    "robot_mass_kg": 10., "gravity_mps2": 9.81,
                    "joint_effort_limits_nm": [23.5] * 12,
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
                pd.DataFrame(energy_rows).to_csv(
                    directory / "paper_energy_conditions.csv", index=False)
                (directory / "paper_claims.json").write_text(json.dumps({
                    "claim_values_released": True}))
                (directory / "paper_energy_claims.json").write_text(json.dumps({
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
            self.assertTrue((output / "energy_policy_effects.csv").is_file())
            self.assertTrue(report["energy"]["qualitative_alignment_pass"])
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
