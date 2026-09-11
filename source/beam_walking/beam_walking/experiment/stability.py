"""Paper-aligned empirical closed-loop return-map analysis.

The paper's convergence quantity is the largest singular value of a hybrid
closed-loop fundamental solution matrix.  For a neural policy and simulator we
estimate the analogous phase-to-phase map with central finite differences.  This
module is simulator independent so the numerical definition can be tested before
any GPU run.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PHYSICAL_DIM = 36
ACTION_DIM = 12
AUGMENTED_DIM = PHYSICAL_DIM + ACTION_DIM
RAW_STATE_DIM = 49  # xyz, quaternion, q, base v, base omega, qdot, last action

TANGENT_LABELS = (
    tuple(f"base_position_{axis}" for axis in "xyz")
    + tuple(f"base_orientation_{axis}" for axis in "xyz")
    + tuple(f"joint_position_{index}" for index in range(12))
    + tuple(f"base_linear_velocity_{axis}" for axis in "xyz")
    + tuple(f"base_angular_velocity_{axis}" for axis in "xyz")
    + tuple(f"joint_velocity_{index}" for index in range(12))
    + tuple(f"last_action_{index}" for index in range(12))
)

# A fixed nondimensionalization is necessary because singular values depend on
# coordinate units.  These scales are frozen before test evaluation.
STATE_SCALES = np.asarray(
    [0.10] * 3       # base position, m
    + [0.10] * 3     # SO(3) tangent, rad
    + [0.50] * 12    # joint position, rad
    + [0.50] * 3     # base linear velocity, m/s
    + [1.00] * 3     # base angular velocity, rad/s
    + [5.00] * 12    # joint velocity, rad/s
    + [1.00] * 12,   # last normalized policy action
    dtype=np.float64,
)

TRAINING_SOURCE_PATHS = (
    "source/beam_walking/beam_walking/experiment/task.py",
    "source/beam_walking/beam_walking/experiment/protocol.py",
    "scripts/beam_experiment.py",
    "scripts/gpu_capacity.py",
    "scripts/watch_training.py",
)
EVALUATION_SOURCE_PATHS = (
    "source/beam_walking/beam_walking/experiment/stability.py",
    "scripts/stability_experiment.py",
    "scripts/analyze_stability.py",
    "scripts/analyze_stability_ensemble.py",
    "scripts/gpu_capacity.py",
)


CONFIRMATORY_REFERENCE_SEED = 10000
CONFIRMATORY_TRAINING_ITERATIONS = 1800
FROZEN_GATE_LIMITS = {
    "periodic_orbit_rms": .02,
    "duty_factor_max_abs_error": .05,
    "stance_width_abs_error_m": .03,
    "forward_speed_abs_error_mps": .04,
    "stance_recall_min": .90,
    "swing_recall_min": .90,
    "lateral_rmse_m": .10,
    "max_lateral_deviation_m": .20,
    "heading_rmse_rad": .10,
    "max_abs_heading_rad": .25,
    "world_lateral_velocity_rmse_mps": .10,
    "body_yaw_rate_rmse_radps": .50,
    "initial_stencil_max_error": 1e-3,
    "initial_condition_number": 1.05,
    "zero_clone_noise_fraction_of_h": .05,
    "settle_group_rms": 1e-3,
    "translation_symmetry_residual": .02,
    "finite_difference_relative_error": .10,
    "reference_valid_rate": .90,
    "global_condition_valid_rate": .90,
    "equivalence_ratio_margin": .20,
}


def canonical_condition_keys():
    """Return the exact 240 pre-registered trace cells."""
    return {
        (.48, gait, speed, width, duty, h)
        for gait, duties in (("trot", (.50, .625, .75)),
                             ("walk", (.75,)))
        for speed in (.25, .30, .35, .40)
        for width in (.10, .20, .30, .40, .50)
        for duty in duties
        for h in (.025, .05, .10)
    }


def confirmatory_protocol(*, condition_keys, references, settle_cycles,
                          perturbation_sizes, zero_clones,
                          master_stencil_size, evaluation_reference_seed,
                          training_iterations_requested,
                          checkpoint_iteration, fresh_training) -> bool:
    """Return true only for the completely frozen confirmatory protocol."""
    keys = [tuple(item) for item in condition_keys]
    return (
        len(keys) == len(set(keys)) == 240
        and set(keys) == canonical_condition_keys()
        and references == 8
        and settle_cycles == 12
        and set(perturbation_sizes) == {.025, .05, .10}
        and zero_clones == 2
        and master_stencil_size == 291
        and evaluation_reference_seed == CONFIRMATORY_REFERENCE_SEED
        and training_iterations_requested == CONFIRMATORY_TRAINING_ITERATIONS
        and checkpoint_iteration == CONFIRMATORY_TRAINING_ITERATIONS - 1
        and fresh_training is True
    )


def training_source_hash(root: str | Path) -> str:
    """Hash all source that fixes PPO training behavior and launch semantics."""
    root = Path(root)
    return hashlib.sha256(b"".join(
        (root / relative).read_bytes()
        for relative in TRAINING_SOURCE_PATHS)).hexdigest()


def evaluation_source_hash(root: str | Path) -> str:
    root = Path(root)
    return hashlib.sha256(b"".join(
        (root / relative).read_bytes()
        for relative in EVALUATION_SOURCE_PATHS)).hexdigest()


ORBITAL_PHYSICAL_INDICES = np.asarray(
    [index for index in range(PHYSICAL_DIM) if index not in (0, 1)], dtype=np.int64
)
ORBITAL_AUGMENTED_INDICES = np.asarray(
    [index for index in range(AUGMENTED_DIM) if index not in (0, 1)], dtype=np.int64
)


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0):
        raise ValueError("Quaternion must be finite and nonzero")
    return q / norm


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply wxyz quaternions with NumPy broadcasting."""
    left, right = _normalize_quaternion(left), _normalize_quaternion(right)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return _normalize_quaternion(np.stack([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ], axis=-1))


def rotation_vector_to_quaternion(vector: np.ndarray) -> np.ndarray:
    """Convert an SO(3) rotation vector to a wxyz quaternion."""
    vector = np.asarray(vector, dtype=np.float64)
    angle = np.linalg.norm(vector, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.empty_like(angle)
    nonzero = angle > 1e-12
    np.divide(np.sin(half), angle, out=scale, where=nonzero)
    scale[~nonzero] = (0.5 - angle * angle / 48.0)[~nonzero]
    return _normalize_quaternion(np.concatenate([np.cos(half), vector * scale], axis=-1))


def quaternion_to_rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to the shortest SO(3) rotation vector."""
    quaternion = _normalize_quaternion(quaternion)
    quaternion = np.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    magnitude = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(magnitude, np.clip(quaternion[..., :1], 0.0, 1.0))
    scale = np.empty_like(magnitude)
    nonzero = magnitude > 1e-12
    np.divide(angle, magnitude, out=scale, where=nonzero)
    scale[~nonzero] = (2.0 + magnitude * magnitude / 3.0)[~nonzero]
    return vector * scale


def state_delta(reference: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Return the 48D tangent displacement from raw reference to raw sample."""
    reference, sample = np.asarray(reference, dtype=np.float64), np.asarray(sample, dtype=np.float64)
    if reference.shape[-1] != RAW_STATE_DIM or sample.shape[-1] != RAW_STATE_DIM:
        raise ValueError(f"Raw states must end in {RAW_STATE_DIM} values")
    reference, sample = np.broadcast_arrays(reference, sample)
    conjugate = reference[..., 3:7].copy()
    conjugate[..., 1:] *= -1
    orientation = quaternion_to_rotation_vector(
        quaternion_multiply(sample[..., 3:7], conjugate)
    )
    return np.concatenate([
        sample[..., 0:3] - reference[..., 0:3], orientation,
        sample[..., 7:19] - reference[..., 7:19],
        sample[..., 19:37] - reference[..., 19:37],
        sample[..., 37:49] - reference[..., 37:49],
    ], axis=-1)


def apply_scaled_perturbation(raw_state: np.ndarray, coordinate: int, amount: float) -> np.ndarray:
    """Apply one nondimensional tangent perturbation to a raw state."""
    if not 0 <= coordinate < AUGMENTED_DIM or not np.isfinite(amount):
        raise ValueError("Invalid perturbation coordinate or amount")
    result = np.asarray(raw_state, dtype=np.float64).copy()
    if result.shape != (RAW_STATE_DIM,):
        raise ValueError(f"Expected one raw state with shape ({RAW_STATE_DIM},)")
    physical = amount * STATE_SCALES[coordinate]
    if coordinate < 3:
        result[coordinate] += physical
    elif coordinate < 6:
        axis = np.zeros(3)
        axis[coordinate - 3] = physical
        result[3:7] = quaternion_multiply(rotation_vector_to_quaternion(axis), result[3:7])
    elif coordinate < 18:
        result[7 + coordinate - 6] += physical
    elif coordinate < 36:
        result[19 + coordinate - 18] += physical
    else:
        result[37 + coordinate - 36] += physical
    return result


def canonical_reference_states(default_joint_positions, references,
                               seed=CONFIRMATORY_REFERENCE_SEED,
                               offset_half_width=.02) -> np.ndarray:
    """Build identical per-condition reference states from the default Go2 pose."""
    joints = np.asarray(default_joint_positions, dtype=np.float64)
    if joints.shape != (12,) or not np.isfinite(joints).all():
        raise ValueError("Canonical reference requires 12 finite default joints")
    if references < 1 or offset_half_width < 0:
        raise ValueError("Invalid canonical reference count or offset width")
    states = np.zeros((references, RAW_STATE_DIM), dtype=np.float64)
    states[:, 0] = -.65
    states[:, 2] = .36
    states[:, 3] = 1.
    states[:, 7:19] = joints
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(
        -offset_half_width, offset_half_width, size=(references, 34))
    for reference in range(references):
        for coordinate, amount in zip(range(2, 36), offsets[reference]):
            states[reference] = apply_scaled_perturbation(
                states[reference], coordinate, float(amount))
    return states


def master_stencil_layout(perturbation_sizes, zero_clones=2):
    """Return local indices for simultaneous h stencils and zero-offset controls."""
    sizes = tuple(float(value) for value in perturbation_sizes)
    if zero_clones < 1 or not sizes or any(value <= 0 for value in sizes):
        raise ValueError("Master stencil needs positive h values and zero clones")
    block = 2 * AUGMENTED_DIM
    mapping = {}
    for h_index, h in enumerate(sizes):
        start = 1 + zero_clones + h_index * block
        mapping[h] = np.concatenate((
            np.asarray([0], dtype=np.int64),
            np.arange(start, start + block, dtype=np.int64),
        ))
    return {
        "size": 1 + zero_clones + len(sizes) * block,
        "baseline": 0,
        "zero_clones": np.arange(1, 1 + zero_clones, dtype=np.int64),
        "h_indices": mapping,
    }


def construct_master_stencil(settled_states: np.ndarray, perturbation_sizes,
                             zero_clones=2, canonical_xy=(-.65, 0.)):
    """Build simultaneous central stencils from per-member settled states.

    Each member keeps its own settled simulator history. This function changes
    only exposed state coordinates, using no-op canonical writes for baseline
    and zero clones and one tangent perturbation for each +/- member.
    """
    settled_states = np.asarray(settled_states, dtype=np.float64)
    layout = master_stencil_layout(perturbation_sizes, zero_clones)
    if settled_states.ndim != 3 or settled_states.shape[1:] != (
        layout["size"], RAW_STATE_DIM
    ):
        raise ValueError("Settled master states have the wrong shape")
    result = settled_states.copy()
    result[..., 0] = canonical_xy[0]
    result[..., 1] = canonical_xy[1]
    for h, indices in layout["h_indices"].items():
        for coordinate in range(AUGMENTED_DIM):
            result[:, indices[1 + 2 * coordinate]] = np.stack([
                apply_scaled_perturbation(state, coordinate, h)
                for state in result[:, indices[1 + 2 * coordinate]]
            ])
            result[:, indices[2 + 2 * coordinate]] = np.stack([
                apply_scaled_perturbation(state, coordinate, -h)
                for state in result[:, indices[2 + 2 * coordinate]]
            ])
    return result, layout


def central_columns(initial_states: np.ndarray, final_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build normalized central-difference columns from baseline,+,- ordering.

    Index zero is the baseline.  For coordinate ``j``, indices ``1+2*j`` and
    ``2+2*j`` are its positive and negative perturbations.
    """
    initial_states = np.asarray(initial_states, dtype=np.float64)
    final_states = np.asarray(final_states, dtype=np.float64)
    expected = 1 + 2 * AUGMENTED_DIM
    if initial_states.shape != (expected, RAW_STATE_DIM) or final_states.shape != initial_states.shape:
        raise ValueError(f"Expected initial/final state arrays with shape ({expected}, {RAW_STATE_DIM})")
    x0, xf = [], []
    for coordinate in range(AUGMENTED_DIM):
        plus, minus = 1 + 2 * coordinate, 2 + 2 * coordinate
        x0.append(0.5 * (state_delta(initial_states[0], initial_states[plus])
                         - state_delta(initial_states[0], initial_states[minus])) / STATE_SCALES)
        xf.append(0.5 * (state_delta(final_states[0], final_states[plus])
                         - state_delta(final_states[0], final_states[minus])) / STATE_SCALES)
    return np.stack(x0, axis=1), np.stack(xf, axis=1)


def fit_return_map(initial_columns: np.ndarray, final_columns: np.ndarray) -> dict:
    """Fit ``xf = Phi x0`` and return numerical diagnostics."""
    initial_columns = np.asarray(initial_columns, dtype=np.float64)
    final_columns = np.asarray(final_columns, dtype=np.float64)
    if initial_columns.ndim != 2 or final_columns.shape[1] != initial_columns.shape[1]:
        raise ValueError("Return-map columns must be two-dimensional with matched samples")
    if not np.isfinite(initial_columns).all() or not np.isfinite(final_columns).all():
        raise ValueError("Return-map columns must be finite")
    phi = final_columns @ np.linalg.pinv(initial_columns, rcond=1e-10)
    _, singular_values, right_vectors = np.linalg.svd(phi, full_matrices=False)
    residual = final_columns - phi @ initial_columns
    denominator = max(float(np.linalg.norm(final_columns)), np.finfo(float).eps)
    return {
        "phi": phi,
        "singular_values": singular_values,
        "worst_case_right_vector": right_vectors[0],
        "chi": float(singular_values[0]),
        "rank": int(np.linalg.matrix_rank(initial_columns)),
        "condition_number": float(np.linalg.cond(initial_columns)),
        "relative_residual": float(np.linalg.norm(residual) / denominator),
    }


def estimate_maps(initial_states: np.ndarray, final_states: np.ndarray) -> dict:
    """Estimate full/physical/orbital maps from one 97-rollout stencil."""
    x0, xf = central_columns(initial_states, final_states)
    augmented = fit_return_map(x0, xf)
    physical = fit_return_map(x0[:PHYSICAL_DIM, :PHYSICAL_DIM], xf[:PHYSICAL_DIM, :PHYSICAL_DIM])
    op = ORBITAL_PHYSICAL_INDICES
    oa = ORBITAL_AUGMENTED_INDICES
    orbital_physical = fit_return_map(x0[np.ix_(op, op)], xf[np.ix_(op, op)])
    orbital_augmented = fit_return_map(x0[np.ix_(oa, oa)], xf[np.ix_(oa, oa)])
    return {
        "augmented": augmented,
        "physical_conditional": physical,
        "orbital_physical_conditional": orbital_physical,
        "orbital_augmented": orbital_augmented,
    }


def relative_matrix_difference(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(left)), np.finfo(float).eps)
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)) / denominator)


def translation_symmetry_fidelity(phi: np.ndarray, limit: float = .02) -> dict:
    """Verify that global x/y are neutral, uncoupled translation modes."""
    phi = np.asarray(phi, dtype=np.float64)
    if phi.shape != (AUGMENTED_DIM, AUGMENTED_DIM):
        raise ValueError("Translation-symmetry check requires the full 48D map")
    residuals, leakages = [], []
    for coordinate in (0, 1):
        target = np.zeros(AUGMENTED_DIM)
        target[coordinate] = 1.
        column = phi[:, coordinate]
        residuals.append(float(np.linalg.norm(column - target)))
        leakages.append(float(np.linalg.norm(np.delete(column, coordinate))))
    maximum_residual = max(residuals)
    maximum_leakage = max(leakages)
    return {
        "translation_x_identity_residual": residuals[0],
        "translation_y_identity_residual": residuals[1],
        "translation_identity_residual_max": maximum_residual,
        "translation_cross_coupling_max": maximum_leakage,
        "translation_symmetry_gate_pass": int(
            maximum_residual <= limit and maximum_leakage <= limit),
    }


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan),
        where=denominator > 0)


def command_fidelity(data: dict, reference: int, speed: float,
                     duty: float, width: float) -> dict:
    """Compute pre-registered command gates from one nominal gait cycle."""
    contacts = np.asarray(data["nominal_contacts"][reference], dtype=bool)
    desired = np.asarray(data["nominal_desired"][reference], dtype=bool)
    feet = np.asarray(data["nominal_feet_body"][reference], dtype=np.float64)
    velocity = np.asarray(
        data["nominal_forward_velocity"][reference], dtype=np.float64)
    failure = np.asarray(data["nominal_failure"][reference], dtype=bool)
    if contacts.ndim != 2 or contacts.shape[1] != 4 or desired.shape != contacts.shape:
        raise ValueError(
            "Nominal contact traces must have shape [references, time, 4]")
    if feet.shape != contacts.shape + (3,) or velocity.shape != contacts.shape[:1]:
        raise ValueError("Malformed nominal foot or velocity traces")

    achieved_df = contacts.mean(axis=0)
    stance_recall = _ratio(
        (contacts & desired).sum(axis=0), desired.sum(axis=0))
    swing_recall = _ratio(
        (~contacts & ~desired).sum(axis=0), (~desired).sum(axis=0))
    signs = np.asarray([1.0, -1.0, 1.0, -1.0])
    stance_y, stance_errors, stance_counts = [], [], []
    for leg in range(4):
        selected = contacts[:, leg]
        stance_counts.append(int(selected.sum()))
        if not selected.any():
            stance_y.append(np.nan)
            stance_errors.append(np.nan)
        else:
            values = feet[selected, leg, 1]
            stance_y.append(float(values.mean()))
            stance_errors.append(float(
                np.mean(np.abs(values - .5 * width * signs[leg]))))
    stance_y = np.asarray(stance_y)
    achieved_width = float(
        np.mean(stance_y[[0, 2]]) - np.mean(stance_y[[1, 3]]))
    foot_lateral_mae = float(np.mean(stance_errors))
    all_phase_mae = float(np.mean(
        np.abs(feet[..., 1] - .5 * width * signs)))
    forward_speed = float(np.mean(velocity))
    lateral_position = np.asarray(
        data["nominal_lateral_position"][reference], dtype=np.float64)
    heading = np.asarray(
        data["nominal_heading"][reference], dtype=np.float64)
    lateral_velocity = np.asarray(
        data["nominal_world_lateral_velocity"][reference], dtype=np.float64)
    yaw_rate = np.asarray(
        data["nominal_body_yaw_rate"][reference], dtype=np.float64)
    lateral_rmse = float(np.sqrt(np.mean(lateral_position ** 2)))
    max_lateral_deviation = float(np.max(np.abs(lateral_position)))
    heading_rmse = float(np.sqrt(np.mean(heading ** 2)))
    max_abs_heading = float(np.max(np.abs(heading)))
    lateral_velocity_rmse = float(np.sqrt(np.mean(lateral_velocity ** 2)))
    yaw_rate_rmse = float(np.sqrt(np.mean(yaw_rate ** 2)))
    df_max_abs_error = float(np.max(np.abs(achieved_df - duty)))
    width_abs_error = abs(achieved_width - width)
    speed_abs_error = abs(forward_speed - speed)
    gate = (
        not failure.any()
        and all(count > 0 for count in stance_counts)
        and np.isfinite(achieved_width)
        and df_max_abs_error <= FROZEN_GATE_LIMITS["duty_factor_max_abs_error"]
        and width_abs_error <= FROZEN_GATE_LIMITS["stance_width_abs_error_m"]
        and speed_abs_error <= FROZEN_GATE_LIMITS["forward_speed_abs_error_mps"]
        and lateral_rmse <= FROZEN_GATE_LIMITS["lateral_rmse_m"]
        and max_lateral_deviation <= FROZEN_GATE_LIMITS["max_lateral_deviation_m"]
        and heading_rmse <= FROZEN_GATE_LIMITS["heading_rmse_rad"]
        and max_abs_heading <= FROZEN_GATE_LIMITS["max_abs_heading_rad"]
        and lateral_velocity_rmse <= FROZEN_GATE_LIMITS["world_lateral_velocity_rmse_mps"]
        and yaw_rate_rmse <= FROZEN_GATE_LIMITS["body_yaw_rate_rmse_radps"]
        and bool(np.all(stance_recall >= FROZEN_GATE_LIMITS["stance_recall_min"]))
        and bool(np.all(swing_recall >= FROZEN_GATE_LIMITS["swing_recall_min"]))
    )
    result = {
        "achieved_df": float(np.mean(achieved_df)),
        "df_max_abs_error": df_max_abs_error,
        "achieved_stance_width": achieved_width,
        "stance_width_abs_error": width_abs_error,
        "stance_foot_lateral_mae": foot_lateral_mae,
        "all_phase_foot_lateral_mae": all_phase_mae,
        "forward_speed": forward_speed,
        "speed_abs_error": speed_abs_error,
        "lateral_rmse": lateral_rmse,
        "max_lateral_deviation": max_lateral_deviation,
        "heading_rmse_rad": heading_rmse,
        "max_abs_heading_rad": max_abs_heading,
        "world_lateral_velocity_rmse": lateral_velocity_rmse,
        "body_yaw_rate_rmse": yaw_rate_rmse,
        "stance_recall_min": float(np.nanmin(stance_recall)),
        "swing_recall_min": float(np.nanmin(swing_recall)),
        "nominal_failure": int(failure.any()),
        "command_gate_pass": int(gate),
    }
    for index, leg in enumerate(("FL", "FR", "RL", "RR")):
        result[f"df_{leg}"] = float(achieved_df[index])
        result[f"stance_samples_{leg}"] = stance_counts[index]
        result[f"stance_y_{leg}"] = float(stance_y[index])
        result[f"stance_recall_{leg}"] = float(stance_recall[index])
        result[f"swing_recall_{leg}"] = float(swing_recall[index])
    return result


def contact_event_signature(initial: np.ndarray, trace: np.ndarray):
    """Return the global ordered hybrid-event groups, ignoring event times."""
    initial = np.asarray(initial, dtype=bool)
    trace = np.asarray(trace, dtype=bool)
    if initial.shape != (4,) or trace.ndim != 2 or trace.shape[1] != 4:
        raise ValueError("Contact signature expects [4] and [time,4]")
    previous = initial.copy()
    signature = []
    for sample in trace:
        changed = np.flatnonzero(sample != previous)
        if len(changed):
            # Changes within one 200 Hz sample are one simultaneous event group.
            signature.append(tuple(
                (int(leg), int(sample[leg])) for leg in changed))
        previous = sample
    return tuple(signature)


def hybrid_topology_gate(data: dict, reference: int) -> bool:
    initial = np.asarray(data["stencil_initial_contacts"][reference], dtype=bool)
    trace = np.asarray(data["stencil_substep_contacts"][reference], dtype=bool)
    zero_initial = np.asarray(data["zero_initial_contacts"][reference], dtype=bool)
    zero_trace = np.asarray(data["zero_substep_contacts"][reference], dtype=bool)
    desired_initial = np.asarray(
        data["desired_initial_contacts"][reference], dtype=bool)
    desired_trace = np.asarray(
        data["desired_substep_contacts"][reference], dtype=bool)
    if (
        initial.shape != (97, 4)
        or trace.ndim != 3 or trace.shape[0] != 97
        or desired_initial.shape != (4,)
        or desired_trace.shape != trace.shape[1:]
    ):
        raise ValueError("Malformed 200 Hz stencil or desired contact traces")
    expected = contact_event_signature(desired_initial, desired_trace)
    baseline = contact_event_signature(initial[0], trace[0])
    signatures = [
        contact_event_signature(initial[index], trace[index])
        for index in range(97)
    ]
    signatures += [
        contact_event_signature(zero_initial[index], zero_trace[index])
        for index in range(len(zero_initial))
    ]
    return (
        np.array_equal(initial[0], desired_initial)
        and baseline == expected
        and all(signature == baseline for signature in signatures)
    )


def numerical_fidelity(data: dict, reference: int, h: float,
                       manifest: dict) -> dict:
    """Gate stencil writes, clone noise, group settling, rank, and conditioning."""
    initial = data["initial_states"][reference]
    final = data["final_states"][reference]
    x0, _ = central_columns(initial, final)
    target = h * np.eye(AUGMENTED_DIM)
    maximum_error = float(np.max(np.abs(x0 - target)))
    relative_error = float(
        np.linalg.norm(x0 - target) / np.linalg.norm(target))
    rank = int(np.linalg.matrix_rank(x0))
    condition_number = float(np.linalg.cond(x0))

    zero_initial = np.asarray(
        data["zero_initial_states"][reference], dtype=np.float64)
    zero_final = np.asarray(
        data["zero_final_states"][reference], dtype=np.float64)
    baseline_initial = np.repeat(
        initial[None, 0], len(zero_initial), axis=0)
    baseline_final = np.repeat(
        final[None, 0], len(zero_final), axis=0)
    zero_input = state_delta(baseline_initial, zero_initial) / STATE_SCALES
    zero_output = state_delta(baseline_final, zero_final) / STATE_SCALES
    zero_input_max = float(np.max(np.abs(zero_input)))
    zero_output_max = float(np.max(np.abs(zero_output)))
    noise_fraction = zero_output_max / h

    # These limits come from frozen source, never editable result metadata.
    max_error_limit = FROZEN_GATE_LIMITS["initial_stencil_max_error"]
    condition_limit = FROZEN_GATE_LIMITS["initial_condition_number"]
    noise_limit = FROZEN_GATE_LIMITS["zero_clone_noise_fraction_of_h"]
    settle_limit = FROZEN_GATE_LIMITS["settle_group_rms"]
    settle_group_rms = float(data["settle_group_rms"][reference])
    passed = (
        rank == AUGMENTED_DIM
        and condition_number <= condition_limit
        and maximum_error <= max_error_limit
        and zero_input_max <= max_error_limit
        and noise_fraction <= noise_limit
        and settle_group_rms <= settle_limit
        and not bool(data["settle_done"][reference])
    )
    return {
        "initial_rank": rank,
        "initial_condition_number": condition_number,
        "initial_stencil_max_error": maximum_error,
        "initial_stencil_relative_error": relative_error,
        "zero_clone_input_max_error": zero_input_max,
        "zero_clone_output_max_error": zero_output_max,
        "zero_clone_noise_fraction_of_h": noise_fraction,
        "settle_group_rms": settle_group_rms,
        "settle_terminated": int(data["settle_done"][reference]),
        "numerical_gate_pass": int(passed),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = .5 * (start + end - 1)
        start = end
    return ranks


def _spearman(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


CONDITION_FIELDS = ("speed", "command_df", "step_width", "period", "gait")
PRIMARY_METRIC = "chi_orbital_augmented_median"
EQUIVALENCE_RATIO_MARGIN = FROZEN_GATE_LIMITS["equivalence_ratio_margin"]


def _condition_key(row):
    return tuple(row[field] for field in CONDITION_FIELDS)


def _pair_effects(rows, factor, low, high, subset=lambda row: True):
    remaining = tuple(field for field in CONDITION_FIELDS if field != factor)
    pairs = {}
    for row in rows:
        if not subset(row) or row[factor] not in (low, high):
            continue
        key = tuple(row[field] for field in remaining)
        pairs.setdefault(key, {})[row[factor]] = row[PRIMARY_METRIC]
    complete = [
        (pair[high] - pair[low], pair[high] / pair[low])
        for pair in pairs.values()
        if low in pair and high in pair
        and np.isfinite(pair[low]) and np.isfinite(pair[high])
        and pair[low] > 0 and pair[high] > 0
    ]
    return pairs, complete


def paper_claim_summary(conditions: list[dict], confirmatory_grid=True) -> dict:
    """Create a single-policy descriptive report with strict selection guards."""
    valid = [row for row in conditions if row["condition_valid"]]
    planned = len(conditions)
    global_rate = len(valid) / planned if planned else 0.
    definitions = (
        ("duty_factor", "command_df", .50, .75,
         lambda row: row["gait"] == "trot"),
        ("narrow_width", "step_width", .30, .10, lambda row: True),
        ("speed", "speed", .25, .40, lambda row: True),
        ("gait", "gait", "trot", "walk",
         lambda row: abs(row["command_df"] - .75) < 1e-8),
    )
    coverages, effects = {}, {}
    for name, factor, low, high, subset in definitions:
        planned_pairs, _ = _pair_effects(
            conditions, factor, low, high, subset)
        _, valid_effects = _pair_effects(valid, factor, low, high, subset)
        expected = sum(
            low in pair and high in pair for pair in planned_pairs.values())
        coverages[name] = {
            "expected_pairs": expected,
            "valid_pairs": len(valid_effects),
            "complete": bool(expected > 0 and len(valid_effects) == expected),
        }
        effects[name] = valid_effects
    release = (
        bool(confirmatory_grid)
        and global_rate >= FROZEN_GATE_LIMITS["global_condition_valid_rate"]
        and all(item["complete"] for item in coverages.values())
    )
    result = {
        "status": (
            "descriptive_single_policy"
            if release else "suppressed_failed_validity_or_coverage_gate"
        ),
        "qualitative_alignment_pass": False,
        "minimum_independent_policies_for_claim": 5,
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_scope": (
            "46D augmented orbital return map with neutral global x/y removed"),
        "unquotiented_full_48d_is_diagnostic": True,
        "translation_reduced_learned_controller_analogue": True,
        "interpretation": (
            "qualitative gait-parameter association alignment only"),
        "equivalence_margin_source": (
            "preregistered study operationalization; not supplied by the paper"),
        "not_tested": [
            "cost_of_transport", "energy_efficiency", "beam_traversal"],
        "equivalence_ratio_margin": EQUIVALENCE_RATIO_MARGIN,
        "planned_conditions": planned,
        "valid_conditions": len(valid),
        "global_condition_valid_rate": global_rate,
        "confirmatory_grid": bool(confirmatory_grid),
        "global_90_percent_gate_pass": bool(global_rate >= FROZEN_GATE_LIMITS["global_condition_valid_rate"]),
        "contrast_coverage": coverages,
        "claim_values_released": release,
    }
    if not release:
        return result

    trot = [row for row in valid if row["gait"] == "trot"]
    narrowing = [row for row in valid if row["step_width"] <= .30]
    result["descriptive_spearman"] = {
        "duty_factor_trot": _spearman(
            [row["command_df"] for row in trot],
            [row[PRIMARY_METRIC] for row in trot]),
        "narrow_width": _spearman(
            [row["step_width"] for row in narrowing],
            [row[PRIMARY_METRIC] for row in narrowing]),
        "speed": _spearman(
            [row["speed"] for row in valid],
            [row[PRIMARY_METRIC] for row in valid]),
    }
    for name, values in effects.items():
        differences = [item[0] for item in values]
        ratios = [item[1] for item in values]
        result[f"{name}_matched_effect"] = {
            "pairs": len(values),
            "median_high_minus_low": float(np.median(differences)),
            "median_high_over_low_ratio": float(np.median(ratios)),
        }
    result["speed_descriptively_within_equivalence_margin"] = bool(
        abs(result["speed_matched_effect"]["median_high_over_low_ratio"] - 1)
        <= EQUIVALENCE_RATIO_MARGIN)
    result["gait_descriptively_within_equivalence_margin"] = bool(
        abs(result["gait_matched_effect"]["median_high_over_low_ratio"] - 1)
        <= EQUIVALENCE_RATIO_MARGIN)
    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _validate_archive(data: dict, expected: dict, manifest: dict):
    required = {
        "initial_states", "final_states", "command", "perturbation_h",
        "cycle_rms", "done", "task_sha256", "evaluation_sha256",
        "checkpoint_sha256", "state_scales", "nominal_contacts",
        "nominal_desired", "nominal_feet_body", "nominal_forward_velocity",
        "nominal_failure", "nominal_lateral_position",
        "nominal_heading", "nominal_world_lateral_velocity",
        "nominal_body_yaw_rate", "settle_group_rms", "settle_done",
        "zero_initial_states", "zero_final_states", "zero_initial_contacts",
        "zero_substep_contacts", "stencil_initial_contacts",
        "stencil_substep_contacts", "desired_initial_contacts",
        "desired_substep_contacts",
    }
    if not required.issubset(data):
        raise ValueError(f"Missing stability fields: {sorted(required - set(data))}")
    forbidden = {"force", "planned_force", "push_plan"}.intersection(data)
    if forbidden:
        raise ValueError(
            f"Stability evaluation must be push-free; found {sorted(forbidden)}")
    if not np.array_equal(data["state_scales"], STATE_SCALES):
        raise ValueError("Trace state scales differ from frozen source scales")
    if not np.array_equal(
        np.asarray(manifest["state_scales"], dtype=float), STATE_SCALES
    ):
        raise ValueError("Manifest state scales differ from frozen source scales")
    identities = {
        "task_sha256": manifest["task_sha256"],
        "evaluation_sha256": manifest["evaluation_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
    }
    for field, value in identities.items():
        if str(data[field]) != value:
            raise ValueError(f"Manifest and trace {field} differ")
    command = np.asarray(data["command"], dtype=float)
    gait_id = ("trot", "walk").index(expected["gait"])
    expected_command = np.asarray([
        expected["speed"], expected["duty_factor"], expected["step_width"],
        expected["period"], gait_id,
    ])
    if command.shape != (5,) or not np.allclose(
        command, expected_command, rtol=0, atol=1e-8
    ):
        raise ValueError("Embedded command does not match manifest condition")
    h = float(data["perturbation_h"])
    if not np.isclose(h, expected["perturbation_h"], rtol=0, atol=1e-10):
        raise ValueError("Embedded perturbation size does not match filename manifest")

    initial, final = data["initial_states"], data["final_states"]
    if initial.ndim != 3 or initial.shape[1:] != (97, RAW_STATE_DIM):
        raise ValueError("Malformed stability state arrays")
    if final.shape != initial.shape or not np.isfinite(initial).all() or not np.isfinite(final).all():
        raise ValueError("Stability states must be finite and shape matched")
    references = initial.shape[0]
    if references != manifest["references"]:
        raise ValueError("Reference count differs from manifest")
    if data["done"].shape != (references, 97):
        raise ValueError("Malformed stability termination flags")
    zero_clones = manifest["zero_clones_per_reference"]
    expected_shapes = {
        "cycle_rms": (references,),
        "settle_group_rms": (references,),
        "settle_done": (references,),
        "zero_initial_states": (references, zero_clones, RAW_STATE_DIM),
        "zero_final_states": (references, zero_clones, RAW_STATE_DIM),
        "zero_initial_contacts": (references, zero_clones, 4),
        "stencil_initial_contacts": (references, 97, 4),
        "desired_initial_contacts": (references, 4),
    }
    for field, shape in expected_shapes.items():
        if data[field].shape != shape:
            raise ValueError(f"Malformed {field}: expected {shape}")
    if data["stencil_substep_contacts"].shape[:2] != (references, 97):
        raise ValueError("Malformed stencil substep contacts")
    if data["zero_substep_contacts"].shape[:2] != (references, zero_clones):
        raise ValueError("Malformed zero-clone substep contacts")
    if data["stencil_substep_contacts"].shape[2:] != data["zero_substep_contacts"].shape[2:]:
        raise ValueError("Stencil and zero-clone contact traces differ in shape")
    control_steps = round(expected["period"] / .02)
    substeps = control_steps * 4
    trace_shapes = {
        "nominal_contacts": (references, control_steps, 4),
        "nominal_desired": (references, control_steps, 4),
        "nominal_feet_body": (references, control_steps, 4, 3),
        "nominal_forward_velocity": (references, control_steps),
        "nominal_failure": (references, control_steps),
        "nominal_lateral_position": (references, control_steps),
        "nominal_heading": (references, control_steps),
        "nominal_world_lateral_velocity": (references, control_steps),
        "nominal_body_yaw_rate": (references, control_steps),
        "stencil_substep_contacts": (references, 97, substeps, 4),
        "zero_substep_contacts": (references, zero_clones, substeps, 4),
        "desired_substep_contacts": (references, substeps, 4),
    }
    for field, shape in trace_shapes.items():
        if data[field].shape != shape:
            raise ValueError(f"Malformed {field}: expected {shape}")
    finite_fields = (
        "settle_group_rms", "nominal_feet_body",
        "nominal_forward_velocity", "nominal_lateral_position",
        "nominal_heading", "nominal_world_lateral_velocity",
        "nominal_body_yaw_rate")
    if any(not np.isfinite(data[field]).all() for field in finite_fields):
        raise ValueError("Stability nominal traces and settle metrics must be finite")


def analyze_stability(directory: str | Path) -> list[dict]:
    """Validate traces, estimate all maps, and write strictly gated summaries."""
    directory = Path(directory)
    manifest = json.loads((directory / "stability_manifest.json").read_text())
    if manifest.get("schema") != "beam_stability_v2":
        raise ValueError("Unsupported stability manifest schema")
    if manifest.get("external_pushes") is not False:
        raise ValueError("Stability manifest must explicitly disable external pushes")
    if manifest.get("paper_metric_primary") != (
            "chi_orb=sigma_max(Phi_orbital_augmented_46D)"):
        raise ValueError("Manifest must preregister the 46D orbital augmented metric")
    if manifest.get("primary_metric_scope") != (
            "translation-reduced learned-controller analogue"):
        raise ValueError("Manifest primary metric scope is missing or changed")
    if manifest.get("unquotiented_full_48d_is_diagnostic") is not True:
        raise ValueError("Manifest must retain the full 48D diagnostic")
    if manifest.get("translation_symmetry_residual_limit") != .02:
        raise ValueError("Translation-symmetry gate must remain frozen at 0.02")
    provenance_path = directory / manifest.get(
        "training_provenance_file", "")
    if not provenance_path.is_file():
        raise ValueError("Archived training provenance is missing")
    provenance_bytes = provenance_path.read_bytes()
    if hashlib.sha256(provenance_bytes).hexdigest() != manifest.get(
            "training_provenance_sha256"):
        raise ValueError("Archived training provenance hash mismatch")
    training_provenance = json.loads(provenance_bytes)
    lineage_seed = training_provenance.get("seed")
    expected_lineage = (
        hashlib.sha256(
            f"{manifest.get('training_source_sha256')}:{lineage_seed}:fresh_v1".encode()
        ).hexdigest()
        if isinstance(lineage_seed, int) else None
    )
    if (
        training_provenance.get("mode") != "train"
        or training_provenance.get("fresh_training") is not True
        or "checkpoint" in training_provenance
        or training_provenance.get("task_sha256") != manifest.get("task_sha256")
        or training_provenance.get("seed") != manifest.get("training_seed")
        or training_provenance.get("training_lineage_id")
            != manifest.get("training_lineage_id")
        or manifest.get("training_lineage_id") != expected_lineage
        or training_provenance.get("training_source_sha256")
            != manifest.get("training_source_sha256")
        or training_provenance.get("training_num_envs")
            != manifest.get("training_num_envs")
        or training_provenance.get("training_iterations_requested")
            != manifest.get("training_iterations_requested")
        or training_provenance.get("checkpoint_selection_rule")
            != manifest.get("checkpoint_selection_rule")
    ):
        raise ValueError("Archived training provenance is not a matching fresh lineage")
    if manifest.get("gate_limits") != FROZEN_GATE_LIMITS:
        raise ValueError("Manifest numeric gate limits differ from frozen source")
    duplicate_limits = {
        "settle_group_rms_limit": "settle_group_rms",
        "zero_clone_noise_fraction_of_h_limit":
            "zero_clone_noise_fraction_of_h",
        "initial_stencil_max_error_limit": "initial_stencil_max_error",
        "initial_condition_number_limit": "initial_condition_number",
        "translation_symmetry_residual_limit":
            "translation_symmetry_residual",
    }
    if any(
        manifest.get(field) != FROZEN_GATE_LIMITS[key]
        for field, key in duplicate_limits.items()
    ):
        raise ValueError("Manifest duplicate numeric limits differ from source")
    if manifest.get("evaluation_reference_seed") != CONFIRMATORY_REFERENCE_SEED:
        # Exploratory runs may use another seed but can never be confirmatory.
        if manifest.get("confirmatory_grid"):
            raise ValueError(
                "Confirmatory protocol requires the frozen evaluation seed")
    expected_conditions = manifest.get("expected_conditions", [])
    if not expected_conditions:
        raise ValueError("Stability manifest has no structured conditions")
    condition_keys = [
        tuple(item[field] for field in (
            "period", "gait", "speed", "step_width",
            "duty_factor", "perturbation_h"))
        for item in expected_conditions]
    if len(condition_keys) != len(set(condition_keys)):
        raise ValueError("Manifest contains duplicate structured conditions")
    recomputed_confirmatory = confirmatory_protocol(
        condition_keys=condition_keys,
        perturbation_sizes={
            item["perturbation_h"] for item in expected_conditions},
        references=manifest.get("references"),
        settle_cycles=manifest.get("settle_cycles"),
        zero_clones=manifest.get("zero_clones_per_reference"),
        master_stencil_size=manifest.get("master_stencil_size"),
        evaluation_reference_seed=manifest.get(
            "evaluation_reference_seed"),
        training_iterations_requested=manifest.get(
            "training_iterations_requested"),
        checkpoint_iteration=manifest.get("checkpoint_iteration"),
        fresh_training=manifest.get("fresh_training"),
    )
    if manifest.get("confirmatory_grid") is not recomputed_confirmatory:
        raise ValueError(
            "Manifest confirmatory flag differs from the frozen protocol")
    expected_names = [item["filename"] for item in expected_conditions]
    if manifest.get("expected_files") != expected_names:
        raise ValueError("Manifest expected file list differs from structured conditions")
    if len(manifest.get("joint_order", [])) != 12 or len(set(manifest["joint_order"])) != 12:
        raise ValueError("Manifest must record 12 unique robot joints")
    if manifest.get("action_order") != manifest["joint_order"]:
        raise ValueError("Action and joint ordering must match for joint-position control")
    actual = sorted(path.name for path in directory.glob("stability_*.npz"))
    if not expected_names or sorted(expected_names) != actual:
        raise ValueError("Stability files must exactly match the condition manifest")
    if len(set(expected_names)) != len(expected_names):
        raise ValueError("Stability manifest contains duplicate filenames")

    metric_names = (
        "chi_augmented", "chi_physical_conditional",
        "chi_orbital_physical", "chi_orbital_augmented",
    )
    map_kinds = (
        "augmented", "physical_conditional",
        "orbital_physical_conditional", "orbital_augmented",
    )
    rows, artifacts, artifact_index, primary_maps = [], {}, [], {}
    for expected in expected_conditions:
        name = expected["filename"]
        with np.load(directory / name) as archive:
            data = {key: archive[key] for key in archive.files}
        _validate_archive(data, expected, manifest)
        speed, duty, width, period, gait_id = data["command"].tolist()
        gait = ("trot", "walk")[int(gait_id)]
        h = float(data["perturbation_h"])
        for reference in range(data["initial_states"].shape[0]):
            base = {
                "file": name, "reference": reference, "speed": speed,
                "command_df": duty, "step_width": width, "period": period,
                "gait": gait, "perturbation_h": h,
                "cycle_rms": float(data["cycle_rms"][reference]),
                "periodic_orbit_gate_pass": int(
                    data["cycle_rms"][reference] <= FROZEN_GATE_LIMITS["periodic_orbit_rms"]),
                "terminated": int(
                    np.asarray(data["done"][reference]).any()),
                "hybrid_topology_gate_pass": int(
                    hybrid_topology_gate(data, reference)),
            }
            base.update(command_fidelity(
                data, reference, speed, duty, width))
            base.update(numerical_fidelity(
                data, reference, h, manifest))
            base.update({metric: np.nan for metric in metric_names})
            base.update({
                "rank_physical": np.nan,
                "rank_orbital_physical": np.nan,
                "rank_orbital_augmented": np.nan,
                "fit_relative_residual": np.nan,
                "translation_x_identity_residual": np.nan,
                "translation_y_identity_residual": np.nan,
                "translation_identity_residual_max": np.nan,
                "translation_cross_coupling_max": np.nan,
                "translation_symmetry_gate_pass": 0,
                "finite_difference_primary_matrix_error": np.nan,
                "finite_difference_primary_chi_error": np.nan,
                "finite_difference_full_matrix_error": np.nan,
                "finite_difference_full_chi_error": np.nan,
                "finite_difference_gate_pass": 0,
                "reference_valid": 0,
            })
            if not base["terminated"]:
                maps = estimate_maps(
                    data["initial_states"][reference],
                    data["final_states"][reference])
                base.update({
                    "chi_augmented": maps["augmented"]["chi"],
                    "chi_physical_conditional":
                        maps["physical_conditional"]["chi"],
                    "chi_orbital_physical":
                        maps["orbital_physical_conditional"]["chi"],
                    "chi_orbital_augmented":
                        maps["orbital_augmented"]["chi"],
                    "rank_physical":
                        maps["physical_conditional"]["rank"],
                    "rank_orbital_physical":
                        maps["orbital_physical_conditional"]["rank"],
                    "rank_orbital_augmented":
                        maps["orbital_augmented"]["rank"],
                    "fit_relative_residual":
                        maps["orbital_augmented"]["relative_residual"],
                })
                base.update(translation_symmetry_fidelity(
                    maps["augmented"]["phi"],
                    FROZEN_GATE_LIMITS[
                        "translation_symmetry_residual"]))
                for kind in map_kinds:
                    result = maps[kind]
                    key = f"r{len(rows):06d}_{kind}"
                    artifacts[f"{key}_phi"] = result["phi"]
                    artifacts[f"{key}_singular_values"] = result[
                        "singular_values"]
                    artifacts[f"{key}_worst_case_right_vector"] = result[
                        "worst_case_right_vector"]
                    labels = {
                        "augmented": TANGENT_LABELS,
                        "physical_conditional": TANGENT_LABELS[:PHYSICAL_DIM],
                        "orbital_physical_conditional": tuple(
                            TANGENT_LABELS[index]
                            for index in ORBITAL_PHYSICAL_INDICES),
                        "orbital_augmented": tuple(
                            TANGENT_LABELS[index]
                            for index in ORBITAL_AUGMENTED_INDICES),
                    }[kind]
                    artifact_index.append({
                        "key": key, "row": len(rows), "file": name,
                        "reference": reference, "kind": kind,
                        "coordinate_labels": labels,
                    })
                primary_maps[len(rows)] = {
                    "orbital": maps["orbital_augmented"]["phi"],
                    "full": maps["augmented"]["phi"],
                }
            rows.append(base)

    groups = {}
    for row_index, maps in primary_maps.items():
        row = rows[row_index]
        key = (
            row["speed"], row["command_df"], row["step_width"],
            row["period"], row["gait"], row["reference"])
        groups.setdefault(key, {})[round(row["perturbation_h"], 6)] = (
            row_index, maps)
    for group in groups.values():
        if not all(h in group for h in (.025, .05, .10)):
            continue
        center_row, center_maps = group[.05]
        primary_chi = rows[center_row]["chi_orbital_augmented"]
        full_chi = rows[center_row]["chi_augmented"]
        primary_matrix_error = max(
            relative_matrix_difference(
                center_maps["orbital"], group[h][1]["orbital"])
            for h in (.025, .10))
        primary_chi_error = max(
            abs(rows[group[h][0]]["chi_orbital_augmented"] - primary_chi)
            / max(abs(primary_chi), np.finfo(float).eps)
            for h in (.025, .10))
        full_matrix_error = max(
            relative_matrix_difference(
                center_maps["full"], group[h][1]["full"])
            for h in (.025, .10))
        full_chi_error = max(
            abs(rows[group[h][0]]["chi_augmented"] - full_chi)
            / max(abs(full_chi), np.finfo(float).eps)
            for h in (.025, .10))
        passed = primary_matrix_error <= FROZEN_GATE_LIMITS["finite_difference_relative_error"] and primary_chi_error <= FROZEN_GATE_LIMITS["finite_difference_relative_error"]
        for row_index, _ in group.values():
            rows[row_index].update({
                "finite_difference_primary_matrix_error":
                    primary_matrix_error,
                "finite_difference_primary_chi_error": primary_chi_error,
                "finite_difference_full_matrix_error": full_matrix_error,
                "finite_difference_full_chi_error": full_chi_error,
                "finite_difference_gate_pass": int(passed),
            })

    for row in rows:
        ranks_pass = (
            row["initial_rank"] == AUGMENTED_DIM
            and row["rank_physical"] == PHYSICAL_DIM
            and row["rank_orbital_physical"] == len(ORBITAL_PHYSICAL_INDICES)
            and row["rank_orbital_augmented"] == len(
                ORBITAL_AUGMENTED_INDICES))
        row["rank_gate_pass"] = int(ranks_pass)
        row["reference_valid"] = int(
            not row["terminated"]
            and row["periodic_orbit_gate_pass"]
            and row["command_gate_pass"]
            and row["hybrid_topology_gate_pass"]
            and row["translation_symmetry_gate_pass"]
            and row["numerical_gate_pass"]
            and row["rank_gate_pass"]
            and row["finite_difference_gate_pass"])

    with (directory / "stability_references.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(directory / "return_maps.npz", **artifacts)
    (directory / "return_map_index.json").write_text(
        json.dumps(_json_safe(artifact_index), indent=2))

    group_fields = CONDITION_FIELDS + ("perturbation_h",)
    summary_groups = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        summary_groups.setdefault(key, []).append(row)
    summary = []
    gate_names = (
        "periodic_orbit_gate_pass", "command_gate_pass",
        "hybrid_topology_gate_pass", "translation_symmetry_gate_pass",
        "numerical_gate_pass", "rank_gate_pass",
        "finite_difference_gate_pass")
    for key, group in summary_groups.items():
        valid = [row for row in group if row["reference_valid"]]
        record = dict(zip(group_fields, key))
        record.update({
            "references": len(group),
            "valid_references": len(valid),
            "reference_valid_rate": len(valid) / len(group),
        })
        for gate in gate_names:
            record[f"{gate}_rate"] = float(
                np.mean([row[gate] for row in group]))
        for metric in metric_names:
            diagnostic = np.asarray([
                row[metric] for row in group if np.isfinite(row[metric])])
            gated = np.asarray([row[metric] for row in valid])
            record[f"{metric}_diagnostic_median"] = (
                float(np.median(diagnostic)) if diagnostic.size else np.nan)
            record[f"{metric}_median"] = (
                float(np.median(gated)) if gated.size else np.nan)
            record[f"{metric}_q025"] = (
                float(np.quantile(gated, .025)) if gated.size else np.nan)
            record[f"{metric}_q975"] = (
                float(np.quantile(gated, .975)) if gated.size else np.nan)
        summary.append(record)
    with (directory / "stability_summary.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    conditions = []
    for row in summary:
        if abs(row["perturbation_h"] - .05) > 1e-8:
            continue
        condition = dict(row)
        condition["condition_valid"] = int(
            row["reference_valid_rate"] >= FROZEN_GATE_LIMITS["reference_valid_rate"])
        conditions.append(condition)
    with (directory / "paper_conditions.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(conditions[0]))
        writer.writeheader()
        writer.writerows(conditions)
    claims = paper_claim_summary(
        conditions, manifest.get("confirmatory_grid", False))
    (directory / "paper_claims.json").write_text(
        json.dumps(_json_safe(claims), indent=2))
    return summary


def _bootstrap_interval(values, seed=9162026, samples=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(
        len(values), size=(samples, len(values)))].mean(axis=1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def analyze_policy_ensemble(directories, output: str | Path):
    """Aggregate matched effects across at least five independent PPO policies."""
    directories = [Path(directory) for directory in directories]
    if len(directories) < 5:
        raise ValueError(
            "Paper claims require at least five independently trained policies")
    manifests, condition_tables, claim_reports = [], [], []
    import pandas as pd
    for directory in directories:
        analyze_stability(directory)
        manifests.append(json.loads(
            (directory / "stability_manifest.json").read_text()))
        condition_tables.append(pd.read_csv(
            directory / "paper_conditions.csv").to_dict("records"))
        claim_reports.append(json.loads(
            (directory / "paper_claims.json").read_text()))
    reference = manifests[0]
    identity_fields = (
        "schema", "task_sha256", "evaluation_sha256", "state_scales",
        "expected_conditions", "joint_order", "action_order",
        "paper_metric_primary", "primary_metric_scope",
        "translation_symmetry_residual_limit", "confirmatory_grid",
        "gate_limits", "references", "zero_clones_per_reference",
        "master_stencil_size", "settle_cycles",
        "reference_initial_offset_half_width_normalized",
        "evaluation_reference_seed", "gait_regime",
        "reference_initialization", "training_source_sha256",
        "training_num_envs", "training_iterations_requested",
        "checkpoint_selection_rule", "checkpoint_iteration")
    for manifest in manifests[1:]:
        if any(manifest[field] != reference[field] for field in identity_fields):
            raise ValueError(
                "Policy ensemble grids, sources, scales, or joint orders differ")
    if len({manifest["checkpoint_sha256"] for manifest in manifests}) != len(manifests):
        raise ValueError("Policy ensemble contains duplicate checkpoints")
    if len({manifest["training_seed"] for manifest in manifests}) != len(manifests):
        raise ValueError("Policy ensemble requires distinct PPO training seeds")
    if not all(manifest.get("fresh_training") is True for manifest in manifests):
        raise ValueError("Policy ensemble requires parent-free fresh training runs")
    lineages = [manifest.get("training_lineage_id") for manifest in manifests]
    if any(not isinstance(lineage, str) or len(lineage) != 64
           for lineage in lineages) or len(set(lineages)) != len(lineages):
        raise ValueError(
            "Policy ensemble requires five distinct recorded training lineages")
    provenance_hashes = [
        manifest.get("training_provenance_sha256") for manifest in manifests]
    if any(not isinstance(value, str) or len(value) != 64
           for value in provenance_hashes) or len(set(provenance_hashes)) != len(manifests):
        raise ValueError(
            "Policy ensemble requires distinct archived training provenance")
    if not all(report["claim_values_released"] for report in claim_reports):
        raise ValueError(
            "Every policy must pass its global validity and contrast coverage gates")

    definitions = (
        ("duty_factor", "command_df", .50, .75,
         lambda row: row["gait"] == "trot", "less_than_one"),
        ("narrow_width", "step_width", .30, .10,
         lambda row: True, "greater_than_one"),
        ("speed", "speed", .25, .40,
         lambda row: True, "equivalent"),
        ("gait", "gait", "trot", "walk",
         lambda row: abs(row["command_df"] - .75) < 1e-8, "equivalent"),
    )
    records = []
    report = {
        "policies": len(directories),
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_scope":
            "46D augmented orbital return map with neutral global x/y removed",
        "translation_reduced_learned_controller_analogue": True,
        "interpretation": (
            "qualitative gait-parameter association alignment only"),
        "equivalence_margin_source": (
            "preregistered study operationalization; not supplied by the paper"),
        "not_tested": [
            "cost_of_transport", "energy_efficiency", "beam_traversal"],
        "equivalence_ratio_margin": EQUIVALENCE_RATIO_MARGIN,
        "qualitative_alignment_pass": False,
        "effects": {},
    }
    all_pass = True
    for name, factor, low, high, subset, decision in definitions:
        policy_log_ratios = []
        for policy, rows in enumerate(condition_tables):
            valid = [row for row in rows if bool(row["condition_valid"])]
            _, effects = _pair_effects(valid, factor, low, high, subset)
            log_ratios = np.log([ratio for _, ratio in effects])
            policy_value = float(np.mean(log_ratios))
            policy_log_ratios.append(policy_value)
            records.append({
                "policy": policy, "directory": str(directories[policy]),
                "training_seed": manifests[policy]["training_seed"],
                "checkpoint_sha256":
                    manifests[policy]["checkpoint_sha256"],
                "factor": name, "mean_log_ratio": policy_value,
                "geometric_mean_ratio": float(np.exp(policy_value)),
                "matched_pairs": len(effects),
            })
        lo, hi = _bootstrap_interval(policy_log_ratios)
        estimate = float(np.mean(policy_log_ratios))
        if decision == "equivalent":
            passed = (
                lo >= np.log(1 - EQUIVALENCE_RATIO_MARGIN)
                and hi <= np.log(1 + EQUIVALENCE_RATIO_MARGIN))
        elif decision == "less_than_one":
            passed = hi < 0
        else:
            passed = lo > 0
        all_pass &= passed
        report["effects"][name] = {
            "decision": decision,
            "mean_ratio": float(np.exp(estimate)),
            "bootstrap_95_percent_ratio_interval": [
                float(np.exp(lo)), float(np.exp(hi))],
            "pass": bool(passed),
        }
    report["qualitative_alignment_pass"] = bool(all_pass)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "policy_effects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "ensemble_claims.json").write_text(
        json.dumps(_json_safe(report), indent=2))
    return report
