"""Paper-aligned empirical closed-loop return-map analysis.

The paper's convergence quantity is the largest singular value of a hybrid
closed-loop fundamental solution matrix.  For a neural policy and simulator we
estimate the analogous phase-to-phase map with central finite differences.  This
module is simulator independent so the numerical definition can be tested before
any GPU run.
"""
from __future__ import annotations

import csv
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


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0)


def command_fidelity(data: dict, reference: int, speed: float,
                     duty: float, width: float) -> dict:
    """Compute pre-registered command gates for one nominal gait cycle."""
    contacts = np.asarray(data["nominal_contacts"][reference], dtype=bool)
    desired = np.asarray(data["nominal_desired"][reference], dtype=bool)
    feet = np.asarray(data["nominal_feet_body"][reference], dtype=np.float64)
    velocity = np.asarray(data["nominal_forward_velocity"][reference], dtype=np.float64)
    failure = np.asarray(data["nominal_failure"][reference], dtype=bool)
    if contacts.ndim != 2 or contacts.shape[1] != 4 or desired.shape != contacts.shape:
        raise ValueError("Nominal contact traces must have shape [references, time, 4]")
    if feet.shape != contacts.shape + (3,) or velocity.shape != contacts.shape[:1]:
        raise ValueError("Malformed nominal foot or velocity traces")
    achieved_df = contacts.mean(axis=0)
    stance_recall = _ratio((contacts & desired).sum(axis=0), desired.sum(axis=0))
    swing_recall = _ratio((~contacts & ~desired).sum(axis=0), (~desired).sum(axis=0))
    signs = np.asarray([1.0, -1.0, 1.0, -1.0])
    achieved_width = float(2.0 * np.mean(feet[..., 1] * signs))
    foot_lateral_mae = float(np.mean(np.abs(feet[..., 1] - 0.5 * width * signs)))
    forward_speed = float(np.mean(velocity))
    df_max_abs_error = float(np.max(np.abs(achieved_df - duty)))
    width_abs_error = abs(achieved_width - width)
    speed_abs_error = abs(forward_speed - speed)
    gate = (
        not failure.any()
        and df_max_abs_error <= 0.05
        and width_abs_error <= 0.03
        and speed_abs_error <= 0.04
        and bool(np.all(stance_recall >= 0.90))
        and bool(np.all(swing_recall >= 0.90))
    )
    result = {
        "achieved_df": float(np.mean(achieved_df)),
        "df_max_abs_error": df_max_abs_error,
        "achieved_width": achieved_width,
        "width_abs_error": width_abs_error,
        "foot_lateral_mae": foot_lateral_mae,
        "forward_speed": forward_speed,
        "speed_abs_error": speed_abs_error,
        "stance_recall_min": float(np.nanmin(stance_recall)),
        "swing_recall_min": float(np.nanmin(swing_recall)),
        "nominal_failure": int(failure.any()),
        "command_gate_pass": int(gate),
    }
    for index, leg in enumerate(("FL", "FR", "RL", "RR")):
        result[f"df_{leg}"] = float(achieved_df[index])
        result[f"stance_recall_{leg}"] = float(stance_recall[index])
        result[f"swing_recall_{leg}"] = float(swing_recall[index])
    return result


def _rank(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks without requiring SciPy."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return float(np.corrcoef(_rank(np.asarray(x)), _rank(np.asarray(y)))[0, 1])


def _paper_claim_summary(conditions: list[dict]) -> dict:
    valid = [row for row in conditions if row["condition_valid"]]
    metric = "chi_orbital_augmented_median"
    result = {
        "metric": metric,
        "interpretation": "lower chi means stronger local phase-to-phase contraction",
        "valid_conditions": len(valid),
        "duty_factor_claim": "higher duty factor should reduce chi",
        "width_claim": "narrowing below nominal stance width should increase chi",
        "speed_claim": "speed should have weak association with chi",
        "gait_claim": "walk and trot should be similar at matched speed, width, period, and DF=0.75",
    }
    trot = [row for row in valid if row["gait"] == "trot"]
    result["duty_factor_spearman_rho"] = _spearman(
        [row["command_df"] for row in trot], [row[metric] for row in trot])
    narrowing = [row for row in valid if row["step_width"] <= 0.30]
    result["narrow_width_spearman_rho"] = _spearman(
        [row["step_width"] for row in narrowing], [row[metric] for row in narrowing])
    result["speed_spearman_rho"] = _spearman(
        [row["speed"] for row in valid], [row[metric] for row in valid])

    def paired_effect(factor, low, high, subset):
        pairs = {}
        remaining = [
            name for name in ("speed", "command_df", "step_width", "period", "gait")
            if name != factor
        ]
        for row in subset:
            if row[factor] not in (low, high):
                continue
            key = tuple(row[name] for name in remaining)
            pairs.setdefault(key, {})[row[factor]] = row[metric]
        effects = [
            pair[high] - pair[low] for pair in pairs.values()
            if low in pair and high in pair
        ]
        return {
            "pairs": len(effects),
            "median_high_minus_low": (
                float(np.median(effects)) if effects else float("nan")
            ),
        }

    result["duty_factor_0.75_minus_0.50"] = {
        **paired_effect("command_df", .50, .75, trot),
        "expected_sign_for_paper_agreement": "negative",
    }
    result["narrow_width_0.10_minus_nominal_0.30"] = {
        **paired_effect("step_width", .30, .10, valid),
        "expected_sign_for_paper_agreement": "positive",
    }
    result["speed_0.40_minus_0.25"] = {
        **paired_effect("speed", .25, .40, valid),
        "paper_expectation": "small magnitude",
    }
    matched = {}
    for row in valid:
        if abs(row["command_df"] - 0.75) > 1e-8:
            continue
        key = (row["speed"], row["step_width"], row["period"])
        matched.setdefault(key, {})[row["gait"]] = row[metric]
    differences = [
        abs(pair["walk"] - pair["trot"]) / max(0.5 * (pair["walk"] + pair["trot"]),
                                               np.finfo(float).eps)
        for pair in matched.values() if {"walk", "trot"}.issubset(pair)
    ]
    result["matched_gait_pairs"] = len(differences)
    result["matched_gait_median_relative_difference"] = (
        float(np.median(differences)) if differences else float("nan")
    )
    return result


def analyze_stability(directory: str | Path) -> list[dict]:
    """Validate traces, estimate return maps, and write gated paper-claim tables."""
    directory = Path(directory)
    manifest_path = directory / "stability_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("external_pushes") is not False:
        raise ValueError("Stability manifest must explicitly disable external pushes")
    expected = manifest.get("expected_files", [])
    actual = sorted(path.name for path in directory.glob("stability_*.npz"))
    if sorted(expected) != actual or not expected:
        raise ValueError("Stability files must exactly match the nonempty manifest")

    required = {
        "initial_states", "final_states", "command", "perturbation_h",
        "cycle_rms", "done", "source_sha256", "checkpoint_sha256",
        "nominal_contacts", "nominal_desired", "nominal_feet_body",
        "nominal_forward_velocity", "nominal_failure",
    }
    metric_names = (
        "chi_augmented", "chi_physical_conditional",
        "chi_orbital_physical", "chi_orbital_augmented",
    )
    rows, map_arrays, map_index = [], {}, []
    identity = None
    for name in expected:
        with np.load(directory / name) as archive:
            data = {key: archive[key] for key in archive.files}
        forbidden = {"force", "planned_force", "push_plan"}.intersection(data)
        if forbidden:
            raise ValueError(f"Stability evaluation must be push-free; found {sorted(forbidden)}")
        if not required.issubset(data):
            raise ValueError(f"Missing stability fields: {sorted(required - set(data))}")
        current_identity = (str(data["source_sha256"]), str(data["checkpoint_sha256"]))
        identity = current_identity if identity is None else identity
        if current_identity != identity:
            raise ValueError("Mixed source/checkpoint identity in stability evaluation")
        if (
            current_identity[0] != manifest.get("source_sha256")
            or current_identity[1] != manifest.get("checkpoint_sha256")
        ):
            raise ValueError("Manifest and trace source/checkpoint identities differ")
        initial, final = data["initial_states"], data["final_states"]
        if initial.ndim != 3 or final.shape != initial.shape or initial.shape[1:] != (97, RAW_STATE_DIM):
            raise ValueError("Malformed stability state arrays")
        if data["done"].shape != initial.shape[:2]:
            raise ValueError("Malformed stability termination flags")
        speed, duty, width, period, gait = data["command"].tolist()
        for reference in range(initial.shape[0]):
            base = {
                "file": name, "reference": reference, "speed": speed,
                "command_df": duty, "step_width": width, "period": period,
                "gait": ("trot", "walk")[int(gait)],
                "perturbation_h": float(data["perturbation_h"]),
                "cycle_rms": float(data["cycle_rms"][reference]),
                "periodic_orbit_gate_pass": int(data["cycle_rms"][reference] <= 0.02),
                "terminated": int(np.asarray(data["done"][reference]).any()),
            }
            base.update(command_fidelity(data, reference, speed, duty, width))
            base.update({metric: float("nan") for metric in metric_names})
            base.update({
                "initial_rank": float("nan"),
                "initial_condition_number": float("nan"),
                "fit_relative_residual": float("nan"),
                "finite_difference_matrix_error": float("nan"),
                "finite_difference_chi_error": float("nan"),
                "finite_difference_gate_pass": 0,
                "reference_valid": 0,
            })
            if not base["terminated"]:
                maps = estimate_maps(initial[reference], final[reference])
                base.update({
                    "chi_augmented": maps["augmented"]["chi"],
                    "chi_physical_conditional": maps["physical_conditional"]["chi"],
                    "chi_orbital_physical": maps["orbital_physical_conditional"]["chi"],
                    "chi_orbital_augmented": maps["orbital_augmented"]["chi"],
                    "initial_rank": maps["augmented"]["rank"],
                    "initial_condition_number": maps["augmented"]["condition_number"],
                    "fit_relative_residual": maps["augmented"]["relative_residual"],
                })
                key = f"map_{len(map_arrays):06d}"
                map_arrays[key] = maps["orbital_augmented"]["phi"]
                map_index.append({
                    "key": key, "row": len(rows), "file": name,
                    "reference": reference, "kind": "orbital_augmented",
                })
            rows.append(base)

    grouped_maps = {}
    for item in map_index:
        row = rows[item["row"]]
        key = (
            row["speed"], row["command_df"], row["step_width"],
            row["period"], row["gait"], row["reference"],
        )
        grouped_maps.setdefault(key, {})[row["perturbation_h"]] = (
            item["row"], map_arrays[item["key"]]
        )
    required_h = (0.025, 0.05, 0.10)
    for group in grouped_maps.values():
        available = {round(h, 6): value for h, value in group.items()}
        if not all(h in available for h in required_h):
            continue
        center_row, center_phi = available[0.05]
        center_chi = rows[center_row]["chi_orbital_augmented"]
        matrix_error = max(
            relative_matrix_difference(center_phi, available[h][1])
            for h in (0.025, 0.10)
        )
        chi_error = max(
            abs(rows[available[h][0]]["chi_orbital_augmented"] - center_chi)
            / max(abs(center_chi), np.finfo(float).eps)
            for h in (0.025, 0.10)
        )
        passed = matrix_error <= 0.10 and chi_error <= 0.10
        for row_index, _ in available.values():
            rows[row_index]["finite_difference_matrix_error"] = matrix_error
            rows[row_index]["finite_difference_chi_error"] = chi_error
            rows[row_index]["finite_difference_gate_pass"] = int(passed)

    for row in rows:
        row["reference_valid"] = int(
            not row["terminated"]
            and row["periodic_orbit_gate_pass"]
            and row["command_gate_pass"]
            and row["finite_difference_gate_pass"]
        )

    with (directory / "stability_references.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(directory / "return_maps.npz", **map_arrays)
    (directory / "return_map_index.json").write_text(json.dumps(map_index, indent=2))

    groups = {}
    group_fields = ("speed", "command_df", "step_width", "period", "gait", "perturbation_h")
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group in groups.items():
        valid = [row for row in group if row["reference_valid"]]
        record = dict(zip(group_fields, key))
        record.update({
            "references": len(group),
            "valid_references": len(valid),
            "reference_valid_rate": len(valid) / len(group),
            "cycle_gate_rate": float(np.mean([row["periodic_orbit_gate_pass"] for row in group])),
            "command_gate_rate": float(np.mean([row["command_gate_pass"] for row in group])),
            "finite_difference_gate_rate": float(np.mean(
                [row["finite_difference_gate_pass"] for row in group])),
        })
        for metric in metric_names:
            values = np.asarray([row[metric] for row in valid], dtype=float)
            record[f"{metric}_median"] = float(np.median(values)) if values.size else np.nan
            record[f"{metric}_q025"] = float(np.quantile(values, .025)) if values.size else np.nan
            record[f"{metric}_q975"] = float(np.quantile(values, .975)) if values.size else np.nan
        summary.append(record)
    with (directory / "stability_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    # The pre-registered central perturbation is h=0.05. A condition is usable
    # only when at least 90% of its independent phase references pass every gate.
    conditions = []
    condition_fields = ("speed", "command_df", "step_width", "period", "gait")
    for row in summary:
        if abs(row["perturbation_h"] - 0.05) > 1e-8:
            continue
        condition = {field: row[field] for field in condition_fields}
        condition.update(row)
        condition["condition_valid"] = int(row["reference_valid_rate"] >= 0.90)
        conditions.append(condition)
    if conditions:
        with (directory / "paper_conditions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(conditions[0]))
            writer.writeheader()
            writer.writerows(conditions)
    claims = _paper_claim_summary(conditions)
    (directory / "paper_claims.json").write_text(
        json.dumps(claims, indent=2).replace("NaN", "null")
    )
    return summary
