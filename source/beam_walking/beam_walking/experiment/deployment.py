"""Frozen domain-randomization profile for the first Go2 hardware candidate."""

import hashlib
import json
from pathlib import Path

STIFFNESS_SCALE_RANGE = (0.60, 1.40)
DAMPING_SCALE_RANGE = (0.50, 1.50)
BASE_MASS_SCALE_RANGE = (0.80, 1.20)
BASE_COM_RANGE_M = {
    "x": (-0.04, 0.04),
    "y": (-0.03, 0.03),
    "z": (-0.02, 0.02),
}
STATIC_FRICTION_RANGE = (0.40, 1.50)
DYNAMIC_FRICTION_RANGE = (0.30, 1.20)
MOTOR_STRENGTH_SCALE_RANGE = (0.70, 1.10)
JOINT_FRICTION_RANGE = (0.0, 0.25)
JOINT_ARMATURE_RANGE = (0.0, 0.02)
ACTUATOR_DELAY_PHYSICS_STEPS = (0, 4)
PHYSICS_DT = 0.005
TRAINING_NUM_ENVS = 3072
TRAINING_ITERATIONS = 1800
STANCE_START_PROBABILITY = 0.10
# Used only to prepare a supported reset pose; sampled gains are restored before
# every policy rollout and checked by the deployment smoke.
CALIBRATION_STIFFNESS = 80.0
CALIBRATION_DAMPING = 3.0
CALIBRATION_QUIET_STEPS = 20
CALIBRATION_MAX_STEPS = 4000
STIFFNESS_EVALUATION_SCALES = (0.60, 0.80, 1.00, 1.20, 1.40)
DAMPING_EVALUATION_SCALES = (0.50, 0.75, 1.00, 1.25, 1.50)
DEPLOYMENT_PROFILE_SCHEMA = "go2_deployment_dr_v1"
DEPLOYMENT_SOURCE_PATHS = (
    "source/beam_walking/beam_walking/experiment/deployment.py",
    "source/beam_walking/beam_walking/experiment/deployment_task.py",
    "source/beam_walking/beam_walking/experiment/deployment_actuator.py",
)


def validate_gain_scales(kp_scale, kd_scale):
    """Validate deterministic evaluation gains without hiding extrapolation."""
    values = (float(kp_scale), float(kd_scale))
    if any(value <= 0 or value > 2 for value in values):
        raise ValueError("Gain scales must be in (0, 2]")
    return values


def deployment_profile():
    """Return JSON-safe provenance for the frozen deployment distribution."""
    return {
        "schema": DEPLOYMENT_PROFILE_SCHEMA,
        "motor_stiffness_scale": list(STIFFNESS_SCALE_RANGE),
        "motor_damping_scale": list(DAMPING_SCALE_RANGE),
        "base_mass_scale": list(BASE_MASS_SCALE_RANGE),
        "base_com_offset_m": {
            axis: list(bounds) for axis, bounds in BASE_COM_RANGE_M.items()
        },
        "static_friction": list(STATIC_FRICTION_RANGE),
        "dynamic_friction": list(DYNAMIC_FRICTION_RANGE),
        "motor_strength_scale": list(MOTOR_STRENGTH_SCALE_RANGE),
        "joint_friction": list(JOINT_FRICTION_RANGE),
        "joint_armature": list(JOINT_ARMATURE_RANGE),
        "actuator_delay_physics_steps": list(ACTUATOR_DELAY_PHYSICS_STEPS),
        "actuator_delay_s": [
            ACTUATOR_DELAY_PHYSICS_STEPS[0] * PHYSICS_DT,
            ACTUATOR_DELAY_PHYSICS_STEPS[1] * PHYSICS_DT,
        ],
        "observation_corruption": "isaaclab_go2_stock_bounded_uniform",
        "external_pushes": False,
        "terrain": "flat_ground",
        "policy_rate_hz": 50,
        "solver_velocity_iterations": 2,
        "external_forces_every_solver_iteration": True,
        "training_num_envs": TRAINING_NUM_ENVS,
        "training_iterations": TRAINING_ITERATIONS,
        "stance_start_probability": STANCE_START_PROBABILITY,
        "reset_calibration_stiffness": CALIBRATION_STIFFNESS,
        "reset_calibration_damping": CALIBRATION_DAMPING,
        "reset_calibration_quiet_steps": CALIBRATION_QUIET_STEPS,
        "reset_calibration_max_steps": CALIBRATION_MAX_STEPS,
        "reset_calibration_gains_restored_before_rollout": True,
    }


def deployment_profile_sha256():
    """Hash the exact frozen values independently of source formatting."""
    encoded = json.dumps(
        deployment_profile(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def deployment_training_source_hash(root, base_training_hash):
    """Bind a training run to the base task and all deployment implementation."""
    root = Path(root)
    digest = hashlib.sha256()
    digest.update(b"go2_deployment_dr_v1\0")
    digest.update(bytes.fromhex(base_training_hash))
    for relative in DEPLOYMENT_SOURCE_PATHS:
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()
