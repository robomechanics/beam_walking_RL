"""CPU-only definitions for exploratory surfaces of the original paper policy."""
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))
from beam_walking.experiment.duty_grid import summarize_condition
from beam_walking.experiment.protocol import GAIT_OFFSETS
from beam_walking.experiment.stability import (
    ORBITAL_AUGMENTED_INDICES, STATE_SCALES, state_delta,
)

CHECKPOINT = ROOT / "results/paper_ppo_forcefix_nominalgains_seed2_3072_20260911/model_1799.pt"
CHECKPOINT_SHA256 = "ef5dae663b9650a3197e589b406101d4a450282e1e358c2af895d0505912127a"
TASK_FILES = tuple("source/beam_walking/beam_walking/experiment/" + name
                   for name in ("task.py", "protocol.py"))
SOURCE_FILES = TASK_FILES + tuple(
    "source/beam_walking/beam_walking/experiment/" + name
    for name in ("analysis.py", "stability.py", "duty_grid.py")
) + tuple("scripts/" + name for name in (
    "collect_policy_surfaces.py", "policy_surface_data.py",
    "make_policy_surfaces.py", "run_old_policy_surfaces.py",
    "evaluation_capacity.py", "gpu_capacity.py"))
WIDTHS = (.10, .20, .30, .40, .50)
DUTIES = (.50, .625, .75)
SPEEDS = (.25, .30, .35, .40)
GAITS = ("trot", "walk")
TRIALS = 32
PERIOD = .48
SMOKE_SEED = 1_300_000
GRID_SEED = 1_310_000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash():
    return hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in SOURCE_FILES)).hexdigest()


def conditions(smoke=False):
    if smoke:
        return [(g, .30, PERIOD, .30, d) for g in GAITS for d in DUTIES]
    return [(g, v, PERIOD, w, d) for g in GAITS for v in SPEEDS
            for w in WIDTHS for d in DUTIES]


def summarize(payload, condition, seed):
    gait, speed, period, width, duty = condition
    rows = summarize_condition(payload, step_width=width, speed=speed,
                               period=period, gait=gait, command_df=duty,
                               seed_start=seed)
    states = np.asarray(payload["phase_zero_states"])
    if states.shape != (len(rows), 5, 49) or not np.isfinite(states).all():
        raise ValueError("Expected five finite phase-zero states per trial")
    rms = np.zeros(len(rows))
    for k in range(4):
        delta = state_delta(states[:, k], states[:, k + 1]) / STATE_SCALES
        rms = np.maximum(rms, np.sqrt(np.mean(
            delta[:, ORBITAL_AUGMENTED_INDICES] ** 2, axis=1)))
    for row, value in zip(rows, rms):
        periodic = bool(value <= .02 and not row["any_failure"])
        row.update(cycle_rms=float(value), periodic_orbit_gate_pass=int(periodic),
                   energy_trial_valid=int(row["compliant"] and periodic),
                   out_of_training_support=int(gait == "walk" and duty != .75))
    return rows


def validate_measurement(payload, condition):
    """Check the physics/control chronology before recording completion."""
    gait, speed, period, width, duty = condition
    shapes = {"contacts": (TRIALS, 4, 24, 4), "desired": (TRIALS, 4, 24, 4),
              "substep_contacts": (TRIALS, 4, 24, 4, 4),
              "force_norm_200hz": (TRIALS, 4, 24, 4, 4),
              "applied_torque": (TRIALS, 4, 96, 12),
              "joint_velocity": (TRIALS, 4, 96, 12),
              "phase_zero_states": (TRIALS, 5, 49),
              "phase_ticks": (TRIALS, 4, 24)}
    for key, shape in shapes.items():
        value = np.asarray(payload[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid shape or nonfinite values in {key}: {value.shape}")
    if not np.array_equal(payload["force_norm_200hz"] > 5, payload["substep_contacts"]):
        raise ValueError("Force history chronology does not match 200 Hz contacts")
    if not np.array_equal(payload["substep_contacts"][:, :, :, -1], payload["contacts"]):
        raise ValueError("Final physics contacts do not match 50 Hz contacts")
    if not np.all(payload["phase_ticks"] == np.arange(24)[None, None, :]):
        raise ValueError("Measured phase sequence is not 0..23")
    # Integer quarter offsets avoid roundoff at stance/swing boundaries.
    phase = ((4 * np.arange(24)[:, None]
              + (4 * np.array(GAIT_OFFSETS[GAITS.index(gait)])).astype(int) * 24) % 96) / 96
    if not np.all(payload["desired"] == (phase < duty)[None, None]):
        raise ValueError("Desired contacts differ from archived phase and command")
    if float(payload["sample_dt"]) != .005 or np.asarray(payload["done"]).any():
        raise ValueError("Unexpected physics step or automatic reset")
