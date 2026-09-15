"""Single-gait specialists commanded over narrow stance widths, 0.05-0.30 m.

Same curriculum, duty anchors, period ranges, swing bounds, PPO and plant as
the gait specialists (unified_specialist.py), with three deliberate changes:

* stance width is sampled over 0.05-0.30 m with anchors every 0.05 m;
* the lateral foot-placement reward uses a 3 cm tolerance instead of 10 cm, so
  the reward still separates neighbouring widths across the narrower range;
* self-collision is enabled, because at 0.05 m the feet are only millimetres
  apart and interpenetrating legs would otherwise go unpenalised.

No domain randomization is used. Randomized actuator delay and motor strength
taught earlier policies to touch down early, which compressed the realized duty
factor (0.50 commanded ran at 0.58-0.67). The width observation keeps the
0.10-0.50 m normalization of every other policy, so 0.05 m maps just below -1,
as in the earlier narrow-stance lineage.
"""

from hashlib import sha256

import torch

from .protocol import (
    ANCHOR_FRACTION,
    CONTROL_DT,
    CORE_CONTROL_STEPS,
    FOUNDATION_CONTROL_STEPS,
    GAITS,
    PERIOD,
    PERIOD_TICKS,
    SPEED_ANCHORS,
    SPEED_RANGE,
)
from .unified_gait import (
    GAIT_MIN_PERIOD_TICKS,
    TROT_DUTY_BINS,
    TROT_DUTY_LEVELS,
    WALK_DUTY_BINS,
    WALK_DUTY_LEVELS,
    files_sha256,
    max_duty_for_period,
)

NARROW_WIDTH_RANGE = (.05, .30)
NARROW_WIDTH_ANCHORS = (.05, .10, .15, .20, .25, .30)
NARROW_WIDTH_TOLERANCE_M = .03
NARROW_FOUNDATION_WIDTH = .15
NARROW_TRAINING_NUM_ENVS = 3072
NARROW_TRAINING_ITERATIONS = 1800
NARROW_TRAINING_SEED = 5
NARROW_SCHEMA = "narrow_specialist_training_v1"
NARROW_TASK_FILES = (
    "source/beam_walking/beam_walking/experiment/task.py",
    "source/beam_walking/beam_walking/experiment/protocol.py",
    "source/beam_walking/beam_walking/experiment/unified_gait.py",
    "source/beam_walking/beam_walking/experiment/narrow_specialist.py",
    "source/beam_walking/beam_walking/experiment/narrow_specialist_task.py",
)
NARROW_TRAINING_SOURCE_FILES = NARROW_TASK_FILES + (
    "scripts/narrow_specialist_experiment.py",
    "scripts/gpu_capacity.py",
    "scripts/watch_training.py",
)
GAIT_DUTY = {"trot": (TROT_DUTY_LEVELS, TROT_DUTY_BINS),
             "walk": (WALK_DUTY_LEVELS, WALK_DUTY_BINS)}


def narrow_extension_record():
    """Provenance block read by evaluate_policy.py to apply the same settings."""
    return {
        "min_step_width": NARROW_WIDTH_RANGE[0],
        "max_step_width": NARROW_WIDTH_RANGE[1],
        "width_tolerance_sigma_m": NARROW_WIDTH_TOLERANCE_M,
        "observation_normalization_range": [.10, .50],
        "self_collisions": True,
        "domain_randomization": False,
    }


def narrow_lineage_id(training_source_sha256, seed, gait):
    if gait not in GAITS:
        raise ValueError(f"Unknown gait: {gait}")
    return sha256(
        f"{training_source_sha256}:{int(seed)}:{gait}:fresh_narrow_specialist_v1".encode()
    ).hexdigest()


def _balanced(count, levels, device):
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    start = torch.randint(levels, (1,), device=device)
    values = (torch.arange(count, device=device) + start) % levels
    return values[torch.randperm(count, device=device)]


def sample_narrow_commands(gait, count, common_control_step, device="cpu"):
    """Staged single-gait commands over narrow stance widths.

    Stage 0: speed 0.30, width 0.15, period 0.48; duty anchors balanced.
    Stage 1: balanced speed x width anchors within every duty anchor, 0.48 s.
    Stage 2: 75% anchors as in stage 1, 25% continuous speed, width, period and
    duty, with duty drawn from the anchor's bin and capped by the swing bound.
    """
    if gait not in GAITS:
        raise ValueError(f"Unknown gait: {gait}")
    if count < 0:
        raise ValueError("Command count must be nonnegative")
    gait_id = GAITS.index(gait)
    levels, bins = GAIT_DUTY[gait]
    values = torch.empty((count, 5), device=device)
    period_ticks = torch.full(
        (count,), round(PERIOD / CONTROL_DT), device=device, dtype=torch.long)
    duty_index = _balanced(count, len(levels), device)
    duty = torch.tensor(levels, device=device)[duty_index]

    if common_control_step < FOUNDATION_CONTROL_STEPS:
        values[:] = torch.tensor(
            [.30, levels[0], NARROW_FOUNDATION_WIDTH, PERIOD, float(gait_id)],
            device=device)
        values[:, 1] = duty
        return values, period_ticks, 0

    core = (count if common_control_step < CORE_CONTROL_STEPS
            else round(ANCHOR_FRACTION * count))
    if core:
        ids = torch.arange(core, device=device)
        values[ids, 1] = duty[ids]
        values[ids, 3] = PERIOD
        values[ids, 4] = float(gait_id)
        widths = torch.tensor(NARROW_WIDTH_ANCHORS, device=device)
        speeds = torch.tensor(SPEED_ANCHORS, device=device)
        combinations = len(SPEED_ANCHORS) * len(NARROW_WIDTH_ANCHORS)
        for level in range(len(levels)):
            member = ids[duty_index[ids] == level]
            combo = _balanced(len(member), combinations, device)
            values[member, 0] = speeds[combo // len(NARROW_WIDTH_ANCHORS)]
            values[member, 2] = widths[combo % len(NARROW_WIDTH_ANCHORS)]

    remaining = count - core
    if remaining:
        tail = torch.arange(core, count, device=device)
        values[tail, 0] = SPEED_RANGE[0] + (SPEED_RANGE[1] - SPEED_RANGE[0]) * torch.rand(
            remaining, device=device)
        values[tail, 2] = NARROW_WIDTH_RANGE[0] + (
            NARROW_WIDTH_RANGE[1] - NARROW_WIDTH_RANGE[0]) * torch.rand(remaining, device=device)
        minimum = GAIT_MIN_PERIOD_TICKS[gait_id]
        span = max(PERIOD_TICKS) - minimum + 1
        period_ticks[tail] = minimum + torch.floor(
            torch.rand(remaining, device=device) * span).long()
        values[tail, 3] = period_ticks[tail] * CONTROL_DT
        values[tail, 4] = float(gait_id)
        cap = max_duty_for_period(
            torch.full((remaining,), gait_id, device=device, dtype=torch.long),
            period_ticks[tail]).to(values.dtype)
        lower = torch.tensor([b[0] for b in bins], device=device)[duty_index[tail]]
        upper = torch.tensor([b[1] for b in bins], device=device)[duty_index[tail]]
        upper = torch.minimum(upper, cap)
        lower = torch.minimum(lower, upper)
        values[tail, 1] = lower + (upper - lower) * torch.rand(remaining, device=device)
        order = torch.randperm(count, device=device)
        values, period_ticks = values[order], period_ticks[order]

    return values, period_ticks, 1 if common_control_step < CORE_CONTROL_STEPS else 2


__all__ = [
    "NARROW_FOUNDATION_WIDTH", "NARROW_SCHEMA", "NARROW_TASK_FILES",
    "NARROW_TRAINING_ITERATIONS", "NARROW_TRAINING_NUM_ENVS",
    "NARROW_TRAINING_SEED", "NARROW_TRAINING_SOURCE_FILES",
    "NARROW_WIDTH_ANCHORS", "NARROW_WIDTH_RANGE", "NARROW_WIDTH_TOLERANCE_M",
    "files_sha256", "narrow_extension_record", "narrow_lineage_id",
    "sample_narrow_commands",
]
