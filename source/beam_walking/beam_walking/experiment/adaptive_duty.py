"""Definitions for the separate adaptive duty-factor experiment.

This module deliberately leaves the paper-facing command strata in
``protocol.py`` unchanged.  The adaptive experiment trains both gaits across
the same duty-factor range so a later selector has meaningful choices for walk
as well as trot.
"""

import torch

from .protocol import (
    ANCHOR_FRACTION,
    CONTROL_DT,
    CORE_CONTROL_STEPS,
    CYCLE_STEPS,
    DUTY_RANGE,
    FOUNDATION_CONTROL_STEPS,
    GAITS,
    MIN_SWING_STEPS,
    PERIOD,
    PERIOD_TICKS,
    SPEED_ANCHORS,
    SPEED_RANGE,
    STEP_WIDTH_ANCHORS,
    STEP_WIDTH_RANGE,
)

ADAPTIVE_DUTY_LEVELS = (.50, .55, .60, .65, .70, .75)
ADAPTIVE_TRAINING_NUM_ENVS = 3072
ADAPTIVE_TRAINING_ITERATIONS = 1800
ADAPTIVE_COMMAND_STRATA = tuple(
    (gait, duty)
    for gait in range(len(GAITS))
    for duty in ADAPTIVE_DUTY_LEVELS
)
ADAPTIVE_TASK_FILES = (
    "source/beam_walking/beam_walking/experiment/task.py",
    "source/beam_walking/beam_walking/experiment/protocol.py",
    "source/beam_walking/beam_walking/experiment/adaptive_duty.py",
    "source/beam_walking/beam_walking/experiment/adaptive_task.py",
)
BASE_COMPATIBLE_TASK_FILES = ADAPTIVE_TASK_FILES[:2]
ADAPTIVE_TRAINING_SOURCE_FILES = ADAPTIVE_TASK_FILES + (
    "scripts/adaptive_duty_experiment.py",
    "scripts/gpu_capacity.py",
    "scripts/watch_training.py",
)


def files_sha256(root, relative_paths):
    """Hash an ordered collection of repository-relative source files."""
    from hashlib import sha256
    from pathlib import Path

    root = Path(root)
    return sha256(b"".join((root / path).read_bytes()
                           for path in relative_paths)).hexdigest()


def adaptive_lineage_id(training_source_sha256, seed):
    """Bind a fresh adaptive checkpoint to exact source and training seed."""
    from hashlib import sha256

    return sha256(
        f"{training_source_sha256}:{int(seed)}:fresh_adaptive_duty_v1".encode()
    ).hexdigest()


def feasible_duty_upper(period_ticks):
    """Return the largest DF that preserves five requested swing ticks."""
    if torch.is_tensor(period_ticks):
        ticks = period_ticks.to(dtype=torch.float32)
        return torch.minimum(
            torch.full_like(ticks, DUTY_RANGE[1]),
            1.0 - MIN_SWING_STEPS / ticks,
        )
    ticks = int(period_ticks)
    if ticks not in PERIOD_TICKS:
        raise ValueError("Period ticks must be in the supported 18--27 range")
    return min(DUTY_RANGE[1], 1.0 - MIN_SWING_STEPS / ticks)


def _balanced_indices(count, levels, device):
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    start = torch.randint(levels, (1,), device=device)
    values = (torch.arange(count, device=device) + start) % levels
    return values[torch.randperm(count, device=device)]


def sample_adaptive_training_commands(count, common_control_step, device="cpu"):
    """Sample balanced walk/trot commands across the full feasible DF range."""
    if count < 0:
        raise ValueError("Command count must be nonnegative")
    values = torch.empty((count, 5), device=device)
    period_ticks = torch.full(
        (count,), CYCLE_STEPS, device=device, dtype=torch.long)
    strata = torch.tensor(ADAPTIVE_COMMAND_STRATA, device=device)

    if common_control_step < FOUNDATION_CONTROL_STEPS:
        stratum = _balanced_indices(count, len(ADAPTIVE_COMMAND_STRATA), device)
        values[:] = torch.tensor([.30, .625, .30, PERIOD, 0.], device=device)
        values[:, 4] = strata[stratum, 0]
        values[:, 1] = strata[stratum, 1]
        return values, period_ticks, 0

    core_count = (
        count if common_control_step < CORE_CONTROL_STEPS
        else round(ANCHOR_FRACTION * count)
    )
    stratum = _balanced_indices(core_count, len(ADAPTIVE_COMMAND_STRATA), device)
    if core_count:
        values[:core_count, 3] = PERIOD
        values[:core_count, 4] = strata[stratum, 0]
        values[:core_count, 1] = strata[stratum, 1]
        combinations = len(SPEED_ANCHORS) * len(STEP_WIDTH_ANCHORS)
        for stratum_index in range(len(ADAPTIVE_COMMAND_STRATA)):
            ids = (stratum == stratum_index).nonzero().flatten()
            combo = _balanced_indices(len(ids), combinations, device)
            width = combo % len(STEP_WIDTH_ANCHORS)
            speed = combo // len(STEP_WIDTH_ANCHORS)
            values[ids, 0] = torch.tensor(SPEED_ANCHORS, device=device)[speed]
            values[ids, 2] = torch.tensor(
                STEP_WIDTH_ANCHORS, device=device)[width]

    remaining = count - core_count
    if remaining:
        target = slice(core_count, count)
        stratum = _balanced_indices(
            remaining, len(ADAPTIVE_COMMAND_STRATA), device)
        values[target, 0] = (
            SPEED_RANGE[0]
            + (SPEED_RANGE[1] - SPEED_RANGE[0])
            * torch.rand(remaining, device=device)
        )
        period_ticks[target] = torch.randint(
            min(PERIOD_TICKS), max(PERIOD_TICKS) + 1,
            (remaining,), device=device,
        )
        values[target, 3] = period_ticks[target] * CONTROL_DT
        values[target, 4] = strata[stratum, 0]

        upper = feasible_duty_upper(period_ticks[target])
        values[target, 1] = DUTY_RANGE[0] + (
            upper - DUTY_RANGE[0]) * torch.rand(remaining, device=device)
        values[target, 2] = (
            STEP_WIDTH_RANGE[0]
            + (STEP_WIDTH_RANGE[1] - STEP_WIDTH_RANGE[0])
            * torch.rand(remaining, device=device)
        )
        order = torch.randperm(count, device=device)
        values, period_ticks = values[order], period_ticks[order]

    return (
        values,
        period_ticks,
        1 if common_control_step < CORE_CONTROL_STEPS else 2,
    )
