"""Registered command distribution for the separate high-duty walk study."""

import torch

from .protocol import (
    ANCHOR_FRACTION,
    CONTROL_DT,
    CORE_CONTROL_STEPS,
    FOUNDATION_CONTROL_STEPS,
    PERIOD,
    SPEED_ANCHORS,
    SPEED_RANGE,
    STEP_WIDTH_ANCHORS,
    STEP_WIDTH_RANGE,
)

HIGH_DUTY_WALK_LEVELS = (.75, .80, .85, .90)
HIGH_DUTY_WALK_PERIOD = .48
HIGH_DUTY_WALK_PERIOD_TICKS = 24
HIGH_DUTY_WALK_MIN_SWING_STEPS = 2
HIGH_DUTY_WALK_TRAINING_NUM_ENVS = 3072
HIGH_DUTY_WALK_TRAINING_ITERATIONS = 1800
HIGH_DUTY_WALK_TRAINING_SEED = 4
HIGH_DUTY_WALK_GRID_SEED = 5_000_000
HIGH_DUTY_WALK_VALIDATION_SEED = 6_000_000
HIGH_DUTY_WALK_TASK_FILES = (
    "source/beam_walking/beam_walking/experiment/task.py",
    "source/beam_walking/beam_walking/experiment/protocol.py",
    "source/beam_walking/beam_walking/experiment/high_duty_walk.py",
    "source/beam_walking/beam_walking/experiment/high_duty_walk_task.py",
)
HIGH_DUTY_WALK_TRAINING_SOURCE_FILES = HIGH_DUTY_WALK_TASK_FILES + (
    "scripts/high_duty_walk_experiment.py",
    "scripts/gpu_capacity.py",
    "scripts/watch_training.py",
)


def files_sha256(root, relative_paths):
    """Hash an ordered collection of repository-relative files."""
    from hashlib import sha256
    from pathlib import Path

    root = Path(root)
    return sha256(b"".join((root / path).read_bytes()
                           for path in relative_paths)).hexdigest()


def high_duty_walk_lineage_id(training_source_sha256, seed):
    """Bind a primary checkpoint to its exact source and seed."""
    from hashlib import sha256

    return sha256(
        f"{training_source_sha256}:{int(seed)}:fresh_high_duty_walk_v1".encode()
    ).hexdigest()


def _balanced_indices(count, levels, device):
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    start = torch.randint(levels, (1,), device=device)
    values = (torch.arange(count, device=device) + start) % levels
    return values[torch.randperm(count, device=device)]


def sample_high_duty_walk_commands(count, common_control_step, device="cpu"):
    """Sample walk commands at 0.48 s across duty factors 0.75--0.90."""
    if count < 0:
        raise ValueError("Command count must be nonnegative")
    values = torch.empty((count, 5), device=device)
    period_ticks = torch.full(
        (count,), HIGH_DUTY_WALK_PERIOD_TICKS, device=device,
        dtype=torch.long,
    )
    duty_levels = torch.tensor(HIGH_DUTY_WALK_LEVELS, device=device)

    if common_control_step < FOUNDATION_CONTROL_STEPS:
        duty = _balanced_indices(count, len(HIGH_DUTY_WALK_LEVELS), device)
        values[:] = torch.tensor(
            [.30, .75, .30, HIGH_DUTY_WALK_PERIOD, 1.], device=device)
        values[:, 1] = duty_levels[duty]
        return values, period_ticks, 0

    core_count = (
        count if common_control_step < CORE_CONTROL_STEPS
        else round(ANCHOR_FRACTION * count)
    )
    duty = _balanced_indices(core_count, len(HIGH_DUTY_WALK_LEVELS), device)
    if core_count:
        values[:core_count, 1] = duty_levels[duty]
        values[:core_count, 3] = HIGH_DUTY_WALK_PERIOD
        values[:core_count, 4] = 1.
        combinations = len(SPEED_ANCHORS) * len(STEP_WIDTH_ANCHORS)
        for duty_index in range(len(HIGH_DUTY_WALK_LEVELS)):
            ids = (duty == duty_index).nonzero().flatten()
            combo = _balanced_indices(len(ids), combinations, device)
            width = combo % len(STEP_WIDTH_ANCHORS)
            speed = combo // len(STEP_WIDTH_ANCHORS)
            values[ids, 0] = torch.tensor(SPEED_ANCHORS, device=device)[speed]
            values[ids, 2] = torch.tensor(STEP_WIDTH_ANCHORS, device=device)[width]

    remaining = count - core_count
    if remaining:
        target = slice(core_count, count)
        values[target, 0] = SPEED_RANGE[0] + (
            SPEED_RANGE[1] - SPEED_RANGE[0]
        ) * torch.rand(remaining, device=device)
        values[target, 1] = HIGH_DUTY_WALK_LEVELS[0] + (
            HIGH_DUTY_WALK_LEVELS[-1] - HIGH_DUTY_WALK_LEVELS[0]
        ) * torch.rand(remaining, device=device)
        values[target, 2] = STEP_WIDTH_RANGE[0] + (
            STEP_WIDTH_RANGE[1] - STEP_WIDTH_RANGE[0]
        ) * torch.rand(remaining, device=device)
        values[target, 3] = HIGH_DUTY_WALK_PERIOD
        values[target, 4] = 1.
        order = torch.randperm(count, device=device)
        values, period_ticks = values[order], period_ticks[order]

    return (
        values,
        period_ticks,
        1 if common_control_step < CORE_CONTROL_STEPS else 2,
    )


assert PERIOD == HIGH_DUTY_WALK_PERIOD
assert round(HIGH_DUTY_WALK_PERIOD / CONTROL_DT) == HIGH_DUTY_WALK_PERIOD_TICKS
assert ((1 - HIGH_DUTY_WALK_LEVELS[-1]) * HIGH_DUTY_WALK_PERIOD_TICKS
        >= HIGH_DUTY_WALK_MIN_SWING_STEPS)
