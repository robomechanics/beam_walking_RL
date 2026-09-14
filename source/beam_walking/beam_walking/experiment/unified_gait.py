"""Registered command distribution for the unified trot/walk controller.

One policy is trained on both gaits under one period distribution. The gaits
keep their own duty-factor ranges: trot covers 0.50--0.75 as in the paper
protocol, walk covers the high-duty 0.75--0.90 range from the separate
high-duty walk study. Period, speed, and width are sampled identically for
both gaits, except that the walk period floor follows from its swing bound.
"""

import torch

from .protocol import (
    ANCHOR_FRACTION,
    CONTROL_DT,
    CORE_CONTROL_STEPS,
    DUTY_ANCHORS,
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

TROT_DUTY_LEVELS = DUTY_ANCHORS                      # .50, .625, .75
WALK_DUTY_LEVELS = (.75, .80, .85, .90)
TROT_DUTY_RANGE = DUTY_RANGE                         # (.50, .75)
WALK_DUTY_RANGE = (WALK_DUTY_LEVELS[0], WALK_DUTY_LEVELS[-1])
# Trot keeps the paper protocol's 0.10 s (five-tick) swing bound. Walk keeps
# the high-duty study's two-tick bound, which is what makes DF 0.90 realizable.
TROT_MIN_SWING_STEPS = MIN_SWING_STEPS               # 5
WALK_MIN_SWING_STEPS = 2
GAIT_MIN_SWING_STEPS = (TROT_MIN_SWING_STEPS, WALK_MIN_SWING_STEPS)
# Both gaits share the 0.36--0.54 s protocol range. Walk needs at least 20
# ticks so that DF 0.90 keeps two swing ticks; trot uses the full range.
TROT_PERIOD_TICKS = (min(PERIOD_TICKS), max(PERIOD_TICKS))   # 18..27
WALK_PERIOD_TICKS = (20, max(PERIOD_TICKS))                  # 20..27
GAIT_MIN_PERIOD_TICKS = (TROT_PERIOD_TICKS[0], WALK_PERIOD_TICKS[0])
# Equal-probability Voronoi bins around each gait's anchors, so continuous
# sampling preserves the balanced stratum exposure of the anchor stage.
TROT_DUTY_BINS = ((.50, .5625), (.5625, .6875), (.6875, .75))
WALK_DUTY_BINS = ((.75, .775), (.775, .825), (.825, .875), (.875, .90))
# Paper-facing strata: (gait id, duty). Gaits are balanced 50/50 first, then
# duty levels are balanced within each gait.
UNIFIED_STRATA = tuple(
    (GAITS.index("trot"), duty) for duty in TROT_DUTY_LEVELS
) + tuple(
    (GAITS.index("walk"), duty) for duty in WALK_DUTY_LEVELS
)

UNIFIED_TRAINING_NUM_ENVS = 3072
UNIFIED_TRAINING_ITERATIONS = 1800
UNIFIED_TRAINING_SEED = 5
UNIFIED_TASK_FILES = (
    "source/beam_walking/beam_walking/experiment/task.py",
    "source/beam_walking/beam_walking/experiment/protocol.py",
    "source/beam_walking/beam_walking/experiment/unified_gait.py",
    "source/beam_walking/beam_walking/experiment/unified_gait_task.py",
)
UNIFIED_TRAINING_SOURCE_FILES = UNIFIED_TASK_FILES + (
    "scripts/unified_gait_experiment.py",
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


def unified_lineage_id(training_source_sha256, seed):
    """Bind a primary checkpoint to its exact source and seed."""
    from hashlib import sha256

    return sha256(
        f"{training_source_sha256}:{int(seed)}:fresh_unified_gait_v1".encode()
    ).hexdigest()


def max_duty_for_period(gait, period_ticks):
    """Largest feasible duty factor per gait at the given period ticks."""
    gait = gait.long()
    upper = torch.tensor(
        [TROT_DUTY_RANGE[1], WALK_DUTY_RANGE[1]],
        device=period_ticks.device)[gait]
    min_swing = torch.tensor(
        GAIT_MIN_SWING_STEPS, device=period_ticks.device,
        dtype=period_ticks.dtype)[gait]
    return torch.minimum(upper, 1. - min_swing / period_ticks)


def _balanced_indices(count, levels, device):
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    start = torch.randint(levels, (1,), device=device)
    values = (torch.arange(count, device=device) + start) % levels
    return values[torch.randperm(count, device=device)]


def _balanced_strata(count, device):
    """Gait balanced 50/50; duty levels balanced within each gait."""
    gait = _balanced_indices(count, len(GAITS), device)
    duty = torch.empty(count, device=device)
    for gait_id, levels in ((0, TROT_DUTY_LEVELS), (1, WALK_DUTY_LEVELS)):
        ids = (gait == gait_id).nonzero().flatten()
        picks = _balanced_indices(len(ids), len(levels), device)
        duty[ids] = torch.tensor(levels, device=device)[picks]
    return gait, duty


def _anchor_fill(values, ids, gait, duty, device):
    """Balanced speed/width anchors within every gait/duty stratum."""
    values[ids, 1] = duty[ids]
    values[ids, 3] = PERIOD
    values[ids, 4] = gait[ids].to(values.dtype)
    combinations = len(SPEED_ANCHORS) * len(STEP_WIDTH_ANCHORS)
    for gait_id, level in UNIFIED_STRATA:
        member = ids[(gait[ids] == gait_id)
                     & (torch.abs(duty[ids] - level) < 1e-9)]
        combo = _balanced_indices(len(member), combinations, device)
        width = combo % len(STEP_WIDTH_ANCHORS)
        speed = combo // len(STEP_WIDTH_ANCHORS)
        values[member, 0] = torch.tensor(SPEED_ANCHORS, device=device)[speed]
        values[member, 2] = torch.tensor(STEP_WIDTH_ANCHORS, device=device)[width]


def sample_unified_commands(count, common_control_step, device="cpu"):
    """Sample staged commands for both gaits with gait-specific duty ranges.

    Stage 0 (foundation): speed .30, width .30, period .48; gait/DF strata only.
    Stage 1 (core): balanced speed/width anchors at period .48.
    Stage 2: 75% anchors plus 25% continuous speed, width, period, and DF.
    Continuous periods span 18--27 ticks for trot and 20--27 ticks for walk;
    continuous DF is drawn from the stratum's Voronoi bin, capped by the gait's
    swing bound at the sampled period.
    """
    if count < 0:
        raise ValueError("Command count must be nonnegative")
    values = torch.empty((count, 5), device=device)
    period_ticks = torch.full(
        (count,), round(PERIOD / CONTROL_DT), device=device, dtype=torch.long)
    gait, duty = _balanced_strata(count, device)

    if common_control_step < FOUNDATION_CONTROL_STEPS:
        values[:] = torch.tensor([.30, .625, .30, PERIOD, 0.], device=device)
        values[:, 1] = duty
        values[:, 4] = gait.to(values.dtype)
        return values, period_ticks, 0

    core_count = (
        count if common_control_step < CORE_CONTROL_STEPS
        else round(ANCHOR_FRACTION * count)
    )
    if core_count:
        _anchor_fill(values, torch.arange(core_count, device=device),
                     gait, duty, device)

    remaining = count - core_count
    if remaining:
        target = torch.arange(core_count, count, device=device)
        tail_gait = gait[target]
        values[target, 0] = SPEED_RANGE[0] + (
            SPEED_RANGE[1] - SPEED_RANGE[0]) * torch.rand(remaining, device=device)
        values[target, 2] = STEP_WIDTH_RANGE[0] + (
            STEP_WIDTH_RANGE[1] - STEP_WIDTH_RANGE[0]
        ) * torch.rand(remaining, device=device)
        minimum_ticks = torch.tensor(
            GAIT_MIN_PERIOD_TICKS, device=device, dtype=torch.long)[tail_gait]
        span = max(PERIOD_TICKS) - minimum_ticks + 1
        period_ticks[target] = minimum_ticks + torch.floor(
            torch.rand(remaining, device=device) * span).long()
        values[target, 3] = period_ticks[target] * CONTROL_DT
        values[target, 4] = tail_gait.to(values.dtype)
        max_df = max_duty_for_period(tail_gait, period_ticks[target])
        lower = torch.empty(remaining, device=device)
        upper = torch.empty(remaining, device=device)
        for gait_id, levels, bins in ((0, TROT_DUTY_LEVELS, TROT_DUTY_BINS),
                                      (1, WALK_DUTY_LEVELS, WALK_DUTY_BINS)):
            for level, (low, high) in zip(levels, bins):
                member = (tail_gait == gait_id) & (
                    torch.abs(duty[target] - level) < 1e-9)
                lower[member] = low
                upper[member] = high
        upper = torch.minimum(upper, max_df)
        lower = torch.minimum(lower, upper)
        values[target, 1] = lower + (upper - lower) * torch.rand(
            remaining, device=device)
        order = torch.randperm(count, device=device)
        values, period_ticks = values[order], period_ticks[order]

    return (
        values, period_ticks,
        1 if common_control_step < CORE_CONTROL_STEPS else 2,
    )


assert round(PERIOD / CONTROL_DT) == 24
assert (1 - WALK_DUTY_LEVELS[-1]) * WALK_PERIOD_TICKS[0] >= WALK_MIN_SWING_STEPS - 1e-9
