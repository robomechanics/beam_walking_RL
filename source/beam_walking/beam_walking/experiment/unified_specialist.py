"""Single-gait specialists trained under the unified protocol.

Two fresh policies, one per gait, sampled exactly like the unified policy
(docs/unified_gait_plan.md) except that every command carries one gait. Trot
keeps DF 0.50--0.75 over 0.36--0.54 s with a five-tick swing bound; walk keeps
DF 0.75--0.90 over 0.40--0.54 s with a two-tick swing bound. Everything else
(curriculum stages, anchors, 75/25 anchor split, plant, reward, PPO) is the
unified protocol, so the only difference between the two runs is the gait and
its duty range.
"""

import torch

from .protocol import GAITS
from .unified_gait import (
    UNIFIED_TRAINING_ITERATIONS,
    UNIFIED_TRAINING_NUM_ENVS,
    UNIFIED_TRAINING_SEED,
    files_sha256,
    sample_unified_commands,
)

SPECIALIST_TRAINING_NUM_ENVS = UNIFIED_TRAINING_NUM_ENVS
SPECIALIST_TRAINING_ITERATIONS = UNIFIED_TRAINING_ITERATIONS
SPECIALIST_TRAINING_SEED = UNIFIED_TRAINING_SEED
SPECIALIST_TASK_FILES = (
    "source/beam_walking/beam_walking/experiment/task.py",
    "source/beam_walking/beam_walking/experiment/protocol.py",
    "source/beam_walking/beam_walking/experiment/unified_gait.py",
    "source/beam_walking/beam_walking/experiment/unified_specialist.py",
    "source/beam_walking/beam_walking/experiment/unified_specialist_task.py",
)
SPECIALIST_TRAINING_SOURCE_FILES = SPECIALIST_TASK_FILES + (
    "scripts/unified_specialist_experiment.py",
    "scripts/gpu_capacity.py",
    "scripts/watch_training.py",
)
SPECIALIST_SCHEMA = "unified_specialist_training_v1"


def specialist_lineage_id(training_source_sha256, seed, gait):
    """Bind a primary checkpoint to its exact source, seed, and gait."""
    from hashlib import sha256

    if gait not in GAITS:
        raise ValueError(f"Unknown gait: {gait}")
    return sha256(
        f"{training_source_sha256}:{int(seed)}:{gait}:fresh_unified_specialist_v1".encode()
    ).hexdigest()


def sample_specialist_commands(gait, count, common_control_step, device="cpu"):
    """Unified commands restricted to one gait.

    The unified sampler balances gaits 50/50 and then balances duty anchors,
    speed/width anchors, and continuous draws within each gait. Drawing twice
    the count and keeping this gait's half reproduces the same per-gait
    distribution with every sample devoted to the specialist's gait.
    """
    if gait not in GAITS:
        raise ValueError(f"Unknown gait: {gait}")
    if count < 0:
        raise ValueError("Command count must be nonnegative")
    gait_id = GAITS.index(gait)
    kept_values, kept_ticks, stage = [], [], 0
    remaining = count
    while remaining > 0:
        values, ticks, stage = sample_unified_commands(
            2 * remaining + 8, common_control_step, device)
        mask = values[:, 4].long() == gait_id
        kept_values.append(values[mask][:remaining])
        kept_ticks.append(ticks[mask][:remaining])
        remaining -= int(mask.sum().clamp(max=remaining))
    if not kept_values:
        values = torch.empty((0, 5), device=device)
        ticks = torch.empty((0,), device=device, dtype=torch.long)
        _, _, stage = sample_unified_commands(0, common_control_step, device)
        return values, ticks, stage
    return torch.cat(kept_values), torch.cat(kept_ticks), stage


__all__ = [
    "SPECIALIST_SCHEMA", "SPECIALIST_TASK_FILES", "SPECIALIST_TRAINING_ITERATIONS",
    "SPECIALIST_TRAINING_NUM_ENVS", "SPECIALIST_TRAINING_SEED",
    "SPECIALIST_TRAINING_SOURCE_FILES", "files_sha256",
    "sample_specialist_commands", "specialist_lineage_id",
]
