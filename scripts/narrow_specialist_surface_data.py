"""Surface-grid definitions for the narrow-stance specialists (0.05-0.30 m).

Everything is inherited from the specialist definitions except the stance-width
levels, the checkpoint binding, the task identity and the environment factory.
The checkpoint is supplied with --checkpoint and its gait read from provenance.
"""
from functools import partial
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))
from beam_walking.experiment.narrow_specialist import (  # noqa: E402
    NARROW_SCHEMA, NARROW_TASK_FILES, NARROW_TRAINING_ITERATIONS,
    NARROW_TRAINING_NUM_ENVS, NARROW_TRAINING_SEED, NARROW_TRAINING_SOURCE_FILES,
    NARROW_WIDTH_ANCHORS, NARROW_WIDTH_RANGE, files_sha256, narrow_lineage_id,
)
from specialist_surface_data import read_training, specialist_gait  # noqa: E402,F401
from unified_surface_data import (  # noqa: E402,F401
    COMPLETE_SCHEMA, DUTIES, DUTIES_BY_GAIT, GAITS, GRID_SEED, PERIOD,
    PERIODS_BY_GAIT, SCHEMA, SMOKE_SEED, SPEEDS, SWEEP_SEED, TRIALS,
    VALIDATION_SEED, feasible_duties, sha256, summarize, validate_measurement,
)

# Training anchors are 0.50/0.625/0.75 for trot, with duty drawn continuously
# over 0.50-0.75 during training. Sweeps and the selector use candidates every
# 0.05 for both gaits.
TRAINING_DUTIES_BY_GAIT = DUTIES_BY_GAIT
DUTIES_BY_GAIT = {"trot": (.50, .55, .60, .65, .70, .75), "walk": (.75, .80, .85, .90)}
DUTIES = tuple(sorted(set(DUTIES_BY_GAIT["trot"]) | set(DUTIES_BY_GAIT["walk"])))


def feasible_duties(gait, period):
    """Candidate duty factors of a gait that respect its swing bound at a period."""
    import torch
    from beam_walking.experiment.protocol import CONTROL_DT
    from beam_walking.experiment.unified_gait import max_duty_for_period
    ticks = round(period / CONTROL_DT)
    cap = float(max_duty_for_period(torch.tensor([GAITS.index(gait)]), torch.tensor([ticks])))
    return tuple(d for d in DUTIES_BY_GAIT[gait] if d <= cap + 1e-6)


WIDTHS = tuple(NARROW_WIDTH_ANCHORS)
CHECKPOINT = None
CHECKPOINT_SHA256 = None
TASK_FILES = NARROW_TASK_FILES
SOURCE_FILES = TASK_FILES + tuple(
    "source/beam_walking/beam_walking/experiment/" + name
    for name in ("analysis.py", "stability.py", "duty_grid.py")
) + tuple("scripts/" + name for name in (
    "collect_policy_surfaces.py", "policy_surface_data.py",
    "unified_surface_data.py", "specialist_surface_data.py",
    "narrow_specialist_surface_data.py", "evaluation_capacity.py", "gpu_capacity.py"))
TRAINING_SEED = NARROW_TRAINING_SEED
TRAINING_SCHEMA = NARROW_SCHEMA
MANIFEST_EXTRA = dict(
    scientific_role="narrow_stance_single_gait_specialist_surface",
    task_variant="specialist",
    width_range_m=list(NARROW_WIDTH_RANGE),
    width_levels_m=list(WIDTHS),
    trot_training_duties=list(TRAINING_DUTIES_BY_GAIT["trot"]),
    walk_training_duties=list(TRAINING_DUTIES_BY_GAIT["walk"]),
    selector_candidate_duties={g: list(DUTIES_BY_GAIT[g]) for g in GAITS},
    walk_unseen_duties=[],
)


def source_hash():
    return hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in SOURCE_FILES)).hexdigest()


def conditions(smoke=False):
    if smoke:
        return [(g, .30, PERIOD, .15, d) for g in GAITS for d in DUTIES_BY_GAIT[g]]
    return [(g, v, PERIOD, w, d) for g in GAITS for v in SPEEDS
            for w in WIDTHS for d in DUTIES_BY_GAIT[g]]


def period_sweep_conditions():
    return [(g, v, p, w, d) for g in GAITS for p in PERIODS_BY_GAIT[g]
            for v in SPEEDS for w in WIDTHS for d in feasible_duties(g, p)]


def accepts_training(training):
    """Fresh narrow specialist, or a push fine-tune of one."""
    from beam_walking.experiment.narrow_specialist_push import PUSH_SCHEMA
    if training.get("schema") == PUSH_SCHEMA:
        return (training.get("fresh_training") is False
                and training.get("warm_start") is True
                and training.get("external_pushes") is True)
    return (training.get("seed") == TRAINING_SEED
            and training.get("fresh_training") is True)


def verify_training(training, saved):
    from beam_walking.experiment.narrow_specialist_push import (
        PUSH_SCHEMA, push_lineage_valid)
    if training.get("schema") == PUSH_SCHEMA:
        specialist_gait(training)
        if not push_lineage_valid(training, saved, ROOT):
            raise ValueError("Checkpoint is not a verified narrow-specialist push fine-tune")
        return
    gait = specialist_gait(training)
    source = files_sha256(ROOT, NARROW_TRAINING_SOURCE_FILES)
    expected = narrow_lineage_id(source, training.get("seed", -1), gait)
    if (training.get("schema") != TRAINING_SCHEMA
            or training.get("mode") != "train"
            or training.get("fresh_training") is not True
            or training.get("primary_training_protocol") is not True
            or training.get("seed") != TRAINING_SEED
            or training.get("training_source_sha256") != source
            or training.get("training_lineage_id") != expected
            or saved.get("training_lineage_id") != expected
            or training.get("training_num_envs") != NARROW_TRAINING_NUM_ENVS
            or training.get("training_iterations_requested") != NARROW_TRAINING_ITERATIONS):
        raise ValueError("Checkpoint is not the final fresh narrow-stance specialist")


def env_classes(training=None):
    from beam_walking.experiment.narrow_specialist_task import (
        NarrowSpecialistEnv, NarrowSpecialistEnvCfg, NarrowSpecialistPPORunnerCfg)
    gait = specialist_gait(training) if training is not None else "trot"
    return (NarrowSpecialistEnv, partial(NarrowSpecialistEnvCfg, specialist_gait=gait),
            NarrowSpecialistPPORunnerCfg)
