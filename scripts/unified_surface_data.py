"""CPU-only definitions for the unified seed-5 policy surface grid.

Same measurement, chronology checks, and summaries as the seed-2 exploratory
surfaces (policy_surface_data.py), applied to the unified trot/walk policy with
its gait-specific duty ranges: trot 0.50/0.625/0.75 and walk
0.75/0.80/0.85/0.90 at period 0.48 s.
"""
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))
from beam_walking.experiment.unified_gait import (  # noqa: E402
    TROT_DUTY_LEVELS, UNIFIED_TASK_FILES, UNIFIED_TRAINING_ITERATIONS,
    UNIFIED_TRAINING_NUM_ENVS, UNIFIED_TRAINING_SEED,
    UNIFIED_TRAINING_SOURCE_FILES, WALK_DUTY_LEVELS, files_sha256,
    unified_lineage_id,
)
from policy_surface_data import (  # noqa: E402
    GAITS, PERIOD, SPEEDS, TRIALS, WIDTHS, sha256, validate_measurement,
)
from policy_surface_data import summarize as _paper_summarize  # noqa: E402

CHECKPOINT = ROOT / "results/unified_gait_seed5_3072_20260914/model_1799.pt"
CHECKPOINT_SHA256 = sha256(CHECKPOINT) if CHECKPOINT.is_file() else None
TASK_FILES = UNIFIED_TASK_FILES
SOURCE_FILES = TASK_FILES + tuple(
    "source/beam_walking/beam_walking/experiment/" + name
    for name in ("analysis.py", "stability.py", "duty_grid.py")
) + tuple("scripts/" + name for name in (
    "collect_policy_surfaces.py", "policy_surface_data.py",
    "unified_surface_data.py", "make_policy_surfaces.py",
    "evaluation_capacity.py", "gpu_capacity.py"))
DUTIES_BY_GAIT = {"trot": tuple(TROT_DUTY_LEVELS), "walk": tuple(WALK_DUTY_LEVELS)}
DUTIES = tuple(sorted(set(TROT_DUTY_LEVELS) | set(WALK_DUTY_LEVELS)))
TRAINING_SEED = UNIFIED_TRAINING_SEED
TRAINING_SCHEMA = "unified_gait_training_v1"
SCHEMA = "unified_surface_v1"
COMPLETE_SCHEMA = "unified_surface_complete_v1"
SMOKE_SEED = 1_500_000
GRID_SEED = 1_510_000
MANIFEST_EXTRA = dict(
    scientific_role="unified_policy_surface_single_checkpoint",
    task_variant="unified",
    trot_training_duties=list(TROT_DUTY_LEVELS),
    walk_training_duties=list(WALK_DUTY_LEVELS),
    walk_unseen_duties=[],
)
__all__ = [
    "CHECKPOINT", "CHECKPOINT_SHA256", "COMPLETE_SCHEMA", "DUTIES",
    "DUTIES_BY_GAIT", "GAITS", "GRID_SEED", "MANIFEST_EXTRA", "PERIOD",
    "ROOT", "SCHEMA", "SMOKE_SEED", "SOURCE_FILES", "SPEEDS", "TASK_FILES",
    "TRAINING_SCHEMA", "TRAINING_SEED", "TRIALS", "WIDTHS", "conditions",
    "env_classes", "sha256", "source_hash", "summarize",
    "validate_measurement", "verify_training",
]


def source_hash():
    return hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in SOURCE_FILES)).hexdigest()


def conditions(smoke=False):
    if smoke:
        return [(g, .30, PERIOD, .30, d) for g in GAITS for d in DUTIES_BY_GAIT[g]]
    return [(g, v, PERIOD, w, d) for g in GAITS for v in SPEEDS
            for w in WIDTHS for d in DUTIES_BY_GAIT[g]]


def summarize(payload, condition, seed):
    """Paper summaries with training support defined by the unified ranges."""
    gait, _, _, _, duty = condition
    rows = _paper_summarize(payload, condition, seed)
    supported = DUTIES_BY_GAIT[gait]
    outside = int(not min(supported) - 1e-9 <= duty <= max(supported) + 1e-9)
    for row in rows:
        row["out_of_training_support"] = outside
    return rows


def verify_training(training, saved):
    """Bind the loaded checkpoint to the fresh unified seed-5 run."""
    training_source_sha256 = files_sha256(ROOT, UNIFIED_TRAINING_SOURCE_FILES)
    expected = unified_lineage_id(training_source_sha256, training.get("seed", -1))
    if (training.get("schema") != TRAINING_SCHEMA
            or training.get("mode") != "train"
            or training.get("fresh_training") is not True
            or training.get("primary_training_protocol") is not True
            or training.get("seed") != TRAINING_SEED
            or training.get("training_source_sha256") != training_source_sha256
            or training.get("training_lineage_id") != expected
            or saved.get("training_lineage_id") != expected
            or training.get("training_num_envs") != UNIFIED_TRAINING_NUM_ENVS
            or training.get("training_iterations_requested") != UNIFIED_TRAINING_ITERATIONS):
        raise ValueError("Checkpoint is not the final fresh unified seed-5 policy")


def env_classes():
    """Isaac-dependent classes; import only after the app is launched."""
    from beam_walking.experiment.unified_gait_task import (
        UnifiedEnv, UnifiedEnvCfg, UnifiedPPORunnerCfg)
    return UnifiedEnv, UnifiedEnvCfg, UnifiedPPORunnerCfg
