"""Surface-grid definitions for the single-gait specialists.

Everything is inherited from the unified definitions (unified_surface_data.py);
only the checkpoint binding, task identity, and environment factory differ.
The checkpoint is supplied on the command line (``--checkpoint``) because there
is one per gait, and its gait is read from the training provenance.
"""
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))
from beam_walking.experiment.protocol import GAITS  # noqa: E402
from beam_walking.experiment.unified_specialist import (  # noqa: E402
    SPECIALIST_SCHEMA, SPECIALIST_TASK_FILES, SPECIALIST_TRAINING_ITERATIONS,
    SPECIALIST_TRAINING_NUM_ENVS, SPECIALIST_TRAINING_SEED,
    SPECIALIST_TRAINING_SOURCE_FILES, files_sha256, specialist_lineage_id,
)
from unified_surface_data import (  # noqa: E402,F401
    COMPLETE_SCHEMA, DUTIES, DUTIES_BY_GAIT, GRID_SEED, PERIOD, PERIODS_BY_GAIT,
    SCHEMA, SMOKE_SEED, SPEEDS, SWEEP_SEED, TRIALS, VALIDATION_SEED, WIDTHS,
    conditions, feasible_duties, period_sweep_conditions, sha256, summarize,
    validate_measurement,
)

# Bound by collect_policy_surfaces.py from --checkpoint before use.
CHECKPOINT = None
CHECKPOINT_SHA256 = None
TASK_FILES = SPECIALIST_TASK_FILES
SOURCE_FILES = TASK_FILES + tuple(
    "source/beam_walking/beam_walking/experiment/" + name
    for name in ("analysis.py", "stability.py", "duty_grid.py")
) + tuple("scripts/" + name for name in (
    "collect_policy_surfaces.py", "policy_surface_data.py",
    "unified_surface_data.py", "specialist_surface_data.py",
    "make_policy_surfaces.py", "evaluation_capacity.py", "gpu_capacity.py"))
TRAINING_SEED = SPECIALIST_TRAINING_SEED
TRAINING_SCHEMA = SPECIALIST_SCHEMA
MANIFEST_EXTRA = dict(
    scientific_role="unified_protocol_single_gait_specialist_surface",
    task_variant="specialist",
    trot_training_duties=list(DUTIES_BY_GAIT["trot"]),
    walk_training_duties=list(DUTIES_BY_GAIT["walk"]),
    walk_unseen_duties=[],
)


def source_hash():
    return hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in SOURCE_FILES)).hexdigest()


def specialist_gait(training):
    gait = training.get("specialist_gait")
    if gait not in GAITS or training.get("gaits") != [gait]:
        raise ValueError("Specialist training provenance must name exactly one gait")
    return gait


def verify_training(training, saved):
    """Bind the loaded checkpoint to a fresh single-gait specialist run."""
    gait = specialist_gait(training)
    training_source_sha256 = files_sha256(ROOT, SPECIALIST_TRAINING_SOURCE_FILES)
    expected = specialist_lineage_id(training_source_sha256, training.get("seed", -1), gait)
    if (training.get("schema") != TRAINING_SCHEMA
            or training.get("mode") != "train"
            or training.get("fresh_training") is not True
            or training.get("primary_training_protocol") is not True
            or training.get("seed") != TRAINING_SEED
            or training.get("training_source_sha256") != training_source_sha256
            or training.get("training_lineage_id") != expected
            or saved.get("training_lineage_id") != expected
            or training.get("training_num_envs") != SPECIALIST_TRAINING_NUM_ENVS
            or training.get("training_iterations_requested") != SPECIALIST_TRAINING_ITERATIONS):
        raise ValueError("Checkpoint is not the final fresh specialist policy")


def env_classes(training=None):
    """Isaac-dependent classes; the env config is bound to the specialist's gait."""
    from beam_walking.experiment.unified_specialist_task import (
        SpecialistEnv, SpecialistEnvCfg, SpecialistPPORunnerCfg)
    gait = specialist_gait(training) if training is not None else "trot"
    return SpecialistEnv, partial(SpecialistEnvCfg, specialist_gait=gait), SpecialistPPORunnerCfg


def read_training(checkpoint):
    return json.loads((Path(checkpoint).parent / "provenance.json").read_text())
