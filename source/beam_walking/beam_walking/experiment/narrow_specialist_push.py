"""Push-trained fine-tunes of the narrow-stance gait specialists.

Each specialist is warm-started from its final checkpoint and trained for the
same number of updates with the ramped base perturbation
(docs/perturbation.md) switched on; the command sampler, plant, reward, and
PPO settings are unchanged. Provenance binds the fine-tune to the parent
checkpoint hash, the gait, the seed, and this source bundle.
"""
from hashlib import sha256
import json
from pathlib import Path

from .protocol import GAITS
from .narrow_specialist import (
    NARROW_SCHEMA as SPECIALIST_SCHEMA,
    NARROW_TASK_FILES as SPECIALIST_TASK_FILES,
    NARROW_TRAINING_ITERATIONS as SPECIALIST_TRAINING_ITERATIONS,
    NARROW_TRAINING_NUM_ENVS as SPECIALIST_TRAINING_NUM_ENVS,
    files_sha256,
)

PUSH_SCHEMA = "narrow_specialist_push_finetune_v1"
PUSH_TRAINING_KIND = "narrow_specialist_push_finetune"
PUSH_TRAINING_NUM_ENVS = SPECIALIST_TRAINING_NUM_ENVS
# 1,200 updates: an earlier push fine-tune reached about 94% of its 1,800-update
# reward by update 1,200, and the shorter run saves time on both gaits.
PUSH_TRAINING_ITERATIONS = 1200
PUSH_TRAINING_SEED = 3   # Same seed the seed-2 push fine-tune used.
PUSH_TASK_FILES = SPECIALIST_TASK_FILES
PUSH_TRAINING_SOURCE_FILES = PUSH_TASK_FILES + (
    "source/beam_walking/beam_walking/experiment/perturbation.py",
    "source/beam_walking/beam_walking/experiment/perturbation_env.py",
    "source/beam_walking/beam_walking/experiment/narrow_specialist_push.py",
    "scripts/narrow_specialist_push_experiment.py",
    "scripts/gpu_capacity.py",
    "scripts/watch_training.py",
)


def push_lineage_id(training_source_sha256, seed, parent_checkpoint_sha256, gait):
    if gait not in GAITS:
        raise ValueError(f"Unknown gait: {gait}")
    return sha256(
        f"{training_source_sha256}:{int(seed)}:{parent_checkpoint_sha256}:{gait}:"
        f"narrow_specialist_push_finetune_v1".encode()).hexdigest()


def verify_parent(checkpoint, gait):
    """The parent must be the final fresh specialist checkpoint of this gait."""
    import torch

    checkpoint = Path(checkpoint).resolve()
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    provenance_path = checkpoint.parent / "provenance.json"
    parent = json.loads(provenance_path.read_text())
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    infos = state.get("infos") or {}
    if (parent.get("schema") != SPECIALIST_SCHEMA
            or parent.get("specialist_gait") != gait
            or parent.get("fresh_training") is not True
            or parent.get("primary_training_protocol") is not True
            or parent.get("training_lineage_id") != infos.get("training_lineage_id")
            or parent.get("task_sha256") != infos.get("task_sha256")
            or state.get("iter") != parent.get("training_iterations_requested", 0) - 1):
        raise ValueError("Parent is not the final fresh specialist checkpoint for this gait")
    return {
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": digest,
        "parent_provenance_sha256": sha256(provenance_path.read_bytes()).hexdigest(),
        "parent_training_lineage_id": parent["training_lineage_id"],
        "parent_task_sha256": parent["task_sha256"],
        "parent_training_seed": parent.get("seed"),
        "parent_checkpoint_iteration": int(state.get("iter")),
        "parent_common_step_counter": int(infos.get("common_step_counter", 0)),
        "parent_state": state,
    }


def push_lineage_valid(training, saved, root):
    """Evaluation-side check that a checkpoint is a verified push fine-tune."""
    root = Path(root)
    try:
        source = files_sha256(root, PUSH_TRAINING_SOURCE_FILES)
    except OSError:
        return False
    expected = push_lineage_id(source, training.get("seed", -1),
                               training.get("parent_checkpoint_sha256", ""),
                               training.get("specialist_gait", ""))
    return bool(
        training.get("schema") == PUSH_SCHEMA
        and training.get("training_kind") == PUSH_TRAINING_KIND
        and training.get("fresh_training") is False
        and training.get("warm_start") is True
        and training.get("initial_weights_match_parent") is True
        and training.get("external_pushes") is True
        and training.get("training_source_sha256") == source
        and training.get("training_lineage_id") == expected
        and saved.get("training_lineage_id") == expected
        and saved.get("training_kind") == PUSH_TRAINING_KIND
        and saved.get("parent_checkpoint_sha256") == training.get("parent_checkpoint_sha256")
        and training.get("checkpoint_selection_rule") == "final_requested_iteration")
