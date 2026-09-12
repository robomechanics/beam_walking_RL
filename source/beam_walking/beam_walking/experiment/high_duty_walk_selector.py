"""Offline labeling and selector policy for the high-duty walk study."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .duty_selector import (
    aggregate_candidates,
    bootstrap_target_intervals,
    select_targets,
    selector_state_sha256,
)
from .high_duty_walk import (
    HIGH_DUTY_WALK_GRID_SEED,
    HIGH_DUTY_WALK_LEVELS,
    HIGH_DUTY_WALK_PERIOD,
    HIGH_DUTY_WALK_VALIDATION_SEED,
)
from .protocol import CONTROL_DT, SPEED_RANGE, STEP_WIDTH_ANCHORS, STEP_WIDTH_RANGE

CONTEXT_FIELDS = ("step_width", "speed", "period", "gait")
REQUIRED_TRIAL_FIELDS = ("seed",) + CONTEXT_FIELDS + (
    "command_df", "compliant", "positive_mechanical_cot",
)
PRIMARY_SPEEDS = (.25, .30, .35, .40)
PRIMARY_TRIALS_PER_CANDIDATE = 32
SELECTOR_RUNTIME_FILES = (
    "source/beam_walking/beam_walking/experiment/protocol.py",
    "source/beam_walking/beam_walking/experiment/high_duty_walk.py",
    "source/beam_walking/beam_walking/experiment/duty_selector.py",
    "source/beam_walking/beam_walking/experiment/high_duty_walk_selector.py",
)


def selector_runtime_sha256(root):
    digest = hashlib.sha256()
    for relative in SELECTOR_RUNTIME_FILES:
        digest.update((Path(root) / relative).read_bytes())
    return digest.hexdigest()


def validate_context_values(step_width, speed):
    values = np.asarray([step_width, speed], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Selector context values must be finite")
    tolerance = 1e-6
    if not STEP_WIDTH_RANGE[0] - tolerance <= step_width <= STEP_WIDTH_RANGE[1] + tolerance:
        raise ValueError("Step width is outside the trained range")
    if not SPEED_RANGE[0] - tolerance <= speed <= SPEED_RANGE[1] + tolerance:
        raise ValueError("Speed is outside the trained range")
    return float(step_width), float(speed)


def read_trial_rows(path):
    path = Path(path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED_TRIAL_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Trial table is missing fields: {sorted(missing)}")
        rows, identities = [], set()
        for raw in reader:
            row = dict(raw)
            row["seed"] = int(raw["seed"])
            for field in ("step_width", "speed", "period", "command_df",
                          "positive_mechanical_cot"):
                row[field] = float(raw[field])
            token = raw["compliant"].strip().lower()
            if token not in {"0", "1", "false", "true", "no", "yes"}:
                raise ValueError(f"Invalid compliant value: {raw['compliant']}")
            row["compliant"] = token in {"1", "true", "yes"}
            if row["gait"] != "walk":
                raise ValueError("High-duty selector accepts walk trials only")
            row["step_width"], row["speed"] = validate_context_values(
                row["step_width"], row["speed"])
            if not np.isclose(row["period"], HIGH_DUTY_WALK_PERIOD, atol=1e-7):
                raise ValueError("High-duty selector period must be 0.48 s")
            if (not np.isfinite(row["command_df"])
                    or row["command_df"] < HIGH_DUTY_WALK_LEVELS[0] - 1e-7
                    or row["command_df"] > HIGH_DUTY_WALK_LEVELS[-1] + 1e-7):
                raise ValueError("Candidate DF is outside [0.75, 0.90]")
            energy = row["positive_mechanical_cot"]
            if np.isinf(energy) or (np.isfinite(energy) and energy < 0):
                raise ValueError("Mechanical CoT must be nonnegative or NaN")
            if row["compliant"] and not np.isfinite(energy):
                raise ValueError("A compliant trial must have finite mechanical CoT")
            identity = (
                row["step_width"], row["speed"], row["period"], row["gait"],
                row["command_df"], row["seed"],
            )
            if identity in identities:
                raise ValueError(f"Duplicate selector trial row: {identity}")
            identities.add(identity)
            rows.append(row)
    if not rows:
        raise ValueError("Trial table contains no rows")
    return rows


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_completed_grid(trials_path, *, require_primary=True):
    """Validate the manifest, archives, matched seeds, and completion hashes."""
    trials_path = Path(trials_path)
    directory = trials_path.parent
    manifest_path = directory / "grid_manifest.json"
    complete_path = directory / "GRID_COMPLETE"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise ValueError("Grid manifest and GRID_COMPLETE marker are required")
    manifest = json.loads(manifest_path.read_text())
    allowed = ({"high_duty_walk_grid_v1"} if require_primary else {
        "high_duty_walk_grid_v1", "high_duty_walk_grid_exploratory_v1"})
    if manifest.get("schema") not in allowed:
        raise ValueError("Unsupported high-duty walk grid schema")
    if (manifest.get("terrain") != "flat_ground"
            or manifest.get("terrain_width_input") is not False):
        raise ValueError("Grid must use flat-ground commanded step width")
    completion = json.loads(complete_path.read_text())
    expected_completion = (
        "high_duty_walk_grid_complete_v1"
        if manifest["schema"] == "high_duty_walk_grid_v1"
        else "high_duty_walk_grid_exploratory_complete_v1")
    if (completion.get("schema") != expected_completion
            or completion.get("manifest_file") != manifest_path.name
            or completion.get("manifest_sha256") != _sha256(manifest_path)
            or completion.get("trial_csv_file") != trials_path.name
            or completion.get("trial_csv_sha256") != _sha256(trials_path)):
        raise ValueError("Grid completion hashes do not match final files")
    conditions = manifest.get("conditions")
    count = manifest.get("trials_per_candidate")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Grid manifest contains no conditions")
    if not isinstance(count, int) or count < 8:
        raise ValueError("Grid has too few trials per candidate")
    for field in ("task_sha256", "checkpoint_sha256", "collector_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Grid manifest has invalid {field}")
    rows = read_trial_rows(trials_path)
    seed_start = manifest.get("evaluation_seed_start")
    expected = {
        (float(item["step_width"]), float(item["speed"]),
         float(item["period"]), str(item["gait"]),
         float(item["command_df"]), seed)
        for item in conditions
        for seed in range(seed_start, seed_start + count)
    }
    actual = {
        (row["step_width"], row["speed"], row["period"], row["gait"],
         row["command_df"], row["seed"])
        for row in rows
    }
    if len(actual) != len(rows) or actual != expected:
        raise ValueError("Trial CSV does not exactly match manifest conditions and seeds")
    filenames = [str(item.get("filename", "")) for item in conditions]
    if not all(filenames) or len(filenames) != len(set(filenames)):
        raise ValueError("Grid condition filenames must be unique")
    if sorted(path.name for path in directory.glob("high_duty_walk_grid_*.npz")) != sorted(filenames):
        raise ValueError("Grid archive set does not exactly match the manifest")
    archive_hashes = completion.get("archive_sha256")
    if not isinstance(archive_hashes, dict) or set(archive_hashes) != set(filenames):
        raise ValueError("Grid completion record has the wrong archive set")
    matched_reset = None
    for item in conditions:
        path = directory / item["filename"]
        if archive_hashes[item["filename"]] != _sha256(path):
            raise ValueError(f"Grid archive hash mismatch: {item['filename']}")
        with np.load(path, allow_pickle=False) as archive:
            for archive_field, manifest_field in (
                    ("task_sha256", "task_sha256"),
                    ("checkpoint_sha256", "checkpoint_sha256"),
                    ("collector_sha256", "collector_sha256")):
                if str(archive[archive_field]) != manifest[manifest_field]:
                    raise ValueError(f"Grid archive identity mismatch: {item['filename']}")
            expected_command = np.asarray([
                item["speed"], item["command_df"], item["step_width"],
                item["period"], 1,
            ])
            if (not np.allclose(
                    archive["command"], expected_command, rtol=0, atol=1e-7)
                    or not np.array_equal(
                        archive["seeds"], np.arange(seed_start, seed_start + count))):
                raise ValueError(f"Grid command/seed mismatch: {item['filename']}")
            reset = (archive["reset_plan"], archive["stance_start"],
                     archive["initial_root"], archive["initial_joints"])
            if tuple(value.shape for value in reset) != (
                    (count, 2), (count,), (count, 13), (count, 12)):
                raise ValueError(f"Grid reset evidence malformed: {item['filename']}")
            if matched_reset is None:
                matched_reset = tuple(value.copy() for value in reset)
            elif (not np.array_equal(reset[0], matched_reset[0])
                  or not np.array_equal(reset[1], matched_reset[1])
                  or not np.allclose(
                      reset[2], matched_reset[2], rtol=0, atol=1e-6)
                  or not np.allclose(
                      reset[3], matched_reset[3], rtol=0, atol=1e-6)):
                raise ValueError("Grid conditions do not share matched reset states")
    if require_primary:
        expected_conditions = {
            (width, speed, HIGH_DUTY_WALK_PERIOD, "walk", duty)
            for width in STEP_WIDTH_ANCHORS
            for speed in PRIMARY_SPEEDS
            for duty in HIGH_DUTY_WALK_LEVELS
        }
        actual_conditions = {identity[:5] for identity in expected}
        if count != PRIMARY_TRIALS_PER_CANDIDATE or actual_conditions != expected_conditions:
            raise ValueError("Primary high-duty grid factors are incomplete or changed")
        if (manifest.get("settle_cycles") != 12
                or manifest.get("measurement_cycles") != 4
                or seed_start != HIGH_DUTY_WALK_GRID_SEED
                or not np.isclose(manifest.get("stance_start_probability", -1), .10)
                or not np.isclose(manifest.get("topology_minimum", -1), .90)
                or not np.isclose(manifest.get("control_dt_s", -1), CONTROL_DT)
                or manifest.get("stance_start_sampling") != "seeded_independent_bernoulli"
                or manifest.get("stance_start_realized") != matched_reset[1].astype(int).tolist()
                or manifest.get("stance_start_realized_count") != int(matched_reset[1].sum())):
            raise ValueError("Primary high-duty grid protocol differs from registration")
    return rows, manifest, {
        "trial_csv_sha256": _sha256(trials_path),
        "grid_manifest_sha256": _sha256(manifest_path),
        "grid_complete_sha256": _sha256(complete_path),
    }


def context_tensor(rows, device="cpu"):
    return torch.tensor([
        [row["step_width"], row["speed"]] for row in rows
    ], dtype=torch.float32, device=device)


class HighDutyWalkSelector(torch.nn.Module):
    """Feedforward policy mapping [step width, speed] to duty factor."""

    def __init__(self, hidden_dims=(32, 32)):
        super().__init__()
        layers = []
        incoming = 2
        for width in hidden_dims:
            layers.extend((torch.nn.Linear(incoming, width), torch.nn.ELU()))
            incoming = width
        layers.append(torch.nn.Linear(incoming, len(HIGH_DUTY_WALK_LEVELS)))
        self.network = torch.nn.Sequential(*layers)

    @staticmethod
    def features(context):
        if context.ndim != 2 or context.shape[1] != 2:
            raise ValueError("Selector context must have shape [batch, 2]")
        for row in context.detach().cpu().numpy():
            validate_context_values(float(row[0]), float(row[1]))
        width = 2 * (context[:, 0:1] - STEP_WIDTH_RANGE[0]) / (
            STEP_WIDTH_RANGE[1] - STEP_WIDTH_RANGE[0]) - 1
        speed = 2 * (context[:, 1:2] - SPEED_RANGE[0]) / (
            SPEED_RANGE[1] - SPEED_RANGE[0]) - 1
        return torch.cat((width, speed), dim=1)

    def logits(self, context):
        return self.network(self.features(context))

    def forward(self, context):
        levels = torch.tensor(
            HIGH_DUTY_WALK_LEVELS, device=context.device, dtype=context.dtype)
        return levels[self.logits(context).argmax(dim=1)]


def fit_selector(targets, seed=0, epochs=3000, learning_rate=3e-3):
    if len(targets) < 2:
        raise ValueError("At least two valid contexts are required")
    torch.manual_seed(seed)
    model = HighDutyWalkSelector()
    context = context_tensor(targets)
    labels = torch.tensor([
        HIGH_DUTY_WALK_LEVELS.index(round(float(row["target_df"]), 2))
        for row in targets
    ], dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(epochs):
        prediction = model.logits(context)
        loss = torch.nn.functional.cross_entropy(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return model, history


@torch.inference_mode()
def predict_rows(model, rows):
    device = next(model.parameters()).device
    values = model(context_tensor(rows, device=device)).cpu().numpy()
    canonical = [min(HIGH_DUTY_WALK_LEVELS,
                     key=lambda level: abs(level - float(value)))
                 for value in values]
    return [dict(row, selected_df=value) for row, value in zip(rows, canonical)]


def context_is_supported(payload, row):
    key = (float(row["step_width"]), float(row["speed"]))
    return any((float(item["step_width"]), float(item["speed"])) == key
               for item in payload["supported_contexts"])


def predict_supported_rows(model, payload, rows, *, require_deployment_ready=True):
    if require_deployment_ready and payload.get("deployment_ready") is not True:
        raise ValueError("Selector has not passed fresh rollout validation")
    unsupported = [row for row in rows if not context_is_supported(payload, row)]
    if unsupported:
        raise ValueError(f"Selector abstains outside labeled contexts: {unsupported}")
    return predict_rows(model, rows)


def load_selector(path, device="cpu"):
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != "high_duty_walk_selector_v1":
        raise ValueError("Unsupported high-duty walk selector schema")
    required = {
        "hidden_dims", "input_fields", "input_ranges", "fixed_period_s",
        "supported_contexts", "deployment_ready", "selector_source_sha256",
        "selector_runtime_sha256", "selector_state_sha256", "output_range",
        "fixed_gait", "grid_task_sha256", "grid_checkpoint_sha256",
        "grid_collector_sha256", "trial_csv_sha256", "grid_manifest_sha256",
        "grid_complete_sha256",
    }
    if required - set(payload):
        raise ValueError(f"Selector checkpoint lacks provenance: {sorted(required - set(payload))}")
    if payload["input_fields"] != ["step_width", "speed"]:
        raise ValueError("Selector input fields differ from this implementation")
    if payload["input_ranges"] != {
            "step_width": list(STEP_WIDTH_RANGE), "speed": list(SPEED_RANGE)}:
        raise ValueError("Selector input ranges differ from this implementation")
    if payload["fixed_period_s"] != HIGH_DUTY_WALK_PERIOD:
        raise ValueError("Selector period differs from this implementation")
    if (payload["fixed_gait"] != "walk"
            or payload["output_range"] != [
                HIGH_DUTY_WALK_LEVELS[0], HIGH_DUTY_WALK_LEVELS[-1]]):
        raise ValueError("Selector fixed gait or output range differs")
    if payload["selector_source_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Selector source differs from the fitted implementation")
    root = Path(__file__).resolve().parents[4]
    if payload["selector_runtime_sha256"] != selector_runtime_sha256(root):
        raise ValueError("Selector runtime source bundle differs from the fit")
    if payload["deployment_ready"] is True:
        fields = {
            "validation_manifest_sha256", "validation_trials_sha256",
            "validation_completion_sha256", "validation_selector_checkpoint_sha256",
            "validation_low_level_checkpoint_sha256", "validation_seed_start",
            "validation_trials_per_context", "validation_all_contexts_passed",
        }
        if (not fields <= set(payload)
                or payload["validation_all_contexts_passed"] is not True
                or payload["validation_seed_start"] != HIGH_DUTY_WALK_VALIDATION_SEED
                or payload["validation_trials_per_context"] != 32
                or payload["validation_low_level_checkpoint_sha256"]
                    != payload["grid_checkpoint_sha256"]):
            raise ValueError("Validated selector lacks successful rollout evidence")
        paths = {
            "manifest": path.parent / "selector_validation_manifest.json",
            "trials": path.parent / "selector_validation_trials.csv",
            "completion": path.parent / "SELECTOR_VALIDATION_COMPLETE",
        }
        if (not all(item.is_file() for item in paths.values())
                or _sha256(paths["manifest"]) != payload["validation_manifest_sha256"]
                or _sha256(paths["trials"]) != payload["validation_trials_sha256"]
                or _sha256(paths["completion"]) != payload["validation_completion_sha256"]):
            raise ValueError("Validated selector evidence is missing or changed")
        manifest = json.loads(paths["manifest"].read_text())
        completion = json.loads(paths["completion"].read_text())
        if (manifest.get("schema") != "high_duty_walk_selector_validation_v1"
                or manifest.get("selector_checkpoint_sha256")
                    != payload["validation_selector_checkpoint_sha256"]
                or manifest.get("checkpoint_sha256")
                    != payload["grid_checkpoint_sha256"]
                or completion.get("schema")
                    != "high_duty_walk_selector_validation_complete_v1"
                or completion.get("manifest_file") != paths["manifest"].name
                or completion.get("manifest_sha256")
                    != payload["validation_manifest_sha256"]
                or completion.get("trial_csv_file") != paths["trials"].name
                or completion.get("trial_csv_sha256")
                    != payload["validation_trials_sha256"]):
            raise ValueError("Validated selector evidence identity mismatch")
        archive_hashes = completion.get("archive_sha256")
        manifest_filenames = [
            str(item.get("filename", ""))
            for item in manifest.get("conditions", [])
        ]
        actual_filenames = sorted(
            item.name for item in path.parent.glob("high_duty_walk_selector_*.npz"))
        if (not isinstance(archive_hashes, dict) or not archive_hashes
                or not all(manifest_filenames)
                or len(manifest_filenames) != len(set(manifest_filenames))
                or set(archive_hashes) != set(manifest_filenames)
                or actual_filenames != sorted(manifest_filenames)):
            raise ValueError("Validated selector archive set is incomplete or changed")
        for filename, expected_hash in archive_hashes.items():
            archive = path.parent / filename
            if not archive.is_file() or _sha256(archive) != expected_hash:
                raise ValueError(f"Validated selector archive hash mismatch: {filename}")
    model = HighDutyWalkSelector(tuple(payload["hidden_dims"])).to(device)
    model.load_state_dict(payload["state_dict"])
    if selector_state_sha256(model.state_dict()) != payload["selector_state_sha256"]:
        raise ValueError("Selector parameter hash mismatch")
    model.eval()
    return model, payload


__all__ = [
    "CONTEXT_FIELDS", "PRIMARY_SPEEDS", "PRIMARY_TRIALS_PER_CANDIDATE",
    "HighDutyWalkSelector", "aggregate_candidates", "bootstrap_target_intervals",
    "fit_selector", "load_selector", "predict_rows", "predict_supported_rows",
    "read_trial_rows", "select_targets", "selector_state_sha256",
    "selector_runtime_sha256", "validate_completed_grid",
]
