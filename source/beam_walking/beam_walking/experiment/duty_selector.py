"""Offline labeling and a small policy for selecting commanded duty factor."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .adaptive_duty import ADAPTIVE_DUTY_LEVELS, feasible_duty_upper
from .protocol import (
    CONTROL_DT, DUTY_RANGE, GAITS, PERIOD_TICKS, SPEED_RANGE,
    STEP_WIDTH_ANCHORS, STEP_WIDTH_RANGE,
)

CONTEXT_FIELDS = ("step_width", "speed", "period", "gait")
REQUIRED_TRIAL_FIELDS = ("seed",) + CONTEXT_FIELDS + (
    "command_df", "compliant", "positive_mechanical_cot",
)
PRIMARY_SPEEDS = (.25, .30, .35, .40)
PRIMARY_PERIODS = (.48,)
PRIMARY_TRIALS_PER_CANDIDATE = 32


def validate_context_values(step_width, speed, period, gait_id):
    """Validate and return canonical scalar selector inputs."""
    values = np.asarray([step_width, speed, period, gait_id], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Selector context values must be finite")
    tolerance = 1e-6
    if not STEP_WIDTH_RANGE[0] - tolerance <= step_width <= STEP_WIDTH_RANGE[1] + tolerance:
        raise ValueError("Step width is outside the trained range")
    if not SPEED_RANGE[0] - tolerance <= speed <= SPEED_RANGE[1] + tolerance:
        raise ValueError("Speed is outside the trained range")
    ticks = round(period / CONTROL_DT)
    if ticks not in PERIOD_TICKS or not np.isclose(
            ticks * CONTROL_DT, period, rtol=0, atol=1e-7):
        raise ValueError("Period must be 0.36--0.54 s in 0.02 s increments")
    rounded_gait = round(gait_id)
    if not np.isclose(gait_id, rounded_gait, rtol=0, atol=1e-7):
        raise ValueError("Gait ID must be an integer")
    if rounded_gait not in range(len(GAITS)):
        raise ValueError("Gait ID must select trot or walk")
    return float(step_width), float(speed), float(period), int(rounded_gait)


def read_trial_rows(path):
    """Read and validate the per-trial grid CSV."""
    path = Path(path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED_TRIAL_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Trial table is missing fields: {sorted(missing)}")
        rows, identities = [], set()
        for raw in reader:
            gait = raw["gait"]
            if gait not in GAITS:
                raise ValueError(f"Unsupported gait: {gait}")
            row = dict(raw)
            row["seed"] = int(raw["seed"])
            for field in ("step_width", "speed", "period", "command_df",
                          "positive_mechanical_cot"):
                row[field] = float(raw[field])
            token = raw["compliant"].strip().lower()
            if token not in {"0", "1", "false", "true", "no", "yes"}:
                raise ValueError(f"Invalid compliant value: {raw['compliant']}")
            row["compliant"] = token in {"1", "true", "yes"}
            width, speed, period, _ = validate_context_values(
                row["step_width"], row["speed"], row["period"],
                GAITS.index(gait))
            row.update(step_width=width, speed=speed, period=period)
            ticks = round(period / CONTROL_DT)
            if (not np.isfinite(row["command_df"])
                    or row["command_df"] < DUTY_RANGE[0]
                    or row["command_df"] > feasible_duty_upper(ticks) + 1e-7):
                raise ValueError("Candidate DF is outside the feasible range")
            energy = row["positive_mechanical_cot"]
            if np.isinf(energy):
                raise ValueError("Mechanical CoT must be finite or NaN for invalid trials")
            if np.isfinite(energy) and energy < 0:
                raise ValueError("Mechanical CoT cannot be negative")
            if row["compliant"] and not np.isfinite(energy):
                raise ValueError("A compliant trial must have finite mechanical CoT")
            identity = _context_key(row) + (row["command_df"], row["seed"])
            if identity in identities:
                raise ValueError(f"Duplicate selector trial row: {identity}")
            identities.add(identity)
            rows.append(row)
    if not rows:
        raise ValueError("Trial table contains no rows")
    return rows


def validate_completed_grid(trials_path, *, require_primary=True):
    """Bind a trial CSV to its completed collector manifest and exact row set."""
    trials_path = Path(trials_path)
    directory = trials_path.parent
    manifest_path = directory / "grid_manifest.json"
    complete_path = directory / "GRID_COMPLETE"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise ValueError("Grid manifest and GRID_COMPLETE marker are required")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    allowed_schemas = ({"adaptive_duty_grid_v1"} if require_primary else
                       {"adaptive_duty_grid_v1",
                        "adaptive_duty_grid_exploratory_v1"})
    if manifest.get("schema") not in allowed_schemas:
        raise ValueError("Unsupported adaptive duty grid schema")
    if manifest.get("terrain") != "flat_ground" or manifest.get(
            "terrain_width_input") is not False:
        raise ValueError("Selector grid must use flat-ground commanded step width")
    count = manifest.get("trials_per_candidate")
    if not isinstance(count, int) or count < 8:
        raise ValueError("Grid has too few trials per candidate")
    try:
        completion = json.loads(complete_path.read_text())
    except (ValueError, OSError) as error:
        raise ValueError("Grid completion record is malformed") from error
    expected_completion_schema = (
        "adaptive_duty_grid_complete_v1"
        if manifest.get("schema") == "adaptive_duty_grid_v1"
        else "adaptive_duty_grid_exploratory_complete_v1")
    if (completion.get("schema") != expected_completion_schema
            or completion.get("manifest_file") != manifest_path.name
            or completion.get("manifest_sha256")
                != hashlib.sha256(manifest_bytes).hexdigest()
            or completion.get("trial_csv_file") != trials_path.name
            or completion.get("trial_csv_sha256")
                != hashlib.sha256(trials_path.read_bytes()).hexdigest()):
        raise ValueError("Grid completion hashes do not match the final files")
    conditions = manifest.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Grid manifest contains no conditions")
    for field in ("task_sha256", "checkpoint_sha256", "collector_sha256"):
        value = manifest.get(field)
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"Grid manifest has invalid {field}")
    rows = read_trial_rows(trials_path)
    seed_start = manifest.get("evaluation_seed_start")
    if not isinstance(seed_start, int):
        raise ValueError("Grid evaluation seed is malformed")
    expected = {
        (float(item["step_width"]), float(item["speed"]),
         float(item["period"]), str(item["gait"]),
         float(item["command_df"]), seed)
        for item in conditions
        for seed in range(seed_start, seed_start + count)
    }
    actual = {
        _context_key(row) + (row["command_df"], row["seed"])
        for row in rows
    }
    if len(actual) != len(rows) or actual != expected:
        raise ValueError(
            "Trial CSV does not exactly match manifest conditions and seeds")
    expected_files = [str(item.get("filename", "")) for item in conditions]
    if (not all(expected_files) or len(expected_files) != len(set(expected_files))):
        raise ValueError("Grid manifest condition filenames must be unique")
    actual_files = sorted(path.name for path in directory.glob("grid_*.npz"))
    if actual_files != sorted(expected_files):
        raise ValueError("Grid archive set does not exactly match the manifest")
    archive_hashes = completion.get("archive_sha256")
    if not isinstance(archive_hashes, dict) or set(archive_hashes) != set(expected_files):
        raise ValueError("Grid completion record has the wrong archive set")
    matched_reset = None
    for filename in expected_files:
        archive_path = directory / filename
        if archive_hashes[filename] != hashlib.sha256(archive_path.read_bytes()).hexdigest():
            raise ValueError(f"Grid archive hash mismatch: {filename}")
        expected_item = next(
            item for item in conditions if item["filename"] == filename)
        with np.load(archive_path, allow_pickle=False) as archive:
            for archive_field, manifest_field in (
                    ("task_sha256", "task_sha256"),
                    ("checkpoint_sha256", "checkpoint_sha256"),
                    ("collector_sha256", "collector_sha256")):
                if str(archive[archive_field]) != manifest[manifest_field]:
                    raise ValueError(
                        f"Grid archive identity mismatch: {filename} {archive_field}")
            expected_command = np.asarray([
                expected_item["speed"], expected_item["command_df"],
                expected_item["step_width"], expected_item["period"],
                GAITS.index(expected_item["gait"]),
            ])
            if (not np.allclose(archive["command"], expected_command,
                                rtol=0, atol=1e-7)
                    or not np.array_equal(
                        archive["seeds"],
                        np.arange(seed_start, seed_start + count))):
                raise ValueError(f"Grid command/seed mismatch: {filename}")
            reset_evidence = (
                archive["reset_plan"], archive["stance_start"],
                archive["initial_root"], archive["initial_joints"])
            expected_shapes = (
                (count, 2), (count,), (count, 13), (count, 12))
            if any(value.shape != shape for value, shape in
                   zip(reset_evidence, expected_shapes)):
                raise ValueError(f"Grid reset evidence is malformed: {filename}")
            if matched_reset is None:
                matched_reset = tuple(value.copy() for value in reset_evidence)
            elif (
                    not np.array_equal(reset_evidence[0], matched_reset[0])
                    or not np.array_equal(reset_evidence[1], matched_reset[1])
                    or not np.allclose(
                        reset_evidence[2], matched_reset[2], rtol=0, atol=1e-6)
                    or not np.allclose(
                        reset_evidence[3], matched_reset[3], rtol=0, atol=1e-6)):
                raise ValueError(
                    "Grid conditions do not share matched reset states")
    if require_primary:
        expected_conditions = {
            (width, speed, period, gait, duty)
            for width in STEP_WIDTH_ANCHORS
            for speed in PRIMARY_SPEEDS
            for period in PRIMARY_PERIODS
            for gait in GAITS
            for duty in ADAPTIVE_DUTY_LEVELS
        }
        actual_conditions = {identity[:5] for identity in expected}
        if count != PRIMARY_TRIALS_PER_CANDIDATE:
            raise ValueError("Primary selector grid requires 32 trials per candidate")
        if actual_conditions != expected_conditions:
            raise ValueError("Primary selector grid is incomplete or has extra conditions")
        if (manifest.get("settle_cycles") != 12
                or manifest.get("measurement_cycles") != 4
                or manifest.get("evaluation_seed_start") != 3_000_000
                or not np.isclose(manifest.get("stance_start_probability", -1), .10)
                or not np.isclose(manifest.get("topology_minimum", -1), .90)
                or not np.isclose(manifest.get("control_dt_s", -1), CONTROL_DT)
                or manifest.get("stance_start_sampling")
                    != "seeded_independent_bernoulli"
                or manifest.get("stance_start_realized")
                    != matched_reset[1].astype(int).tolist()
                or manifest.get("stance_start_realized_count")
                    != int(matched_reset[1].sum())):
            raise ValueError("Primary selector grid protocol differs from registration")
    return rows, manifest, {
        "trial_csv_sha256": hashlib.sha256(trials_path.read_bytes()).hexdigest(),
        "grid_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "grid_complete_sha256": hashlib.sha256(complete_path.read_bytes()).hexdigest(),
    }


def _context_key(row):
    return (
        float(row["step_width"]), float(row["speed"]),
        float(row["period"]), str(row["gait"]),
    )


def aggregate_candidates(rows):
    """Summarize compliance and compliant-trial energy for every DF candidate."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[_context_key(row) + (float(row["command_df"]),)].append(row)
    candidates = []
    for key, trials in sorted(grouped.items()):
        compliant = [trial for trial in trials if trial["compliant"]]
        energy = np.asarray([
            trial["positive_mechanical_cot"] for trial in compliant
            if np.isfinite(trial["positive_mechanical_cot"])
        ])
        candidates.append({
            "step_width": key[0], "speed": key[1], "period": key[2],
            "gait": key[3], "command_df": key[4],
            "trials": len(trials), "compliant_trials": len(compliant),
            "finite_energy_compliant_trials": len(energy),
            "compliance_rate": len(compliant) / len(trials),
            "finite_energy_compliant_rate": len(energy) / len(trials),
            "median_positive_mechanical_cot": (
                float(np.median(energy)) if len(energy) else float("nan")),
        })
    return candidates


def select_targets(rows, min_compliance_rate=.90, min_trials=8):
    """Choose the lowest-energy candidate among sufficiently compliant DFs.

    Contexts with no eligible candidate are reported separately and are never
    silently turned into training labels.
    """
    if not 0 < min_compliance_rate <= 1:
        raise ValueError("Minimum compliance rate must be in (0, 1]")
    if min_trials < 1:
        raise ValueError("Minimum trials must be positive")
    by_context = defaultdict(list)
    for candidate in aggregate_candidates(rows):
        by_context[_context_key(candidate)].append(candidate)
    targets, rejected = [], []
    for context, candidates in sorted(by_context.items()):
        eligible = [
            item for item in candidates
            if item["trials"] >= min_trials
            and item["compliance_rate"] >= min_compliance_rate
            and item["finite_energy_compliant_rate"] >= min_compliance_rate
            and np.isfinite(item["median_positive_mechanical_cot"])
        ]
        if not eligible:
            rejected.append({
                "step_width": context[0], "speed": context[1],
                "period": context[2], "gait": context[3],
                "reason": "no_candidate_passed_compliance_and_energy_gate",
            })
            continue
        winner = min(
            eligible,
            key=lambda item: (
                item["median_positive_mechanical_cot"], item["command_df"]),
        )
        targets.append({
            "step_width": context[0], "speed": context[1],
            "period": context[2], "gait": context[3],
            "target_df": winner["command_df"],
            "target_positive_mechanical_cot":
                winner["median_positive_mechanical_cot"],
            "target_compliance_rate": winner["compliance_rate"],
            "eligible_candidates": len(eligible),
        })
    return targets, rejected


def bootstrap_target_intervals(rows, min_compliance_rate=.90, min_trials=8,
                               samples=1000, seed=0):
    """Estimate selection uncertainty by resampling trials within each DF cell."""
    if samples < 1:
        raise ValueError("Bootstrap sample count must be positive")
    grouped = defaultdict(list)
    for row in rows:
        grouped[_context_key(row) + (float(row["command_df"]),)].append(row)
    by_context = defaultdict(dict)
    for key, trials in grouped.items():
        by_context[key[:4]][key[4]] = trials
    rng = np.random.default_rng(seed)
    intervals = []
    for context, candidates in sorted(by_context.items()):
        by_seed = {}
        expected_seeds = None
        for duty, trials in candidates.items():
            mapping = {int(row["seed"]): row for row in trials}
            if len(mapping) != len(trials):
                raise ValueError(
                    f"Duplicate trial seed in context {context}, DF {duty}")
            seeds = set(mapping)
            if expected_seeds is None:
                expected_seeds = seeds
            elif seeds != expected_seeds:
                raise ValueError(
                    f"DF candidates do not share matched seeds in context {context}")
            by_seed[duty] = mapping
        ordered_seeds = np.asarray(sorted(expected_seeds), dtype=int)
        selected = []
        for _ in range(samples):
            sampled_seeds = ordered_seeds[
                rng.integers(0, len(ordered_seeds), size=len(ordered_seeds))]
            resampled = []
            for mapping in by_seed.values():
                # Copy rows because a paired bootstrap can repeat a seed. Give
                # each draw a unique temporary seed so ordinary aggregation
                # remains unchanged while every DF sees the same reset draw.
                resampled.extend(
                    dict(mapping[int(seed_value)], seed=draw)
                    for draw, seed_value in enumerate(sampled_seeds))
            targets, _ = select_targets(
                resampled, min_compliance_rate=min_compliance_rate,
                min_trials=min_trials)
            if targets:
                selected.append(targets[0]["target_df"])
        values = np.asarray(selected, dtype=float)
        intervals.append({
            "step_width": context[0], "speed": context[1],
            "period": context[2], "gait": context[3],
            "bootstrap_valid_fraction": len(values) / samples,
            "target_df_ci_low": (
                float(np.quantile(values, .025)) if len(values) else float("nan")),
            "target_df_ci_high": (
                float(np.quantile(values, .975)) if len(values) else float("nan")),
        })
    return intervals


def context_tensor(rows, device="cpu"):
    """Convert context records to [width, speed, period, gait-id] tensors."""
    return torch.tensor([
        [row["step_width"], row["speed"], row["period"],
         GAITS.index(row["gait"])]
        for row in rows
    ], dtype=torch.float32, device=device)


class DutyFactorSelector(torch.nn.Module):
    """Feedforward policy mapping gait commands to a feasible duty factor."""

    def __init__(self, hidden_dims=(32, 32)):
        super().__init__()
        layers = []
        incoming = 5
        for width in hidden_dims:
            layers.extend((torch.nn.Linear(incoming, width), torch.nn.ELU()))
            incoming = width
        layers.append(torch.nn.Linear(incoming, 1))
        self.network = torch.nn.Sequential(*layers)

    @staticmethod
    def features(context):
        if context.ndim != 2 or context.shape[1] != 4:
            raise ValueError("Selector context must have shape [batch, 4]")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("Selector context values must be finite")
        for row in context.detach().cpu().numpy():
            validate_context_values(*[float(value) for value in row])
        width = 2 * (context[:, 0:1] - STEP_WIDTH_RANGE[0]) / (
            STEP_WIDTH_RANGE[1] - STEP_WIDTH_RANGE[0]) - 1
        speed = 2 * (context[:, 1:2] - SPEED_RANGE[0]) / (
            SPEED_RANGE[1] - SPEED_RANGE[0]) - 1
        period = 2 * (context[:, 2:3] - .36) / (.54 - .36) - 1
        gait = torch.nn.functional.one_hot(
            context[:, 3].long(), num_classes=len(GAITS)).to(context.dtype)
        return torch.cat((width, speed, period, gait), dim=1)

    def forward(self, context):
        raw = torch.sigmoid(self.network(self.features(context))).squeeze(1)
        period_ticks = torch.round(context[:, 2] / .02)
        upper = feasible_duty_upper(period_ticks).to(
            device=context.device, dtype=context.dtype)
        return DUTY_RANGE[0] + raw * (upper - DUTY_RANGE[0])


def fit_selector(targets, seed=0, epochs=3000, learning_rate=3e-3):
    """Fit a selector to grid-derived labels and return model and loss history."""
    if len(targets) < 2:
        raise ValueError("At least two valid contexts are required to fit a selector")
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("Epochs and learning rate must be positive")
    torch.manual_seed(seed)
    model = DutyFactorSelector()
    context = context_tensor(targets)
    labels = torch.tensor(
        [row["target_df"] for row in targets], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(epochs):
        prediction = model(context)
        loss = torch.nn.functional.mse_loss(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return model, history


def selector_state_sha256(state_dict):
    """Hash ordered selector parameters independently of checkpoint packaging."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode())
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_selector(path, device="cpu"):
    """Load a selector checkpoint created by ``fit_duty_selector.py``."""
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != "flat_duty_selector_v1":
        raise ValueError("Unsupported duty-selector checkpoint schema")
    required = {
        "hidden_dims", "input_fields", "input_ranges", "gait_id_mapping",
        "control_dt_s", "supported_contexts", "deployment_ready",
        "selector_source_sha256", "trial_csv_sha256",
        "grid_manifest_sha256", "grid_complete_sha256",
        "grid_task_sha256", "grid_checkpoint_sha256",
        "grid_collector_sha256", "selector_state_sha256",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Selector checkpoint lacks provenance: {sorted(missing)}")
    if payload["gait_id_mapping"] != {
            name: index for index, name in enumerate(GAITS)}:
        raise ValueError("Selector gait mapping differs from this implementation")
    if payload["input_fields"] != [
            "step_width", "speed", "period", "gait_id"]:
        raise ValueError("Selector input fields differ from this implementation")
    if payload["input_ranges"] != {
            "step_width": list(STEP_WIDTH_RANGE),
            "speed": list(SPEED_RANGE), "period": [.36, .54]}:
        raise ValueError("Selector input ranges differ from this implementation")
    if payload["control_dt_s"] != CONTROL_DT:
        raise ValueError("Selector control interval differs from this implementation")
    current_source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if payload["selector_source_sha256"] != current_source:
        raise ValueError("Selector source differs from the fitted implementation")
    if payload["deployment_ready"] is True:
        validation_fields = {
            "validation_manifest_sha256", "validation_trials_sha256",
            "validation_low_level_checkpoint_sha256", "validation_seed_start",
            "validation_trials_per_context", "validation_all_contexts_passed",
            "validation_selector_checkpoint_sha256",
            "validation_completion_sha256",
        }
        if (not validation_fields <= set(payload)
                or payload["validation_all_contexts_passed"] is not True
                or payload.get("exploratory") is not False
                or payload["validation_trials_per_context"] != 32
                or payload["validation_seed_start"] != 4_000_000
                or payload["validation_low_level_checkpoint_sha256"]
                    != payload["grid_checkpoint_sha256"]):
            raise ValueError("Validated selector lacks successful rollout evidence")
        validation_paths = {
            "manifest": path.parent / "selector_validation_manifest.json",
            "trials": path.parent / "selector_validation_trials.csv",
            "completion": path.parent / "SELECTOR_VALIDATION_COMPLETE",
        }
        if not all(value.is_file() for value in validation_paths.values()):
            raise ValueError("Validated selector evidence files are missing")
        if (hashlib.sha256(validation_paths["manifest"].read_bytes()).hexdigest()
                != payload["validation_manifest_sha256"]
                or hashlib.sha256(validation_paths["trials"].read_bytes()).hexdigest()
                != payload["validation_trials_sha256"]
                or hashlib.sha256(validation_paths["completion"].read_bytes()).hexdigest()
                != payload["validation_completion_sha256"]):
            raise ValueError("Validated selector evidence hash mismatch")
        validation_manifest = json.loads(validation_paths["manifest"].read_text())
        validation_completion = json.loads(
            validation_paths["completion"].read_text())
        if (validation_manifest.get("selector_checkpoint_sha256")
                != payload["validation_selector_checkpoint_sha256"]
                or validation_manifest.get("checkpoint_sha256")
                    != payload["grid_checkpoint_sha256"]
                or validation_completion.get("schema")
                    != "adaptive_duty_selector_validation_complete_v1"
                or validation_completion.get("manifest_sha256")
                    != payload["validation_manifest_sha256"]
                or validation_completion.get("trial_csv_sha256")
                    != payload["validation_trials_sha256"]):
            raise ValueError("Validated selector evidence identity mismatch")
        validation_archives = validation_completion.get("archive_sha256")
        if not isinstance(validation_archives, dict) or not validation_archives:
            raise ValueError("Validated selector archive hashes are missing")
        for filename, expected_hash in validation_archives.items():
            archive_path = path.parent / filename
            if (not archive_path.is_file()
                    or hashlib.sha256(archive_path.read_bytes()).hexdigest()
                        != expected_hash):
                raise ValueError(f"Validated selector archive hash mismatch: {filename}")
    model = DutyFactorSelector(tuple(payload["hidden_dims"])).to(device)
    model.load_state_dict(payload["state_dict"])
    if selector_state_sha256(model.state_dict()) != payload["selector_state_sha256"]:
        raise ValueError("Selector parameter hash mismatch")
    model.eval()
    return model, payload


def context_is_supported(payload, row):
    """Return whether a context exactly matches a labeled selector context."""
    key = _context_key(row)
    return any(_context_key(candidate) == key
               for candidate in payload["supported_contexts"])


def predict_supported_rows(model, payload, rows, *, require_deployment_ready=True):
    """Predict only at labeled contexts and optionally require fresh validation."""
    if require_deployment_ready and payload.get("deployment_ready") is not True:
        raise ValueError("Selector has not passed fresh rollout validation")
    unsupported = [row for row in rows if not context_is_supported(payload, row)]
    if unsupported:
        raise ValueError(f"Selector abstains outside labeled contexts: {unsupported}")
    return predict_rows(model, rows)


@torch.inference_mode()
def predict_rows(model, rows):
    device = next(model.parameters()).device
    predictions = model(context_tensor(rows, device=device)).cpu().numpy()
    return [dict(row, selected_df=float(value))
            for row, value in zip(rows, predictions)]
