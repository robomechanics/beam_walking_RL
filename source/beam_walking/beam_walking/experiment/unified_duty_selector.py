"""One duty-factor selector for trot and walk on the unified policy.

The selector maps [step width, speed, period, gait] to a duty factor. Trot
chooses among 0.50/0.625/0.75 and walk among 0.75/0.80/0.85/0.90, the same
candidates the unified policy was trained on (docs/unified_gait_plan.md),
restricted at each period to the candidates that respect the gait's swing
bound (trot needs five swing ticks, walk two). Labels come from the unified
period-sweep surface grid (scripts/collect_policy_surfaces.py --task unified
--period-sweep): for each context the label is the lowest median positive
mechanical CoT candidate among those passing the 90% compliance and
finite-energy gates. The classifier's output is masked to the gait's feasible
candidates at the requested period, so it can never request a walk duty
below 0.75, a trot duty above 0.75, or a duty with too little swing time.
"""

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
from .protocol import CONTROL_DT, GAITS, PERIOD_TICKS, SPEED_RANGE, STEP_WIDTH_RANGE
from .unified_gait import (
    GAIT_MIN_PERIOD_TICKS,
    TROT_DUTY_LEVELS,
    WALK_DUTY_LEVELS,
    max_duty_for_period,
)

CONTEXT_FIELDS = ("step_width", "speed", "period", "gait")
REQUIRED_TRIAL_FIELDS = ("seed",) + CONTEXT_FIELDS + (
    "command_df", "compliant", "positive_mechanical_cot",
)
PERIOD_RANGE = (min(PERIOD_TICKS) * CONTROL_DT, max(PERIOD_TICKS) * CONTROL_DT)
# Stance-width domain of the selector. Defaults to the standard 0.10-0.50 m;
# narrow-stance grids set it from their manifest, and checkpoints restore it.
WIDTH_RANGE = tuple(STEP_WIDTH_RANGE)


def set_width_range(width_range):
    global WIDTH_RANGE
    low, high = (float(value) for value in width_range)
    if not 0 < low < high:
        raise ValueError(f"Invalid selector width range: {width_range}")
    WIDTH_RANGE = (low, high)
GAIT_LEVELS = {"trot": tuple(TROT_DUTY_LEVELS), "walk": tuple(WALK_DUTY_LEVELS)}
# Class index space: every (gait, duty) candidate of the grid being fitted.
LEVELS = tuple((gait, duty) for gait in GAITS for duty in GAIT_LEVELS[gait])


def set_gait_levels(gait_levels):
    """Candidate duty factors per gait. Grids that sweep other candidates set them
    from their manifest, and checkpoints restore them."""
    global LEVELS
    levels = {gait: tuple(float(duty) for duty in gait_levels[gait]) for gait in GAITS}
    if any(not values or list(values) != sorted(values) for values in levels.values()):
        raise ValueError(f"Invalid selector candidate levels: {gait_levels}")
    GAIT_LEVELS.clear()
    GAIT_LEVELS.update(levels)
    LEVELS = tuple((gait, duty) for gait in GAITS for duty in GAIT_LEVELS[gait])
PRIMARY_TRIALS_PER_CANDIDATE = 32
SELECTOR_SCHEMA = "unified_duty_selector_v2"
SURFACE_SCHEMA = "unified_surface_v1"
SURFACE_COMPLETE_SCHEMA = "unified_surface_complete_v1"
VALIDATION_SCHEMA = "unified_selector_validation_v1"
VALIDATION_COMPLETE_SCHEMA = "unified_selector_validation_complete_v1"
SELECTOR_RUNTIME_FILES = (
    "source/beam_walking/beam_walking/experiment/protocol.py",
    "source/beam_walking/beam_walking/experiment/unified_gait.py",
    "source/beam_walking/beam_walking/experiment/duty_selector.py",
    "source/beam_walking/beam_walking/experiment/unified_duty_selector.py",
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selector_runtime_sha256(root):
    root = Path(root)
    return hashlib.sha256(b"".join(
        (root / relative).read_bytes() for relative in SELECTOR_RUNTIME_FILES
    )).hexdigest()


def period_ticks(period):
    ticks = round(float(period) / CONTROL_DT)
    if ticks not in PERIOD_TICKS or not np.isclose(ticks * CONTROL_DT, period, atol=1e-6):
        raise ValueError("Period must be 0.36-0.54 s in 0.02 s increments")
    return ticks


def validate_context_values(step_width, speed, period, gait):
    values = np.asarray([step_width, speed, period], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Selector context must be finite")
    if not WIDTH_RANGE[0] - 1e-6 <= values[0] <= WIDTH_RANGE[1] + 1e-6:
        raise ValueError(f"Step width is outside [{WIDTH_RANGE[0]:.2f}, {WIDTH_RANGE[1]:.2f}] m")
    if not SPEED_RANGE[0] - 1e-6 <= values[1] <= SPEED_RANGE[1] + 1e-6:
        raise ValueError("Speed is outside [0.25, 0.40] m/s")
    if gait not in GAITS:
        raise ValueError(f"Unknown gait: {gait}")
    ticks = period_ticks(values[2])
    if ticks < GAIT_MIN_PERIOD_TICKS[GAITS.index(gait)]:
        raise ValueError(f"Period {values[2]:.2f} s is below the {gait} training floor")
    return float(values[0]), float(values[1]), float(ticks * CONTROL_DT), str(gait)


def duty_cap(gait, period):
    """Largest feasible duty for a gait at a period (swing bound)."""
    ticks = period_ticks(period)
    return float(max_duty_for_period(
        torch.tensor([GAITS.index(gait)]), torch.tensor([ticks])))


def feasible_levels(gait, period):
    cap = duty_cap(gait, period)
    return tuple(d for d in GAIT_LEVELS[gait] if d <= cap + 1e-6)


def level_index(gait, duty, period=None):
    for index, (level_gait, level_duty) in enumerate(LEVELS):
        if level_gait == gait and abs(level_duty - float(duty)) < 1e-6:
            if period is not None and level_duty > duty_cap(gait, period) + 1e-6:
                raise ValueError(f"{gait} DF {duty} is infeasible at {period:.2f} s")
            return index
    raise ValueError(f"{gait} DF {duty} is not a unified candidate")


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
            (row["step_width"], row["speed"], row["period"],
             row["gait"]) = validate_context_values(
                row["step_width"], row["speed"], row["period"], row["gait"])
            level_index(row["gait"], row["command_df"], row["period"])
            energy = row["positive_mechanical_cot"]
            if np.isinf(energy) or (np.isfinite(energy) and energy < 0):
                raise ValueError("Mechanical CoT must be nonnegative or NaN")
            if row["compliant"] and not np.isfinite(energy):
                raise ValueError("A compliant trial must have finite mechanical CoT")
            identity = (row["step_width"], row["speed"], row["period"],
                        row["gait"], row["command_df"], row["seed"])
            if identity in identities:
                raise ValueError(f"Duplicate selector trial row: {identity}")
            identities.add(identity)
            rows.append(row)
    if not rows:
        raise ValueError("Trial table contains no rows")
    return rows


def validate_completed_surface_grid(trials_path):
    """Check the unified surface grid's manifest, completion, and archive hashes."""
    trials_path = Path(trials_path)
    directory = trials_path.parent
    manifest_path = directory / "surface_manifest.json"
    complete_path = directory / "SURFACE_COMPLETE"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise ValueError("Surface manifest and SURFACE_COMPLETE marker are required")
    manifest = json.loads(manifest_path.read_text())
    completion = json.loads(complete_path.read_text())
    if (manifest.get("schema") != SURFACE_SCHEMA
            or manifest.get("smoke") is not False
            or manifest.get("task_variant") not in ("unified", "specialist")
            or manifest.get("terrain") != "flat_ground"
            or completion.get("schema") != SURFACE_COMPLETE_SCHEMA
            or completion.get("manifest_sha256") != _sha256(manifest_path)
            or completion.get("trials_sha256") != _sha256(trials_path)):
        raise ValueError("A complete unified surface grid with matching hashes is required")
    archive_hashes = completion.get("archive_sha256") or {}
    names = {entry["filename"] for entry in manifest["conditions"]}
    if set(archive_hashes) != names:
        raise ValueError("Surface archive set does not match the manifest")
    for name, digest in archive_hashes.items():
        if _sha256(directory / name) != digest:
            raise ValueError(f"Surface archive hash mismatch: {name}")
    rows = read_trial_rows(trials_path)
    expected = {
        (float(e["step_width"]), float(e["speed"]), round(float(e["period"]), 6),
         str(e["gait"]), float(e["command_df"]))
        for e in manifest["conditions"]}
    observed = {(r["step_width"], r["speed"], round(r["period"], 6), r["gait"],
                 r["command_df"]) for r in rows}
    if observed != expected:
        raise ValueError("Trial table conditions differ from the manifest")
    trials = int(manifest["trials_per_condition"])
    if trials != PRIMARY_TRIALS_PER_CANDIDATE:
        raise ValueError("Primary selector fit requires 32 matched trials per candidate")
    hashes = {
        "grid_manifest_sha256": completion["manifest_sha256"],
        "trial_csv_sha256": completion["trials_sha256"],
        "grid_complete_sha256": _sha256(complete_path),
    }
    return rows, manifest, hashes


def context_tensor(rows, device="cpu"):
    return torch.tensor([
        [row["step_width"], row["speed"], row["period"], GAITS.index(row["gait"])]
        for row in rows
    ], dtype=torch.float32, device=device)


class UnifiedDutySelector(torch.nn.Module):
    """Feedforward classifier over the unified (gait, duty) candidates.

    Input [step width, speed, period, gait id]. Logits of candidates that
    belong to the other gait, or that violate the gait's swing bound at the
    requested period, are masked out before the argmax.
    """

    def __init__(self, hidden_dims=(32, 32)):
        super().__init__()
        layers = []
        incoming = 3 + len(GAITS)
        for width in hidden_dims:
            layers.extend((torch.nn.Linear(incoming, width), torch.nn.ELU()))
            incoming = width
        layers.append(torch.nn.Linear(incoming, len(LEVELS)))
        self.network = torch.nn.Sequential(*layers)
        self.register_buffer("level_gait", torch.tensor(
            [GAITS.index(gait) for gait, _ in LEVELS], dtype=torch.long))
        self.register_buffer("level_duty", torch.tensor(
            [duty for _, duty in LEVELS], dtype=torch.float32))

    @staticmethod
    def features(context):
        if context.ndim != 2 or context.shape[1] != 4:
            raise ValueError("Selector context must have shape [batch, 4]")
        for row in context.detach().cpu().numpy():
            gait_id = int(round(float(row[3])))
            if abs(row[3] - gait_id) > 1e-6 or not 0 <= gait_id < len(GAITS):
                raise ValueError("Gait id must be 0 (trot) or 1 (walk)")
            validate_context_values(float(row[0]), float(row[1]), float(row[2]),
                                    GAITS[gait_id])
        width = 2 * (context[:, 0:1] - WIDTH_RANGE[0]) / (
            WIDTH_RANGE[1] - WIDTH_RANGE[0]) - 1
        speed = 2 * (context[:, 1:2] - SPEED_RANGE[0]) / (
            SPEED_RANGE[1] - SPEED_RANGE[0]) - 1
        period = 2 * (context[:, 2:3] - PERIOD_RANGE[0]) / (
            PERIOD_RANGE[1] - PERIOD_RANGE[0]) - 1
        gait = torch.nn.functional.one_hot(
            context[:, 3].round().long(), num_classes=len(GAITS)).to(context.dtype)
        return torch.cat((width, speed, period, gait), dim=1)

    def allowed(self, context):
        gait = context[:, 3].round().long()
        ticks = torch.round(context[:, 2] / CONTROL_DT).long()
        cap = max_duty_for_period(gait, ticks).to(context.dtype)
        same_gait = self.level_gait[None, :] == gait[:, None]
        feasible = self.level_duty[None, :] <= cap[:, None] + 1e-6
        return same_gait & feasible

    def logits(self, context):
        raw = self.network(self.features(context))
        return raw.masked_fill(~self.allowed(context), float("-inf"))

    def forward(self, context):
        return self.level_duty[self.logits(context).argmax(dim=1)]


def fit_selector(targets, seed=0, epochs=3000, learning_rate=3e-3):
    if len(targets) < 2:
        raise ValueError("At least two valid contexts are required")
    torch.manual_seed(seed)
    model = UnifiedDutySelector()
    context = context_tensor(targets)
    labels = torch.tensor([
        level_index(row["gait"], row["target_df"], row["period"]) for row in targets
    ], dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(epochs):
        loss = torch.nn.functional.cross_entropy(model.logits(context), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return model, history


@torch.inference_mode()
def predict_rows(model, rows):
    device = next(model.parameters()).device
    values = model(context_tensor(rows, device=device)).cpu().numpy()
    canonical = [
        min(feasible_levels(row["gait"], row["period"]),
            key=lambda level: abs(level - float(value)))
        for row, value in zip(rows, values)]
    return [dict(row, selected_df=value) for row, value in zip(rows, canonical)]


def _context_id(row):
    return (float(row["step_width"]), float(row["speed"]),
            round(float(row["period"]), 6), str(row["gait"]))


def context_is_supported(payload, row):
    key = _context_id(row)
    return any(_context_id(item) == key for item in payload["supported_contexts"])


def predict_supported_rows(model, payload, rows, *, require_deployment_ready=True):
    if require_deployment_ready and payload.get("deployment_ready") is not True:
        raise ValueError("Selector has not passed fresh rollout validation")
    unsupported = [row for row in rows if not context_is_supported(payload, row)]
    if unsupported:
        raise ValueError(f"Selector abstains outside labeled contexts: {unsupported}")
    return predict_rows(model, rows)


def selector_payload(model, targets, rejected, manifest, hashes, *, root,
                     min_compliance_rate, min_trials, exploratory, grids=None):
    """Payload for one grid (manifest, hashes) or several (``grids``).

    ``grids`` is a list of {"gaits": [...], "manifest": ..., "hashes": ...}
    entries, one per low-level checkpoint (the specialists). The grid_* and
    hash fields then become gait-keyed dictionaries.
    """
    source_path = Path(__file__)
    if grids:
        def by_gait(getter):
            out = {}
            for grid in grids:
                for gait in grid["gaits"]:
                    if gait in out:
                        raise ValueError(f"Gait {gait} appears in more than one grid")
                    out[gait] = getter(grid)
            return out
        hashes = {key: by_gait(lambda g, key=key: g["hashes"][key])
                  for key in ("grid_manifest_sha256", "trial_csv_sha256",
                              "grid_complete_sha256")}
        grid_fields = {
            "grid_task_sha256": by_gait(lambda g: g["manifest"]["task_sha256"]),
            "grid_checkpoint_sha256": by_gait(lambda g: g["manifest"]["checkpoint_sha256"]),
            "grid_collector_sha256": by_gait(lambda g: g["manifest"]["collector_sha256"]),
        }
    else:
        grid_fields = {
            "grid_task_sha256": manifest["task_sha256"],
            "grid_checkpoint_sha256": manifest["checkpoint_sha256"],
            "grid_collector_sha256": manifest["collector_sha256"],
        }
    return {
        "schema": SELECTOR_SCHEMA,
        "state_dict": model.state_dict(), "hidden_dims": [32, 32],
        "input_fields": ["step_width", "speed", "period", "gait_id"],
        "input_ranges": {
            "step_width": list(WIDTH_RANGE), "speed": list(SPEED_RANGE),
            "period": list(PERIOD_RANGE), "gait_id": [0, len(GAITS) - 1]},
        "gait_levels": {gait: list(levels) for gait, levels in GAIT_LEVELS.items()},
        "gait_min_period_ticks": list(GAIT_MIN_PERIOD_TICKS),
        "label_rule": (
            "minimum median positive mechanical CoT among candidates passing "
            "the registered compliance and finite-energy gates"),
        "minimum_compliance_rate": min_compliance_rate,
        "minimum_trials_per_candidate": min_trials,
        "supported_contexts": [
            {"step_width": row["step_width"], "speed": row["speed"],
             "period": row["period"], "gait": row["gait"]} for row in targets],
        "rejected_contexts": rejected,
        "deployment_ready": False,
        "requires_fresh_rollout_validation": True,
        "exploratory": exploratory,
        "selector_source_sha256": _sha256(source_path),
        "selector_runtime_sha256": selector_runtime_sha256(root),
        "selector_state_sha256": selector_state_sha256(model.state_dict()),
        **hashes,
        **grid_fields,
    }


def _check_validation_directory(directory, payload, expected_selector_sha256,
                                expected_checkpoint_sha256):
    directory = Path(directory)
    paths = {
        "manifest": directory / "surface_manifest.json",
        "trials": directory / "surface_trials.csv",
        "completion": directory / "SURFACE_COMPLETE",
    }
    if not all(item.is_file() for item in paths.values()):
        raise ValueError(f"Validation evidence is missing in {directory}")
    manifest = json.loads(paths["manifest"].read_text())
    completion = json.loads(paths["completion"].read_text())
    if (manifest.get("schema") != VALIDATION_SCHEMA
            or manifest.get("selector_checkpoint_sha256") != expected_selector_sha256
            or manifest.get("checkpoint_sha256") != expected_checkpoint_sha256
            or completion.get("schema") != VALIDATION_COMPLETE_SCHEMA
            or completion.get("manifest_sha256") != _sha256(paths["manifest"])
            or completion.get("trials_sha256") != _sha256(paths["trials"])):
        raise ValueError(f"Validation evidence identity mismatch in {directory}")
    return manifest, {key: _sha256(path) for key, path in paths.items()}


def load_selector(path, device="cpu"):
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != SELECTOR_SCHEMA:
        raise ValueError("Unsupported unified selector schema")
    required = {
        "hidden_dims", "input_fields", "input_ranges", "gait_levels",
        "gait_min_period_ticks", "supported_contexts", "deployment_ready",
        "selector_source_sha256", "selector_runtime_sha256",
        "selector_state_sha256", "grid_task_sha256", "grid_checkpoint_sha256",
        "grid_collector_sha256", "trial_csv_sha256", "grid_manifest_sha256",
        "grid_complete_sha256",
    }
    if required - set(payload):
        raise ValueError(f"Selector checkpoint lacks provenance: {sorted(required - set(payload))}")
    set_width_range(payload["input_ranges"]["step_width"])
    if payload["input_fields"] != ["step_width", "speed", "period", "gait_id"]:
        raise ValueError("Selector input fields differ from this implementation")
    set_gait_levels(payload["gait_levels"])
    if payload["gait_min_period_ticks"] != list(GAIT_MIN_PERIOD_TICKS):
        raise ValueError("Selector period floors differ from this implementation")
    if payload["selector_source_sha256"] != _sha256(Path(__file__)):
        raise ValueError("Selector source differs from the fitted implementation")
    root = Path(__file__).resolve().parents[4]
    if payload["selector_runtime_sha256"] != selector_runtime_sha256(root):
        raise ValueError("Selector runtime source bundle differs from the fit")
    if payload["deployment_ready"] is True and "validation_evidence" in payload:
        # One evidence folder per low-level checkpoint, copied next to the
        # promoted selector by validate_unified_selector.py.
        if payload.get("validation_all_contexts_passed") is not True:
            raise ValueError("Validated selector lacks successful rollout evidence")
        grid_checkpoints = payload["grid_checkpoint_sha256"]
        for evidence in payload["validation_evidence"]:
            expected_ckpt = (grid_checkpoints[evidence["gaits"][0]]
                             if isinstance(grid_checkpoints, dict) else grid_checkpoints)
            if evidence["low_level_checkpoint_sha256"] != expected_ckpt:
                raise ValueError("Validation evidence checkpoint differs from the fit")
            _, digests = _check_validation_directory(
                path.parent / evidence["directory"], payload,
                payload["validation_selector_checkpoint_sha256"], expected_ckpt)
            if (digests["manifest"] != evidence["manifest_sha256"]
                    or digests["trials"] != evidence["trials_sha256"]
                    or digests["completion"] != evidence["completion_sha256"]):
                raise ValueError("Validated selector evidence is missing or changed")
    elif payload["deployment_ready"] is True:
        fields = {
            "validation_manifest_sha256", "validation_trials_sha256",
            "validation_completion_sha256", "validation_selector_checkpoint_sha256",
            "validation_low_level_checkpoint_sha256", "validation_seed_start",
            "validation_trials_per_context", "validation_all_contexts_passed",
        }
        if (not fields <= set(payload)
                or payload["validation_all_contexts_passed"] is not True
                or payload["validation_low_level_checkpoint_sha256"]
                    != payload["grid_checkpoint_sha256"]):
            raise ValueError("Validated selector lacks successful rollout evidence")
        paths = {
            "manifest": path.parent / "surface_manifest.json",
            "trials": path.parent / "surface_trials.csv",
            "completion": path.parent / "SURFACE_COMPLETE",
        }
        if (not all(item.is_file() for item in paths.values())
                or _sha256(paths["manifest"]) != payload["validation_manifest_sha256"]
                or _sha256(paths["trials"]) != payload["validation_trials_sha256"]
                or _sha256(paths["completion"]) != payload["validation_completion_sha256"]):
            raise ValueError("Validated selector evidence is missing or changed")
        manifest = json.loads(paths["manifest"].read_text())
        if (manifest.get("schema") != VALIDATION_SCHEMA
                or manifest.get("selector_checkpoint_sha256")
                    != payload["validation_selector_checkpoint_sha256"]
                or manifest.get("checkpoint_sha256") != payload["grid_checkpoint_sha256"]):
            raise ValueError("Validated selector evidence identity mismatch")
    model = UnifiedDutySelector(tuple(payload["hidden_dims"])).to(device)
    model.load_state_dict(payload["state_dict"])
    if selector_state_sha256(model.state_dict()) != payload["selector_state_sha256"]:
        raise ValueError("Selector parameter hash mismatch")
    model.eval()
    return model, payload


__all__ = [
    "CONTEXT_FIELDS", "GAIT_LEVELS", "LEVELS", "PERIOD_RANGE", "set_gait_levels",
    "PRIMARY_TRIALS_PER_CANDIDATE", "SELECTOR_SCHEMA", "VALIDATION_SCHEMA",
    "VALIDATION_COMPLETE_SCHEMA", "UnifiedDutySelector",
    "aggregate_candidates", "bootstrap_target_intervals", "duty_cap",
    "feasible_levels", "fit_selector", "level_index", "load_selector",
    "predict_rows", "predict_supported_rows", "read_trial_rows",
    "select_targets", "selector_payload", "selector_runtime_sha256",
    "selector_state_sha256", "validate_completed_surface_grid",
    "validate_context_values",
]
