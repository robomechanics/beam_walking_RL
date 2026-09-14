"""Offline analysis of measured flat-ground gait-command trials."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .protocol import GAIT_OFFSETS

LEGS = ("FL", "FR", "RL", "RR")
OFFSETS = (0., .5, .5, 0.)
PERIOD = .48
DT = .02
LEGACY_NOMINAL_SCHEMA = "nominal_flat_gait_eval_v2"
V3_NOMINAL_SCHEMA = "nominal_flat_gait_eval_v3"
NOMINAL_SCHEMA = "nominal_flat_gait_eval_v4"
FORCE_THRESHOLDS_N = (2., 5., 10.)
FORCE_SAMPLE_DT = .005
FORCE_SAMPLES_PER_CONTROL = 4
NOMINAL_EVALUATOR_PATHS = (
    "scripts/evaluate_policy.py",
    "scripts/evaluation_capacity.py",
    "scripts/analyze_beam.py",
    "source/beam_walking/beam_walking/experiment/analysis.py",
    "scripts/gpu_capacity.py",
    "source/beam_walking/beam_walking/experiment/deployment.py",
    "source/beam_walking/beam_walking/experiment/deployment_task.py",
    "source/beam_walking/beam_walking/experiment/deployment_actuator.py",
)


def nominal_evaluation_source_hash(root):
    root = Path(root)
    return hashlib.sha256(b"".join(
        (root / relative).read_bytes()
        for relative in NOMINAL_EVALUATOR_PATHS)).hexdigest()


def oldest_first_force_history(history):
    """Reverse Isaac's newest-first history axis without changing other axes."""
    if getattr(history, "ndim", 0) < 2:
        raise ValueError("Force history requires an environment and history axis")
    if hasattr(history, "flip"):
        return history.flip(1)
    return np.flip(np.asarray(history), axis=1).copy()


def include_post_step_video_frame(step, was_active, done, stride=2):
    """Exclude terminal steps because Isaac has already reset them on return."""
    if stride < 1:
        raise ValueError("Video stride must be positive")
    return step % stride == 0 and bool(was_active) and not bool(done)


def nominal_archive_payload(traces, metadata):
    """Build an unambiguous nominal-rollout archive.

    The environment's legacy ``speed`` transition is body-frame forward
    velocity. Store it under an explicit trace name so ``command_speed`` can
    remain a scalar run-identity field.
    """
    payload = {}
    for key, values in traces.items():
        archive_key = "body_forward_velocity" if key == "speed" else key
        if archive_key in payload:
            raise ValueError(f"Duplicate nominal archive field: {archive_key}")
        payload[archive_key] = np.stack(values)
    overlap = set(payload) & set(metadata)
    if overlap:
        raise ValueError(f"Trace/metadata field collision: {sorted(overlap)}")
    payload.update(metadata)
    if "body_forward_velocity" not in payload or "command_speed" not in payload:
        raise ValueError(
            "Nominal archive requires body-forward trace and scalar command speed")
    return payload


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0., center - radius), min(1., center + radius)


def complete_cycle_df(times, contacts, offset, eligible=None, period=PERIOD,
                      sample_dt=DT):
    """Integrate end-of-step contact over fully observed cycles after settling.

    Each sample at tick k represents (k-1,k]. Fractional walk phase offsets
    receive interval-overlap weights; the closing boundary must be observed.
    """
    ticks = np.rint(np.asarray(times) / sample_dt).astype(int)
    count = round(period / sample_dt)
    if count < 1 or not np.isclose(count * sample_dt, period):
        raise ValueError("Period must be a positive integer number of control steps")
    if not len(ticks):
        return []
    if not np.all(np.diff(ticks) > 0):
        raise ValueError("Contact times must be strictly increasing")
    values = []
    for cycle in range(int(np.floor(ticks[0] / count + offset)) - 1,
                       int(np.floor(ticks[-1] / count + offset)) + 1):
        lower, upper = (cycle - offset) * count, (cycle + 1 - offset) * count
        if lower < count or lower < ticks[0] - 1 or upper > ticks[-1]:
            continue
        required = np.arange(int(np.floor(lower)) + 1, int(np.ceil(upper)) + 1)
        selected = np.isin(ticks, required)
        if not np.array_equal(ticks[selected], required):
            continue
        if eligible is not None and not np.asarray(eligible)[selected].all():
            continue
        weights = np.minimum(ticks[selected], upper) - np.maximum(ticks[selected] - 1, lower)
        values.append(float(np.sum(np.asarray(contacts)[selected] * weights) / count))
    return values


def topology_cycle_fraction(times, contacts, reference, gait, period=PERIOD,
                            sample_dt=FORCE_SAMPLE_DT, tolerance=DT):
    """Fraction of complete cycles retaining the 5 N hybrid-event topology."""
    ticks = np.rint(np.asarray(times) / sample_dt).astype(int)
    count = round(period / sample_dt)
    tolerance_ticks = round(tolerance / sample_dt)
    contacts = np.asarray(contacts, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if (contacts.shape != reference.shape or contacts.shape != (len(ticks), 4)
            or not np.array_equal(np.diff(ticks), np.ones(max(0, len(ticks) - 1), dtype=int))):
        raise ValueError("Topology inputs must be aligned contiguous four-leg contacts")
    if gait not in ("trot", "walk"):
        raise ValueError("Unsupported gait topology")

    eligible = good = 0
    first_cycle = max(1, int(np.ceil((ticks[0] + 1) / count)))
    last_cycle = int(np.floor(ticks[-1] / count))
    for cycle in range(first_cycle, last_cycle):
        lower, upper = cycle * count, (cycle + 1) * count
        selected = (ticks >= lower - 1) & (ticks <= upper)
        required = np.arange(lower - 1, upper + 1)
        if not np.array_equal(ticks[selected], required):
            continue
        eligible += 1
        candidate_events, reference_events = {}, {}
        for label, values, destination in (
                ("candidate", contacts[selected], candidate_events),
                ("reference", reference[selected], reference_events)):
            changes = values[1:] != values[:-1]
            event_ticks = required[1:]
            for leg in range(4):
                rising = event_ticks[changes[:, leg] & values[1:, leg]]
                falling = event_ticks[changes[:, leg] & ~values[1:, leg]]
                rising = rising[rising > lower]
                falling = falling[falling > lower]
                destination[(leg, "rise")] = rising
                destination[(leg, "fall")] = falling
        cycle_good = True
        for leg in range(4):
            for event in ("rise", "fall"):
                observed = candidate_events[(leg, event)]
                baseline = reference_events[(leg, event)]
                if (len(observed) != 1 or len(baseline) != 1
                        or abs(int(observed[0]) - int(baseline[0])) > tolerance_ticks):
                    cycle_good = False
        if cycle_good and gait == "trot":
            for events in (candidate_events, reference_events):
                for event in ("rise", "fall"):
                    if (abs(int(events[(0, event)][0]) - int(events[(3, event)][0]))
                            > tolerance_ticks
                            or abs(int(events[(1, event)][0]) - int(events[(2, event)][0]))
                            > tolerance_ticks):
                        cycle_good = False
        if cycle_good and gait == "walk":
            for event in ("rise", "fall"):
                candidate_order = sorted(
                    range(4), key=lambda leg: candidate_events[(leg, event)][0])
                reference_order = sorted(
                    range(4), key=lambda leg: reference_events[(leg, event)][0])
                if candidate_order != reference_order:
                    cycle_good = False
        good += int(cycle_good)
    return (good / eligible if eligible else np.nan), good, eligible


def schedule_centered_topology_fraction(
        times, contacts, desired, gait, period=PERIOD,
        sample_dt=FORCE_SAMPLE_DT, tolerance=DT, *,
        initial_contacts=None, initial_desired=None, circular=False):
    """Match observed contact events uniquely to scheduled gait events.

    Events are owned by their nearest same-leg, same-direction scheduled event,
    rather than by a fixed cycle interval.  This makes an event just before a
    nominal cycle boundary belong to the event just after that boundary.  The
    matching remains strict: missing, duplicate, late, or unmatched events fail
    the affected cycle.  ``circular`` evaluates a one-period Poincare trace on
    the phase circle and is used by the stability and energy evaluators.
    """
    ticks_float = np.asarray(times, dtype=float) / sample_dt
    ticks = np.rint(ticks_float).astype(int)
    count = round(period / sample_dt)
    tolerance_ticks = round(tolerance / sample_dt)
    contacts = np.asarray(contacts, dtype=bool)
    desired = np.asarray(desired, dtype=bool)
    if (count < 1 or not np.isclose(count * sample_dt, period)
            or tolerance_ticks < 0
            or not np.isclose(tolerance_ticks * sample_dt, tolerance)):
        raise ValueError("Invalid topology period or event tolerance")
    if (contacts.shape != desired.shape
            or contacts.shape != (len(ticks), 4)
            or not np.allclose(ticks_float, ticks, atol=1e-6)
            or not np.array_equal(
                np.diff(ticks), np.ones(max(0, len(ticks) - 1), dtype=int))):
        raise ValueError(
            "Topology inputs must be aligned contiguous four-leg contacts")
    if gait not in ("trot", "walk"):
        raise ValueError("Unsupported gait topology")
    if circular and len(ticks) != count:
        raise ValueError("Circular topology requires exactly one gait period")
    if (initial_contacts is None) != (initial_desired is None):
        raise ValueError("Both topology initial states must be supplied together")
    if initial_contacts is not None:
        initial_contacts = np.asarray(initial_contacts, dtype=bool)
        initial_desired = np.asarray(initial_desired, dtype=bool)
        if initial_contacts.shape != (4,) or initial_desired.shape != (4,):
            raise ValueError("Topology initial contact states must have shape [4]")

    def events(values, initial):
        if initial is None:
            current, previous, event_ticks = values[1:], values[:-1], ticks[1:]
        else:
            current, previous, event_ticks = values, np.concatenate(
                [initial[None], values[:-1]], axis=0), ticks
        changes = current != previous
        return {
            (leg, direction): event_ticks[
                changes[:, leg] & (current[:, leg] == state)]
            for leg in range(4)
            for direction, state in (("rise", True), ("fall", False))
        }

    observed = events(contacts, initial_contacts)
    expected = events(desired, initial_desired)

    if circular:
        # Express the trace on [0, count).  Circular distance assigns an event
        # at the end of the period to the scheduled event at its beginning.
        origin = ticks[0] - 1

        def phase(value):
            return (int(value) - origin) % count

        def signed_distance(value, target):
            delta = (phase(value) - phase(target)) % count
            return delta - count if delta > count / 2 else delta

        cycle_expected = {key: value for key, value in expected.items()}
        if any(len(value) != 1 for value in cycle_expected.values()):
            raise ValueError(
                "Circular desired schedule must have one rise and fall per leg")
        matched = {}
        cycle_good = True
        for key, targets in cycle_expected.items():
            target = int(targets[0])
            candidates = np.asarray(observed[key], dtype=int)
            # Every observed event is owned by this sole circular event.
            if len(candidates) != 1:
                cycle_good = False
                continue
            delta = signed_distance(int(candidates[0]), target)
            if abs(delta) > tolerance_ticks:
                cycle_good = False
            matched[key] = phase(target) + delta
        cycles = [(0, cycle_expected, matched, cycle_good)]
    else:
        all_expected = np.concatenate(
            [value for value in expected.values()]) if expected else np.asarray([])
        if not len(all_expected):
            return np.nan, 0, 0
        first_cycle = int(np.floor((int(all_expected.min()) - 1) / count))
        last_cycle = int(np.floor((int(all_expected.max()) - 1) / count))
        cycles = []
        for cycle in range(first_cycle, last_cycle + 1):
            cycle_expected = {
                key: value[((value - 1) // count) == cycle]
                for key, value in expected.items()
            }
            if any(len(value) != 1 for value in cycle_expected.values()):
                continue
            # The first/last scheduled cycles need neighbouring same-direction
            # events to own boundary jitter unambiguously.
            if cycle <= first_cycle or cycle >= last_cycle:
                continue
            matched, cycle_good = {}, True
            for key, targets in cycle_expected.items():
                target = int(targets[0])
                candidates = np.asarray(observed[key], dtype=int)
                reference_events = np.asarray(expected[key], dtype=int)
                owned = []
                for candidate in candidates:
                    distances = np.abs(reference_events - candidate)
                    nearest = np.flatnonzero(distances == distances.min())
                    # Attribute a midpoint tie to both neighbours so the
                    # unmatched extra transition makes both cycles fail.
                    if target in {int(reference_events[index])
                                  for index in nearest}:
                        owned.append(int(candidate))
                if len(owned) != 1 or abs(owned[0] - target) > tolerance_ticks:
                    cycle_good = False
                elif len(owned) == 1:
                    matched[key] = owned[0]
            cycles.append((cycle, cycle_expected, matched, cycle_good))

    good = 0
    for _, scheduled, matched, initially_good in cycles:
        cycle_good = initially_good and len(matched) == 8
        if cycle_good and gait == "trot":
            for event in ("rise", "fall"):
                if (abs(matched[(0, event)] - matched[(3, event)])
                        > tolerance_ticks
                        or abs(matched[(1, event)] - matched[(2, event)])
                        > tolerance_ticks):
                    cycle_good = False
        if cycle_good and gait == "walk":
            for event in ("rise", "fall"):
                observed_order = sorted(
                    range(4), key=lambda leg: matched[(leg, event)])
                scheduled_order = sorted(
                    range(4), key=lambda leg: int(scheduled[(leg, event)][0]))
                cyclic_orders = [
                    scheduled_order[index:] + scheduled_order[:index]
                    for index in range(4)]
                if observed_order not in cyclic_orders:
                    cycle_good = False
        good += int(cycle_good)
    eligible = len(cycles)
    return (good / eligible if eligible else np.nan), good, eligible


def validate_manifest(directory):
    """Reject missing cells, mixed checkpoints, and unmatched trial plans."""
    directory = Path(directory)
    manifest = json.loads((directory / "evaluation_manifest.json").read_text())
    schema = manifest.get("schema")
    if schema not in (
            None, LEGACY_NOMINAL_SCHEMA, V3_NOMINAL_SCHEMA, NOMINAL_SCHEMA):
        raise ValueError(f"Unsupported evaluation manifest schema: {schema}")
    versioned = schema in (
        LEGACY_NOMINAL_SCHEMA, V3_NOMINAL_SCHEMA, NOMINAL_SCHEMA)
    force_versioned = schema in (V3_NOMINAL_SCHEMA, NOMINAL_SCHEMA)
    v4 = schema == NOMINAL_SCHEMA
    frame = manifest.get("step_width_frame", "world")
    if frame not in ("world", "body"):
        raise ValueError("Invalid manifest step_width_frame")
    required = [
        "expected_conditions", "seeds", "period", "gait",
        "source_sha256", "checkpoint_sha256",
    ]
    if any(key not in manifest for key in required):
        raise ValueError("Evaluation manifest is missing required run identity")
    if versioned:
        v2_required = [
            "evaluator_sha256", "training_provenance_sha256",
            "split", "condition_reset_seed",
        ]
        if any(key not in manifest for key in v2_required):
            raise ValueError("Versioned manifest is missing required provenance")
        if manifest.get("terrain") != "flat_ground":
            raise ValueError("Versioned evaluation must use flat ground")
        if manifest.get("external_pushes") is not False:
            raise ValueError("Versioned evaluation must disable external pushes")
        if manifest["split"] not in ("validation", "test"):
            raise ValueError("Invalid held-out split")
    cells = manifest["expected_conditions"]
    conditions = [(cell["step_width"], cell["df"], cell["disturbed"]) for cell in cells]
    if len(conditions) != len(set(conditions)):
        raise ValueError("Duplicate evaluation condition in manifest")
    if versioned and any(bool(cell["disturbed"]) for cell in cells):
        raise ValueError("Versioned evaluation conditions must all be nominal")
    names = [cell["filename"] for cell in cells]
    if not names or len(names) != len(set(names)):
        raise ValueError("Expected filenames must be nonempty and unique")
    if set(names) != {path.name for path in directory.glob("trial_*.npz")}:
        raise ValueError("Measured files do not exactly match the evaluation manifest")
    seeds = np.asarray(manifest["seeds"])
    if not len(seeds) or len(seeds) != len(set(seeds.tolist())):
        raise ValueError("Trial seeds must be nonempty and unique")
    if versioned:
        low, high = ((10000, 1000000) if manifest["split"] == "validation"
                     else (1000000, 2000000))
        if seeds[0] < low or seeds[-1] >= high or not np.array_equal(
                seeds, np.arange(seeds[0], seeds[0] + len(seeds))):
            raise ValueError("Versioned seeds must be contiguous and inside the held-out split")
        provenance_path = directory / "training_provenance.json"
        if (not provenance_path.is_file() or hashlib.sha256(
                provenance_path.read_bytes()).hexdigest()
                != manifest["training_provenance_sha256"]):
            raise ValueError("Archived training provenance hash mismatch")
        complete_path = directory / "evaluation_complete.json"
        if not complete_path.is_file():
            raise ValueError("Versioned evaluation is missing its completion marker")
        complete = json.loads(complete_path.read_text())
        expected_complete = {
            "complete": True, "schema": schema,
            "conditions": len(names), "trials_per_condition": len(seeds),
            "source_sha256": manifest["source_sha256"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "evaluator_sha256": manifest["evaluator_sha256"],
            "training_provenance_sha256":
                manifest["training_provenance_sha256"],
        }
        # Evaluations made after the perturbation feature also record the
        # push flag and profile (None for nominal runs) in the completion
        # marker; older markers lack them. Either way they must match the
        # manifest.
        for key in ("external_pushes", "perturbation"):
            if key in complete:
                if complete[key] != manifest.get(key):
                    raise ValueError(f"Completion marker {key} differs from manifest")
                expected_complete[key] = complete[key]
        deployment_evaluation = manifest.get("deployment_evaluation_profile")
        if deployment_evaluation is not None:
            from .deployment import (
                deployment_profile, deployment_profile_sha256,
            )
            required_deployment = (
                "deployment_training_required", "deployment_training_verified",
                "deployment_profile", "deployment_profile_sha256",
                "deployment_training_source_sha256", "kp_scale", "kd_scale",
            )
            if any(key not in manifest for key in required_deployment):
                raise ValueError("Deployment evaluation identity is incomplete")
            if deployment_evaluation not in (
                    "nominal", "randomized", "fixed_gains"):
                raise ValueError("Unknown deployment evaluation profile")
            if (manifest["deployment_training_required"] is not True
                    or manifest["deployment_training_verified"] is not True
                    or manifest["deployment_profile"] != deployment_profile()
                    or manifest["deployment_profile_sha256"]
                        != deployment_profile_sha256()):
                raise ValueError("Frozen deployment profile identity mismatch")
            if deployment_evaluation == "fixed_gains":
                if (manifest["kp_scale"] is None
                        or manifest["kd_scale"] is None):
                    raise ValueError("Fixed-gain deployment identity is incomplete")
            elif (manifest["kp_scale"] is not None
                  or manifest["kd_scale"] is not None):
                raise ValueError("Non-fixed deployment profile contains gains")
            training = json.loads(provenance_path.read_text())
            if (training.get("deployment_profile") != manifest["deployment_profile"]
                    or training.get("deployment_profile_sha256")
                        != manifest["deployment_profile_sha256"]
                    or training.get("training_source_sha256")
                        != manifest["deployment_training_source_sha256"]):
                raise ValueError("Archived deployment training identity mismatch")
            expected_complete.update({
                key: manifest[key] for key in (
                    "deployment_evaluation_profile",
                    "deployment_training_verified",
                    "deployment_profile_sha256",
                    "deployment_training_source_sha256",
                    "kp_scale", "kd_scale",
                )
            })
        if complete != expected_complete:
            raise ValueError("Versioned evaluation completion marker mismatch")
    if force_versioned:
        force_identity = {
            "force_instrumented": True,
            "force_field": "foot_force_norm_200hz", "force_units": "N",
            "force_sample_dt": FORCE_SAMPLE_DT,
            "force_samples_per_control": FORCE_SAMPLES_PER_CONTROL,
            "force_history_order": "oldest_to_newest",
            "force_leg_order": list(LEGS),
            "contact_thresholds_n": list(FORCE_THRESHOLDS_N),
            "policy_contact_threshold_n": 5.,
            "terminal_force_capture": "recorder_pre_reset",
            "scientific_role": "command_fidelity_prerequisite",
            "not_closed_loop_chi_evidence": True,
        }
        for key, expected in force_identity.items():
            if key not in manifest or not np.array_equal(
                    np.asarray(manifest[key]), np.asarray(expected)):
                raise ValueError(f"Force metadata mismatch: {key}")
        if v4:
            topology_identity = {
                "topology_definition":
                    "schedule_centered_unique_circular_events_v1",
                "topology_event_tolerance_s": DT,
                "topology_primary_threshold_n": 5.,
                "topology_gate_aggregation": "mean_trial_cycle_fraction",
                "topology_sensitivity_thresholds_are_diagnostic": True,
            }
            for key, expected in topology_identity.items():
                if key not in manifest or not np.array_equal(
                        np.asarray(manifest[key]), np.asarray(expected)):
                    raise ValueError(f"V4 topology metadata mismatch: {key}")
    reference, push_reference = None, None
    for cell in cells:
        path = directory / cell["filename"]
        if path.name != cell["filename"]:
            raise ValueError("Manifest filenames must be local basenames")
        with np.load(path) as data:
            data_frame, _, _ = validate_width_frame(data)
            if data_frame != frame:
                raise ValueError(f"Width frame mismatch: {path.name}")
            for key in ["period", "gait", "source_sha256", "checkpoint_sha256"]:
                if key not in data or data[key].item() != manifest[key]:
                    raise ValueError(f"Run identity mismatch: {path.name}, {key}")
            if versioned:
                identity = (
                    "schema", "terrain", "external_pushes", "split",
                    "evaluator_sha256", "training_provenance_sha256",
                    "condition_reset_seed",
                )
                for key in identity:
                    if key not in data or data[key].item() != manifest[key]:
                        raise ValueError(f"Versioned identity mismatch: {path.name}, {key}")
                forbidden = {"force", "planned_force", "push_plan"} & set(data.files)
                if forbidden:
                    raise ValueError(
                        f"Versioned nominal archive contains push fields: {sorted(forbidden)}")
                if manifest.get("deployment_evaluation_profile") is not None:
                    deployment_identity = (
                        "deployment_evaluation_profile",
                        "deployment_training_required",
                        "deployment_training_verified",
                        "deployment_profile_sha256",
                        "deployment_training_source_sha256",
                    )
                    for key in deployment_identity:
                        if key not in data or data[key].item() != manifest[key]:
                            raise ValueError(
                                f"Deployment identity mismatch: {path.name}, {key}")
                    if ("deployment_profile" not in data
                            or json.loads(data["deployment_profile"].item())
                                != manifest["deployment_profile"]):
                        raise ValueError(
                            f"Deployment profile mismatch: {path.name}")
                    for key in ("kp_scale", "kd_scale"):
                        saved = float(data[key])
                        expected = manifest[key]
                        if ((expected is None and not np.isnan(saved))
                                or (expected is not None
                                    and not np.isclose(saved, expected))):
                            raise ValueError(
                                f"Deployment gain mismatch: {path.name}, {key}")
            if force_versioned:
                for key in force_identity:
                    if key not in data or not np.array_equal(
                            np.asarray(data[key]), np.asarray(manifest[key])):
                        raise ValueError(f"Force identity mismatch: {path.name}, {key}")
                if v4:
                    for key in topology_identity:
                        if key not in data or not np.array_equal(
                                np.asarray(data[key]),
                                np.asarray(manifest[key])):
                            raise ValueError(
                                f"V4 topology identity mismatch: {path.name}, {key}")
                valid = np.asarray(data["valid"], dtype=bool)
                done = np.asarray(data["done"], dtype=bool)
                forces = np.asarray(data[manifest["force_field"]])
                preserved = np.asarray(data["terminal_force_preserved"], dtype=bool)
                if forces.shape != valid.shape + (FORCE_SAMPLES_PER_CONTROL, 4):
                    raise ValueError(f"Malformed 200 Hz force shape: {path.name}")
                if forces.dtype.kind != "f" or not np.isfinite(forces).all() or (forces < 0).any():
                    raise ValueError(f"Invalid 200 Hz forces: {path.name}")
                if preserved.shape != valid.shape or not np.array_equal(
                        preserved, valid & done):
                    raise ValueError(f"Terminal force preservation mismatch: {path.name}")
                contacts = np.asarray(data["contacts"], dtype=bool)
                if contacts.shape != valid.shape + (4,) or not np.array_equal(
                        forces[..., -1, :][valid] > 5., contacts[valid]):
                    raise ValueError(f"5 N force/contact reconstruction mismatch: {path.name}")
            if "speed" in manifest:
                if "command_speed" in data:
                    saved_speed = data["command_speed"].item()
                elif "speed" in data and data["speed"].ndim == 0:
                    saved_speed = data["speed"].item()
                elif "commands" in data and data["commands"].ndim == 3:
                    saved_speed = data["commands"][0, 0, 0].item()
                else:
                    raise ValueError(
                        f"Run identity mismatch: {path.name}, command speed")
                if not np.isclose(saved_speed, manifest["speed"]):
                    raise ValueError(f"Run identity mismatch: {path.name}, speed")
            for key in ["step_width", "df", "disturbed"]:
                if key not in data or data[key].item() != cell[key]:
                    raise ValueError(f"Condition mismatch: {path.name}, {key}")
            if not np.array_equal(data["seeds"], seeds):
                raise ValueError(f"Trial seed mismatch: {path.name}")
            if not np.isclose(float(data["control_dt"]), DT):
                raise ValueError("Unsupported contact sampling interval")
            plan_key = "reset_plan" if "reset_plan" in data else "push_plan"
            paired = {
                key: data[key].copy()
                for key in ["initial_root", "initial_joints", plan_key]
            }
            if reference is None:
                reference = paired
            elif any(not np.allclose(
                    paired[key], reference[key], atol=1e-6) for key in paired):
                raise ValueError(
                    f"Matched reset state or push plan mismatch: {path.name}")
            if not cell["disturbed"] and "planned_force" in data and np.any(data["planned_force"]):
                raise ValueError("Nominal condition contains a planned push")
            if cell["disturbed"]:
                if "planned_force" not in data:
                    raise ValueError("Legacy disturbed condition is missing its planned force")
                if push_reference is None:
                    push_reference = data["planned_force"].copy()
                elif not np.allclose(data["planned_force"], push_reference):
                    raise ValueError("Disturbed comparisons require identical planned forces")
    return [directory / name for name in names]


def validate_width_frame(data):
    """Verify declared axes and return independently recomputed body coordinates.

    Missing frame metadata denotes legacy world-axis measurements. Optional
    orientation is wxyz and maps body vectors into the world frame.
    """
    frame = str(np.asarray(data.get("step_width_frame", "world")).item())
    if frame not in ("world", "body"):
        raise ValueError("Invalid step_width_frame")
    if frame == "body" and any(key not in data for key in ("root_quat", "feet_body")):
        raise ValueError("Body width requires root_quat and feet_body")
    if "root_quat" not in data:
        if "feet_body" in data:
            raise ValueError("Cannot verify feet_body without root_quat")
        return frame, None, None
    valid = np.asarray(data["valid"], dtype=bool)
    quat = np.asarray(data["root_quat"])
    feet, body = np.asarray(data["feet"]), np.asarray(data["body"])
    if (quat.shape != valid.shape + (4,) or feet.shape != valid.shape + (4, 3)
            or body.shape != valid.shape + (3,)):
        raise ValueError("Malformed root_quat, feet, or body shape")
    if (not np.isfinite(quat[valid]).all()
            or not np.allclose(np.linalg.norm(quat[valid], axis=-1), 1., atol=1e-4)):
        raise ValueError("root_quat must contain finite unit wxyz quaternions")
    relative = feet - body[..., None, :]
    # Inverse quaternion rotation, using the conjugated vector component.
    vector = -quat[..., None, 1:]
    cross = 2 * np.cross(vector, relative)
    recomputed = relative + quat[..., None, :1] * cross + np.cross(vector, cross)
    if not np.isfinite(recomputed[valid]).all():
        raise ValueError("Nonfinite reconstructed body foot coordinates")
    if "feet_body" in data:
        recorded = np.asarray(data["feet_body"])
        if recorded.shape != feet.shape or not np.allclose(recorded[valid], recomputed[valid], atol=2e-5, rtol=1e-4):
            raise ValueError("feet_body does not match world feet/root_quat rotation")
    w, x, y, z = np.moveaxis(quat, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return frame, recomputed, yaw


def trial_rows(path):
    with np.load(path) as archive:
        d = {key: archive[key] for key in archive.files}
    frame, body_feet, yaw = validate_width_frame(d)
    period, gait = float(d["period"]), str(d["gait"])
    offsets_by_gait = {"trot": GAIT_OFFSETS[0], "walk": GAIT_OFFSETS[1]}
    if gait not in offsets_by_gait or not np.allclose(d["phase_offsets"], offsets_by_gait[gait]):
        raise ValueError(f"Invalid gait or phase offsets: {path}")
    body_speed_key = (
        "body_forward_velocity" if "body_forward_velocity" in d else "speed")
    body_speed = np.asarray(d[body_speed_key])
    if body_speed.shape != d["valid"].shape:
        raise ValueError(f"Malformed body-forward velocity trace: {path}")
    rows = []
    for trial in range(d["valid"].shape[1]):
        mask = d["valid"][:, trial].astype(bool)
        count = int(mask.sum())
        if count == 0 or not mask[:count].all() or mask[count:].any():
            raise ValueError(f"Invalid first-episode mask: {path}, trial {trial}")
        done = d["done"][mask, trial].astype(bool)
        if done[:-1].any() or not done[-1]:
            raise ValueError(f"Incomplete or multiple terminal episodes: {path}, trial {trial}")
        t, body, feet = (d[key][mask, trial] for key in ["time", "body", "feet"])
        ct, requested, cmds = (d[key][mask, trial] for key in ["contacts", "desired", "commands"])
        if not np.allclose(np.diff(t), DT, atol=2e-6) or not np.isfinite(body).all():
            raise ValueError(f"Invalid trajectory samples: {path}")
        if not np.allclose(cmds, cmds[0]) or not np.allclose(cmds[:, 3], period):
            raise ValueError(f"Evaluation commands must remain fixed: {path}")
        if not np.allclose(cmds[:, 1], float(d["df"])) or not np.allclose(cmds[:, 2], float(d["step_width"])):
            raise ValueError(f"Command metadata mismatch: {path}")
        if not np.allclose(cmds[:, 4], ("trot", "walk").index(gait)):
            raise ValueError(f"Gait command metadata mismatch: {path}")
        settled = t >= period
        stance = settled[:, None] & ct
        threshold_metrics = {}
        archive_schema = str(np.asarray(d.get("schema", "")).item())
        if archive_schema in (V3_NOMINAL_SCHEMA, NOMINAL_SCHEMA):
            force = np.asarray(d["foot_force_norm_200hz"])[mask, trial]
            high_times = (t[:, None] + FORCE_SAMPLE_DT * np.arange(
                1 - FORCE_SAMPLES_PER_CONTROL, 1)[None, :]).reshape(-1)
            high_requested = np.repeat(requested, FORCE_SAMPLES_PER_CONTROL, axis=0)
            high_settled = high_times >= period
            contacts_by_threshold = {
                threshold: (force > threshold).reshape(-1, 4)
                for threshold in FORCE_THRESHOLDS_N
            }
            for threshold in FORCE_THRESHOLDS_N:
                tag = f"force{int(threshold)}n"
                high_contacts = contacts_by_threshold[threshold]
                last_contacts = force[:, -1] > threshold
                if archive_schema == NOMINAL_SCHEMA:
                    topology_fraction, topology_good, topology_cycles = (
                        schedule_centered_topology_fraction(
                            high_times, high_contacts, high_requested, gait,
                            period=period))
                else:
                    # V3 remains reproducible under its archived fixed-bin,
                    # 5 N reference definition.
                    topology_fraction, topology_good, topology_cycles = (
                        topology_cycle_fraction(
                            high_times, high_contacts,
                            contacts_by_threshold[5.], gait, period=period))
                threshold_metrics[f"{tag}_topology_fraction"] = topology_fraction
                threshold_metrics[f"{tag}_topology_good_cycles"] = topology_good
                threshold_metrics[f"{tag}_topology_cycles"] = topology_cycles
                for leg, name in enumerate(LEGS):
                    values = complete_cycle_df(
                        high_times, high_contacts[:, leg],
                        float(d["phase_offsets"][leg]), period=period,
                        sample_dt=FORCE_SAMPLE_DT)
                    threshold_metrics[f"{tag}_df_{name}"] = (
                        float(np.mean(values)) if values else np.nan)
                    threshold_metrics[f"{tag}_cycles_{name}"] = len(values)
                    for state, label in ((True, "stance_recall"),
                                         (False, "swing_recall")):
                        selected = (high_requested[:, leg] == state) & high_settled
                        threshold_metrics[f"{tag}_{label}_{name}"] = (
                            float((high_contacts[selected, leg] == state).mean())
                            if selected.any() else np.nan)
                for axes, y in [
                    ("world", feet[:, :, 1] - body[:, 1:2]),
                    ("body", body_feet[mask, trial, :, 1]
                     if body_feet is not None else None),
                ]:
                    width, mae = np.nan, np.nan
                    if y is not None:
                        leg_y = [
                            float(y[settled & last_contacts[:, leg], leg].mean())
                            if (settled & last_contacts[:, leg]).any() else np.nan
                            for leg in range(4)
                        ]
                        width = float(
                            np.mean([leg_y[0], leg_y[2]])
                            - np.mean([leg_y[1], leg_y[3]]))
                        errors = np.abs(
                            y - cmds[:, 2:3] * np.array([1, -1, 1, -1]) / 2)
                        selected = settled[:, None] & last_contacts
                        mae = float(errors[selected].mean()) if selected.any() else np.nan
                    threshold_metrics[f"{tag}_{axes}_achieved_width"] = width
                    threshold_metrics[f"{tag}_{axes}_foot_lateral_mae"] = mae
        motion_metrics = {}
        motion_keys = ("body_lateral_velocity", "body_yaw_rate")
        present_motion = [key in d for key in motion_keys]
        if any(present_motion) and not all(present_motion):
            raise ValueError(f"Body lateral velocity and yaw rate must be recorded together: {path}")
        if all(present_motion):
            for key in motion_keys:
                raw = np.asarray(d[key])
                if raw.shape != d["valid"].shape or not np.isfinite(raw[d["valid"]]).all():
                    raise ValueError(f"Malformed {key}: {path}")
                values = raw[mask, trial][settled]
                motion_metrics[f"{key}_mean"] = float(values.mean()) if values.size else np.nan
                motion_metrics[f"{key}_rmse"] = float(np.sqrt(np.mean(values ** 2))) if values.size else np.nan
        if "world_lateral_velocity" in d:
            raw = np.asarray(d["world_lateral_velocity"])
            if raw.shape != d["valid"].shape or not np.isfinite(raw[d["valid"]]).all():
                raise ValueError(f"Malformed world_lateral_velocity: {path}")
            values = raw[mask, trial][settled]
            motion_metrics["world_lateral_velocity_mean"] = float(values.mean()) if values.size else np.nan
            motion_metrics["world_lateral_velocity_rmse"] = float(np.sqrt(np.mean(values ** 2))) if values.size else np.nan
        width_metrics = {}
        for axes, y in [("world", feet[:, :, 1] - body[:, 1:2]),
                        ("body", body_feet[mask, trial, :, 1] if body_feet is not None else None)]:
            width, mae = np.nan, np.nan
            if y is not None:
                leg_y = [float(y[settled & ct[:, j], j].mean()) if (settled & ct[:, j]).any() else np.nan for j in range(4)]
                width = float(np.mean([leg_y[0], leg_y[2]]) - np.mean([leg_y[1], leg_y[3]]))
                errors = np.abs(y - cmds[:, 2:3] * np.array([1, -1, 1, -1]) / 2)
                mae = float(errors[stance].mean()) if stance.any() else np.nan
            width_metrics[f"{axes}_achieved_width"] = width
            width_metrics[f"{axes}_foot_lateral_mae"] = mae
        failure = bool(d["failure"][mask, trial][-1])
        success = bool(d["success"][mask, trial][-1]) and not failure
        row = {
            "file": path.name, "trial": trial, "seed": int(d["seeds"][trial]),
            "step_width": float(d["step_width"]), "command_df": float(d["df"]),
            "period": period, "gait": gait, "disturbed": bool(d["disturbed"]),
            "success": int(success), "failure": int(failure), "timeout": int(not success and not failure),
            "duration": float(t[-1]), "traversal_time": float(t[-1]) if success else np.nan,
            "course_distance": float(np.clip(body[:, 0].max(), 0, 3.35)),
            "forward_distance": float(max(0., body[:, 0].max() + .65)),
            "mean_speed": float(body_speed[mask, trial].mean()),
            "forward_speed": float(d["forward_velocity"][mask, trial][settled].mean()) if settled.any() else np.nan,
            "settled_body_forward_speed": float(body_speed[mask, trial][settled].mean()) if settled.any() else np.nan,
            "command_speed": float(cmds[0, 0]), "command_width": float(cmds[0, 2]),
            "step_width_frame": frame,
            "achieved_width": width_metrics[f"{frame}_achieved_width"],
            "foot_lateral_mae": width_metrics[f"{frame}_foot_lateral_mae"],
            **width_metrics,
            "heading_rmse_rad": float(np.sqrt(np.mean(yaw[mask, trial] ** 2))) if yaw is not None else np.nan,
            "max_abs_heading_rad": float(np.abs(yaw[mask, trial]).max()) if yaw is not None else np.nan,
            "settled_samples": int(settled.sum()),
            "lateral_rmse": float(np.sqrt(np.mean(body[:, 1] ** 2))),
            "max_lateral_deviation": float(np.max(np.abs(body[:, 1]))),
            "timing_accuracy": float((ct[settled] == requested[settled]).mean()) if settled.any() else np.nan,
            **motion_metrics,
            "applied_impulse": (float(np.linalg.norm(d["force"][mask, trial], axis=-1).sum() * DT)
                                if "force" in d else 0.0),
            "planned_force": (float(np.linalg.norm(d["planned_force"][trial]))
                              if "planned_force" in d else 0.0),
            "planned_push_duration": (float(d["push_plan"][trial, 5])
                                      if "push_plan" in d else 0.0),
            "planned_push_time": (float(d["push_plan"][trial, 2])
                                      if "push_plan" in d else np.nan),
            **threshold_metrics,
        }
        for leg, name in enumerate(LEGS):
            for values_key, prefix in [("contacts", "df"), ("desired", "schedule_df")]:
                values = complete_cycle_df(t, d[values_key][mask, trial, leg], float(d["phase_offsets"][leg]), period=period)
                row[f"{prefix}_{name}"] = float(np.mean(values)) if values else np.nan
                if prefix == "df":
                    row[f"cycles_{name}"] = len(values)
            for state, label in [(True, "stance_recall"), (False, "swing_recall")]:
                selected = (requested[:, leg] == state) & settled
                row[f"{label}_{name}"] = float((ct[selected, leg] == state).mean()) if selected.any() else np.nan
        rows.append(row)
    return rows


def analyze(directory):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    files = validate_manifest(directory)
    manifest = json.loads((directory / "evaluation_manifest.json").read_text())
    v4 = manifest.get("schema") == NOMINAL_SCHEMA
    table = pd.DataFrame([row for path in files for row in trial_rows(path)])
    for key in ["period", "gait", "command_speed", "step_width_frame"]:
        if table[key].nunique() != 1:
            raise ValueError(f"Analyze one fixed {key} per directory")
    table["achieved_df"] = table[[f"df_{x}" for x in LEGS]].mean(axis=1)
    table["df_max_abs_error"] = table[[f"df_{x}" for x in LEGS]].sub(table.command_df, axis=0).abs().max(axis=1)
    # Frozen straight-path gates. Missing legacy diagnostics fail rather than
    # silently passing a controller that turns or weaves.
    for key in (
        "heading_rmse_rad", "max_abs_heading_rad",
        "world_lateral_velocity_rmse", "body_yaw_rate_rmse",
    ):
        if key not in table:
            table[key] = np.nan
    invariant_locomotion_pass = (
        table.settled_samples.ge(np.rint(table.period / DT))
        & table.lateral_rmse.le(.10)
        & table.max_lateral_deviation.le(.20)
        & table.heading_rmse_rad.le(.10)
        & table.max_abs_heading_rad.le(.25)
        & table.world_lateral_velocity_rmse.le(.10)
        & table.body_yaw_rate_rmse.le(.50)
        & (table.forward_speed - table.command_speed).abs().le(.04))
    primary_contact_pass = (
        table[[f"cycles_{x}" for x in LEGS]].ge(1).all(axis=1)
        & table[[f"df_{x}" for x in LEGS]].sub(table.command_df, axis=0).abs().le(.05).all(axis=1)
        & table[[f"swing_recall_{x}" for x in LEGS]].ge(.9).all(axis=1)
        & table[[f"stance_recall_{x}" for x in LEGS]].ge(.9).all(axis=1)
        & table.foot_lateral_mae.le(.015)
        & (table.achieved_width - table.command_width).abs().le(.03)
    )
    # Preserve the preregistered frozen 50 Hz/5 N gate exactly.
    table["compliance_pass"] = invariant_locomotion_pass & primary_contact_pass
    endpoint_pass = table.success.eq(1) & table.failure.eq(0)
    force_instrumented = all(
        f"force{int(threshold)}n_df_{leg}" in table
        for threshold in FORCE_THRESHOLDS_N for leg in LEGS)
    if force_instrumented:
        for threshold in FORCE_THRESHOLDS_N:
            tag = f"force{int(threshold)}n"
            table[f"{tag}_achieved_df"] = table[
                [f"{tag}_df_{leg}" for leg in LEGS]].mean(axis=1)
            table[f"{tag}_df_max_abs_error"] = table[
                [f"{tag}_df_{leg}" for leg in LEGS]
            ].sub(table.command_df, axis=0).abs().max(axis=1)
            contact_pass = (
                invariant_locomotion_pass & endpoint_pass
                & table[[f"{tag}_cycles_{leg}" for leg in LEGS]].ge(1).all(axis=1)
                & table[[f"{tag}_df_{leg}" for leg in LEGS]]
                    .sub(table.command_df, axis=0).abs().le(.05).all(axis=1)
                & table[[f"{tag}_swing_recall_{leg}" for leg in LEGS]]
                    .ge(.9).all(axis=1)
                & table[[f"{tag}_stance_recall_{leg}" for leg in LEGS]]
                    .ge(.9).all(axis=1)
                & table[f"{tag}_{table.step_width_frame.iloc[0]}_foot_lateral_mae"].le(.015)
                & (table[f"{tag}_{table.step_width_frame.iloc[0]}_achieved_width"]
                   - table.command_width).abs().le(.03))
            table[f"{tag}_contact_pass"] = contact_pass
            table[f"{tag}_compliance_pass"] = (
                contact_pass if v4 else
                contact_pass & table[f"{tag}_topology_fraction"].ge(.9))
        for leg in LEGS:
            threshold_columns = [
                f"force{int(threshold)}n_df_{leg}"
                for threshold in FORCE_THRESHOLDS_N]
            table[f"force_df_span_{leg}"] = (
                table[threshold_columns].max(axis=1)
                - table[threshold_columns].min(axis=1))
        table["force_df_span_pass"] = table[
            [f"force_df_span_{leg}" for leg in LEGS]
        ].le(DT / table.period, axis=0).all(axis=1)
        table["force_robust_trial_pass"] = (
            table[[f"force{int(threshold)}n_compliance_pass"
                   for threshold in FORCE_THRESHOLDS_N]].all(axis=1)
            & table.force_df_span_pass)
    table.to_csv(directory / "trials.csv", index=False)
    records, threshold_records = [], []
    for (width, df, disturbed), group in table.groupby(["step_width", "command_df", "disturbed"]):
        lo, hi = wilson(int(group.success.sum()), len(group))
        record = {"step_width": width, "command_df": df, "disturbed": disturbed, "n": len(group),
                  "successes": int(group.success.sum()), "failures": int(group.failure.sum()),
                  "timeouts": int(group.timeout.sum()), "success_rate": group.success.mean(),
                  "ci_low": lo, "ci_high": hi, "period": group.period.iloc[0], "gait": group.gait.iloc[0],
                  "step_width_frame": group.step_width_frame.iloc[0]}
        metrics = ["course_distance", "forward_distance", "mean_speed", "forward_speed", "duration",
                   "traversal_time", "achieved_df", "achieved_width", "command_width", "command_speed",
                   "timing_accuracy", "applied_impulse", "df_max_abs_error", "lateral_rmse",
                   "max_lateral_deviation", "foot_lateral_mae", "settled_samples",
                   "world_achieved_width", "world_foot_lateral_mae", "body_achieved_width",
                   "body_foot_lateral_mae", "heading_rmse_rad", "max_abs_heading_rad"]
        metrics += [key for key in ["world_lateral_velocity_mean", "world_lateral_velocity_rmse",
                                    "body_lateral_velocity_mean", "body_lateral_velocity_rmse",
                                    "body_yaw_rate_mean", "body_yaw_rate_rmse"] if key in group]
        for metric in metrics:
            record[metric] = group[metric].mean()
        for leg in LEGS:
            for prefix in ["df", "schedule_df", "stance_recall", "swing_recall"]:
                record[f"{prefix}_{leg}"] = group[f"{prefix}_{leg}"].mean()
            record[f"trials_with_cycles_{leg}"] = int((group[f"cycles_{leg}"] > 0).sum())
        record["compliant_trial_fraction"] = group.compliance_pass.mean()
        record["compliance_screen_pass"] = bool(record["compliant_trial_fraction"] >= .8)
        if force_instrumented:
            threshold_screens = []
            frame_name = group.step_width_frame.iloc[0]
            for threshold in FORCE_THRESHOLDS_N:
                tag = f"force{int(threshold)}n"
                fraction = group[f"{tag}_compliance_pass"].mean()
                record[f"{tag}_compliant_trial_fraction"] = fraction
                record[f"{tag}_compliance_screen_pass"] = bool(fraction >= .8)
                record[f"{tag}_achieved_df"] = group[f"{tag}_achieved_df"].mean()
                record[f"{tag}_achieved_width"] = group[
                    f"{tag}_{frame_name}_achieved_width"].mean()
                record[f"{tag}_topology_fraction"] = group[
                    f"{tag}_topology_fraction"].mean()
                topology_values = group[f"{tag}_topology_fraction"].to_numpy()
                topology_bootstrap = topology_values[np.random.default_rng(
                    8102026 + int(round(width * 1000))
                    + int(round(df * 1000)) + int(threshold)
                ).integers(len(topology_values), size=(10000, len(topology_values)))
                ].mean(axis=1)
                record[f"{tag}_topology_ci_low"] = float(
                    np.quantile(topology_bootstrap, .025))
                record[f"{tag}_topology_ci_high"] = float(
                    np.quantile(topology_bootstrap, .975))
                record[f"{tag}_topology_screen_pass"] = bool(
                    record[f"{tag}_topology_fraction"] >= .9)
                threshold_screen = bool(
                    fraction >= .8 and (
                        not v4 or record[f"{tag}_topology_screen_pass"]))
                record[f"{tag}_threshold_screen_pass"] = threshold_screen
                topology_good = int(group[f"{tag}_topology_good_cycles"].sum())
                topology_cycles = int(group[f"{tag}_topology_cycles"].sum())
                pooled_topology = (
                    topology_good / topology_cycles if topology_cycles else np.nan)
                record[f"{tag}_topology_good_cycles"] = topology_good
                record[f"{tag}_topology_cycles"] = topology_cycles
                record[f"{tag}_pooled_topology_fraction"] = pooled_topology
                threshold_records.append({
                    "step_width": width, "command_df": df,
                    "threshold_n": threshold,
                    "achieved_df": group[f"{tag}_achieved_df"].mean(),
                    "max_per_leg_df_abs_error": group[
                        [f"{tag}_df_{leg}" for leg in LEGS]
                    ].sub(group.command_df, axis=0).abs().max(axis=1).mean(),
                    "min_stance_recall": group[
                        [f"{tag}_stance_recall_{leg}" for leg in LEGS]
                    ].min(axis=1).mean(),
                    "min_swing_recall": group[
                        [f"{tag}_swing_recall_{leg}" for leg in LEGS]
                    ].min(axis=1).mean(),
                    "achieved_width_error": (
                        group[f"{tag}_{frame_name}_achieved_width"]
                        - group.command_width).abs().mean(),
                    "compliant_trial_fraction": fraction,
                    "topology_fraction": group[f"{tag}_topology_fraction"].mean(),
                    "topology_ci_low": record[f"{tag}_topology_ci_low"],
                    "topology_ci_high": record[f"{tag}_topology_ci_high"],
                    "topology_good_cycles": topology_good,
                    "topology_cycles": topology_cycles,
                    "pooled_topology_fraction": pooled_topology,
                    "topology_screen_pass": bool(
                        record[f"{tag}_topology_screen_pass"]),
                    "threshold_screen_pass": threshold_screen,
                })
                threshold_screens.append(threshold_screen)
            record["force_df_span_trial_fraction"] = group.force_df_span_pass.mean()
            record["force_robust_trial_fraction"] = group.force_robust_trial_pass.mean()
            record["force_threshold_conclusion_unchanged"] = bool(
                all(value == record["force5n_threshold_screen_pass"]
                    for value in threshold_screens))
            record["force5n_200hz_vs_frozen50hz_unchanged"] = bool(
                record["force5n_compliance_screen_pass"]
                == record["compliance_screen_pass"])
            record["combined_cell_pass"] = (
                combined_cell_acceptance_v4(record) if v4
                else combined_cell_acceptance(record, threshold_screens))
        records.append(record)
    summary = pd.DataFrame(records)
    summary.to_csv(directory / "summary.csv", index=False)
    if threshold_records:
        threshold_table = pd.DataFrame(threshold_records)
        span_lookup = summary.set_index(["step_width", "command_df"])[
            "force_df_span_trial_fraction"]
        threshold_table["df_span_trial_fraction"] = [
            span_lookup.loc[(row.step_width, row.command_df)]
            for row in threshold_table.itertuples()]
        threshold_table["combined_cell_pass"] = [
            bool(summary[(summary.step_width == row.step_width)
                         & (summary.command_df == row.command_df)]
                 .combined_cell_pass.iloc[0])
            for row in threshold_table.itertuples()]
        threshold_table.to_csv(directory / "force_threshold_summary.csv", index=False)
    rng, contrasts = np.random.default_rng(8102026), []
    for (width, disturbed), group in table.groupby(["step_width", "disturbed"]):
        pivot = group.pivot(index="seed", columns="command_df", values="success")
        if pivot.isna().any().any():
            raise ValueError("Duty-factor comparisons require identical trial seed sets")
        if len(pivot.columns) < 2:
            continue
        low, high = min(pivot.columns), max(pivot.columns)
        diffs = (pivot[high] - pivot[low]).to_numpy()
        bootstrap = diffs[rng.integers(len(diffs), size=(10000, len(diffs)))].mean(axis=1)
        high_ci = wilson(int(pivot[high].sum()), len(pivot), z=2.241402727604947)
        low_ci = wilson(int(pivot[low].sum()), len(pivot), z=2.241402727604947)
        contrasts.append({"step_width": float(width), "disturbed": bool(disturbed), "low_df": float(low), "high_df": float(high),
            "success_difference": float(diffs.mean()), "ci_low": float(np.quantile(bootstrap, .025)),
            "ci_high": float(np.quantile(bootstrap, .975)), "bootstrap_degenerate": bool(np.ptp(bootstrap) == 0),
            "conservative_ci_low": high_ci[0] - low_ci[1], "conservative_ci_high": high_ci[1] - low_ci[0]})
    is_force_versioned = json.loads(
        (directory / "evaluation_manifest.json").read_text()).get(
        "schema") in (V3_NOMINAL_SCHEMA, NOMINAL_SCHEMA)
    contrast_name = (
        "endpoint_success_diagnostics.json"
        if is_force_versioned else "paired_contrasts.json")
    (directory / contrast_name).write_text(json.dumps({
        "scientific_role": "command_fidelity_prerequisite",
        "not_closed_loop_chi_evidence": True,
        "contrasts": contrasts,
    } if is_force_versioned else contrasts, indent=2))

    def save(fig, name):
        fig.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(directory / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    conditions = sorted(summary.disturbed.unique())
    width_frame = table.step_width_frame.iloc[0]
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 3.6), squeeze=False)
    for ax, disturbed in zip(axes[0], conditions):
        for width, group in summary[summary.disturbed == disturbed].groupby("step_width"):
            ax.errorbar(group.command_df, group.success_rate,
                        yerr=[group.success_rate - group.ci_low, group.ci_high - group.success_rate],
                        marker="o", capsize=3, label=f"{width:.2f} m")
        ax.set(title="Pushes" if disturbed else "Nominal", xlabel="Commanded DF",
               ylabel="Straight crossing success (95% Wilson CI)", ylim=(-.03, 1.03))
        ax.legend(title=f"{width_frame.title()}-axis step width")
    save(fig, "endpoint_success_vs_df"
         if is_force_versioned else "success_vs_df")
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 3.8), squeeze=False)
    for ax, disturbed in zip(axes[0], conditions):
        grid = summary[summary.disturbed == disturbed].pivot(index="step_width", columns="command_df", values="success_rate")
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i in range(len(grid)):
            for j in range(len(grid.columns)):
                ax.text(j, i, f"{grid.iloc[i, j]:.0%}", ha="center", color="white" if grid.iloc[i, j] < .55 else "black")
        ax.set(xticks=range(len(grid.columns)), xticklabels=grid.columns, yticks=range(len(grid)),
               yticklabels=grid.index, xlabel="Commanded DF", ylabel=f"{width_frame.title()}-axis step width (m)",
               title="Pushes" if disturbed else "Nominal")
        fig.colorbar(im, ax=ax, label="Success")
    save(fig, "success_heatmap")
    limits = [table.command_df.min() - .02, table.command_df.max() + .02]
    fig, axes = plt.subplots(len(conditions), 4, figsize=(15, 3.7 * len(conditions)), squeeze=False)
    for row, disturbed in enumerate(conditions):
        for width, group in summary[summary.disturbed == disturbed].groupby("step_width"):
            for col, key in enumerate(["achieved_df", "achieved_width", "forward_speed", "lateral_rmse"]):
                axes[row, col].plot(group.command_df, group[key], "o-", label=f"{width:.2f} m")
            axes[row, 1].axhline(width, ls=":", alpha=.4)
        axes[row, 0].plot(limits, limits, "k--", label="Command")
        axes[row, 2].axhline(table.command_speed.iloc[0], color="k", ls="--", label="Command")
        axes[row, 3].axhline(0., color="k", ls="--", label="Straight path")
        for col, label in enumerate(["Achieved DF", f"{width_frame.title()}-axis stance width (m)", "Forward speed (m/s)", "Lateral RMSE (m)"]):
            axes[row, col].set(xlabel="Commanded DF", ylabel=label, title="Pushes" if disturbed else "Nominal")
            axes[row, col].legend(fontsize=7, title=f"{width_frame.title()} width")
    save(fig, "command_compliance")
    fig, axes = plt.subplots(len(conditions), 4, figsize=(13, 3 * len(conditions)), squeeze=False)
    for row, disturbed in enumerate(conditions):
        for col, leg in enumerate(LEGS):
            for width, group in summary[summary.disturbed == disturbed].groupby("step_width"):
                axes[row, col].plot(group.command_df, group[f"df_{leg}"], "o-", label=f"{width:.2f} m")
            axes[row, col].plot(limits, limits, "k--")
            axes[row, col].set(title=f"{leg}, {'push' if disturbed else 'nominal'}", xlabel="Commanded DF", ylabel="Achieved DF")
            axes[row, col].legend(fontsize=7, title=f"{width_frame.title()} width")
    save(fig, "per_leg_df")
    fig, axes = plt.subplots(1, len(conditions), figsize=(7 * len(conditions), 4), squeeze=False)
    for ax, disturbed in zip(axes[0], conditions):
        for path in files:
            with np.load(path) as data:
                if bool(data["disturbed"]) != disturbed:
                    continue
                for trial in range(data["valid"].shape[1]):
                    xy = data["body"][data["valid"][:, trial], trial, :2]
                    ax.plot(xy[:, 0], xy[:, 1], color="tab:blue", alpha=.12, lw=.6)
        ax.axhline(0., color="black", ls="--", label="Commanded straight path")
        ax.set(xlabel="Forward position (m)", ylabel="Lateral displacement (m)", title="Pushes" if disturbed else "Nominal")
        ax.legend()
    save(fig, "straight_path_tracking")
    print(summary.to_string(index=False))
    return summary


def nominal_study_readiness(trot_directory, walk_directory):
    """Validate the exact 20-cell command-fidelity prerequisite study."""
    import pandas as pd
    directories = [Path(trot_directory), Path(walk_directory)]
    manifests = []
    reset_identities = []
    for directory in directories:
        files = validate_manifest(directory)
        manifest = json.loads((directory / "evaluation_manifest.json").read_text())
        manifests.append(manifest)
        with np.load(files[0]) as data:
            reset_identities.append({
                "reset_plan": np.asarray(data["reset_plan"]),
                "initial_root": np.asarray(data["initial_root"]),
                "initial_joints": np.asarray(data["initial_joints"]),
                "stance_start": np.asarray(data["stance_start"])[0],
            })
    if {manifest["gait"] for manifest in manifests} != {"trot", "walk"}:
        raise ValueError("Study requires one trot and one four-beat walk evaluation")
    for manifest in manifests:
        if (manifest.get("schema") != NOMINAL_SCHEMA
                or not np.isclose(manifest["period"], .48)
                or not np.isclose(manifest["speed"], .30)):
            raise ValueError("Study requires force-instrumented period .48/speed .30 data")
    for key in (
        "source_sha256", "checkpoint_sha256", "evaluator_sha256", "seeds",
        "training_provenance_sha256", "split", "condition_reset_seed",
        "step_width_frame",
    ):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Study provenance mismatch: {key}")
    deployment_keys = (
        "deployment_evaluation_profile", "deployment_training_required",
        "deployment_training_verified", "deployment_profile",
        "deployment_profile_sha256", "deployment_training_source_sha256",
        "kp_scale", "kd_scale",
    )
    if any(key in manifest for manifest in manifests for key in deployment_keys):
        if any(key not in manifest for manifest in manifests
               for key in deployment_keys):
            raise ValueError("Study deployment identity is incomplete")
        for key in deployment_keys:
            if manifests[0][key] != manifests[1][key]:
                raise ValueError(f"Study deployment mismatch: {key}")
    validate_matched_study_resets(*reset_identities)
    expected = {
        ("trot", width, duty)
        for width in (.1, .2, .3, .4, .5)
        for duty in (.5, .625, .75)
    } | {("walk", width, .75) for width in (.1, .2, .3, .4, .5)}
    observed = {
        (manifest["gait"], float(cell["step_width"]), float(cell["df"]))
        for manifest in manifests for cell in manifest["expected_conditions"]
    }
    if observed != expected:
        raise ValueError(f"Study cell mismatch: missing={expected-observed}, extra={observed-expected}")
    if len(manifests[0]["seeds"]) != 64:
        raise ValueError("Study requires exactly 64 held-out trials per cell")
    summaries = [pd.read_csv(directory / "summary.csv") for directory in directories]
    combined = pd.concat(summaries, ignore_index=True)
    if (len(combined) != 20 or "combined_cell_pass" not in combined
            or not combined.n.eq(64).all()):
        raise ValueError("Study summaries must contain exactly 20 gated cells")
    passed = bool(combined.combined_cell_pass.astype(bool).all())
    readiness = {
        "ready": passed, "cells": 20,
        "passed_cells": int(combined.combined_cell_pass.astype(bool).sum()),
        "scientific_role": "command_fidelity_prerequisite",
        "not_closed_loop_chi_evidence": True,
        "checkpoint_sha256": manifests[0]["checkpoint_sha256"],
        "evaluator_sha256": manifests[0]["evaluator_sha256"],
    }
    if manifests[0].get("deployment_evaluation_profile") is not None:
        readiness.update({key: manifests[0][key] for key in deployment_keys})
    return readiness


def validate_matched_study_resets(first, second):
    """Reject cross-gait comparisons with different initial populations."""
    required = ("reset_plan", "initial_root", "initial_joints", "stance_start")
    if any(key not in first or key not in second for key in required):
        raise ValueError("Study reset identity is incomplete")
    for key in required:
        left, right = np.asarray(first[key]), np.asarray(second[key])
        matches = (np.array_equal(left, right) if key == "stance_start"
                   else np.allclose(left, right, atol=1e-7))
        if not matches:
            raise ValueError(
                f"Study gait comparisons require matched reset identity: {key}")


def combined_cell_acceptance(record, threshold_screens):
    """Apply the frozen cell-level prerequisite gates without averaging them."""
    return bool(
        record["compliance_screen_pass"]
        and all(threshold_screens)
        and all(record[f"force{int(threshold)}n_pooled_topology_fraction"] >= .9
                for threshold in FORCE_THRESHOLDS_N)
        and record["force_df_span_trial_fraction"] >= .9
        and record["force_robust_trial_fraction"] >= .8)


def combined_cell_acceptance_v4(record):
    """Apply the V4 prerequisite with the policy's 5 N contact as primary.

    The 2 N and 10 N reconstructions remain threshold-sensitivity diagnostics;
    they are correlated views of the same trials and are not multiplied into
    the primary acceptance decision.
    """
    return bool(
        record["compliance_screen_pass"]
        and record["force5n_compliance_screen_pass"]
        and record["force5n_topology_screen_pass"])
