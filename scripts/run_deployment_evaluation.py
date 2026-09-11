"""Run and summarize the held-out deployment robustness matrix."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))
from beam_walking.experiment.deployment import (
    DAMPING_EVALUATION_SCALES, STIFFNESS_EVALUATION_SCALES,
)
from beam_walking.experiment.analysis import validate_manifest


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=20_000)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

if not args.checkpoint.is_file():
    parser.error("Checkpoint does not exist")
if args.num_envs != 64:
    parser.error("The frozen deployment matrix requires 64 held-out trials per cell")
if not 10_000 <= args.seed or args.seed + args.num_envs > 1_000_000:
    parser.error("Deployment evaluation must use the validation seed block")
TEST_SEED = 1_500_000
VALIDATION_SEED = 20_000
ORCHESTRATOR_PATH = Path(__file__).resolve()


def profiles():
    yield "nominal", None, None
    yield "randomized", None, None
    for kp in STIFFNESS_EVALUATION_SCALES:
        for kd in DAMPING_EVALUATION_SCALES:
            yield f"kp{kp:.1f}_kd{kd:.1f}", kp, kd


def evaluation_command(profile, gait, output, kp, kd, split, seed):
    command = [
        sys.executable, str(ROOT / "scripts/evaluate_policy.py"),
        "--checkpoint", str(args.checkpoint.resolve()),
        "--output", str(output.resolve()),
        "--num_envs", str(args.num_envs),
        "--seed", str(seed), "--split", split,
        "--gait", gait, "--speed", ".30", "--period", ".48",
        "--require_deployment_checkpoint", "--device", args.device,
    ]
    if args.headless:
        command.append("--headless")
    if profile == "randomized":
        command.extend(["--deployment_profile", "randomized"])
    elif profile != "nominal":
        command.extend([
            "--deployment_profile", "fixed_gains",
            "--kp_scale", str(kp), "--kd_scale", str(kd),
        ])
    return command


def validate_completed(directory, profile, gait, kp, kd, split, seed,
                       checkpoint_sha256):
    """Accept a resumed result only when its complete identity matches."""
    validate_manifest(directory)
    manifest = json.loads((directory / "evaluation_manifest.json").read_text())
    expected_seeds = list(range(seed, seed + args.num_envs))
    expected_profile = "fixed_gains" if kp is not None else profile
    expected = {
        "checkpoint_sha256": checkpoint_sha256,
        "gait": gait, "period": .48, "speed": .30,
        "split": split, "seeds": expected_seeds,
        "deployment_evaluation_profile": expected_profile,
        "deployment_training_required": True,
        "deployment_training_verified": True,
        "kp_scale": kp, "kd_scale": kd,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Resumed evaluation identity mismatch in {directory}: {key}")


def run_profile(profile, kp, kd, split, seed, name=None):
    directories = {}
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    for gait in ("trot", "walk"):
        output = args.output / (f"{name}_{gait}" if name else f"{profile}_{gait}")
        complete = output / "evaluation_complete.json"
        if complete.is_file():
            validate_completed(
                output, profile, gait, kp, kd, split, seed,
                checkpoint_sha256)
        else:
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(
                    f"Refusing to overwrite incomplete evaluation {output}")
            subprocess.run(
                evaluation_command(profile, gait, output, kp, kd, split, seed),
                cwd=ROOT, check=True)
            validate_completed(
                output, profile, gait, kp, kd, split, seed,
                checkpoint_sha256)
        directories[gait] = output
    subprocess.run([
        sys.executable, str(ROOT / "scripts/analyze_beam.py"),
        str(directories["trot"]), "--study-with",
        str(directories["walk"]),
    ], cwd=ROOT, check=True)
    readiness = json.loads(
        (directories["trot"] / "study_readiness.json").read_text())
    summary = pd.concat([
        pd.read_csv(directories[gait] / "summary.csv")
        for gait in ("trot", "walk")
    ], ignore_index=True)
    return directories, readiness, summary


def main():
    if args.seed != VALIDATION_SEED:
        parser.error(
            f"The frozen deployment matrix requires validation seed {VALIDATION_SEED}")
    orchestrator_sha256 = hashlib.sha256(
        ORCHESTRATOR_PATH.read_bytes()).hexdigest()
    expected_names = {
        f"{profile}_{gait}"
        for profile, _, _ in profiles() for gait in ("trot", "walk")
    } | {"selected_test_trot", "selected_test_walk",
         "deployment_evaluation_progress.json", "deployment_robustness.json",
         "run_deployment_evaluation.py"}
    if args.output.exists():
        unexpected = {
            path.name for path in args.output.iterdir()
            if path.name not in expected_names
        }
        if unexpected:
            parser.error(
                f"Output contains files outside the frozen matrix: {sorted(unexpected)}")
    args.output.mkdir(parents=True, exist_ok=True)
    progress_path = args.output / "deployment_evaluation_progress.json"
    archived_orchestrator = args.output / "run_deployment_evaluation.py"
    if archived_orchestrator.is_file():
        if hashlib.sha256(archived_orchestrator.read_bytes()).hexdigest() \
                != orchestrator_sha256:
            raise RuntimeError("Archived deployment selector source mismatch")
    else:
        shutil.copy2(ORCHESTRATOR_PATH, archived_orchestrator)
    if progress_path.is_file():
        previous_progress = json.loads(progress_path.read_text())
        if previous_progress.get("orchestrator_sha256") != orchestrator_sha256:
            raise RuntimeError("Deployment progress selector hash mismatch")
    final_path = args.output / "deployment_robustness.json"
    if final_path.is_file():
        previous_final = json.loads(final_path.read_text())
        if previous_final.get("orchestrator_sha256") != orchestrator_sha256:
            raise RuntimeError("Deployment report selector hash mismatch")
    records = []
    progress_path.write_text(json.dumps({
        "schema": "go2_deployment_robustness_matrix_v1",
        "orchestrator_sha256": orchestrator_sha256,
        "orchestrator_source": "run_deployment_evaluation.py",
        "checkpoint_sha256": hashlib.sha256(
            args.checkpoint.read_bytes()).hexdigest(),
        "profiles": records,
    }, indent=2))
    for profile, kp, kd in profiles():
        directories, readiness, summary = run_profile(
            profile, kp, kd, "validation", args.seed)
        record = {
            "profile": profile, "kp_scale": kp, "kd_scale": kd,
            "cells": int(len(summary)),
            "passed_cells": int(summary.combined_cell_pass.astype(bool).sum()),
            "all_cells_pass": bool(summary.combined_cell_pass.astype(bool).all()),
            "trot_directory": str(directories["trot"].resolve()),
            "walk_directory": str(directories["walk"].resolve()),
            "checkpoint_sha256": readiness["checkpoint_sha256"],
            "minimum_compliant_trial_fraction": float(
                summary.compliant_trial_fraction.min()),
            "mean_compliant_trial_fraction": float(
                summary.compliant_trial_fraction.mean()),
            "minimum_force5n_topology_fraction": float(
                summary.force5n_topology_fraction.min()),
            "mean_force5n_topology_fraction": float(
                summary.force5n_topology_fraction.mean()),
            "mean_normalized_tracking_error": float(
                ((summary.forward_speed - summary.command_speed).abs() / .04
                 + (summary.body_achieved_width - summary.command_width).abs() / .03
                 + summary.heading_rmse_rad / .10) .mean()),
        }
        records.append(record)
        progress_path.write_text(json.dumps({
            "schema": "go2_deployment_robustness_matrix_v1",
            "orchestrator_sha256": orchestrator_sha256,
            "orchestrator_source": "run_deployment_evaluation.py",
            "checkpoint_sha256": records[0]["checkpoint_sha256"],
            "profiles": records,
        }, indent=2))

    checkpoint_hashes = {record["checkpoint_sha256"] for record in records}
    if len(checkpoint_hashes) != 1:
        raise RuntimeError("Deployment profiles did not evaluate one checkpoint")
    fixed = [record for record in records if record["kp_scale"] is not None]
    ranked = sorted(fixed, key=lambda record: (
        -record["passed_cells"],
        -record["minimum_compliant_trial_fraction"],
        -record["mean_force5n_topology_fraction"],
        record["mean_normalized_tracking_error"],
        abs(record["kp_scale"] - 1) + abs(record["kd_scale"] - 1),
    ))
    selected = ranked[0]
    deployment_gain_selection_pass = bool(selected["all_cells_pass"])
    test_directories, test_readiness, test_summary = run_profile(
        selected["profile"], selected["kp_scale"], selected["kd_scale"],
        "test", TEST_SEED, name="selected_test")
    selected_test_all_cells_pass = bool(
        test_summary.combined_cell_pass.astype(bool).all())
    nominal_pass = next(
        record["all_cells_pass"] for record in records
        if record["profile"] == "nominal")
    randomized_pass = next(
        record["all_cells_pass"] for record in records
        if record["profile"] == "randomized")
    ready_for_port = bool(
        deployment_gain_selection_pass and selected_test_all_cells_pass
        and test_readiness["ready"] and nominal_pass and randomized_pass)
    report = {
        "schema": "go2_deployment_robustness_matrix_v1",
        "orchestrator_sha256": orchestrator_sha256,
        "orchestrator_source": "run_deployment_evaluation.py",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": records[0]["checkpoint_sha256"],
        "validation_seed_start": args.seed,
        "trials_per_cell": args.num_envs,
        "stiffness_scales_evaluated": list(STIFFNESS_EVALUATION_SCALES),
        "damping_scales_evaluated": list(DAMPING_EVALUATION_SCALES),
        "profiles": records,
        "profiles_passed": sum(record["all_cells_pass"] for record in records),
        "total_profiles": len(records),
        "deployment_matrix_pass": all(
            record["all_cells_pass"] for record in records),
        "gain_selection_rule": (
            "max passed cells; max worst-cell compliant-trial fraction; "
            "max mean 5N topology fraction; min normalized speed/width/heading "
            "error; closest-to-nominal tie break"),
        "selected_kp_scale": selected["kp_scale"],
        "selected_kd_scale": selected["kd_scale"],
        "selected_kp": 25.0 * selected["kp_scale"],
        "selected_kd": 0.5 * selected["kd_scale"],
        "deployment_gain_selection_pass": deployment_gain_selection_pass,
        "selected_test_seed_start": TEST_SEED,
        "selected_test_trot_directory":
            str(test_directories["trot"].resolve()),
        "selected_test_walk_directory":
            str(test_directories["walk"].resolve()),
        "selected_test_passed_cells": int(
            test_summary.combined_cell_pass.astype(bool).sum()),
        "selected_test_all_cells_pass": selected_test_all_cells_pass,
        "nominal_validation_pass": bool(nominal_pass),
        "randomized_validation_pass": bool(randomized_pass),
        "gain_candidate_ready_for_port_validation":
            ready_for_port,
        "test_seed_block_used": True,
    }
    (args.output / "deployment_robustness.json").write_text(
        json.dumps(report, indent=2))
    print("DEPLOYMENT_ROBUSTNESS", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
