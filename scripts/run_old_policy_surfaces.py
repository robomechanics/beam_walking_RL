"""Wait for prior GPU work, then run reviewed old-policy smoke/grid/figures."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from evaluation_capacity import check_evaluation_capacity
from policy_surface_data import (
    CHECKPOINT_SHA256, ROOT, SMOKE_SEED, TRIALS, conditions,
    sha256, source_hash, summarize, validate_measurement,
)


def smoke_valid(directory, frozen_hash):
    manifest_path = directory / "surface_manifest.json"
    trials_path = directory / "surface_trials.csv"
    record = json.loads((directory / "SURFACE_COMPLETE").read_text())
    manifest = json.loads(manifest_path.read_text())
    if (record["manifest_sha256"] != sha256(manifest_path)
            or record["trials_sha256"] != sha256(trials_path)
            or manifest.get("smoke") is not True
            or manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or manifest.get("collector_sha256") != frozen_hash
            or len(manifest["conditions"]) != 6):
        raise ValueError("Smoke completion/identity mismatch")
    expected = set(conditions(True))
    entries = manifest["conditions"]
    keys = ("gait", "speed", "period", "step_width", "command_df")
    if {tuple(e[k] for k in keys) for e in entries} != expected:
        raise ValueError("Smoke condition grid mismatch")
    if set(record["archive_sha256"]) != {e["filename"] for e in entries}:
        raise ValueError("Smoke archive set mismatch")
    rows = []
    for entry in entries:
        name = entry["filename"]
        if sha256(directory / name) != record["archive_sha256"][name]:
            raise ValueError("Smoke raw archive hash mismatch")
        condition = tuple(entry[k] for k in keys)
        with np.load(directory / name, allow_pickle=False) as payload:
            if (str(payload["checkpoint_sha256"]) != CHECKPOINT_SHA256
                    or str(payload["collector_sha256"]) != frozen_hash
                    or not np.array_equal(payload["seeds"], np.arange(SMOKE_SEED, SMOKE_SEED + TRIALS))):
                raise ValueError("Smoke archive identity mismatch")
            validate_measurement(payload, condition)
            rows.extend(summarize(payload, condition, SMOKE_SEED))
    data = pd.DataFrame(rows)
    pd.testing.assert_frame_equal(pd.read_csv(trials_path), data, check_dtype=False,
                                  check_exact=False, rtol=1e-10, atol=1e-12)
    reference = data[(data.gait == "trot") & data.command_df.isin([.625, .75])]
    if not (reference.groupby("command_df").compliant.mean() >= .90).all():
        raise ValueError("Trained trot reference failed smoke command/energy sanity")
    return dict(instrumentation_pass=True, matched_conditions=6,
                trials=len(data), trained_reference_compliance=reference.groupby(
                    "command_df").compliant.mean().to_dict(),
                unseen_walk_performance_is_not_a_smoke_gate=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewed-sha256", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "results") or output.exists():
        parser.error("Output must be a new directory inside repository results")
    if source_hash() != args.reviewed_sha256:
        parser.error("Reviewed source hash mismatch")
    output.mkdir(parents=True)
    state = {"checkpoint_sha256": CHECKPOINT_SHA256,
             "reviewed_sha256": args.reviewed_sha256, "training_started": False}

    def update(status, **extra):
        state.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(output / "status.json")
        print(json.dumps(state), flush=True)

    def frozen():
        if source_hash() != args.reviewed_sha256:
            raise RuntimeError("Reviewed source changed; re-review required")

    deadline = time.monotonic() + 2 * 3600

    def wait_capacity():
        while time.monotonic() < deadline:
            frozen()
            # The pre-existing deployment handoff has priority. Waiting for its
            # terminal state avoids racing two simulator startups on a free GPU.
            prior = ROOT / "results/deployment_smoke_queue_seed3_20260911/status.json"
            if prior.exists():
                queued = json.loads(prior.read_text())
                if queued.get("state") in ("waiting_for_adaptive", "waiting_for_capacity", "running_smoke"):
                    update("waiting_for_existing_queue", prior_state=queued["state"])
                    time.sleep(30)
                    continue
            try:
                capacity = check_evaluation_capacity(TRIALS, "cuda:0", False)
            except (RuntimeError, subprocess.CalledProcessError) as error:
                update("waiting_for_capacity", reason=str(error))
                time.sleep(30)
                continue
            return capacity
        raise RuntimeError("Evaluation wait expired after two hours")

    try:
        for mode in ("smoke", "grid"):
            capacity = wait_capacity()
            frozen()
            command = [sys.executable, "-u", str(ROOT / "scripts/collect_policy_surfaces.py"),
                       "--headless", "--output", str(output / mode)]
            if mode == "smoke":
                command.append("--smoke")
            update("running_" + mode, command=command, capacity=capacity)
            with (output / f"{mode}.log").open("x") as log:
                result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0 or not (output / mode / "SURFACE_COMPLETE").is_file():
                raise RuntimeError(f"{mode} did not complete; inspect {mode}.log")
            if mode == "smoke":
                checks = smoke_valid(output / mode, args.reviewed_sha256)
                (output / "smoke_checks.json").write_text(json.dumps(checks, indent=2))
                update("smoke_passed", smoke_checks=checks)
        frozen()
        update("making_figures")
        command = [sys.executable, str(ROOT / "scripts/make_policy_surfaces.py"),
                   "--input", str(output / "grid"), "--output",
                   str(ROOT / "PAPER_GRAPHS/old_policy_walk_df")]
        with (output / "figures.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        update("complete", figures=str(ROOT / "PAPER_GRAPHS/old_policy_walk_df"))
    except Exception as error:
        update("blocked", reason=str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
