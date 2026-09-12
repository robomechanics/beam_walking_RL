"""Queue the registered high-duty walk pipeline behind current GPU work."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))
RESULTS = ROOT / "results"
RUNS = {
    "smoke": RESULTS / "high_duty_walk_smoke_seed4_20260912",
    "training": RESULTS / "high_duty_walk_seed4_3072_20260912",
    "grid": RESULTS / "high_duty_walk_grid_seed4_20260912",
    "selector": RESULTS / "high_duty_walk_selector_seed4_20260912",
    "validation": RESULTS / "high_duty_walk_selector_validation_seed4_20260912",
}
STATUS = RESULTS / "high_duty_walk_pipeline_20260912.json"
SOURCE_FILES = (
    "docs/high_duty_walk_plan.md",
    "source/beam_walking/beam_walking/experiment/task.py",
    "source/beam_walking/beam_walking/experiment/protocol.py",
    "source/beam_walking/beam_walking/experiment/duty_grid.py",
    "source/beam_walking/beam_walking/experiment/duty_selector.py",
    "source/beam_walking/beam_walking/experiment/stability.py",
    "source/beam_walking/beam_walking/experiment/high_duty_walk.py",
    "source/beam_walking/beam_walking/experiment/high_duty_walk_task.py",
    "source/beam_walking/beam_walking/experiment/high_duty_walk_selector.py",
    "scripts/high_duty_walk_experiment.py",
    "scripts/collect_high_duty_walk_grid.py",
    "scripts/fit_high_duty_walk_selector.py",
    "scripts/select_high_duty_walk.py",
    "scripts/gpu_capacity.py",
    "scripts/evaluation_capacity.py",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hash():
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def write_status(state, **fields):
    payload = {
        "schema": "high_duty_walk_pipeline_v1",
        "state": state, "updated_at": now(),
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "runs": {key: str(value) for key, value in RUNS.items()},
        **fields,
    }
    STATUS.write_text(json.dumps(payload, indent=2))
    print("PIPELINE_STATUS", json.dumps(payload), flush=True)


def capacity_snapshot():
    gpu = subprocess.run([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=True).stdout.strip()
    available_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:"))
    return gpu, available_kib // 1024


def wait_for_capacity():
    last_report = 0.
    while True:
        try:
            gpu, ram_mib = capacity_snapshot()
            ready = not gpu and ram_mib >= 14_336
        except (OSError, subprocess.SubprocessError, StopIteration) as error:
            gpu, ram_mib, ready = f"capacity_error:{error}", 0, False
        if ready:
            write_status("capacity_ready", available_system_mib=ram_mib)
            return
        if time.monotonic() - last_report >= 600:
            write_status(
                "waiting_for_capacity", available_system_mib=ram_mib,
                active_compute_processes=gpu.splitlines() if gpu else [])
            last_report = time.monotonic()
        time.sleep(60)


def verify_source():
    actual = source_hash()
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"Reviewed source changed while queued: {EXPECTED_SOURCE_SHA256} -> {actual}")


def run_stage(name, command):
    verify_source()
    write_status(name, command=command)
    subprocess.run(command, cwd=ROOT, check=True)
    verify_source()


def write_readme():
    from beam_walking.experiment.high_duty_walk_selector import load_selector

    verify_source()
    validation = RUNS["validation"]
    selector = RUNS["selector"]
    report = json.loads((selector / "fit_report.json").read_text())
    rows = (validation / "selector_validation_summary.csv").read_text().splitlines()
    checkpoint = validation / "validated_high_duty_walk_selector.pt"
    _, payload = load_selector(checkpoint)
    if payload.get("deployment_ready") is not True:
        raise RuntimeError("Fresh selector validation did not pass every promotion gate")
    text = f"""# Validated high-duty walk selector — seed 4

The independent flat-ground PPO controller was trained from scratch for 1,800
updates with 3,072 environments. It covers walk at period 0.48 s and commanded
duty factors 0.75, 0.80, 0.85, and 0.90 across full step widths 0.10–0.50 m
and speeds 0.25–0.40 m/s.

The grid used 2,560 matched held-out trials. The selector maps commanded step
width and speed to one of the four tested duty factors by choosing the
lowest-energy candidate that passed the registered compliance gates. It fitted
{report['contexts']} contexts and rejected {report['rejected_contexts']}.
Fresh rollout validation contains {max(0, len(rows) - 1)} context summaries.

Main artifacts:

- `selected_vs_achieved_duty_factor.png`: selected and achieved DF versus step width.
- `selector_validation_summary.csv`: one row per validated context.
- `selector_validation_trials.csv`: fresh held-out validation trials.
- `{checkpoint.name}`: validated selector policy.
- `SELECTOR_VALIDATION_COMPLETE`: hashed validation completion record.

The selector outputs a command optimum for this trained controller and protocol;
it is not a universal biomechanical optimum.
"""
    (validation / "README.md").write_text(text)
    verify_source()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for path in RUNS.values():
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f"Pipeline output is not empty: {path}")
    commands = [
        ("smoke", [
            sys.executable, "scripts/high_duty_walk_experiment.py", "smoke",
            "--num_envs", "64", "--seed", "4", "--steps", "48",
            "--output", str(RUNS["smoke"]), "--headless"]),
        ("training", [
            sys.executable, "scripts/high_duty_walk_experiment.py", "train",
            "--num_envs", "3072", "--iterations", "1800", "--seed", "4",
            "--output", str(RUNS["training"]), "--no_watcher", "--headless"]),
        ("grid", [
            sys.executable, "scripts/collect_high_duty_walk_grid.py",
            "--checkpoint", str(RUNS["training"] / "model_1799.pt"),
            "--output", str(RUNS["grid"]), "--headless"]),
        ("selector_fit", [
            sys.executable, "scripts/fit_high_duty_walk_selector.py",
            str(RUNS["grid"] / "duty_grid_trials.csv"),
            "--output", str(RUNS["selector"])]),
        ("validation", [
            sys.executable, "scripts/collect_high_duty_walk_grid.py",
            "--checkpoint", str(RUNS["training"] / "model_1799.pt"),
            "--selector-checkpoint",
            str(RUNS["selector"] / "high_duty_walk_selector.pt"),
            "--output", str(RUNS["validation"]), "--headless"]),
    ]
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    wait_for_capacity()
    for name, command in commands:
        run_stage(name, command)
    write_readme()
    verify_source()
    write_status("complete")


EXPECTED_SOURCE_SHA256 = source_hash()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status("failed", error=repr(error))
        raise
