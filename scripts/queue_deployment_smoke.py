"""Wait for the adaptive run, then execute the council-approved diagnostic once.

This queue never starts training. A failed calibration requires another review.
Run outside the process sandbox so /proc and NVIDIA report the host processes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gpu_capacity import check_capacity

ROOT = Path(__file__).resolve().parents[1]


def process_token(pid):
    """Distinguish an active process from a reused PID or exited zombie."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] == "Z" else fields[19]


def source_hashes():
    files = list((ROOT / "source/beam_walking/beam_walking/experiment").glob("*.py"))
    files += [ROOT / "scripts" / name for name in (
        "beam_experiment.py", "gpu_capacity.py", "watch_training.py",
        "queue_deployment_smoke.py")]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def adaptive_completed(directory):
    """A vanished PID alone does not establish successful training."""
    try:
        report = json.loads((directory / "throughput.json").read_text())
    except FileNotFoundError:
        return False
    return (report.get("mode") == "train" and report.get("updates") == 1800
            and report.get("num_envs") == 3072
            and (directory / "model_1799.pt").is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-run", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--smoke-output", type=Path, required=True)
    args = parser.parse_args()
    adaptive, queue, output = [path.resolve() for path in (
        args.adaptive_run, args.queue_dir, args.smoke_output)]
    for path in (adaptive, queue, output):
        if not path.is_relative_to(ROOT / "results"):
            parser.error("All queue paths must be inside this repository's results")
    if output.exists():
        parser.error("Smoke output must not exist")
    queue.mkdir(parents=True, exist_ok=False)
    status = {"queue_pid": os.getpid(), "adaptive_run": str(adaptive),
              "smoke_output": str(output), "training_queued": False}

    def update(state, **extra):
        status.update(state=state, updated_at=datetime.now(timezone.utc).isoformat(),
                      **extra)
        temporary = queue / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(queue / "status.json")
        print(json.dumps(status), flush=True)

    try:
        watcher = json.loads((adaptive / "watcher_process.json").read_text())
        pid = int(watcher["training_pid"])
        token = process_token(pid)
        if token is not None:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            argv = [value.decode() for value in argv if value]
            if (not any(Path(value).name == "adaptive_duty_experiment.py"
                        for value in argv) or "train" not in argv
                    or "--output" not in argv):
                raise RuntimeError("Adaptive PID does not match the training command")
            target = Path(argv[argv.index("--output") + 1])
            cwd = Path(f"/proc/{pid}/cwd").resolve()
            if (cwd / target).resolve() != adaptive:
                raise RuntimeError("Adaptive PID output directory mismatch")
        frozen = source_hashes()
        command = [sys.executable, str(ROOT / "scripts/beam_experiment.py"),
                   "smoke", "--deployment_dr", "--num_envs", "64",
                   "--steps", "48", "--seed", "3", "--headless",
                   "--output", str(output)]
        plan = {"adaptive_pid": pid, "adaptive_start_ticks": token,
                "source_sha256": frozen, "command": command,
                "scope": "one reviewed diagnostic smoke; no training"}
        (queue / "plan.json").write_text(json.dumps(plan, indent=2))
        deadline = time.monotonic() + 24 * 3600
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError("Queue expired after 24 hours")
            if source_hashes() != frozen:
                raise RuntimeError("Reviewed source changed while queued; re-review required")
            if token is not None and process_token(pid) == token:
                update("waiting_for_adaptive", adaptive_pid=pid)
                time.sleep(30)
                continue
            if not adaptive_completed(adaptive):
                raise RuntimeError("Adaptive process exited without completion artifacts")
            try:
                capacity = check_capacity("smoke", 64, "cuda:0", False)
            except (RuntimeError, subprocess.CalledProcessError) as error:
                update("waiting_for_capacity", reason=str(error))
                time.sleep(30)
                continue
            if source_hashes() != frozen:
                raise RuntimeError("Reviewed source changed before launch")
            if output.exists():
                raise RuntimeError("Smoke output appeared while queued; refusing overwrite")
            update("running_smoke", capacity=capacity)
            # The entrypoint repeats capacity checks immediately before Isaac starts.
            with (queue / "smoke.log").open("x") as log:
                result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
            checks_path = output / "smoke_checks.json"
            # Isaac may exit zero even after a Python exception; require its artifact.
            if result.returncode != 0 or not checks_path.is_file():
                update("smoke_failed_review_required", returncode=result.returncode,
                       diagnostics=str(output / "deployment_calibration_diagnostics.json"))
                return 1
            checks = json.loads(checks_path.read_text())
            if (checks.get("finite_observations_rewards") is not True
                    or checks.get("policy_observation_dimension") != 68
                    or checks.get("action_dimension") != 12
                    or checks.get("first_contact_counts") != [4] * 64
                    or checks.get("first_failures") != [False] * 64
                    or not checks.get("deployment_runtime_summary")):
                raise RuntimeError("Smoke artifact does not prove required runtime checks")
            update("smoke_passed_awaiting_training_review")
            return 0
    except Exception as error:
        update("blocked", reason=str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
