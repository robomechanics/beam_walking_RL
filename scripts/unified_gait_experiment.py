"""Train the unified flat-ground trot/walk controller.

One policy, both gaits, one period distribution (0.36--0.54 s), trot duty
0.50--0.75 and walk duty 0.75--0.90. See docs/unified_gait_plan.md.
"""

import argparse
import faulthandler
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))

from beam_walking.experiment.unified_gait import (  # noqa: E402
    TROT_DUTY_RANGE,
    TROT_MIN_SWING_STEPS,
    TROT_PERIOD_TICKS,
    UNIFIED_STRATA,
    UNIFIED_TASK_FILES,
    UNIFIED_TRAINING_ITERATIONS,
    UNIFIED_TRAINING_NUM_ENVS,
    UNIFIED_TRAINING_SEED,
    UNIFIED_TRAINING_SOURCE_FILES,
    WALK_DUTY_RANGE,
    WALK_MIN_SWING_STEPS,
    WALK_PERIOD_TICKS,
    files_sha256,
    unified_lineage_id,
)
from beam_walking.experiment.protocol import (  # noqa: E402
    CYCLE_STEPS,
    advance_phase_ticks,
)
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("smoke", "benchmark", "train"))
parser.add_argument("--num_envs", type=int, default=UNIFIED_TRAINING_NUM_ENVS)
parser.add_argument("--iterations", type=int, default=UNIFIED_TRAINING_ITERATIONS)
parser.add_argument("--seed", type=int, default=UNIFIED_TRAINING_SEED)
parser.add_argument("--development", action="store_true",
                    help="Allow non-primary training parameters")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--stance_start_probability", type=float, default=.10)
parser.add_argument("--no_watcher", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.mode == "benchmark" and args.iterations == 1800:
    args.iterations = 20
if args.num_envs < 1 or args.iterations < 1:
    parser.error("Environment and iteration counts must be positive")
if not 0 <= args.seed < 10_000:
    parser.error("Training and smoke seeds must be in [0, 10000)")
if not 0 <= args.stance_start_probability <= 1:
    parser.error("Stance-start probability must be in [0, 1]")
if args.output.exists() and any(args.output.iterdir()):
    parser.error("Output directory must be new or empty")
if args.mode == "train" and not args.development and (
        args.num_envs != UNIFIED_TRAINING_NUM_ENVS
        or args.iterations != UNIFIED_TRAINING_ITERATIONS
        or args.seed != UNIFIED_TRAINING_SEED
        or abs(args.stance_start_probability - .10) > 1e-12):
    parser.error(
        "Primary unified training requires 3072 environments, 1800 updates, "
        "seed 5, and 0.10 grounded-start probability")

from gpu_capacity import check_capacity  # noqa: E402

capacity = check_capacity(
    args.mode, args.num_envs, args.device or "cuda:0", False)
app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from beam_walking.experiment.unified_gait_task import (  # noqa: E402
    UnifiedEnv,
    UnifiedEnvCfg,
    UnifiedPPORunnerCfg,
)
from beam_walking.experiment.task import command  # noqa: E402

def snapshot_sources(output):
    snapshot = output / "source_snapshot"
    for relative in UNIFIED_TRAINING_SOURCE_FILES:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)


@torch.inference_mode()
def smoke(env, wrapped):
    env.capture = True
    observation, _ = wrapped.reset()
    if observation["policy"].shape != (args.num_envs, 68):
        raise RuntimeError(
            f"Expected 68 policy observations, got {observation['policy'].shape}")
    if env.action_manager.action.shape != (args.num_envs, 12):
        raise RuntimeError("Unified controller must emit 12 joint actions")
    gait_command = command(env)
    sampled_strata = {
        (int(gait), round(float(duty), 3))
        for gait, duty in zip(gait_command.values[:, 4], gait_command.values[:, 1])
    }
    required = {(gait, round(duty, 3)) for gait, duty in UNIFIED_STRATA}
    if args.num_envs >= len(required) and not required <= sampled_strata:
        raise RuntimeError(
            f"Smoke reset did not cover every unified gait/DF stratum: {sampled_strata}")
    histories, reset_histories = [], []
    first_grounded_contacts = None
    for step in range(max(args.steps, CYCLE_STEPS)):
        before_ticks = gait_command.phase_ticks.clone()
        before_periods = gait_command.period_ticks.clone()
        before_phase = gait_command.phase.clone()
        before_commands = gait_command.values.clone()
        observation, reward, done, _ = wrapped.step(torch.zeros(
            args.num_envs, 12, device=env.device))
        done = done.bool()
        if not bool(torch.isfinite(observation["policy"]).all()
                    and torch.isfinite(reward).all()):
            raise RuntimeError("Unified smoke produced a nonfinite value")
        torch.testing.assert_close(env.transition["phase"], before_phase)
        torch.testing.assert_close(env.transition["commands"], before_commands)
        expected = advance_phase_ticks(before_ticks, before_periods, done)
        torch.testing.assert_close(gait_command.phase_ticks, expected)
        changed = (gait_command.values != before_commands).any(dim=1)
        if bool((changed & ~done & (expected != 0)).any()):
            raise RuntimeError("Unified command changed away from a cycle boundary")
        if step == 0 and bool(gait_command.stance_start.any()):
            grounded = gait_command.stance_start
            first_grounded_contacts = env.transition["contacts"][grounded].sum(1)
            if (not bool((first_grounded_contacts == 4).all())
                    or bool(env.transition["failure"][grounded].any())):
                raise RuntimeError("Grounded unified reset did not begin supported")
        histories.append(before_ticks.cpu())
        reset_histories.append(done.cpu())
    history = torch.stack(histories[:CYCLE_STEPS])
    resets = torch.stack(reset_histories[:CYCLE_STEPS])
    expected = torch.arange(CYCLE_STEPS)[:, None]
    complete = (~resets).all(0) & (history == expected).all(0)
    if not bool(complete.any()):
        raise RuntimeError("Unified smoke found no complete phase-locked cycle")
    return {
        "policy_observation_dimension": 68,
        "action_dimension": 12,
        "finite_observations_rewards": True,
        "phase_alignment_verified": True,
        "all_unified_strata_present": required <= sampled_strata,
        "boundary_only_command_changes": True,
        "grounded_reset_first_contact_counts": (
            first_grounded_contacts.cpu().tolist()
            if first_grounded_contacts is not None else []),
        "sampled_strata": sorted([list(item) for item in sampled_strata]),
        "complete_cycle_env": int(complete.nonzero()[0]),
    }


def main():
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_sources(args.output)
    (args.output / "capacity.json").write_text(json.dumps(capacity, indent=2))

    cfg = UnifiedEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.stance_start_probability = args.stance_start_probability
    cfg.sim.device = args.device or "cuda:0"
    cfg.events.motor_gain_randomization = None
    env = UnifiedEnv(cfg)
    wrapped = None
    try:
        if args.stance_start_probability:
            env.calibrate_stance()
            (args.output / "settled_stance.json").write_text(json.dumps({
                key: value.detach().cpu().tolist()
                for key, value in env.settled_stance.items()
            }, indent=2))
        wrapped = RslRlVecEnvWrapper(env, clip_actions=5.)
        agent = UnifiedPPORunnerCfg()
        agent.seed = args.seed
        agent.device = cfg.sim.device
        agent.max_iterations = args.iterations
        dump_yaml(str(args.output / "env.yaml"), cfg)
        dump_yaml(str(args.output / "agent.yaml"), agent)
        task_sha256 = files_sha256(ROOT, UNIFIED_TASK_FILES)
        training_source_sha256 = files_sha256(
            ROOT, UNIFIED_TRAINING_SOURCE_FILES)
        lineage = unified_lineage_id(training_source_sha256, args.seed)
        provenance = {
            "schema": "unified_gait_training_v1",
            "argv": sys.argv, "mode": args.mode, "seed": args.seed,
            "terrain": "flat_ground", "terrain_width_input": False,
            "step_width_is_commanded_foot_separation": True,
            "gaits": ["trot", "walk"],
            "gait_balance": "equal",
            "trot_duty_range": list(TROT_DUTY_RANGE),
            "walk_duty_range": list(WALK_DUTY_RANGE),
            "balanced_strata": [[gait, duty] for gait, duty in UNIFIED_STRATA],
            "anchor_period_s": .48,
            "trot_period_control_ticks": list(TROT_PERIOD_TICKS),
            "walk_period_control_ticks": list(WALK_PERIOD_TICKS),
            "trot_minimum_requested_swing_control_ticks": TROT_MIN_SWING_STEPS,
            "walk_minimum_requested_swing_control_ticks": WALK_MIN_SWING_STEPS,
            "motor_gain_randomization": False,
            "stance_start_probability": args.stance_start_probability,
            "fresh_training": args.mode == "train",
            "primary_training_protocol": bool(
                args.mode == "train" and not args.development),
            "warm_start": False,
            "task_sha256": task_sha256,
            "training_source_sha256": training_source_sha256,
            "training_lineage_id": (
                lineage if args.mode == "train" else None),
            "training_num_envs": args.num_envs,
            "training_iterations_requested": args.iterations,
            "checkpoint_selection_rule": "final_requested_iteration",
            "watcher_enabled": bool(args.mode == "train" and not args.no_watcher),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "isaaclab", "isaacsim", "rsl-rl-lib")},
            "gpu": torch.cuda.get_device_name(),
        }
        (args.output / "provenance.json").write_text(
            json.dumps(provenance, indent=2))

        faulthandler.cancel_dump_traceback_later()
        if args.mode == "smoke":
            result = smoke(env, wrapped)
            (args.output / "smoke_checks.json").write_text(
                json.dumps(result, indent=2))
            print("UNIFIED_GAIT_SMOKE_OK", json.dumps(result), flush=True)
            return

        runner = OnPolicyRunner(
            wrapped, agent.to_dict(), log_dir=str(args.output), device=env.device)
        original_save = runner.save

        def save_with_identity(path, infos=None):
            original_save(path, {
                "common_step_counter": env.common_step_counter,
                "task_sha256": task_sha256,
                "training_lineage_id": lineage,
            })

        runner.save = save_with_identity
        started = time.perf_counter()
        watcher = watcher_log = None
        if args.mode == "train" and not args.no_watcher:
            watcher_log = (args.output / "watcher.log").open("w")
            watcher = subprocess.Popen([
                sys.executable, str(ROOT / "scripts/watch_training.py"),
                "--run", str(args.output.resolve()),
                "--output", str((args.output / "watcher_report.json").resolve()),
                "--watch", "--pid", str(os.getpid()), "--interval", "20",
                "--report_interval", "600",
            ], stdout=watcher_log, stderr=subprocess.STDOUT)
            (args.output / "watcher_process.json").write_text(json.dumps({
                "pid": watcher.pid, "training_pid": os.getpid(),
                "report_interval_s": 600,
                "reports": "Initial, every 10 minutes, and final",
                "automatically_started": True,
            }, indent=2))
        try:
            runner.learn(
                num_learning_iterations=args.iterations,
                init_at_random_ep_len=False)
        finally:
            if getattr(runner, "writer", None) is not None:
                runner.writer.flush()
            if watcher is not None:
                watcher.terminate()
                try:
                    watcher.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    watcher.kill()
                    watcher.wait(timeout=5)
                # Preserve a final independent read even if the continuous
                # watcher failed to start or exited unexpectedly.
                subprocess.run([
                    sys.executable, str(ROOT / "scripts/watch_training.py"),
                    "--run", str(args.output.resolve()),
                    "--output", str(
                        (args.output / "watcher_final.json").resolve()),
                    "--final-report",
                ], stdout=watcher_log, stderr=subprocess.STDOUT,
                    timeout=45, check=False)
                watcher_log.close()
        elapsed = time.perf_counter() - started
        throughput = {
            "mode": args.mode, "num_envs": args.num_envs,
            "updates": args.iterations,
            "rollout_steps": agent.num_steps_per_env,
            "elapsed_s": elapsed,
            "environment_steps_per_s": (
                args.iterations * args.num_envs * agent.num_steps_per_env / elapsed),
            "seconds_per_update": elapsed / args.iterations,
            "torch_peak_allocated_mib":
                torch.cuda.max_memory_allocated() / 1024 ** 2,
        }
        (args.output / "throughput.json").write_text(
            json.dumps(throughput, indent=2))
        print("UNIFIED_GAIT_TRAINING_COMPLETE", json.dumps(throughput), flush=True)
    finally:
        if wrapped is not None:
            wrapped.close()
        else:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
