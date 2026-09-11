"""Run flat-ground gait-command PPO in the beam_walking project."""
import argparse
from pathlib import Path
import sys
import faulthandler

faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))
from beam_walking.experiment.protocol import (GAITS, GAIT_OFFSETS, PERIOD_TICKS, CONTROL_DT,
    CYCLE_STEPS, MIN_SWING_STEPS, STEP_WIDTH_FRAME, validate_scientific_gait_duties,
    advance_phase_ticks)
from beam_walking.experiment.stability import training_source_hash
from beam_walking.experiment.deployment import (
    STANCE_START_PROBABILITY, TRAINING_ITERATIONS, TRAINING_NUM_ENVS,
)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["smoke", "benchmark", "train", "evaluate"])
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=1800)
parser.add_argument("--seed", type=int)
parser.add_argument("--split", choices=["validation", "test"], default="validation")
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--output", type=Path, default=ROOT / "results/ppo_flat")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--stance_start_probability", type=float, default=.10)
parser.add_argument("--video", action="store_true")
parser.add_argument("--no_watcher", action="store_true",
                    help="Disable the automatic read-only training watcher")
parser.add_argument(
    "--deployment_dr", action="store_true",
    help="Train the frozen hardware-oriented dynamics/sensor randomization profile")
parser.add_argument("--camera_env", type=int, default=0)
parser.add_argument("--dfs", type=float, nargs="+")
parser.add_argument("--gait", choices=GAITS, default="trot", help="Fixed gait for one evaluation grid")
parser.add_argument("--period", type=float, default=.48, help="Evaluation period, 0.36-0.54 s in 0.02 s increments")
parser.add_argument("--speed", type=float, default=.30, help="Fixed evaluation speed, 0.25-0.40 m/s")
parser.add_argument("--step_widths", type=float, nargs="+", default=[.10, .20, .30, .40, .50],
                    help="Full left-right foot separations for flat-ground evaluation")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.dfs is None:
    args.dfs = [.75] if args.mode == "evaluate" and args.gait == "walk" else [.5, .625, .75]
if not 0. <= args.stance_start_probability <= 1.:
    parser.error("Stance-start probability must be in [0,1]")
if args.mode == "benchmark" and args.iterations == 1800:
    args.iterations = 20
if args.deployment_dr and args.mode == "evaluate":
    parser.error("Use scripts/evaluate_policy.py for deployment robustness evaluation")
if args.deployment_dr and args.checkpoint is not None:
    parser.error("The first deployment-DR policy must be trained from scratch")
if args.deployment_dr and args.mode == "train":
    if args.num_envs != TRAINING_NUM_ENVS:
        parser.error(
            f"Deployment training requires exactly {TRAINING_NUM_ENVS} environments")
    if args.iterations != TRAINING_ITERATIONS:
        parser.error(
            f"Deployment training requires exactly {TRAINING_ITERATIONS} updates")
    if args.stance_start_probability != STANCE_START_PROBABILITY:
        parser.error(
            "Deployment training requires exactly 10% grounded stance starts")
    if args.no_watcher:
        parser.error("Deployment training requires the automatic watcher")
period_ticks = round(args.period / CONTROL_DT)
if period_ticks not in PERIOD_TICKS or abs(period_ticks * CONTROL_DT - args.period) > 1e-6:
    parser.error("Period must be 0.36-0.54 s in 0.02 s increments")
if any(not .10 <= width <= .50 for width in args.step_widths):
    parser.error("Step width must be between 0.10 and 0.50 m")
if not .25 <= args.speed <= .40:
    parser.error("Evaluation speed must be between 0.25 and 0.40 m/s")
if any(df < .5 or df > min(.75, 1 - MIN_SWING_STEPS / period_ticks) + 1e-7 for df in args.dfs):
    parser.error("DF/period combination must leave at least 0.10 s requested swing; use a longer period or lower DF")
if len(set(args.step_widths)) != len(args.step_widths) or len(set(args.dfs)) != len(args.dfs):
    parser.error("Evaluation widths and DFs must not contain duplicates")
if args.mode == "evaluate":
    try:
        validate_scientific_gait_duties(args.gait, args.dfs)
    except ValueError as error:
        parser.error(str(error))
if args.seed is None:
    args.seed = (10000 if args.split == "validation" else 1000000) if args.mode == "evaluate" else 42
if args.mode == "evaluate":
    low, high = (10000, 1000000) if args.split == "validation" else (1000000, 2000000)
    if not low <= args.seed or args.seed + args.num_envs > high:
        parser.error("Evaluation seeds must stay in the selected held-out split")
elif not 0 <= args.seed < 10000:
    parser.error("Training/smoke seeds must be in [0,10000)")
if args.output.exists() and any(args.output.iterdir()):
    parser.error("Output directory is not empty; use a fresh directory to preserve provenance")
from gpu_capacity import check_capacity
capacity = check_capacity(args.mode, args.num_envs, args.device or "cuda:0", args.video)
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app
faulthandler.enable()
faulthandler.dump_traceback_later(45, repeat=True)

import hashlib
import importlib.metadata
import json
import subprocess
import shutil
import time
import os
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml
from beam_walking.experiment.task import BeamEnv, BeamEnvCfg, BeamPPORunnerCfg, command
from beam_walking.experiment.deployment import (
    deployment_profile, deployment_profile_sha256,
    deployment_training_source_hash,
)
from beam_walking.experiment.deployment_task import (
    DeploymentBeamEnv, DeploymentBeamEnvCfg, DeploymentBeamPPORunnerCfg,
    deployment_runtime_summary,
)

active_env = None

def main():
    global active_env
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / "source_snapshot"
    shutil.copytree(ROOT / "source/beam_walking/beam_walking/experiment", snapshot / "experiment", ignore=shutil.ignore_patterns("__pycache__"))
    for filename in ["beam_experiment.py", "gpu_capacity.py", "analyze_beam.py",
                     "watch_training.py", "stability_experiment.py", "analyze_stability.py",
                     "analyze_stability_ensemble.py"]:
        source = ROOT / "scripts" / filename
        if source.exists():
            shutil.copy2(source, snapshot / filename)
    (args.output / "capacity.json").write_text(json.dumps(capacity, indent=2))
    cfg = DeploymentBeamEnvCfg() if args.deployment_dr else BeamEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.stance_start_probability = args.stance_start_probability
    cfg.sim.device = args.device or "cuda:0"
    if args.mode == "evaluate":
        cfg.events.motor_gain_randomization = None
    env_class = DeploymentBeamEnv if args.deployment_dr else BeamEnv
    env = env_class(cfg, render_mode="rgb_array" if args.video else None)
    active_env = env
    if args.stance_start_probability > 0:
        try:
            env.calibrate_stance()
        except BaseException:
            if hasattr(env, "deployment_calibration_diagnostics"):
                (args.output / "deployment_calibration_diagnostics.json").write_text(
                    json.dumps(
                        env.deployment_calibration_diagnostics, indent=2))
            env.close()
            raise
        (args.output / "settled_stance.json").write_text(json.dumps(
            {key: value.detach().cpu().tolist() for key, value in env.settled_stance.items()}, indent=2))
        if hasattr(env, "deployment_calibration_diagnostics"):
            (args.output / "deployment_calibration_diagnostics.json").write_text(
                json.dumps(env.deployment_calibration_diagnostics, indent=2))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=5.)
    faulthandler.cancel_dump_traceback_later()
    agent = (DeploymentBeamPPORunnerCfg()
             if args.deployment_dr else BeamPPORunnerCfg())
    agent.seed = args.seed
    agent.device = cfg.sim.device
    agent.max_iterations = args.iterations
    dump_yaml(str(args.output / "env.yaml"), cfg)
    dump_yaml(str(args.output / "agent.yaml"), agent)
    metadata = {"argv": sys.argv, "seed": args.seed, "mode": args.mode, "split": args.split,
        "step_width_frame": STEP_WIDTH_FRAME,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "versions": {n: importlib.metadata.version(n) for n in ["torch", "isaaclab", "isaacsim", "rsl-rl-lib"]},
        "gpu": torch.cuda.get_device_name(),
        "task_sha256": hashlib.sha256(b"".join(
            (ROOT / "source/beam_walking/beam_walking/experiment" / name).read_bytes()
            for name in ["task.py", "protocol.py"])).hexdigest(),
        "training_source_sha256": training_source_hash(ROOT),
        "training_num_envs": args.num_envs,
        "training_iterations_requested": args.iterations,
        "stance_start_probability": args.stance_start_probability,
        "checkpoint_selection_rule": "final_requested_iteration",
        "watcher_enabled": bool(args.mode == "train" and not args.no_watcher),
        "deployment_domain_randomization": bool(args.deployment_dr),
        "deployment_profile": deployment_profile() if args.deployment_dr else None}
    if args.deployment_dr:
        metadata["base_training_source_sha256"] = metadata["training_source_sha256"]
        metadata["training_source_sha256"] = deployment_training_source_hash(
            ROOT, metadata["base_training_source_sha256"])
        metadata["deployment_profile_sha256"] = deployment_profile_sha256()
    metadata["fresh_training"] = bool(
        args.mode == "train" and args.checkpoint is None)
    metadata["training_lineage_id"] = (
        hashlib.sha256(
            f"{metadata['training_source_sha256']}:{args.seed}:fresh_v1".encode()
        ).hexdigest()
        if metadata["fresh_training"] else None)
    if args.checkpoint:
        metadata["checkpoint"] = str(args.checkpoint.resolve())
        metadata["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2))
    if args.mode == "smoke":
        env.capture = True
        original_stance_probability = env.cfg.stance_start_probability
        if args.deployment_dr:
            env.cfg.stance_start_probability = 1.0
        obs, _ = wrapped.reset()
        env.cfg.stance_start_probability = original_stance_probability
        runtime_summary = (
            deployment_runtime_summary(env) if args.deployment_dr else None)
        if obs["policy"].shape != (args.num_envs, 68):
            raise RuntimeError(
                f"Expected 68 policy observations, got {obs['policy'].shape}")
        if env.action_manager.action.shape != (args.num_envs, 12):
            raise RuntimeError(
                f"Expected 12 joint-position actions, got "
                f"{env.action_manager.action.shape}")
        robot = env.scene["robot"]
        kp_ratios, kd_ratios = [], []
        for actuator in robot.actuators.values():
            indices = actuator.joint_indices
            current_kp = actuator.stiffness
            current_kd = actuator.damping
            nominal_kp = robot.data.default_joint_stiffness[:, indices]
            nominal_kd = robot.data.default_joint_damping[:, indices]
            kp_ratios.append(current_kp / nominal_kp)
            kd_ratios.append(current_kd / nominal_kd)
        kp_ratios = torch.cat([value.reshape(-1) for value in kp_ratios])
        kd_ratios = torch.cat([value.reshape(-1) for value in kd_ratios])
        gain_ratios = torch.cat([kp_ratios, kd_ratios])
        if args.deployment_dr:
            if (float(kp_ratios.min()) < .6 - 1e-6
                    or float(kp_ratios.max()) > 1.4 + 1e-6
                    or float(kd_ratios.min()) < .5 - 1e-6
                    or float(kd_ratios.max()) > 1.5 + 1e-6
                    or float(kp_ratios.std()) <= .01
                    or float(kd_ratios.std()) <= .01):
                raise RuntimeError(
                    "Deployment gain randomization is outside its frozen range "
                    "or did not vary across joints/environments")
        elif not bool(torch.allclose(
                gain_ratios, torch.ones_like(gain_ratios),
                rtol=0., atol=1e-6)):
            raise RuntimeError("Paper-correlation run must use nominal Kp/Kd gains")
        if args.steps < CYCLE_STEPS:
            raise ValueError(
                f"Chronology smoke requires at least one {CYCLE_STEPS}-step gait cycle")
        first_contacts = None
        first_failures = None
        phase_tick_history = []
        reset_history = []
        for _ in range(args.steps):
            c = command(env)
            before_ticks = c.phase_ticks.clone()
            before_periods = c.period_ticks.clone()
            before_phase = c.phase.clone()
            before_commands = c.values.clone()
            obs, reward, done, info = wrapped.step(
                torch.zeros(args.num_envs, 12, device=env.device))
            done_mask = done.to(torch.bool)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
            force_shape = tuple(
                env.scene["contact_forces"].data.net_forces_w_history[
                    :, :, c.sensor_feet].norm(dim=-1).shape)
            if force_shape != (args.num_envs, cfg.decimation, 4):
                raise RuntimeError(
                    f"Expected high-rate foot forces {(args.num_envs, cfg.decimation, 4)}, "
                    f"got {force_shape}")
            if any(not bool(torch.isfinite(value)) for key, value in
                   env.training_metrics.items() if key.startswith("Gait/substep_")):
                raise RuntimeError("High-rate contact training metrics are not finite")
            torch.testing.assert_close(env.transition["phase"], before_phase)
            torch.testing.assert_close(env.transition["commands"], before_commands)
            expected_ticks = advance_phase_ticks(
                before_ticks, before_periods, done_mask)
            torch.testing.assert_close(command(env).phase_ticks, expected_ticks)
            changed = (command(env).values != before_commands).any(dim=1)
            assert not bool((changed & ~done_mask & (expected_ticks != 0)).any()), (
                "A continuing environment changed command away from a cycle boundary")
            phase_tick_history.append(before_ticks.cpu())
            reset_history.append(done_mask.cpu())
            if first_contacts is None:
                first_contacts = env.transition["contacts"].sum(dim=1).cpu().tolist()
                first_failures = env.transition["failure"].cpu().tolist()
                if args.deployment_dr or args.stance_start_probability == 1:
                    assert all(count == 4 for count in first_contacts), "Grounded reset did not start with four supported feet"
                    assert not any(first_failures), "Grounded reset spuriously failed"
        history = torch.stack(phase_tick_history[:CYCLE_STEPS])
        resets = torch.stack(reset_history[:CYCLE_STEPS])
        expected_cycle = torch.arange(CYCLE_STEPS)[:, None]
        candidates = (~resets).all(dim=0) & (history == expected_cycle).all(dim=0)
        if not bool(candidates.any()):
            raise RuntimeError("Chronology smoke found no complete surviving phase 0..23 cycle")
        chronology_env = int(candidates.nonzero()[0])
        reset_obs, _ = wrapped.reset()
        if not bool((command(env).phase_ticks == 0).all()):
            raise RuntimeError("Explicit smoke reset did not return every environment to phase zero")
        assert torch.isfinite(reset_obs["policy"]).all()
        (args.output / "smoke_checks.json").write_text(json.dumps({
            "first_contact_counts": first_contacts, "first_failures": first_failures,
            "steps": args.steps, "finite_observations_rewards": True,
            "policy_observation_dimension": 68,
            "action_dimension": 12,
            "phase_alignment_verified": True,
            "complete_cycle_env": chronology_env,
            "complete_cycle_phase_ticks": history[:, chronology_env].tolist(),
            "explicit_reset_phase_zero": True,
            "foot_force_history_shape": list(force_shape),
            "high_rate_contact_reward_finite": True,
            "motor_gain_ratio_min": float(gain_ratios.min().cpu()),
            "motor_gain_ratio_max": float(gain_ratios.max().cpu()),
            "motor_kp_ratio_min": float(kp_ratios.min().cpu()),
            "motor_kp_ratio_max": float(kp_ratios.max().cpu()),
            "motor_kd_ratio_min": float(kd_ratios.min().cpu()),
            "motor_kd_ratio_max": float(kd_ratios.max().cpu()),
            "motor_gain_randomization": bool(args.deployment_dr),
            "deployment_profile": deployment_profile()
                if args.deployment_dr else None,
            "deployment_runtime_summary": runtime_summary,
            "observation_corruption": bool(
                cfg.observations.policy.enable_corruption)}, indent=2))
        print("SMOKE_OK", obs["policy"].shape, "feet", command(env).feet, flush=True)
    else:
        runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=str(args.output), device=env.device)
        if args.checkpoint:
            saved_info = runner.load(
                str(args.checkpoint),
                load_optimizer=args.mode in ("train", "benchmark"))
            if not saved_info or not saved_info.get("task_sha256"):
                raise ValueError("Checkpoint has no source identity; cannot resume")
            if saved_info["task_sha256"] != metadata["task_sha256"]:
                raise ValueError(
                    "Checkpoint task differs from the current 68D experiment; "
                    "train a fresh controller or use its saved source")
            metadata["checkpoint_task_sha256"] = saved_info["task_sha256"]
            metadata["checkpoint_iteration"] = runner.current_learning_iteration
            (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2))
            if args.mode in ["train", "benchmark"]:
                env.common_step_counter = (saved_info or {}).get("common_step_counter", (runner.current_learning_iteration + 1) * agent.num_steps_per_env)
                runner.current_learning_iteration += 1
                wrapped.reset()
        if args.mode in ["train", "benchmark"]:
            original_save = runner.save
            def save_with_progress(path, infos=None):
                original_save(path, {"common_step_counter": env.common_step_counter,
                    "task_sha256": metadata["task_sha256"],
                    "training_source_sha256": metadata["training_source_sha256"],
                    "deployment_profile_schema": (
                        metadata["deployment_profile"]["schema"]
                        if args.deployment_dr else None),
                    "deployment_profile_sha256": (
                        metadata.get("deployment_profile_sha256"))})
            runner.save = save_with_progress
            # Random episode-length initialization would break synchronized reset/phase semantics.
            remaining = args.iterations - runner.current_learning_iteration
            if remaining <= 0:
                raise ValueError("--iterations is the TOTAL desired update count and is already reached")
            started = time.perf_counter()
            watcher = None
            watcher_log = None
            if args.mode == "train" and not args.no_watcher:
                watcher_log = (args.output / "watcher.log").open("w")
                watcher = subprocess.Popen([
                    sys.executable, str(ROOT / "scripts/watch_training.py"),
                    "--run", str(args.output.resolve()), "--output", str((args.output / "watcher_report.json").resolve()),
                    "--watch", "--pid", str(os.getpid()), "--interval", "20", "--report_interval", "600"],
                    stdout=watcher_log, stderr=subprocess.STDOUT)
                (args.output / "watcher_process.json").write_text(json.dumps({
                    "pid": watcher.pid, "training_pid": os.getpid(), "report_interval_s": 600,
                    "reports": "Initial, every 10 minutes, and final", "automatically_started": True}, indent=2))
            try:
                runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
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
                    # An independent final read also covers watcher startup/crash.
                    subprocess.run([sys.executable, str(ROOT / "scripts/watch_training.py"),
                        "--run", str(args.output.resolve()), "--output", str((args.output / "watcher_final.json").resolve())],
                        stdout=watcher_log, stderr=subprocess.STDOUT, timeout=45, check=False)
                    watcher_log.close()
            elapsed = time.perf_counter() - started
            (args.output / "throughput.json").write_text(json.dumps({
                "mode": args.mode, "num_envs": args.num_envs, "updates": remaining,
                "rollout_steps": agent.num_steps_per_env, "elapsed_s": elapsed,
                "environment_steps_per_s": remaining * args.num_envs * agent.num_steps_per_env / elapsed,
                "seconds_per_update": elapsed / remaining,
                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
                "stance_start_probability": args.stance_start_probability,
                "not_evidence_of_policy_convergence": True}, indent=2))
        else:
            if args.checkpoint is None:
                raise ValueError("Evaluation requires a frozen trained checkpoint")
            evaluate(env, wrapped, runner.get_inference_policy(device=env.device), metadata)
    wrapped.close()


@torch.inference_mode()
def evaluate(env, wrapped, policy, metadata):
    """Common random numbers across every DF/width condition; record first episodes only."""
    env.capture = True
    c = command(env)
    n = env.num_envs
    # The evaluation seed range is separate from training. Each trial owns its RNG.
    reset_plan = []
    stance_starts = []
    for i in range(n):
        rng = np.random.default_rng(args.seed + i)
        reset_plan.append([rng.uniform(-.04, .04), rng.uniform(-.01, .01)])
        stance_starts.append(rng.random() < args.stance_start_probability)
    reset_plan = np.asarray(reset_plan)
    expected_conditions = []
    for width in args.step_widths:
        for df in args.dfs:
            filename = f"trial_s{width:.3f}_d{df:.3f}_v{args.speed:.3f}_{args.gait}_p{args.period:.2f}_nominal.npz"
            expected_conditions.append({"filename": filename, "step_width": width, "df": df,
                                        "disturbed": False})
    manifest = {"terrain": "flat_ground", "step_width_frame": STEP_WIDTH_FRAME,
                "expected_conditions": expected_conditions,
                "seeds": list(range(args.seed, args.seed + n)), "period": args.period,
                "speed": args.speed, "gait": args.gait,
                "source_sha256": metadata["task_sha256"], "checkpoint_sha256": metadata["checkpoint_sha256"]}
    (args.output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2))
    for width in args.step_widths:
        for df in args.dfs:
            c.evaluation = {"df": df, "speed": args.speed,
                "gait": GAITS.index(args.gait), "period": args.period, "step_width": width,
                "stance_start": torch.tensor(stance_starts, device=env.device, dtype=torch.bool),
                "yaw": torch.tensor(reset_plan[:, 0], device=env.device, dtype=torch.float32),
                "lateral": torch.tensor(reset_plan[:, 1], device=env.device, dtype=torch.float32)}
            obs, _ = wrapped.reset()
            initial_root = env.scene["robot"].data.root_state_w.cpu().numpy().copy()
            initial_joints = env.scene["robot"].data.joint_pos.cpu().numpy().copy()
            if args.video:
                if not 0 <= args.camera_env < n:
                    raise ValueError("camera_env must index one of the evaluated environments")
                origin = env.scene.env_origins[args.camera_env].cpu().numpy()
                center = origin + np.array([1.5, 0., .2])
                env.sim.set_camera_view(eye=center + np.array([2.5, -4., 2.4]), target=center)
            active = torch.ones(n, device=env.device, dtype=torch.bool)
            frames, traces = [], {}
            for step in range(env.max_episode_length):
                with torch.inference_mode():
                    obs, _, done, _ = wrapped.step(policy(obs))
                record = {**env.transition, "valid": active.clone(), "done": done.bool().clone()}
                for key, value in record.items():
                    traces.setdefault(key, []).append(value.cpu().numpy())
                if args.video and step % 2 == 0 and active[args.camera_env]:
                    frames.append(env.render())
                active &= ~done.bool()
                if not active.any():
                    break
            stem = f"trial_s{width:.3f}_d{df:.3f}_v{args.speed:.3f}_{args.gait}_p{args.period:.2f}_nominal"
            np.savez_compressed(args.output / f"{stem}.npz", **{k: np.stack(v) for k, v in traces.items()},
                df=df, speed=args.speed, disturbed=False, terrain="flat_ground", control_dt=CONTROL_DT,
                step_width_frame=STEP_WIDTH_FRAME,
                source_sha256=metadata["task_sha256"], checkpoint_sha256=metadata["checkpoint_sha256"],
                period=args.period, gait=args.gait, step_width=width,
                phase_offsets=np.asarray(GAIT_OFFSETS[GAITS.index(args.gait)]),
                seeds=np.arange(args.seed, args.seed + n), reset_plan=reset_plan,
                initial_root=initial_root, initial_joints=initial_joints)
            if frames:
                import imageio.v2 as imageio
                success_trace = np.stack(traces["success"])[:, args.camera_env]
                valid_trace = np.stack(traces["valid"])[:, args.camera_env]
                failed = np.stack(traces["failure"])[:, args.camera_env][valid_trace][-1]
                outcome = "failure" if failed else ("success" if success_trace[valid_trace][-1] else "timeout")
                imageio.mimsave(args.output / f"{stem}_seed{args.seed + args.camera_env}_{outcome}.mp4", frames, fps=25)
            print("EVALUATED", stem, "trials", n, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        if active_env is not None:
            active_env.close()
        raise
    finally:
        app.close()
