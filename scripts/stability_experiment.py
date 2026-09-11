"""Estimate paper-style closed-loop gait convergence on flat ground without pushes."""
import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np

faulthandler.enable()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))
from beam_walking.experiment.protocol import (
    CONTROL_DT, GAITS, MIN_SWING_STEPS, PERIOD_TICKS,
)
from beam_walking.experiment.stability import (
    AUGMENTED_DIM, CONFIRMATORY_TRAINING_ITERATIONS, FROZEN_GATE_LIMITS,
    ORBITAL_AUGMENTED_INDICES, RAW_STATE_DIM, STATE_SCALES,
    canonical_reference_states, confirmatory_protocol,
    construct_master_stencil, evaluation_source_hash, master_stencil_layout,
    state_delta, training_source_hash,
)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--references", type=int, default=8)
parser.add_argument("--settle_cycles", type=int, default=12)
parser.add_argument(
    "--perturbation_sizes", type=float, nargs="+", default=[.025, .05, .10])
parser.add_argument(
    "--step_widths", type=float, nargs="+", default=[.10, .20, .30, .40, .50])
parser.add_argument("--dfs", type=float, nargs="+", default=[.50, .625, .75])
parser.add_argument(
    "--speeds", "--speed", type=float, nargs="+", default=[.25, .30, .35, .40])
parser.add_argument(
    "--gaits", "--gait", choices=GAITS, nargs="+", default=list(GAITS))
parser.add_argument(
    "--periods", "--period", type=float, nargs="+", default=[.48])
parser.add_argument("--seed", type=int, default=10000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.references < 1 or args.settle_cycles < 4:
    parser.error("Use at least one reference and four settling cycles")
if any(not np.isfinite(h) or h <= 0 or h > .2 for h in args.perturbation_sizes):
    parser.error("Perturbation sizes must be finite and in (0, 0.2]")
if len(set(args.perturbation_sizes)) != len(args.perturbation_sizes):
    parser.error("Perturbation sizes must be unique")
if not all(
    any(np.isclose(h, required) for h in args.perturbation_sizes)
    for required in (.025, .05, .10)
):
    parser.error("Perturbation sizes must include 0.025, 0.05, and 0.10")
if not args.checkpoint.is_file():
    parser.error("Checkpoint does not exist or is not a file")
condition_period_ticks = {}
for period in args.periods:
    ticks = round(period / CONTROL_DT)
    if ticks not in PERIOD_TICKS or not np.isclose(ticks * CONTROL_DT, period):
        parser.error("Periods must be 0.36-0.54 s in 0.02 s increments")
    condition_period_ticks[period] = ticks
if any(df < .5 or df > .75 for df in args.dfs):
    parser.error("Duty factors must be in [0.50, 0.75]")
if any(width < .10 or width > .50 for width in args.step_widths):
    parser.error("Step width must be in [0.10, 0.50] m")
if any(speed < .25 or speed > .40 for speed in args.speeds):
    parser.error("Speeds must be in [0.25, 0.40] m/s")
for values, label in (
    (args.step_widths, "widths"), (args.dfs, "DFs"),
    (args.speeds, "speeds"), (args.gaits, "gaits"),
    (args.periods, "periods"),
):
    if len(set(values)) != len(values):
        parser.error(f"Stability {label} must be unique")
if args.output.exists() and any(args.output.iterdir()):
    parser.error("Output directory must be new or empty")


def valid_condition(period, gait, duty):
    """Keep the no-flight walk and at least 0.10 seconds of requested swing."""
    ticks = condition_period_ticks[period]
    if gait == "walk" and not np.isclose(duty, .75):
        return False
    return duty <= 1 - MIN_SWING_STEPS / ticks + 1e-7


ZERO_CLONES = 2
LAYOUT = master_stencil_layout(args.perturbation_sizes, ZERO_CLONES)
MASTER_STENCIL = LAYOUT["size"]
num_envs = MASTER_STENCIL * args.references

from gpu_capacity import check_capacity
capacity = check_capacity("evaluate", num_envs, args.device or "cuda:0", False)
app = AppLauncher(args).app

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from beam_walking.experiment.task import (
    BeamEnv, BeamEnvCfg, BeamPPORunnerCfg, command,
)

active_env = None


def raw_state(env):
    """Return local 49D state: xyz,q,12 joints,base v/omega,12 qdot,last action."""
    robot = env.scene["robot"]
    root = robot.data.root_state_w.clone()
    root[:, :3] -= env.scene.env_origins
    return torch.cat([
        root[:, :7], robot.data.joint_pos, root[:, 7:13],
        robot.data.joint_vel, env.action_manager.action,
    ], dim=1)


def write_raw_states(env, raw):
    """Write one raw state for every environment without changing commands."""
    robot = env.scene["robot"]
    ids = torch.arange(num_envs, device=env.device)
    root = torch.cat([
        raw[:, :3] + env.scene.env_origins,
        raw[:, 3:7], raw[:, 19:25],
    ], dim=1)
    robot.write_root_state_to_sim(root, ids)
    robot.write_joint_state_to_sim(
        raw[:, 7:19], raw[:, 25:37], env_ids=ids)
    env.action_manager._action[:] = raw[:, 37:49]
    env.action_manager._prev_action[:] = raw[:, 37:49]


def reference_ids(device):
    return torch.arange(args.references, device=device) * MASTER_STENCIL


@torch.inference_mode()
def settle_condition(env, wrapped, policy, speed, gait, period,
                     period_ticks, width, duty):
    """Settle every clone from identical per-reference physical histories."""
    c = command(env)
    zeros = torch.zeros(num_envs, device=env.device)
    c.evaluation = {
        "df": duty, "speed": speed, "gait": GAITS.index(gait),
        "period": period, "step_width": width,
        "stance_start": torch.zeros(
            num_envs, device=env.device, dtype=torch.bool),
        "yaw": zeros, "lateral": zeros,
    }
    wrapped.reset()
    refs = reference_ids(env.device)

    # Apply matched small initial-state offsets to each reference, then broadcast
    # each state to every member of that reference's master stencil. All members
    # subsequently accumulate the same contact-solver history for 12 periods.
    reference_raw = canonical_reference_states(
        env.scene["robot"].data.default_joint_pos[0].cpu().numpy(),
        args.references, seed=args.seed, offset_half_width=.02)
    broadcast = np.repeat(
        reference_raw[:, None, :], MASTER_STENCIL, axis=1
    ).reshape(num_envs, RAW_STATE_DIM)
    write_raw_states(env, torch.as_tensor(
        broadcast, device=env.device, dtype=torch.float32))
    c.contact_cache[:] = c.contact_cache[refs].repeat_interleave(MASTER_STENCIL)
    c.nonfoot_cache[:] = c.nonfoot_cache[refs].repeat_interleave(MASTER_STENCIL)
    c.phase_ticks[:] = 0
    env.episode_length_buf[:] = 0
    env.substep_failure[:] = False
    obs = wrapped.get_observations()

    phase_states = []
    settle_done = torch.zeros(
        args.references, device=env.device, dtype=torch.bool)
    for cycle in range(args.settle_cycles):
        for _ in range(period_ticks):
            obs, _, done, _ = wrapped.step(policy(obs))
            settle_done |= done.bool().reshape(
                args.references, MASTER_STENCIL).any(dim=1)
        if cycle >= args.settle_cycles - 4:
            phase_states.append(
                raw_state(env).reshape(
                    args.references, MASTER_STENCIL, RAW_STATE_DIM
                )[:, 0].cpu().numpy())

    cycle_rms = np.zeros(args.references)
    for previous, current in zip(phase_states[:-1], phase_states[1:]):
        delta = state_delta(previous, current) / STATE_SCALES
        value = np.sqrt(np.mean(
            delta[:, ORBITAL_AUGMENTED_INDICES] ** 2, axis=1))
        cycle_rms = np.maximum(cycle_rms, value)
    cycle_rms[settle_done.cpu().numpy()] = np.inf

    settled = raw_state(env).reshape(
        args.references, MASTER_STENCIL, RAW_STATE_DIM).cpu().numpy()
    baseline = np.repeat(settled[:, :1], MASTER_STENCIL, axis=1)
    spread = state_delta(baseline, settled) / STATE_SCALES
    settle_group_rms = np.sqrt(np.mean(
        spread[..., ORBITAL_AUGMENTED_INDICES] ** 2, axis=-1
    )).max(axis=1)
    return settled, cycle_rms, settle_group_rms, settle_done.cpu().numpy()


@torch.inference_mode()
def collect_condition(env, wrapped, policy, period_ticks, settled_result):
    """Roll all h stencils and zero-offset controls concurrently for one period."""
    settled, cycle_rms, settle_group_rms, settle_done = settled_result
    master, layout = construct_master_stencil(
        settled, args.perturbation_sizes, ZERO_CLONES)
    write_raw_states(env, torch.as_tensor(
        master.reshape(num_envs, RAW_STATE_DIM),
        device=env.device, dtype=torch.float32))
    c = command(env)
    if not bool(torch.all(c.phase_ticks == 0)):
        raise RuntimeError("Stability reference is not phase locked at zero")
    initial_contacts = c.contact_cache.clone()
    initial_desired = c.desired[reference_ids(env.device)].clone()
    initial = raw_state(env).reshape(
        args.references, MASTER_STENCIL, RAW_STATE_DIM).cpu().numpy()
    obs = wrapped.get_observations()
    refs = reference_ids(env.device)
    done_latched = torch.zeros(
        num_envs, device=env.device, dtype=torch.bool)
    contact_trace, desired_trace = [], []
    feet_trace, velocity_trace, failure_trace = [], [], []
    lateral_trace, heading_trace = [], []
    lateral_velocity_trace, yaw_rate_trace = [], []
    stencil_substeps = []
    env.capture_substeps = True
    for _ in range(period_ticks):
        obs, _, done, _ = wrapped.step(policy(obs))
        done_latched |= done.bool()
        transition = env.transition
        contact_trace.append(transition["contacts"][refs].cpu().numpy())
        desired_trace.append(transition["desired"][refs].cpu().numpy())
        feet_trace.append(transition["feet_body"][refs].cpu().numpy())
        velocity_trace.append(
            transition["forward_velocity"][refs].cpu().numpy())
        lateral_trace.append(
            transition["body"][refs, 1].cpu().numpy())
        quat = transition["root_quat"][refs]
        yaw = torch.atan2(
            2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1 - 2 * (quat[:, 2].square() + quat[:, 3].square()))
        heading_trace.append(yaw.cpu().numpy())
        lateral_velocity_trace.append(
            transition["world_lateral_velocity"][refs].cpu().numpy())
        yaw_rate_trace.append(
            transition["body_yaw_rate"][refs].cpu().numpy())
        failure_trace.append(transition["failure"][refs].cpu().numpy())
        if len(env.substep_contacts) != env.cfg.decimation:
            raise RuntimeError("Missing 200 Hz contact samples")
        stencil_substeps.append(
            torch.stack(env.substep_contacts, dim=1).cpu().numpy())
    env.capture_substeps = False

    final = raw_state(env).reshape(
        args.references, MASTER_STENCIL, RAW_STATE_DIM).cpu().numpy()
    done_master = done_latched.reshape(
        args.references, MASTER_STENCIL).cpu().numpy()
    contact_master = np.concatenate(
        stencil_substeps, axis=1
    ).reshape(args.references, MASTER_STENCIL, -1, 4)
    initial_contact_master = initial_contacts.reshape(
        args.references, MASTER_STENCIL, 4).cpu().numpy()
    common = {
        "cycle_rms": cycle_rms,
        "settle_group_rms": settle_group_rms,
        "settle_done": settle_done,
        "nominal_contacts": np.stack(contact_trace, axis=1),
        "nominal_desired": np.stack(desired_trace, axis=1),
        "desired_initial_contacts": initial_desired.cpu().numpy(),
        "desired_substep_contacts": np.repeat(
            np.stack(desired_trace, axis=1), env.cfg.decimation, axis=1),
        "nominal_feet_body": np.stack(feet_trace, axis=1),
        "nominal_forward_velocity": np.stack(velocity_trace, axis=1),
        "nominal_lateral_position": np.stack(lateral_trace, axis=1),
        "nominal_heading": np.stack(heading_trace, axis=1),
        "nominal_world_lateral_velocity":
            np.stack(lateral_velocity_trace, axis=1),
        "nominal_body_yaw_rate": np.stack(yaw_rate_trace, axis=1),
        "nominal_failure": np.stack(failure_trace, axis=1),
        "zero_initial_states": initial[:, layout["zero_clones"]],
        "zero_final_states": final[:, layout["zero_clones"]],
        "zero_initial_contacts": initial_contact_master[
            :, layout["zero_clones"]],
        "zero_substep_contacts": contact_master[:, layout["zero_clones"]],
    }
    result = {}
    for h, local in layout["h_indices"].items():
        result[h] = {
            **common,
            "initial_states": initial[:, local],
            "final_states": final[:, local],
            "done": done_master[:, local],
            "stencil_initial_contacts": initial_contact_master[:, local],
            "stencil_substep_contacts": contact_master[:, local],
        }
    return result


def main():
    global active_env
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = BeamEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device or "cuda:0"
    cfg.events.motor_gain_randomization = None
    env = BeamEnv(cfg)
    active_env = env
    env.capture = True
    env.calibrate_stance()
    wrapped = RslRlVecEnvWrapper(env, clip_actions=5.)

    agent = BeamPPORunnerCfg()
    agent.seed = args.seed
    agent.device = cfg.sim.device
    runner = OnPolicyRunner(
        wrapped, agent.to_dict(), log_dir=None, device=env.device)
    saved = runner.load(str(args.checkpoint), load_optimizer=False)
    task_hash = hashlib.sha256(b"".join(
        (ROOT / "source/beam_walking/beam_walking/experiment" / name).read_bytes()
        for name in ["task.py", "protocol.py"]
    )).hexdigest()
    if not saved or saved.get("task_sha256") != task_hash:
        raise ValueError(
            "Stability checkpoint must match the current task/protocol source")
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    training_provenance_path = args.checkpoint.parent / "provenance.json"
    if not training_provenance_path.is_file():
        raise ValueError("Checkpoint training provenance.json is required")
    training_provenance = json.loads(training_provenance_path.read_text())
    current_training_source = training_source_hash(ROOT)
    expected_lineage = (
        hashlib.sha256(
            f"{current_training_source}:{training_provenance.get('seed')}:fresh_v1".encode()
        ).hexdigest()
        if isinstance(training_provenance.get("seed"), int) else None)
    if (
        training_provenance.get("mode") != "train"
        or training_provenance.get("task_sha256") != task_hash
        or not isinstance(training_provenance.get("seed"), int)
        or training_provenance.get("fresh_training") is not True
        or "checkpoint" in training_provenance
        or training_provenance.get("training_lineage_id") != expected_lineage
        or training_provenance.get("training_source_sha256")
            != current_training_source
        or training_provenance.get("training_iterations_requested")
            != CONFIRMATORY_TRAINING_ITERATIONS
        or training_provenance.get("checkpoint_selection_rule")
            != "final_requested_iteration"
        or not isinstance(training_provenance.get("training_num_envs"), int)
        or training_provenance.get("training_num_envs") < 1
        or runner.current_learning_iteration
            != CONFIRMATORY_TRAINING_ITERATIONS - 1
    ):
        raise ValueError(
            "Stability claims require a fresh, parent-free training lineage "
            "with matching task source and integer seed")
    training_provenance_bytes = training_provenance_path.read_bytes()
    training_provenance_hash = hashlib.sha256(
        training_provenance_bytes).hexdigest()
    eval_hash = evaluation_source_hash(ROOT)
    policy = runner.get_inference_policy(device=env.device)

    conditions = []
    for period in args.periods:
        for gait in args.gaits:
            for speed in args.speeds:
                for width in args.step_widths:
                    for duty in args.dfs:
                        if not valid_condition(period, gait, duty):
                            continue
                        for h in args.perturbation_sizes:
                            filename = (
                                f"stability_s{width:.3f}_d{duty:.3f}_"
                                f"v{speed:.3f}_{gait}_p{period:.2f}_"
                                f"h{h:.3f}.npz")
                            conditions.append({
                                "filename": filename, "speed": speed,
                                "duty_factor": duty, "step_width": width,
                                "period": period, "gait": gait,
                                "perturbation_h": h,
                            })
    if not conditions:
        raise ValueError(
            "No physically valid gait/DF/period conditions were requested")
    manifest = {
        "schema": "beam_stability_v2",
        "paper_metric_primary":
            "chi_orb=sigma_max(Phi_orbital_augmented_46D)",
        "primary_metric_scope":
            "translation-reduced learned-controller analogue",
        "unquotiented_full_48d_is_diagnostic": True,
        "single_policy_results_are_descriptive": True,
        "minimum_policies_for_claim": 5,
        "confirmatory_grid": confirmatory_protocol(
            condition_keys=[
                (item["period"], item["gait"], item["speed"],
                 item["step_width"], item["duty_factor"],
                 item["perturbation_h"])
                for item in conditions],
            references=args.references, settle_cycles=args.settle_cycles,
            perturbation_sizes=args.perturbation_sizes,
            zero_clones=ZERO_CLONES, master_stencil_size=MASTER_STENCIL,
            evaluation_reference_seed=args.seed,
            training_iterations_requested=training_provenance[
                "training_iterations_requested"],
            checkpoint_iteration=runner.current_learning_iteration,
            fresh_training=training_provenance["fresh_training"]),
        "terrain": "flat_ground", "external_pushes": False,
        "state_dimension_physical": 36,
        "state_dimension_augmented": 48,
        "state_scales": STATE_SCALES.tolist(),
        "tangent_labels": list(__import__(
            "beam_walking.experiment.stability",
            fromlist=["TANGENT_LABELS"]).TANGENT_LABELS),
        "references": args.references,
        "zero_clones_per_reference": ZERO_CLONES,
        "master_stencil_size": MASTER_STENCIL,
        "settle_cycles": args.settle_cycles,
        "settle_group_rms_limit":
            FROZEN_GATE_LIMITS["settle_group_rms"],
        "zero_clone_noise_fraction_of_h_limit":
            FROZEN_GATE_LIMITS["zero_clone_noise_fraction_of_h"],
        "initial_stencil_max_error_limit":
            FROZEN_GATE_LIMITS["initial_stencil_max_error"],
        "initial_condition_number_limit":
            FROZEN_GATE_LIMITS["initial_condition_number"],
        "translation_symmetry_residual_limit":
            FROZEN_GATE_LIMITS["translation_symmetry_residual"],
        "gate_limits": FROZEN_GATE_LIMITS,
        "evaluation_reference_seed": args.seed,
        "expected_conditions": conditions,
        "expected_files": [item["filename"] for item in conditions],
        "reference_initial_offset_half_width_normalized": .02,
        "reference_initialization":
            "canonical_default_plus_matched_offsets_v1",
        "training_source_sha256": current_training_source,
        "training_num_envs": training_provenance["training_num_envs"],
        "training_iterations_requested":
            training_provenance["training_iterations_requested"],
        "checkpoint_selection_rule": "final_requested_iteration",
        "checkpoint_iteration": runner.current_learning_iteration,
        "gait_regime": {
            "trot_df": [.50, .75], "walk_df": [.75, .75],
            "minimum_swing_s": MIN_SWING_STEPS * CONTROL_DT,
        },
        "task_sha256": task_hash,
        "evaluation_sha256": eval_hash,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint": str(args.checkpoint.resolve()),
        "training_seed": training_provenance["seed"],
        "fresh_training": True,
        "training_lineage_id": training_provenance["training_lineage_id"],
        "training_provenance_source":
            str(training_provenance_path.resolve()),
        "training_provenance_file": "training_provenance.json",
        "training_provenance_sha256": training_provenance_hash,
        "joint_order": list(env.scene["robot"].joint_names),
        "action_order": list(env.scene["robot"].joint_names),
        "capacity": capacity,
        "argv": sys.argv,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "isaaclab", "isaacsim", "rsl-rl-lib"]
        },
    }
    (args.output / "stability_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    (args.output / "training_provenance.json").write_bytes(
        training_provenance_bytes)
    shutil.copytree(
        ROOT / "source/beam_walking/beam_walking/experiment",
        args.output / "source_snapshot" / "experiment",
        ignore=shutil.ignore_patterns("__pycache__"))
    for filename in (
        "stability_experiment.py", "analyze_stability.py",
        "analyze_stability_ensemble.py", "gpu_capacity.py",
    ):
        shutil.copy2(
            ROOT / "scripts" / filename,
            args.output / "source_snapshot" / filename)

    by_key = {
        (
            item["period"], item["gait"], item["speed"],
            item["step_width"], item["duty_factor"], item["perturbation_h"],
        ): item["filename"]
        for item in conditions
    }
    for period in args.periods:
        ticks = condition_period_ticks[period]
        for gait in args.gaits:
            for speed in args.speeds:
                for width in args.step_widths:
                    for duty in args.dfs:
                        if not valid_condition(period, gait, duty):
                            continue
                        settled = settle_condition(
                            env, wrapped, policy, speed, gait, period, ticks,
                            width, duty)
                        payloads = collect_condition(
                            env, wrapped, policy, ticks, settled)
                        for h, payload in payloads.items():
                            name = by_key[
                                (period, gait, speed, width, duty, h)]
                            np.savez_compressed(
                                args.output / name, **payload,
                                command=np.asarray([
                                    speed, duty, width, period,
                                    GAITS.index(gait)]),
                                perturbation_h=h,
                                state_scales=STATE_SCALES,
                                task_sha256=task_hash,
                                evaluation_sha256=eval_hash,
                                checkpoint_sha256=checkpoint_hash)
                            print(
                                "STABILITY_EVALUATED", name, "references",
                                args.references, flush=True)
    from beam_walking.experiment.stability import analyze_stability
    analyze_stability(args.output)
    wrapped.close()


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
