"""Run held-out nominal flat-ground evaluation for a frozen gait-command PPO."""
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
    CONTROL_DT, GAITS, GAIT_OFFSETS, MIN_SWING_STEPS, PERIOD_TICKS,
    STEP_WIDTH_FRAME, validate_scientific_gait_duties,
)
from beam_walking.experiment.perturbation import (
    add_perturbation_arguments, perturbation_from_args, summarize_outcomes)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=10000)
parser.add_argument("--split", choices=("validation", "test"), default="validation")
parser.add_argument("--stance_start_probability", type=float, default=.10)
parser.add_argument("--video", action="store_true")
parser.add_argument("--capture_joints", action="store_true",
                    help="Also archive per-step policy observations/actions, joint states, "
                         "applied torques and per-foot contact force vectors (sim2sim debugging)")
parser.add_argument("--camera_env", type=int, default=0)
parser.add_argument("--dfs", type=float, nargs="+")
parser.add_argument("--gait", choices=GAITS, default="trot")
parser.add_argument("--period", type=float, default=.48)
parser.add_argument("--speed", type=float, default=.30)
parser.add_argument("--step_widths", type=float, nargs="+", default=[.10, .20, .30, .40, .50])
parser.add_argument(
    "--deployment_profile", choices=("nominal", "randomized", "fixed_gains"),
    default="nominal",
    help="Plant/noise profile for deployment-policy validation")
parser.add_argument("--kp_scale", type=float)
parser.add_argument("--kd_scale", type=float)
parser.add_argument(
    "--require_deployment_checkpoint", action="store_true",
    help="Reject checkpoints not trained with the frozen deployment DR profile")
parser.add_argument(
    "--task", choices=("paper", "unified", "specialist", "narrow_specialist"), default="paper",
    help="Task variant the checkpoint was trained on: the paper sampler "
         "(walk DF .75 only) or the unified trot/walk sampler "
         "(walk DF .75-.90, docs/unified_gait_plan.md)")
add_perturbation_arguments(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.dfs is None:
    if args.gait == "walk":
        args.dfs = [.75, .80, .85, .90] if args.task != "paper" else [.75]
    else:
        args.dfs = [.50, .625, .75]
if args.task != "paper" and args.deployment_profile != "nominal":
    parser.error("The unified/specialist tasks support only the nominal plant profile")
if args.num_envs < 1:
    parser.error("--num_envs must be positive")
if args.deployment_profile == "fixed_gains":
    if args.kp_scale is None or args.kd_scale is None:
        parser.error("fixed_gains requires both --kp_scale and --kd_scale")
elif args.kp_scale is not None or args.kd_scale is not None:
    parser.error("Gain scales are valid only with --deployment_profile fixed_gains")
if not 0 <= args.stance_start_probability <= 1:
    parser.error("Stance-start probability must be in [0,1]")
try:
    # Opt-in ramped base wrench. Perturbed archives are labelled and use a
    # distinct schema so they are never mistaken for nominal protocol data.
    perturbation_cfg = perturbation_from_args(args)
except ValueError as error:
    parser.error(str(error))
ticks = round(args.period / CONTROL_DT)
if ticks not in PERIOD_TICKS or not np.isclose(ticks * CONTROL_DT, args.period):
    parser.error("Period must be 0.36-0.54 s in 0.02 s increments")
# Narrow-beam extension: a checkpoint trained with --min_step_width (and the
# matching placement tolerance / lateral heading term) records those settings
# in its provenance. Apply them so evaluation observations match training and
# the width grid may extend below 0.10 m.
import json as _json
_narrow = {}
try:
    _narrow = _json.loads((args.checkpoint.resolve().parent / "provenance.json").read_text()
                          ).get("narrow_beam_extension") or {}
except (OSError, ValueError):
    _narrow = {}
_min_width = _narrow.get("min_step_width") or .10
if any(not _min_width - 1e-9 <= width <= .50 for width in args.step_widths):
    parser.error(f"Step width must be in [{_min_width:.2f},0.50] m")
if not .25 <= args.speed <= .40:
    parser.error("Speed must be in [0.25,0.40] m/s")
if args.task != "paper" and args.gait == "walk":
    # Unified walk support: DF .75-.90 with two swing ticks, period >= 20 ticks.
    from beam_walking.experiment.unified_gait import (
        WALK_DUTY_RANGE, WALK_MIN_SWING_STEPS, WALK_PERIOD_TICKS)
    if ticks < WALK_PERIOD_TICKS[0]:
        parser.error("Unified walk was trained at periods of 0.40 s and longer")
    if any(df < WALK_DUTY_RANGE[0] - 1e-7
           or df > min(WALK_DUTY_RANGE[1], 1 - WALK_MIN_SWING_STEPS / ticks) + 1e-7
           for df in args.dfs):
        parser.error("Unified walk DF must be in [0.75, 0.90] with two swing ticks")
elif any(df < .50 or df > min(.75, 1 - MIN_SWING_STEPS / ticks) + 1e-7 for df in args.dfs):
    parser.error("DF/period must leave at least 0.10 s of requested swing")
if len(set(args.step_widths)) != len(args.step_widths) or len(set(args.dfs)) != len(args.dfs):
    parser.error("Widths and duty factors must be unique")
if args.task == "paper":
    try:
        validate_scientific_gait_duties(args.gait, args.dfs)
    except ValueError as error:
        parser.error(str(error))
low, high = (10000, 1000000) if args.split == "validation" else (1000000, 2000000)
if not low <= args.seed or args.seed + args.num_envs > high:
    parser.error("Evaluation seeds must stay in the selected held-out split")
if not args.checkpoint.is_file():
    parser.error("Checkpoint does not exist")
if args.output.exists() and any(args.output.iterdir()):
    parser.error("Output directory must be new or empty")

from evaluation_capacity import check_evaluation_capacity
capacity = check_evaluation_capacity(args.num_envs, args.device or "cuda:0", args.video)
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import torch
from isaaclab.managers import (
    DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg,
)
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from beam_walking.experiment.analysis import (
    NOMINAL_SCHEMA, include_post_step_video_frame, nominal_archive_payload,
    nominal_evaluation_source_hash, oldest_first_force_history,
)
from beam_walking.experiment.task import BeamEnv, BeamEnvCfg, BeamPPORunnerCfg, command
from beam_walking.experiment.deployment import (
    DEPLOYMENT_SOURCE_PATHS,
    STANCE_START_PROBABILITY, TRAINING_ITERATIONS, TRAINING_NUM_ENVS,
    deployment_profile, deployment_profile_sha256,
    deployment_training_source_hash, validate_gain_scales,
    deployment_finetune_lineage_valid,
)
from beam_walking.experiment.deployment_task import (
    DeploymentBeamEnv, DeploymentBeamEnvCfg, motor_gain_event,
)
from beam_walking.experiment.stability import TRAINING_SOURCE_PATHS, training_source_hash

# Science sources must match the live tree byte for byte; launch-only scripts
# may drift (e.g. new flags) without invalidating a finished checkpoint.
SCIENCE_SOURCE_PATHS = (
    "source/beam_walking/beam_walking/experiment/task.py",
    "source/beam_walking/beam_walking/experiment/protocol.py",
    *DEPLOYMENT_SOURCE_PATHS,
)


def snapshot_source_path(snapshot, relative):
    name = Path(relative).name
    return snapshot / "experiment" / name if relative.startswith("source/") else snapshot / name


def training_source_check(training, snapshot):
    """Verify the recorded training-source hash against the live tree or, if only
    launch scripts changed since, against the checkpoint's own source snapshot."""
    recorded = training.get("training_source_sha256")
    live = deployment_training_source_hash(ROOT, training_source_hash(ROOT))
    if recorded == live:
        return {"verified": True, "against": "live_tree", "drifted_launch_scripts": []}
    try:
        snapshot_hash = deployment_training_source_hash(
            ROOT, hashlib.sha256(b"".join(
                snapshot_source_path(snapshot, rel).read_bytes()
                for rel in TRAINING_SOURCE_PATHS)).hexdigest())
        # deployment_training_source_hash reads DEPLOYMENT_SOURCE_PATHS from
        # ROOT; those must be identical in the snapshot for the hash to apply.
        science_match = all(
            snapshot_source_path(snapshot, rel).read_bytes() == (ROOT / rel).read_bytes()
            for rel in SCIENCE_SOURCE_PATHS)
        drifted = [rel for rel in TRAINING_SOURCE_PATHS
                   if snapshot_source_path(snapshot, rel).read_bytes() != (ROOT / rel).read_bytes()]
    except OSError:
        return {"verified": False, "against": "missing_snapshot", "drifted_launch_scripts": None}
    verified = bool(snapshot_hash == recorded and science_match
                    and all(rel not in SCIENCE_SOURCE_PATHS for rel in drifted))
    return {"verified": verified, "against": "checkpoint_source_snapshot",
            "snapshot_training_source_sha256": snapshot_hash,
            "live_training_source_sha256": live,
            "drifted_launch_scripts": drifted}


FORCE_THRESHOLDS_N = (2., 5., 10.)
FORCE_SAMPLE_DT = .005
FORCE_SAMPLES_PER_CONTROL = 4


class EvaluationForceRecorder(RecorderTerm):
    """Preserve terminal contact history before Isaac resets an environment."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        gait = command(env)
        self.sensor_feet = gait.sensor_feet
        self.terminal = torch.zeros(
            env.num_envs, FORCE_SAMPLES_PER_CONTROL, 4,
            dtype=torch.float32, device=env.device)
        self.captured = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def force_history(self):
        history = self._env.scene["contact_forces"].data.net_forces_w_history
        forces = history[:, :, self.sensor_feet].norm(dim=-1)
        if forces.shape[1:] != (FORCE_SAMPLES_PER_CONTROL, 4):
            raise RuntimeError(
                f"Unexpected contact-force history shape {tuple(forces.shape)}")
        # Isaac stores newest first; archives use chronological oldest-to-newest.
        return oldest_first_force_history(forces).to(torch.float32)

    def record_pre_step(self):
        self.captured.zero_()
        return None, None

    def record_pre_reset(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self._env.device, dtype=torch.long)
        self.terminal[ids] = self.force_history()[ids].clone()
        self.captured[ids] = True
        return None, None

    def post_step_history(self, done):
        result = self.force_history().clone()
        if bool(done.any()):
            if not bool(self.captured[done].all()):
                raise RuntimeError("Terminal contact forces were not captured pre-reset")
            result[done] = self.terminal[done]
        return result


@configclass
class EvaluationForceRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = EvaluationForceRecorder


@configclass
class EvaluationRecorderManagerCfg(RecorderManagerBaseCfg):
    force_history = EvaluationForceRecorderCfg()
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    export_in_close = False


def source_hash(paths):
    return hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()


@torch.inference_mode()
def evaluate(env, wrapped, policy, provenance):
    env.capture = True
    gait_command = command(env)
    n = env.num_envs
    reset_plan, stance_starts = [], []
    for index in range(n):
        rng = np.random.default_rng(args.seed + index)
        reset_plan.append([rng.uniform(-.04, .04), rng.uniform(-.01, .01)])
        stance_starts.append(rng.random() < args.stance_start_probability)
    reset_plan = np.asarray(reset_plan)
    disturbed = perturbation_cfg is not None
    suffix = "perturbed" if disturbed else "nominal"
    schema = f"{NOMINAL_SCHEMA}_perturbed" if disturbed else NOMINAL_SCHEMA
    perturbation_profile = perturbation_cfg.profile() if disturbed else None
    expected = []
    for width in args.step_widths:
        for duty in args.dfs:
            stem = (
                f"trial_s{width:.3f}_d{duty:.3f}_v{args.speed:.3f}_"
                f"{args.gait}_p{args.period:.2f}_{suffix}")
            expected.append({
                "filename": f"{stem}.npz", "step_width": width,
                "df": duty, "disturbed": disturbed,
            })
    manifest = {
        "schema": schema,
        "terrain": "flat_ground", "external_pushes": disturbed,
        "perturbation": perturbation_profile,
        "split": args.split,
        "condition_reset_seed": provenance["condition_reset_seed"],
        "training_provenance_sha256":
            provenance["training_provenance_sha256"],
        "step_width_frame": STEP_WIDTH_FRAME,
        "expected_conditions": expected,
        "seeds": list(range(args.seed, args.seed + n)),
        "period": args.period, "speed": args.speed, "gait": args.gait,
        "source_sha256": provenance["task_sha256"],
        "evaluator_sha256": provenance["evaluator_sha256"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "force_instrumented": True,
        "force_field": "foot_force_norm_200hz", "force_units": "N",
        "force_sample_dt": FORCE_SAMPLE_DT,
        "force_samples_per_control": FORCE_SAMPLES_PER_CONTROL,
        "force_history_order": "oldest_to_newest",
        "force_leg_order": ["FL", "FR", "RL", "RR"],
        "contact_thresholds_n": list(FORCE_THRESHOLDS_N),
        "policy_contact_threshold_n": 5.,
        "terminal_force_capture": "recorder_pre_reset",
        "scientific_role": "command_fidelity_prerequisite",
        "not_closed_loop_chi_evidence": True,
        "topology_definition": "schedule_centered_unique_circular_events_v1",
        "topology_event_tolerance_s": .02,
        "topology_primary_threshold_n": 5.,
        "topology_gate_aggregation": "mean_trial_cycle_fraction",
        "topology_sensitivity_thresholds_are_diagnostic": True,
    }
    if provenance.get("deployment_evaluation_profile") is not None:
        manifest.update({
            key: provenance[key] for key in (
                "deployment_evaluation_profile", "deployment_training_required",
                "deployment_training_verified", "deployment_profile",
                "deployment_profile_sha256", "deployment_training_source_sha256",
                "kp_scale", "kd_scale",
            )
        })
    (args.output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2))

    matched_initial = None
    outcome_summaries = {}
    for width in args.step_widths:
        for duty in args.dfs:
            if disturbed:
                # Common disturbance sequence across every width/DF condition.
                env.perturbation.reseed(args.seed)
            gait_command.evaluation = {
                "df": duty, "speed": args.speed,
                "gait": GAITS.index(args.gait), "period": args.period,
                "step_width": width,
                "stance_start": torch.tensor(
                    stance_starts, device=env.device, dtype=torch.bool),
                "yaw": torch.tensor(
                    reset_plan[:, 0], device=env.device, dtype=torch.float32),
                "lateral": torch.tensor(
                    reset_plan[:, 1], device=env.device, dtype=torch.float32),
            }
            wrapped.seed(provenance["condition_reset_seed"])
            obs, _ = wrapped.reset()
            initial_root = env.scene["robot"].data.root_state_w.cpu().numpy().copy()
            initial_joints = env.scene["robot"].data.joint_pos.cpu().numpy().copy()
            initial_pair = (initial_root, initial_joints)
            if matched_initial is None:
                matched_initial = tuple(value.copy() for value in initial_pair)
            elif any(not np.allclose(current, reference, atol=1e-6)
                     for current, reference in zip(initial_pair, matched_initial)):
                raise RuntimeError(
                    "Condition reset did not reproduce matched root/joint states")
            stem = (
                f"trial_s{width:.3f}_d{duty:.3f}_v{args.speed:.3f}_"
                f"{args.gait}_p{args.period:.2f}_{suffix}")
            video_writer = None
            temporary_video = None
            if args.video:
                if not 0 <= args.camera_env < n:
                    raise ValueError("camera_env must index an evaluated environment")
                origin = env.scene.env_origins[args.camera_env].cpu().numpy()
                center = origin + np.array([1.5, 0., .2])
                env.sim.set_camera_view(
                    eye=center + np.array([2.5, -4., 2.4]), target=center)
                import imageio.v2 as imageio
                temporary_video = args.output / (
                    f"{stem}_seed{args.seed + args.camera_env}_recording.mp4")
                video_writer = imageio.get_writer(temporary_video, fps=25)
                video_writer.append_data(env.render())
            active = torch.ones(n, device=env.device, dtype=torch.bool)
            force_recorder = env.recorder_manager._terms["force_history"]
            traces = {}
            try:
                for step in range(env.max_episode_length):
                    action = policy(obs)
                    if args.capture_joints:
                        # Observation the policy just acted on and its raw action.
                        obs_tensor = obs["policy"] if not torch.is_tensor(obs) else obs
                        pre = {"policy_obs": obs_tensor.clone(), "policy_action": action.clone()}
                    obs, _, done, _ = wrapped.step(action)
                    done = done.bool()
                    force_history = force_recorder.post_step_history(done)
                    if not torch.equal(force_history[:, -1] > 5., env.transition["contacts"]):
                        raise RuntimeError(
                            "Archived 5 N terminal contact does not match policy contact")
                    record = {
                        **env.transition, "valid": active.clone(),
                        "done": done.clone(),
                        "foot_force_norm_200hz": force_history,
                        "terminal_force_preserved": (active & done).clone(),
                    }
                    if args.capture_joints:
                        robot = env.scene["robot"]
                        gait = command(env)
                        record.update(pre)
                        record["joint_pos"] = robot.data.joint_pos.clone()
                        record["joint_vel"] = robot.data.joint_vel.clone()
                        record["joint_pos_target"] = robot.data.joint_pos_target.clone()
                        record["applied_torque"] = robot.data.applied_torque.clone()
                        record["foot_force_w"] = env.scene["contact_forces"].data.net_forces_w[
                            :, gait.sensor_feet].clone()
                        record["root_lin_vel_b"] = robot.data.root_lin_vel_b.clone()
                        record["root_ang_vel_b"] = robot.data.root_ang_vel_b.clone()
                        record["projected_gravity_b"] = robot.data.projected_gravity_b.clone()
                    for key, value in record.items():
                        traces.setdefault(key, []).append(value.cpu().numpy())
                    if args.video and include_post_step_video_frame(
                            step, active[args.camera_env], done[args.camera_env]):
                        video_writer.append_data(env.render())
                    active &= ~done.bool()
                    if not bool(active.any()):
                        break
            finally:
                if video_writer is not None:
                    video_writer.close()
            archive_metadata = {
                "schema": schema,
                "df": duty, "command_speed": args.speed,
                "disturbed": disturbed, "terrain": "flat_ground",
                "external_pushes": disturbed, "split": args.split,
                "perturbation": np.asarray(json.dumps(perturbation_profile, sort_keys=True)),
                "condition_reset_seed": provenance["condition_reset_seed"],
                "training_provenance_sha256":
                    provenance["training_provenance_sha256"],
                "control_dt": CONTROL_DT,
                "step_width_frame": STEP_WIDTH_FRAME,
                "source_sha256": provenance["task_sha256"],
                "evaluator_sha256": provenance["evaluator_sha256"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "force_instrumented": True,
                "force_field": "foot_force_norm_200hz", "force_units": "N",
                "force_sample_dt": FORCE_SAMPLE_DT,
                "force_samples_per_control": FORCE_SAMPLES_PER_CONTROL,
                "force_history_order": "oldest_to_newest",
                "force_leg_order": np.asarray(["FL", "FR", "RL", "RR"]),
                "contact_thresholds_n": np.asarray(FORCE_THRESHOLDS_N),
                "policy_contact_threshold_n": 5.,
                "terminal_force_capture": "recorder_pre_reset",
                "scientific_role": "command_fidelity_prerequisite",
                "not_closed_loop_chi_evidence": True,
                "topology_definition":
                    "schedule_centered_unique_circular_events_v1",
                "topology_event_tolerance_s": .02,
                "topology_primary_threshold_n": 5.,
                "topology_gate_aggregation":
                    "mean_trial_cycle_fraction",
                "topology_sensitivity_thresholds_are_diagnostic": True,
                "period": args.period, "gait": args.gait,
                "step_width": width,
                "phase_offsets": np.asarray(GAIT_OFFSETS[GAITS.index(args.gait)]),
                "seeds": np.arange(args.seed, args.seed + n),
                "reset_plan": reset_plan,
                "initial_root": initial_root,
                "initial_joints": initial_joints,
            }
            if provenance.get("deployment_evaluation_profile") is not None:
                archive_metadata.update({
                    "deployment_evaluation_profile":
                        provenance["deployment_evaluation_profile"],
                    "deployment_training_required":
                        provenance["deployment_training_required"],
                    "deployment_training_verified":
                        provenance["deployment_training_verified"],
                    "deployment_profile": np.asarray(
                        json.dumps(provenance["deployment_profile"], sort_keys=True)),
                    "deployment_profile_sha256":
                        provenance["deployment_profile_sha256"],
                    "deployment_training_source_sha256":
                        provenance["deployment_training_source_sha256"],
                    "kp_scale": (np.nan if provenance["kp_scale"] is None
                                 else provenance["kp_scale"]),
                    "kd_scale": (np.nan if provenance["kd_scale"] is None
                                 else provenance["kd_scale"]),
                })
            archive = nominal_archive_payload(traces, archive_metadata)
            np.savez_compressed(args.output / f"{stem}.npz", **archive)
            if disturbed:
                outcome_summaries[stem] = {"step_width": width, "df": duty,
                                           **summarize_outcomes(traces)}
            if temporary_video is not None:
                success_trace = archive["success"][:, args.camera_env]
                valid_trace = archive["valid"][:, args.camera_env].astype(bool)
                failed = bool(archive["failure"][:, args.camera_env][valid_trace][-1])
                outcome = (
                    "failure" if failed else
                    ("success" if bool(success_trace[valid_trace][-1]) else "timeout"))
                temporary_video.replace(
                    args.output / f"{stem}_seed{args.seed + args.camera_env}_{outcome}.mp4")
            print("EVALUATED", stem, "trials", n, flush=True)

    expected_names = {item["filename"] for item in expected}
    actual_names = {path.name for path in args.output.glob("trial_*.npz")}
    if actual_names != expected_names:
        raise RuntimeError(
            f"Evaluation archive mismatch: missing={sorted(expected_names-actual_names)}, "
            f"extra={sorted(actual_names-expected_names)}")
    if disturbed:
        (args.output / "perturbation_summary.json").write_text(json.dumps(
            {"perturbation": perturbation_profile, "conditions": outcome_summaries},
            indent=2))
        for stem, summary in outcome_summaries.items():
            print("PERTURBED_OUTCOME", stem, json.dumps(summary), flush=True)
    completion = {
        "complete": True, "schema": schema,
        "external_pushes": disturbed, "perturbation": perturbation_profile,
        "conditions": len(expected_names), "trials_per_condition": n,
        "source_sha256": provenance["task_sha256"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "evaluator_sha256": provenance["evaluator_sha256"],
        "training_provenance_sha256":
            provenance["training_provenance_sha256"],
    }
    if provenance.get("deployment_evaluation_profile") is not None:
        completion.update({
            key: provenance[key] for key in (
                "deployment_evaluation_profile", "deployment_training_verified",
                "deployment_profile_sha256", "deployment_training_source_sha256",
                "kp_scale", "kd_scale",
            )
        })
    (args.output / "evaluation_complete.json").write_text(
        json.dumps(completion, indent=2))


def main():
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir(parents=True)
    for source in (
        ROOT / "scripts/evaluate_policy.py",
        ROOT / "scripts/gpu_capacity.py",
        ROOT / "scripts/evaluation_capacity.py",
        ROOT / "scripts/analyze_beam.py",
        ROOT / "source/beam_walking/beam_walking/experiment/analysis.py",
        ROOT / "source/beam_walking/beam_walking/experiment/task.py",
        ROOT / "source/beam_walking/beam_walking/experiment/protocol.py",
        ROOT / "source/beam_walking/beam_walking/experiment/deployment.py",
        ROOT / "source/beam_walking/beam_walking/experiment/deployment_task.py",
        ROOT / "source/beam_walking/beam_walking/experiment/deployment_actuator.py",
        ROOT / "source/beam_walking/beam_walking/experiment/perturbation.py",
        ROOT / "source/beam_walking/beam_walking/experiment/perturbation_env.py",
    ):
        shutil.copy2(source, snapshot / source.name)
    (args.output / "capacity.json").write_text(json.dumps(capacity, indent=2))

    from beam_walking.experiment import protocol as _protocol, task as _task
    if _narrow.get("min_step_width"):
        _protocol.NARROW_WIDTH_MIN = float(_narrow["min_step_width"])
    if _narrow.get("width_tolerance_sigma_m"):
        _protocol.WIDTH_SCORE_VARIANCE = float(_narrow["width_tolerance_sigma_m"]) ** 2
    _task.LATERAL_HEADING_GAIN = float(_narrow.get("lateral_heading_gain_rad_per_m") or 0.0)
    _task.HEADING_COST_WEIGHT = float(_narrow.get("heading_cost_weight") or 1.0)
    _task.HEADING_COST_ON_OBSERVATION = bool(_narrow.get("heading_cost_on_observation", False))
    _task.CENTERING_SCALE = float(_narrow.get("centering_scale_m") or 0.25)
    _task.CENTERING_WEIGHT = float(_narrow.get("centering_weight") or 1.0)
    if args.task == "unified":
        from beam_walking.experiment.unified_gait_task import (
            UnifiedEnv, UnifiedEnvCfg)
        cfg = UnifiedEnvCfg()
    elif args.task == "specialist":
        from beam_walking.experiment.unified_specialist_task import (
            SpecialistEnv as UnifiedEnv, SpecialistEnvCfg)
        _training = _json.loads(
            (args.checkpoint.resolve().parent / "provenance.json").read_text())
        if _training.get("specialist_gait") != args.gait:
            raise ValueError(
                f"Checkpoint is a {_training.get('specialist_gait')} specialist; "
                f"--gait {args.gait} does not match")
        cfg = SpecialistEnvCfg(specialist_gait=args.gait)
    elif args.task == "narrow_specialist":
        from beam_walking.experiment.narrow_specialist_task import (
            NarrowSpecialistEnv as UnifiedEnv, NarrowSpecialistEnvCfg)
        _training = _json.loads(
            (args.checkpoint.resolve().parent / "provenance.json").read_text())
        if _training.get("specialist_gait") != args.gait:
            raise ValueError(
                f"Checkpoint is a {_training.get('specialist_gait')} specialist; "
                f"--gait {args.gait} does not match")
        cfg = NarrowSpecialistEnvCfg(specialist_gait=args.gait)
    else:
        cfg = (BeamEnvCfg() if args.deployment_profile == "nominal"
               else DeploymentBeamEnvCfg())
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.stance_start_probability = args.stance_start_probability
    cfg.sim.device = args.device or "cuda:0"
    if args.deployment_profile == "fixed_gains":
        kp_scale, kd_scale = validate_gain_scales(args.kp_scale, args.kd_scale)
        cfg.events.motor_gain_randomization = motor_gain_event(
            (kp_scale, kp_scale), (kd_scale, kd_scale))
    elif args.deployment_profile == "nominal":
        cfg.events.motor_gain_randomization = None
    cfg.recorders = EvaluationRecorderManagerCfg()
    env_class = (BeamEnv if args.deployment_profile == "nominal"
                 else DeploymentBeamEnv)
    if args.task in ("unified", "specialist", "narrow_specialist"):
        env_class = UnifiedEnv
    if perturbation_cfg is not None:
        from beam_walking.experiment.perturbation_env import perturbed_env_class
        env_class = perturbed_env_class(env_class, perturbation_cfg)
    env = env_class(cfg, render_mode="rgb_array" if args.video else None)
    wrapped = None
    try:
        if args.stance_start_probability > 0:
            env.calibrate_stance()
            (args.output / "settled_stance.json").write_text(json.dumps({
                key: value.detach().cpu().tolist()
                for key, value in env.settled_stance.items()
            }, indent=2))
        wrapped = RslRlVecEnvWrapper(env, clip_actions=5.)
        agent = BeamPPORunnerCfg()
        agent.seed = args.seed
        agent.device = cfg.sim.device
        runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=env.device)
        saved = runner.load(str(args.checkpoint), load_optimizer=False)
        if args.task == "unified":
            from beam_walking.experiment.unified_gait import UNIFIED_TASK_FILES
            task_hash = source_hash([ROOT / p for p in UNIFIED_TASK_FILES])
        elif args.task == "specialist":
            from beam_walking.experiment.unified_specialist import SPECIALIST_TASK_FILES
            task_hash = source_hash([ROOT / p for p in SPECIALIST_TASK_FILES])
        elif args.task == "narrow_specialist":
            from beam_walking.experiment.narrow_specialist import NARROW_TASK_FILES
            task_hash = source_hash([ROOT / p for p in NARROW_TASK_FILES])
        else:
            task_hash = source_hash([
                ROOT / "source/beam_walking/beam_walking/experiment/task.py",
                ROOT / "source/beam_walking/beam_walking/experiment/protocol.py",
            ])
        if not saved or saved.get("task_sha256") != task_hash:
            raise ValueError("Checkpoint task source does not match this evaluator")
        training_path = args.checkpoint.parent / "provenance.json"
        if not training_path.is_file():
            raise ValueError("Checkpoint training provenance.json is required")
        training_bytes = training_path.read_bytes()
        training = json.loads(training_bytes)
        expected_profile = deployment_profile()
        expected_profile_hash = deployment_profile_sha256()
        source_check = training_source_check(
            training, args.checkpoint.parent / "source_snapshot")
        current_deployment_source = (
            training.get("training_source_sha256") if source_check["verified"]
            else deployment_training_source_hash(ROOT, training_source_hash(ROOT)))
        deployment_training_verified = bool(
            training.get("deployment_domain_randomization") is True
            and training.get("deployment_profile") == expected_profile
            and training.get("deployment_profile_sha256")
                == expected_profile_hash
            and training.get("training_source_sha256")
                == current_deployment_source
            and training.get("training_num_envs") == TRAINING_NUM_ENVS
            and training.get("training_iterations_requested")
                == TRAINING_ITERATIONS
            and training.get("stance_start_probability")
                == STANCE_START_PROBABILITY
            and training.get("watcher_enabled") is True)
        if ((args.require_deployment_checkpoint
             or args.deployment_profile != "nominal")
                and not deployment_training_verified):
            raise ValueError(
                "Deployment evaluation requires a checkpoint trained with the "
                "frozen deployment DR profile")
        training_provenance_hash = hashlib.sha256(training_bytes).hexdigest()
        (args.output / "training_provenance.json").write_bytes(training_bytes)
        specialist_push_verified = False
        if args.task == "specialist":
            from beam_walking.experiment.unified_specialist_push import push_lineage_valid
            specialist_push_verified = push_lineage_valid(training, saved, ROOT)
        elif args.task == "narrow_specialist":
            from beam_walking.experiment.narrow_specialist_push import push_lineage_valid
            specialist_push_verified = push_lineage_valid(training, saved, ROOT)
        if (
            training.get("task_sha256") != task_hash
            or not (training.get("fresh_training") is True
                    or specialist_push_verified
                    or (deployment_training_verified
                        and deployment_finetune_lineage_valid(training, saved)))
            or training.get("checkpoint_selection_rule") != "final_requested_iteration"
            or runner.current_learning_iteration
                != training.get("training_iterations_requested", 0) - 1
        ):
            raise ValueError("Evaluation requires the final checkpoint with verified training lineage")
        if deployment_training_verified and (
                saved.get("deployment_profile_schema")
                    != expected_profile["schema"]
                or saved.get("deployment_profile_sha256")
                    != expected_profile_hash
                or saved.get("training_source_sha256")
                    != training.get("training_source_sha256")):
            raise ValueError(
                "Checkpoint metadata does not bind the frozen deployment profile")
        checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        evaluator_hash = nominal_evaluation_source_hash(ROOT)
        deployment_evaluation = bool(
            args.require_deployment_checkpoint
            or args.deployment_profile != "nominal")
        provenance = {
            "mode": "evaluate", "split": args.split, "seed": args.seed,
            "argv": sys.argv, "terrain": "flat_ground",
            "task_variant": args.task,
            "specialist_push_finetune": specialist_push_verified,
            "external_pushes": False,
            "condition_reset_seed": args.seed + 3000000,
            "training_provenance": str(training_path.resolve()),
            "training_provenance_file": "training_provenance.json",
            "training_provenance_sha256": training_provenance_hash,
            "training_source_sha256": training["training_source_sha256"],
            "task_sha256": task_hash, "evaluator_sha256": evaluator_hash,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_iteration": runner.current_learning_iteration,
            "training_source_check": source_check,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "isaaclab", "isaacsim", "rsl-rl-lib")
            },
        }
        if deployment_evaluation:
            provenance.update({
                "deployment_evaluation_profile": args.deployment_profile,
                "deployment_training_required": True,
                "deployment_training_verified": deployment_training_verified,
                "deployment_profile": expected_profile,
                "deployment_profile_sha256": expected_profile_hash,
                "deployment_training_source_sha256":
                    current_deployment_source,
                "kp_scale": (float(args.kp_scale)
                             if args.deployment_profile == "fixed_gains"
                             else None),
                "kd_scale": (float(args.kd_scale)
                             if args.deployment_profile == "fixed_gains"
                             else None),
            })
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
        evaluate(env, wrapped, runner.get_inference_policy(device=env.device), provenance)
    finally:
        if wrapped is not None:
            wrapped.close()
        else:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # app.close() can end the process before Python reports the failure.
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()
