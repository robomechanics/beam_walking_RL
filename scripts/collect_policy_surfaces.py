"""Explore walking DF generalization of the exact frozen seed-2 paper policy.

This separate evaluator cannot release paper claims or compute chi. Training
sources and the confirmatory walking-DF restriction are unchanged.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

# --task selects the frozen seed-2 paper definitions (default) or the unified
# seed-5 definitions; everything downstream reads from the chosen module.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--task", choices=("paper", "unified", "specialist", "narrow_specialist"), default="paper")
_pre.add_argument("--checkpoint", type=Path)
_pre_args = _pre.parse_known_args()[0]
TASK_VARIANT = _pre_args.task
import importlib  # noqa: E402
_data = importlib.import_module({
    "paper": "policy_surface_data", "unified": "unified_surface_data",
    "specialist": "specialist_surface_data",
    "narrow_specialist": "narrow_specialist_surface_data"}[TASK_VARIANT])
if _pre_args.checkpoint is not None:
    # Specialists have one checkpoint per gait, bound here from the command line.
    if TASK_VARIANT == "paper":
        raise SystemExit("--checkpoint is not accepted for the frozen paper task")
    _data.CHECKPOINT = _pre_args.checkpoint.resolve()
    _data.CHECKPOINT_SHA256 = _data.sha256(_data.CHECKPOINT)
elif TASK_VARIANT in ("specialist", "narrow_specialist"):
    raise SystemExit(f"--task {TASK_VARIANT} requires --checkpoint")
CHECKPOINT, CHECKPOINT_SHA256 = _data.CHECKPOINT, _data.CHECKPOINT_SHA256
GRID_SEED, SMOKE_SEED, ROOT = _data.GRID_SEED, _data.SMOKE_SEED, _data.ROOT
SOURCE_FILES, TASK_FILES, TRIALS = _data.SOURCE_FILES, _data.TASK_FILES, _data.TRIALS
conditions, sha256, source_hash = _data.conditions, _data.sha256, _data.source_hash
summarize, validate_measurement = _data.summarize, _data.validate_measurement
TRAINING_SEED = getattr(_data, "TRAINING_SEED", 2)
SCHEMA = getattr(_data, "SCHEMA", "old_policy_surface_exploratory_v1")
COMPLETE_SCHEMA = getattr(_data, "COMPLETE_SCHEMA", "old_policy_surface_complete_v1")
MANIFEST_EXTRA = getattr(_data, "MANIFEST_EXTRA", dict(
    scientific_role="exploratory_generalization_single_frozen_policy",
    walk_training_duties=[.75], walk_unseen_duties=[.50, .625]))
from beam_walking.experiment.protocol import CONTROL_DT, GAITS, leg_phase
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--task", choices=("paper", "unified", "specialist", "narrow_specialist"), default="paper")
parser.add_argument("--checkpoint", type=Path,
                    help="Specialist task: the gait's checkpoint (model_1799.pt)")
parser.add_argument("--gaits", choices=GAITS, nargs="+",
                    help="Restrict conditions to these gaits (specialists: the policy's gait)")
parser.add_argument("--period-sweep", action="store_true",
                    help="Unified task only: collect every gait x period x speed x "
                         "width x feasible duty condition for the selector")
parser.add_argument("--selector-checkpoint", type=Path,
                    help="Unified task only: roll out the selector's chosen DF per "
                         "supported context on fresh seeds instead of the fixed grid")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.checkpoint = CHECKPOINT
args.num_envs = TRIALS
args.seed = SMOKE_SEED if args.smoke else GRID_SEED
CONDITIONS = conditions(args.smoke)
SELECTOR_PAYLOAD = SELECTOR_SHA256 = None
if args.period_sweep:
    if TASK_VARIANT == "paper" or args.smoke or args.selector_checkpoint:
        parser.error("--period-sweep requires --task unified/specialist alone")
    CONDITIONS = _data.period_sweep_conditions()
    args.seed = _data.SWEEP_SEED
if args.selector_checkpoint is not None:
    if TASK_VARIANT == "paper" or args.smoke:
        parser.error("--selector-checkpoint requires --task unified/specialist and no --smoke")
    from beam_walking.experiment.unified_duty_selector import (
        VALIDATION_COMPLETE_SCHEMA, VALIDATION_SCHEMA, load_selector,
        predict_supported_rows)
    _model, SELECTOR_PAYLOAD = load_selector(args.selector_checkpoint)
    if SELECTOR_PAYLOAD.get("exploratory") is not False:
        parser.error("Only a primary selector can enter rollout validation")
    _grid_ckpt = SELECTOR_PAYLOAD["grid_checkpoint_sha256"]
    if isinstance(_grid_ckpt, dict):
        # One grid per gait (specialists): validate only the gaits this
        # checkpoint was fitted from.
        _fitted_gaits = [g for g, v in _grid_ckpt.items() if v == CHECKPOINT_SHA256]
        if not _fitted_gaits:
            parser.error("Selector was fitted for different low-level checkpoints")
        args.gaits = sorted(set(args.gaits or _fitted_gaits) & set(_fitted_gaits))
        if not args.gaits:
            parser.error("Requested gaits were not fitted from this checkpoint")
    elif _grid_ckpt != CHECKPOINT_SHA256:
        parser.error("Selector was fitted for a different low-level checkpoint")
    SELECTOR_SHA256 = sha256(args.selector_checkpoint)
    _rows = predict_supported_rows(
        _model, SELECTOR_PAYLOAD,
        # Contexts carry their own period; never override it.
        [dict(row) for row in SELECTOR_PAYLOAD["supported_contexts"]],
        require_deployment_ready=False)
    CONDITIONS = [(row["gait"], row["speed"], row["period"], row["step_width"],
                   row["selected_df"]) for row in _rows]
    args.seed = _data.VALIDATION_SEED
    SCHEMA, COMPLETE_SCHEMA = VALIDATION_SCHEMA, VALIDATION_COMPLETE_SCHEMA
if args.gaits:
    if TASK_VARIANT == "paper":
        parser.error("--gaits is not accepted for the frozen paper task")
    CONDITIONS = [c for c in CONDITIONS if c[0] in args.gaits]
    if not CONDITIONS:
        parser.error("No conditions remain for the requested gaits")
if TASK_VARIANT in ("specialist", "narrow_specialist"):
    _gait = _data.specialist_gait(_data.read_training(CHECKPOINT))
    if {c[0] for c in CONDITIONS} != {_gait}:
        parser.error(f"Specialist checkpoint is a {_gait} policy; pass --gaits {_gait}")
args.settle_cycles = 12
args.measurement_cycles = 4
args.stance_start_probability = .10
args.output = args.output.resolve()
if not args.output.is_relative_to(ROOT / "results"):
    parser.error("Output must be inside this repository's results directory")
if args.output.exists():
    parser.error("Output directory must be new")
EXPECTED_CHECKPOINT = CHECKPOINT_SHA256
if EXPECTED_CHECKPOINT is None or sha256(CHECKPOINT) != EXPECTED_CHECKPOINT:
    parser.error("Checkpoint differs from the registered policy for this task")
training_bytes = (CHECKPOINT.parent / "provenance.json").read_bytes()
training = json.loads(training_bytes)
task_hash = hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in TASK_FILES)).hexdigest()
if hasattr(_data, "accepts_training"):
    if training.get("task_sha256") != task_hash or not _data.accepts_training(training):
        parser.error("Policy task/training provenance mismatch")
elif (training.get("task_sha256") != task_hash or training.get("seed") != TRAINING_SEED
        or training.get("fresh_training") is not True):
    parser.error("Policy task/training provenance mismatch")
from evaluation_capacity import check_evaluation_capacity
capacity = check_evaluation_capacity(TRIALS, args.device or "cuda:0", False)
app = AppLauncher(args).app

import torch
from isaaclab.managers import (
    DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg,
)
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from beam_walking.experiment.task import BeamEnv, BeamEnvCfg, BeamPPORunnerCfg, command
if TASK_VARIANT == "unified":
    BeamEnv, BeamEnvCfg, BeamPPORunnerCfg = _data.env_classes()
elif TASK_VARIANT in ("specialist", "narrow_specialist"):
    BeamEnv, BeamEnvCfg, BeamPPORunnerCfg = _data.env_classes(training)


def raw_state(env):
    robot = env.scene["robot"]
    root = robot.data.root_state_w.clone()
    root[:, :3] -= env.scene.env_origins
    return torch.cat([root[:, :7], robot.data.joint_pos, root[:, 7:13],
                      robot.data.joint_vel, env.action_manager.action], dim=1)


class EnergyRecorder(RecorderTerm):
    """Capture clipped torque and left-endpoint velocity at 200 Hz."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ids = None
        self.torque = []
        self.velocity = []

    def arm(self, env_ids):
        if self.ids is not None:
            raise RuntimeError("Energy recorder is already armed")
        self.ids = env_ids.clone()
        self.torque = []
        self.velocity = []

    def record_post_physics_decimation_step(self):
        if self.ids is not None:
            robot = self._env.scene["robot"]
            self.torque.append(robot.data.applied_torque[self.ids].clone())
            self.velocity.append(robot.data.joint_vel[self.ids].clone())
        return None, None

    def take(self, expected_samples):
        if self.ids is None:
            raise RuntimeError("Energy recorder is not armed")
        if len(self.torque) != expected_samples:
            raise RuntimeError(
                f"Expected {expected_samples} physics samples, got {len(self.torque)}")
        torque = torch.stack(self.torque, dim=1).cpu().numpy()
        velocity = torch.stack(self.velocity, dim=1).cpu().numpy()
        self.ids = None
        self.torque = []
        self.velocity = []
        return torque, velocity


@configclass
class EnergyRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = EnergyRecorder


@configclass
class DutyGridRecorderCfg(RecorderManagerBaseCfg):
    energy = EnergyRecorderCfg()
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    export_in_close = False


@torch.inference_mode()
def collect_condition(env, wrapped, policy, condition, reset_plan,
                      stance_starts, robot_mass, gravity):
    gait, speed, period, width, duty = condition
    ticks = round(period / CONTROL_DT)
    gait_command = command(env)
    gait_command.evaluation = {
        "df": duty, "speed": speed, "period": period,
        "step_width": width, "gait": GAITS.index(gait),
        "stance_start": stance_starts,
        "yaw": reset_plan[:, 0], "lateral": reset_plan[:, 1],
    }
    wrapped.seed(args.seed + 1_000_000)
    observation, _ = wrapped.reset()
    initial = (
        env.scene["robot"].data.root_state_w.cpu().numpy().copy(),
        env.scene["robot"].data.joint_pos.cpu().numpy().copy(),
        env.scene["robot"].data.joint_vel.cpu().numpy().copy(),
    )
    ids = torch.arange(args.num_envs, device=env.device)
    measured = {key: [] for key in (
        "applied_torque", "joint_velocity", "x_boundaries",
        "initial_contacts", "initial_desired", "phase_ticks",
        "contacts", "substep_contacts", "force_norm_200hz", "desired", "feet_body",
        "forward_velocity", "lateral_position", "heading",
        "world_lateral_velocity", "body_yaw_rate", "failure", "done",
    )}
    energy_recorder = env.recorder_manager._terms["energy"]
    done_seen = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    failure_seen = torch.zeros(
        args.num_envs, device=env.device, dtype=torch.bool)
    phase_states = []
    for cycle in range(args.settle_cycles):
        measuring = cycle >= args.settle_cycles - args.measurement_cycles
        cycle_trace = {key: [] for key in (
            "phase_ticks", "contacts", "substep_contacts", "force_norm_200hz", "desired",
            "feet_body", "forward_velocity", "lateral_position", "heading",
            "world_lateral_velocity", "body_yaw_rate", "failure", "done",
        )}
        if measuring:
            if not bool((gait_command.phase_ticks[~done_seen] == 0).all()):
                raise RuntimeError("Measurement cycle did not start at phase zero")
            if not phase_states:
                phase_states.append(raw_state(env).cpu().numpy())
            env.capture_substeps = True
            energy_recorder.arm(ids)
            start_x = env.scene["robot"].data.root_pos_w[:, 0].clone()
            measured["initial_contacts"].append(
                gait_command.contact_cache.cpu().numpy())
            previous_phase = leg_phase(
                (gait_command.phase_ticks - 1) % gait_command.period_ticks,
                gait_command.period_ticks, gait_command.values[:, 4].long())
            measured["initial_desired"].append(
                (previous_phase < gait_command.values[:, 1:2]).cpu().numpy())
        for _ in range(ticks):
            if measuring:
                cycle_trace["phase_ticks"].append(
                    gait_command.phase_ticks.cpu().numpy())
            observation, _, done, _ = wrapped.step(policy(observation))
            done_seen |= done.bool()
            if bool(done_seen.any()):
                raise RuntimeError(
                    "Unexpected auto-reset during fixed-duration duty collection")
            failure_seen |= env.transition["failure"]
            if measuring:
                transition = env.transition
                cycle_trace["contacts"].append(
                    transition["contacts"].cpu().numpy())
                if len(env.substep_contacts) != env.cfg.decimation:
                    raise RuntimeError("Missing 200 Hz contact samples")
                cycle_trace["substep_contacts"].append(
                    torch.stack(env.substep_contacts, dim=1).cpu().numpy())
                force = env.scene["contact_forces"].data.net_forces_w_history
                cycle_trace["force_norm_200hz"].append(
                    force[:, :, gait_command.sensor_feet].norm(dim=-1).flip(1).cpu().numpy())
                cycle_trace["desired"].append(
                    transition["desired"].cpu().numpy())
                cycle_trace["feet_body"].append(
                    transition["feet_body"].cpu().numpy())
                cycle_trace["forward_velocity"].append(
                    transition["forward_velocity"].cpu().numpy())
                cycle_trace["lateral_position"].append(
                    transition["body"][:, 1].cpu().numpy())
                quaternion = transition["root_quat"]
                heading = torch.atan2(
                    2 * (quaternion[:, 0] * quaternion[:, 3]
                         + quaternion[:, 1] * quaternion[:, 2]),
                    1 - 2 * (quaternion[:, 2].square()
                             + quaternion[:, 3].square()))
                cycle_trace["heading"].append(heading.cpu().numpy())
                cycle_trace["world_lateral_velocity"].append(
                    transition["world_lateral_velocity"].cpu().numpy())
                cycle_trace["body_yaw_rate"].append(
                    transition["body_yaw_rate"].cpu().numpy())
                cycle_trace["failure"].append(
                    transition["failure"].cpu().numpy())
                cycle_trace["done"].append(done.bool().cpu().numpy())
        if measuring:
            torque, velocity = energy_recorder.take(ticks * env.cfg.decimation)
            end_x = env.scene["robot"].data.root_pos_w[:, 0].cpu().numpy()
            measured["applied_torque"].append(torque)
            measured["joint_velocity"].append(velocity)
            measured["x_boundaries"].append(np.stack(
                (start_x.cpu().numpy(), end_x), axis=-1))
            for key, values in cycle_trace.items():
                measured[key].append(np.stack(values, axis=1))
            phase_states.append(raw_state(env).cpu().numpy())
            env.capture_substeps = False
    payload = {
        key: np.stack(values, axis=1) for key, values in measured.items()
    }
    payload.update({
        "phase_zero_states": np.stack(phase_states, axis=1),
        "sample_dt": env.cfg.sim.dt,
        "robot_mass_kg": robot_mass, "gravity_mps2": gravity,
        "command": np.asarray([speed, duty, width, period, GAITS.index(gait)]),
        "seeds": np.arange(args.seed, args.seed + args.num_envs),
        "any_failure": failure_seen.cpu().numpy(),
        "reset_plan": reset_plan.cpu().numpy(),
        "stance_start": stance_starts.cpu().numpy(),
        "initial_root": initial[0], "initial_joints": initial[1],
        "initial_joint_velocity": initial[2],
    })
    return payload, initial


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    for relative in SOURCE_FILES:
        target = args.output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    (args.output / "training_provenance.json").write_bytes(training_bytes)
    (args.output / "capacity.json").write_text(json.dumps(capacity, indent=2))
    cfg = BeamEnvCfg()
    cfg.scene.num_envs = TRIALS
    cfg.seed = args.seed
    cfg.stance_start_probability = .10
    cfg.sim.device = args.device or "cuda:0"
    cfg.events.motor_gain_randomization = None
    # Retain every physical failure via the per-substep latch, without resetting
    # and splicing subsequent episodes into a fixed-duration measurement.
    cfg.terminations.support_failure = None
    cfg.terminations.crossing = None
    cfg.episode_length_s = 100.
    cfg.recorders = DutyGridRecorderCfg()
    env = BeamEnv(cfg)
    wrapped = None
    try:
        env.capture = True
        env.calibrate_stance()
        (args.output / "settled_stance.json").write_text(json.dumps({
            k: v.detach().cpu().tolist() for k, v in env.settled_stance.items()
        }, indent=2))
        wrapped = RslRlVecEnvWrapper(env, clip_actions=5.)
        agent = BeamPPORunnerCfg()
        agent.seed, agent.device = args.seed, cfg.sim.device
        runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=env.device)
        saved = runner.load(str(CHECKPOINT), load_optimizer=False)
        if (not saved or saved.get("task_sha256") != task_hash
                or saved.get("common_step_counter") != 86400
                or runner.current_learning_iteration != 1799):
            raise ValueError("Loaded policy identity/iteration mismatch")
        if TASK_VARIANT in ("unified", "specialist", "narrow_specialist"):
            _data.verify_training(training, saved)
        policy = runner.get_inference_policy(device=env.device)
        robot_mass = float(env.scene["robot"].data.default_mass[0].sum().cpu())
        gravity = abs(float(cfg.sim.gravity[2]))
        resets, grounded = [], []
        for seed in range(args.seed, args.seed + TRIALS):
            rng = np.random.default_rng(seed)
            resets.append([rng.uniform(-.04, .04), rng.uniform(-.01, .01)])
            grounded.append(rng.random() < .10)
        reset_plan = torch.tensor(resets, device=env.device)
        stance_starts = torch.tensor(grounded, device=env.device, dtype=torch.bool)
        frozen_hash = source_hash()
        prefix = "selector" if SELECTOR_PAYLOAD is not None else "surface"
        entries = [dict(gait=g, speed=v, period=p, step_width=w, command_df=d,
                        filename=(f"{prefix}_{g}_v{v:.3f}_w{w:.3f}_d{d:.3f}.npz"
                                  if TASK_VARIANT == "paper" else
                                  f"{prefix}_{g}_v{v:.3f}_p{p:.2f}_w{w:.3f}_d{d:.3f}.npz"))
                   for g, v, p, w, d in CONDITIONS]
        manifest = dict(
            schema=SCHEMA, smoke=args.smoke,
            paper_claims_allowed=False, chi_computed=False,
            selector_checkpoint=(str(args.selector_checkpoint.resolve())
                                 if SELECTOR_PAYLOAD is not None else None),
            selector_checkpoint_sha256=SELECTOR_SHA256,
            period_sweep=bool(args.period_sweep),
            gaits=sorted({c[0] for c in CONDITIONS}),
            **MANIFEST_EXTRA,
            terrain="flat_ground", external_pushes=False, nominal_motor_gains=True,
            conditions=entries, trials_per_condition=TRIALS,
            total_cycles=12, discarded_cycles=8, measurement_cycles=4,
            periodicity_definition="max normalized 46D orbital RMS over four consecutive phase-zero differences",
            periodicity_limit=.02, energy_valid_trial_rate_min=.90,
            automatic_support_failure_termination=False,
            failure_handling="latch failures over all twelve cycles; exclude from valid endpoints",
            seed_start=args.seed, seeds=list(range(args.seed, args.seed + TRIALS)),
            stance_start_probability=.10, stance_start_realized_count=sum(grounded),
            checkpoint=str(CHECKPOINT), checkpoint_sha256=EXPECTED_CHECKPOINT,
            training_provenance_sha256=hashlib.sha256(training_bytes).hexdigest(),
            task_sha256=task_hash, collector_sha256=frozen_hash,
        )
        manifest_path = args.output / "surface_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        all_rows, archive_hashes, matched_initial = [], {}, None
        for condition, entry in zip(CONDITIONS, entries):
            if source_hash() != frozen_hash:
                raise RuntimeError("Reviewed sources changed during evaluation")
            payload, initial = collect_condition(
                env, wrapped, policy, condition, reset_plan, stance_starts,
                robot_mass, gravity)
            if matched_initial is None:
                matched_initial = tuple(x.copy() for x in initial)
            elif any(not np.allclose(x, ref, atol=1e-6, rtol=0)
                     for x, ref in zip(initial, matched_initial)):
                raise RuntimeError("Reset states differ across matched conditions")
            validate_measurement(payload, condition)
            archive = args.output / entry["filename"]
            np.savez_compressed(archive, **payload,
                                checkpoint_sha256=EXPECTED_CHECKPOINT,
                                task_sha256=task_hash, collector_sha256=frozen_hash)
            archive_hashes[entry["filename"]] = sha256(archive)
            rows = summarize(payload, condition, args.seed)
            all_rows.extend(rows)
            write_rows(args.output / "surface_trials.partial.csv", all_rows)
            print("SURFACE_EVALUATED", entry["filename"],
                  "valid_energy_trials", sum(r["energy_trial_valid"] for r in rows),
                  "of", len(rows), flush=True)
        trial_path = args.output / "surface_trials.csv"
        write_rows(trial_path, all_rows)
        complete = dict(schema=COMPLETE_SCHEMA,
                        manifest_sha256=sha256(manifest_path),
                        trials_sha256=sha256(trial_path), archive_sha256=archive_hashes,
                        conditions=len(entries), trials=len(all_rows))
        (args.output / "SURFACE_COMPLETE").write_text(json.dumps(complete, indent=2))
        print("SURFACE_COMPLETE", len(entries), "conditions", flush=True)
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
