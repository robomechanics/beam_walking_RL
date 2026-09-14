"""Collect and validate the fixed high-duty walk selector grid."""

import argparse
import csv
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

from beam_walking.experiment.high_duty_walk import (  # noqa: E402
    HIGH_DUTY_WALK_LEVELS,
    HIGH_DUTY_WALK_GRID_SEED,
    HIGH_DUTY_WALK_PERIOD,
    HIGH_DUTY_WALK_TRAINING_ITERATIONS,
    HIGH_DUTY_WALK_TRAINING_NUM_ENVS,
    HIGH_DUTY_WALK_TRAINING_SEED,
    HIGH_DUTY_WALK_VALIDATION_SEED,
    HIGH_DUTY_WALK_TASK_FILES,
    HIGH_DUTY_WALK_TRAINING_SOURCE_FILES,
    high_duty_walk_lineage_id,
    files_sha256,
)
from beam_walking.experiment.protocol import (  # noqa: E402
    CONTROL_DT,
    GAITS,
    leg_phase,
)
from beam_walking.experiment.high_duty_walk_selector import (  # noqa: E402
    load_selector,
    predict_supported_rows,
)
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--selector-checkpoint", type=Path,
                    help="Validate a fitted selector on a fresh seed block")
parser.add_argument("--smoke", action="store_true",
                    help="Run one recorder/reset/energy condition only")
parser.add_argument("--exploratory", action="store_true",
                    help="Allow non-primary collection settings")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--settle_cycles", type=int, default=12)
parser.add_argument("--measurement_cycles", type=int, default=4)
parser.add_argument("--step_widths", type=float, nargs="+",
                    default=[.10, .20, .30, .40, .50])
parser.add_argument("--dfs", type=float, nargs="+",
                    default=list(HIGH_DUTY_WALK_LEVELS))
parser.add_argument("--speeds", type=float, nargs="+",
                    default=[.25, .30, .35, .40])
parser.add_argument("--periods", type=float, nargs="+", default=[.48])
parser.add_argument("--gaits", choices=("walk",), nargs="+", default=["walk"])
parser.add_argument(
    "--task", choices=("high_duty_walk", "unified"), default="high_duty_walk",
    help="Low-level task the checkpoint was trained on; 'unified' evaluates "
         "the walk candidates on the unified trot/walk policy")
parser.add_argument("--seed", type=int)
parser.add_argument("--stance_start_probability", type=float, default=.10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

TRAINING_SCHEMA = "high_duty_walk_training_v1"
if args.task == "unified":
    # Same grid, gates, and selector; the low-level policy, its task hash,
    # lineage rule, and training seed come from the unified protocol.
    from beam_walking.experiment.unified_gait import (  # noqa: E402
        UNIFIED_TASK_FILES, UNIFIED_TRAINING_SEED,
        UNIFIED_TRAINING_SOURCE_FILES, unified_lineage_id)
    HIGH_DUTY_WALK_TASK_FILES = UNIFIED_TASK_FILES
    HIGH_DUTY_WALK_TRAINING_SOURCE_FILES = UNIFIED_TRAINING_SOURCE_FILES
    high_duty_walk_lineage_id = unified_lineage_id
    HIGH_DUTY_WALK_TRAINING_SEED = UNIFIED_TRAINING_SEED
    TRAINING_SCHEMA = "unified_gait_training_v1"

if not args.checkpoint.is_file():
    parser.error("Checkpoint does not exist")
if args.selector_checkpoint is not None and not args.selector_checkpoint.is_file():
    parser.error("Selector checkpoint does not exist")
if args.smoke and args.selector_checkpoint:
    parser.error("Collector smoke and selector validation are separate modes")
if args.selector_checkpoint and args.exploratory:
    parser.error("Exploratory selectors cannot be promoted by validation")
if args.seed is None:
    args.seed = (HIGH_DUTY_WALK_GRID_SEED - 100_000 if args.smoke else
                 (HIGH_DUTY_WALK_VALIDATION_SEED
                  if args.selector_checkpoint else HIGH_DUTY_WALK_GRID_SEED))
if args.output.exists() and any(args.output.iterdir()):
    parser.error("Output directory must be new or empty")
if args.num_envs < 8:
    parser.error("Use at least eight held-out trials per candidate")
if args.measurement_cycles < 2 or args.settle_cycles < args.measurement_cycles:
    parser.error("Settling must include at least two measured cycles")
seed_range = ((HIGH_DUTY_WALK_GRID_SEED - 100_000,
               HIGH_DUTY_WALK_GRID_SEED) if args.smoke else
              ((HIGH_DUTY_WALK_VALIDATION_SEED,
                HIGH_DUTY_WALK_VALIDATION_SEED + 1_000_000)
               if args.selector_checkpoint else
               (HIGH_DUTY_WALK_GRID_SEED,
                HIGH_DUTY_WALK_GRID_SEED + 1_000_000)))
if args.seed < seed_range[0] or args.seed + args.num_envs > seed_range[1]:
    parser.error(f"Evaluation seeds must stay in [{seed_range[0]}, {seed_range[1]})")
if not 0 <= args.stance_start_probability <= 1:
    parser.error("Stance-start probability must be in [0, 1]")
if not args.smoke and not args.exploratory:
    if (args.num_envs != 32 or args.settle_cycles != 12
            or args.measurement_cycles != 4
            or not np.isclose(args.stance_start_probability, .10)):
        parser.error(
            "Primary grid/validation requires 32 trials, 12 settle cycles, "
            "4 measured cycles, and 0.10 grounded-start probability")
    expected_seed = (HIGH_DUTY_WALK_VALIDATION_SEED
                     if args.selector_checkpoint else HIGH_DUTY_WALK_GRID_SEED)
    if args.seed != expected_seed:
        parser.error(f"Primary mode requires seed start {expected_seed}")
    if not args.selector_checkpoint and (
            args.step_widths != [.10, .20, .30, .40, .50]
            or args.dfs != list(HIGH_DUTY_WALK_LEVELS)
            or args.speeds != [.25, .30, .35, .40]
            or args.periods != [HIGH_DUTY_WALK_PERIOD]
            or args.gaits != ["walk"]):
        parser.error("Primary grid factor levels are fixed; use --exploratory")
for values, label in ((args.step_widths, "width"), (args.dfs, "DF"),
                      (args.speeds, "speed"), (args.periods, "period"),
                      (args.gaits, "gait")):
    if len(values) != len(set(values)):
        parser.error(f"Duplicate {label} values are not allowed")
if any(not .10 <= value <= .50 for value in args.step_widths):
    parser.error("Step widths must be in [0.10, 0.50] m")
if any(not .25 <= value <= .40 for value in args.speeds):
    parser.error("Speeds must be in [0.25, 0.40] m/s")
if any(not HIGH_DUTY_WALK_LEVELS[0] <= value <= HIGH_DUTY_WALK_LEVELS[-1]
       for value in args.dfs):
    parser.error("Duty factors must be in [0.75, 0.90]")
period_ticks = {}
for period in args.periods:
    ticks = round(period / CONTROL_DT)
    if (not np.isclose(period, HIGH_DUTY_WALK_PERIOD)
            or not np.isclose(ticks * CONTROL_DT, period)):
        parser.error("High-duty walk uses the fixed 0.48 s period")
    period_ticks[period] = ticks
selector_model = selector_payload = None
if args.selector_checkpoint:
    selector_model, selector_payload = load_selector(args.selector_checkpoint)
    if selector_payload.get("exploratory") is not False:
        parser.error("Only a primary-grid selector can enter rollout validation")
    contexts = selector_payload["supported_contexts"]
    predictions = predict_supported_rows(
        selector_model, selector_payload, contexts,
        require_deployment_ready=False)
    conditions = [
        ("walk", row["speed"], HIGH_DUTY_WALK_PERIOD,
         row["step_width"], row["selected_df"])
        for row in predictions
    ]
else:
    conditions = [
        (gait, speed, period, width, duty)
        for gait in args.gaits
        for speed in args.speeds
        for period in args.periods
        for width in args.step_widths
        for duty in args.dfs
    ]
if args.smoke:
    conditions = [("walk", .30, HIGH_DUTY_WALK_PERIOD, .30, .75)]
if not conditions:
    parser.error("No feasible grid conditions were requested")

from evaluation_capacity import check_evaluation_capacity  # noqa: E402
capacity = check_evaluation_capacity(
    args.num_envs, args.device or "cuda:0", False)
app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.managers import (  # noqa: E402
    DatasetExportMode,
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from beam_walking.experiment.high_duty_walk_task import (  # noqa: E402
    HighDutyWalkEnv,
    HighDutyWalkEnvCfg,
    HighDutyWalkPPORunnerCfg,
)
if args.task == "unified":
    from beam_walking.experiment.unified_gait_task import (  # noqa: E402
        UnifiedEnv as HighDutyWalkEnv,
        UnifiedEnvCfg as HighDutyWalkEnvCfg,
        UnifiedPPORunnerCfg as HighDutyWalkPPORunnerCfg,
    )
from beam_walking.experiment.duty_grid import summarize_condition  # noqa: E402
from beam_walking.experiment.task import command  # noqa: E402

SNAPSHOT_FILES = HIGH_DUTY_WALK_TASK_FILES + (
    "source/beam_walking/beam_walking/experiment/duty_grid.py",
    "source/beam_walking/beam_walking/experiment/duty_selector.py",
    "source/beam_walking/beam_walking/experiment/high_duty_walk_selector.py",
    "source/beam_walking/beam_walking/experiment/analysis.py",
    "source/beam_walking/beam_walking/experiment/stability.py",
    "scripts/collect_high_duty_walk_grid.py",
    "scripts/fit_high_duty_walk_selector.py",
    "scripts/select_high_duty_walk.py",
    "scripts/evaluation_capacity.py",
    "scripts/gpu_capacity.py",
)


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


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completion_record(schema, manifest_path, trial_path, archive_names):
    return {
        "schema": schema,
        "manifest_file": manifest_path.name,
        "manifest_sha256": file_sha256(manifest_path),
        "trial_csv_file": trial_path.name,
        "trial_csv_sha256": file_sha256(trial_path),
        "archive_sha256": {
            name: file_sha256(args.output / name) for name in archive_names},
    }


def snapshot_sources(output):
    for relative in SNAPSHOT_FILES:
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)


def write_validation_graph(rows, output):
    """Plot one achieved-duty curve per speed with a single legend entry."""
    import matplotlib.pyplot as plt

    grouped = {}
    for row in rows:
        key = (row["speed"], row["step_width"], row["command_df"])
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key, trials in sorted(grouped.items()):
        achieved = np.asarray([row["achieved_df"] for row in trials])
        summaries.append({
            "gait": "walk", "speed": key[0],
            "period": HIGH_DUTY_WALK_PERIOD, "step_width": key[1],
            "selected_df": key[2],
            "achieved_df_median": float(np.median(achieved)),
            "achieved_df_q025": float(np.quantile(achieved, .025)),
            "achieved_df_q975": float(np.quantile(achieved, .975)),
            "compliance_rate": float(np.mean([
                row["compliant"] for row in trials])),
            "finite_energy_compliant_trials": int(sum(
                row["compliant"] and np.isfinite(row["positive_mechanical_cot"])
                for row in trials)),
            "trials": len(trials),
        })
    write_rows(output / "selector_validation_summary.csv", summaries)
    colors = ("#0072B2", "#009E73", "#E69F00", "#CC79A7")
    markers = ("o", "s", "^", "D")
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    for index, speed in enumerate(sorted({row["speed"] for row in summaries})):
        group = sorted(
            (row for row in summaries if row["speed"] == speed),
            key=lambda row: row["step_width"])
        width = np.asarray([row["step_width"] for row in group])
        selected = np.asarray([row["selected_df"] for row in group])
        achieved = np.asarray([row["achieved_df_median"] for row in group])
        low = np.asarray([row["achieved_df_q025"] for row in group])
        high = np.asarray([row["achieved_df_q975"] for row in group])
        axis.plot(
            width, selected, linestyle="--", marker=markers[index],
            markerfacecolor="white", color=colors[index], linewidth=1.5,
            markersize=7)
        axis.errorbar(
            width, achieved, yerr=np.vstack((achieved - low, high - achieved)),
            fmt=f"{markers[index]}-", color=colors[index], capsize=3,
            linewidth=2, markersize=6, label=f"{speed:.2f} m/s")
    axis.set_xlabel("Commanded full step width (m)")
    axis.set_ylabel("Achieved duty factor")
    axis.set_ylim(.73, .93)
    axis.set_yticks((*HIGH_DUTY_WALK_LEVELS, 22 / 24))
    axis.grid(alpha=.25)
    axis.legend(title="Commanded speed", frameon=False, ncol=2)
    axis.set_title(
        "High-duty walk validation, period 0.48 s\n"
        "Dashed open markers: selected; solid filled markers: achieved")
    fig.tight_layout()
    fig.savefig(output / "selected_vs_achieved_duty_factor.png", dpi=180)
    fig.savefig(output / "selected_vs_achieved_duty_factor.pdf")
    plt.close(fig)
    return summaries


@torch.inference_mode()
def collect_condition(env, wrapped, policy, condition, reset_plan,
                      stance_starts, robot_mass, gravity):
    gait, speed, period, width, duty = condition
    ticks = period_ticks[period]
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
    )
    ids = torch.arange(args.num_envs, device=env.device)
    measured = {key: [] for key in (
        "applied_torque", "joint_velocity", "x_boundaries",
        "initial_contacts", "initial_desired", "phase_ticks",
        "contacts", "substep_contacts", "desired", "feet_body",
        "forward_velocity", "lateral_position", "heading",
        "world_lateral_velocity", "body_yaw_rate", "failure", "done",
    )}
    energy_recorder = env.recorder_manager._terms["energy"]
    done_seen = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
    failure_seen = torch.zeros(
        args.num_envs, device=env.device, dtype=torch.bool)
    for cycle in range(args.settle_cycles):
        measuring = cycle >= args.settle_cycles - args.measurement_cycles
        cycle_trace = {key: [] for key in (
            "phase_ticks", "contacts", "substep_contacts", "desired",
            "feet_body", "forward_velocity", "lateral_position", "heading",
            "world_lateral_velocity", "body_yaw_rate", "failure", "done",
        )}
        if measuring:
            if not bool((gait_command.phase_ticks[~done_seen] == 0).all()):
                raise RuntimeError("Measurement cycle did not start at phase zero")
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
            env.capture_substeps = False
    payload = {
        key: np.stack(values, axis=1) for key, values in measured.items()
    }
    payload.update({
        "sample_dt": env.cfg.sim.dt,
        "robot_mass_kg": robot_mass, "gravity_mps2": gravity,
        "command": np.asarray([speed, duty, width, period, GAITS.index(gait)]),
        "seeds": np.arange(args.seed, args.seed + args.num_envs),
        "any_failure": failure_seen.cpu().numpy(),
        "reset_plan": reset_plan.cpu().numpy(),
        "stance_start": stance_starts.cpu().numpy(),
        "initial_root": initial[0], "initial_joints": initial[1],
    })
    return payload, initial


def main():
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_sources(args.output)
    (args.output / "capacity.json").write_text(json.dumps(capacity, indent=2))

    cfg = HighDutyWalkEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.stance_start_probability = args.stance_start_probability
    cfg.sim.device = args.device or "cuda:0"
    cfg.events.motor_gain_randomization = None
    # Fixed-duration measurement must never cross an automatic reset. Physical
    # failure remains captured at every step and invalidates that trial.
    cfg.terminations.support_failure = None
    cfg.terminations.crossing = None
    cfg.episode_length_s = 100.
    cfg.recorders = DutyGridRecorderCfg()
    env = HighDutyWalkEnv(cfg)
    wrapped = None
    try:
        env.capture = True
        env.calibrate_stance()
        wrapped = RslRlVecEnvWrapper(env, clip_actions=5.)
        agent = HighDutyWalkPPORunnerCfg()
        agent.seed = args.seed
        agent.device = cfg.sim.device
        runner = OnPolicyRunner(
            wrapped, agent.to_dict(), log_dir=None, device=env.device)
        saved = runner.load(str(args.checkpoint), load_optimizer=False)
        task_sha256 = files_sha256(ROOT, HIGH_DUTY_WALK_TASK_FILES)
        if not saved or saved.get("task_sha256") != task_sha256:
            raise ValueError("Checkpoint does not match the requested low-level task")
        training_path = args.checkpoint.parent / "provenance.json"
        if not training_path.is_file():
            raise ValueError("High-duty walk training provenance is required")
        training_bytes = training_path.read_bytes()
        training = json.loads(training_bytes)
        training_source_sha256 = files_sha256(
            ROOT, HIGH_DUTY_WALK_TRAINING_SOURCE_FILES)
        expected_lineage = high_duty_walk_lineage_id(
            training_source_sha256, training.get("seed", -1))
        expected_control_steps = (
            training.get("training_iterations_requested", 0)
            * agent.num_steps_per_env)
        if (training.get("schema") != TRAINING_SCHEMA
                or training.get("mode") != "train"
                or training.get("fresh_training") is not True
                or training.get("primary_training_protocol") is not True
                or training.get("task_sha256") != task_sha256
                or training.get("training_source_sha256")
                    != training_source_sha256
                or not isinstance(training.get("seed"), int)
                or training.get("seed") != HIGH_DUTY_WALK_TRAINING_SEED
                or training.get("training_lineage_id") != expected_lineage
                or saved.get("training_lineage_id") != expected_lineage
                or saved.get("common_step_counter") != expected_control_steps
                or training.get("training_num_envs")
                    != HIGH_DUTY_WALK_TRAINING_NUM_ENVS
                or training.get("training_iterations_requested")
                    != HIGH_DUTY_WALK_TRAINING_ITERATIONS
                or training.get("checkpoint_selection_rule")
                    != "final_requested_iteration"
                or runner.current_learning_iteration
                    != training.get("training_iterations_requested", 0) - 1):
            raise ValueError(
                "Grid requires the final checkpoint of the fresh high-duty walk run")
        policy = runner.get_inference_policy(device=env.device)
        checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        selector_sha256 = None
        if args.selector_checkpoint:
            selector_sha256 = hashlib.sha256(
                args.selector_checkpoint.read_bytes()).hexdigest()
            if (selector_payload["grid_task_sha256"] != task_sha256
                    or selector_payload["grid_checkpoint_sha256"]
                        != checkpoint_sha256):
                raise ValueError(
                    "Selector was fitted for a different low-level checkpoint")
        robot = env.scene["robot"]
        robot_mass = float(robot.data.default_mass[0].sum().cpu())
        gravity = abs(float(cfg.sim.gravity[2]))

        rng = np.random.default_rng(args.seed)
        reset_plan_np = np.column_stack((
            rng.uniform(-.05, .05, args.num_envs),
            rng.uniform(-.015, .015, args.num_envs),
        )).astype(np.float32)
        stance_np = rng.random(args.num_envs) < args.stance_start_probability
        reset_plan = torch.as_tensor(reset_plan_np, device=env.device)
        stance_starts = torch.as_tensor(
            stance_np, device=env.device, dtype=torch.bool)
        prefix = ("high_duty_walk_selector" if args.selector_checkpoint
                  else "high_duty_walk_grid")
        manifest = {
            "schema": (
                "high_duty_walk_grid_smoke_v1" if args.smoke else
                ("high_duty_walk_selector_validation_v1"
                 if args.selector_checkpoint else
                 ("high_duty_walk_grid_exploratory_v1" if args.exploratory
                  else "high_duty_walk_grid_v1"))),
            "scientific_scope": "flat_ground_commanded_step_width",
            "task_variant": args.task,
            "terrain": "flat_ground", "terrain_width_input": False,
            "objective": "minimum positive mechanical CoT among candidates passing compliance gates",
            "gait": "walk",
            "candidate_duty_factors": list(HIGH_DUTY_WALK_LEVELS),
            "fixed_period_s": HIGH_DUTY_WALK_PERIOD,
            "command_gate_limits_source": "experiment.stability.FROZEN_GATE_LIMITS",
            "topology_definition": "schedule_centered_unique_circular_events_v1",
            "topology_minimum": .90,
            "control_dt_s": CONTROL_DT, "physics_dt_s": cfg.sim.dt,
            "trials_per_candidate": args.num_envs,
            "settle_cycles": args.settle_cycles,
            "measurement_cycles": args.measurement_cycles,
            "stance_start_probability": args.stance_start_probability,
            "stance_start_sampling": "seeded_independent_bernoulli",
            "stance_start_realized_count": int(stance_np.sum()),
            "stance_start_realized": stance_np.astype(int).tolist(),
            "automatic_support_failure_termination": False,
            "automatic_crossing_termination": False,
            "conditions": [{
                "gait": gait, "speed": speed, "period": period,
                "step_width": width, "command_df": duty,
                "filename": (f"{prefix}_{gait}_v{speed:.3f}_p{period:.2f}_"
                             f"w{width:.3f}_d{duty:.3f}.npz"),
            } for gait, speed, period, width, duty in conditions],
            "evaluation_seed_start": args.seed,
            "evaluation_seed_range": list(seed_range),
            "condition_reset_seed": args.seed + 1_000_000,
            "task_sha256": task_sha256,
            "collector_sha256": files_sha256(ROOT, SNAPSHOT_FILES),
            "training_source_sha256": training_source_sha256,
            "training_lineage_id": expected_lineage,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "selector_checkpoint": (
                str(args.selector_checkpoint.resolve())
                if args.selector_checkpoint else None),
            "selector_checkpoint_sha256": selector_sha256,
            "training_provenance_sha256": hashlib.sha256(training_bytes).hexdigest(),
            "capacity": capacity,
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "isaaclab", "isaacsim", "rsl-rl-lib")},
        }
        manifest_name = (
            "duty_grid_smoke_manifest.json" if args.smoke else
            ("selector_validation_manifest.json"
             if args.selector_checkpoint else "grid_manifest.json"))
        (args.output / manifest_name).write_text(
            json.dumps(manifest, indent=2))
        (args.output / "training_provenance.json").write_bytes(training_bytes)

        all_rows = []
        matched_initial = None
        for condition in conditions:
            payload, initial = collect_condition(
                env, wrapped, policy, condition, reset_plan, stance_starts,
                robot_mass, gravity)
            if matched_initial is None:
                matched_initial = tuple(value.copy() for value in initial)
            elif any(not np.allclose(current, reference, rtol=0, atol=1e-6)
                     for current, reference in zip(initial, matched_initial)):
                raise RuntimeError("Condition reset states are not reproducible")
            gait, speed, period, width, duty = condition
            filename = (f"{prefix}_{gait}_v{speed:.3f}_p{period:.2f}_"
                        f"w{width:.3f}_d{duty:.3f}.npz")
            np.savez_compressed(
                args.output / filename, **payload,
                task_sha256=task_sha256,
                checkpoint_sha256=checkpoint_sha256,
                collector_sha256=manifest["collector_sha256"])
            rows = summarize_condition(
                payload, step_width=width, speed=speed, period=period,
                gait=gait, command_df=duty, seed_start=args.seed)
            all_rows.extend(rows)
            table_stem = (
                "duty_grid_smoke_trials" if args.smoke else
                ("selector_validation_trials" if args.selector_checkpoint
                 else "duty_grid_trials"))
            write_rows(args.output / f"{table_stem}.partial.csv", all_rows)
            print("DUTY_GRID_EVALUATED", filename, "trials", len(rows), flush=True)
        if args.smoke:
            trial_path = args.output / "duty_grid_smoke_trials.csv"
            write_rows(trial_path, all_rows)
            record = completion_record(
                "high_duty_walk_grid_smoke_complete_v1",
                args.output / manifest_name, trial_path,
                [item["filename"] for item in manifest["conditions"]])
            (args.output / "DUTY_GRID_SMOKE_COMPLETE").write_text(
                json.dumps(record, indent=2))
            print("DUTY_GRID_SMOKE_COMPLETE", len(all_rows), "trials", flush=True)
        elif args.selector_checkpoint:
            trial_path = args.output / "selector_validation_trials.csv"
            write_rows(trial_path, all_rows)
            summaries = write_validation_graph(all_rows, args.output)
            deployment_ready = bool(
                len(summaries) == len(selector_payload["supported_contexts"])
                and all(row["trials"] == args.num_envs
                        and row["compliance_rate"] >= .90
                        and row["finite_energy_compliant_trials"]
                            / row["trials"] >= .90
                        and abs(row["achieved_df_median"]
                                - row["selected_df"]) <= .05
                        for row in summaries))
            validation_record = completion_record(
                "high_duty_walk_selector_validation_complete_v1",
                args.output / manifest_name, trial_path,
                [item["filename"] for item in manifest["conditions"]])
            completion_path = args.output / "SELECTOR_VALIDATION_COMPLETE"
            completion_path.write_text(json.dumps(validation_record, indent=2))
            validated = dict(selector_payload)
            validated.update({
                "schema": "high_duty_walk_selector_v1",
                "deployment_ready": deployment_ready,
                "requires_fresh_rollout_validation": not deployment_ready,
                "validation_manifest_sha256": hashlib.sha256(
                    (args.output / manifest_name).read_bytes()).hexdigest(),
                "validation_trials_sha256": hashlib.sha256(
                    trial_path.read_bytes()).hexdigest(),
                "validation_selector_checkpoint_sha256": selector_sha256,
                "validation_completion_sha256": file_sha256(completion_path),
                "validation_low_level_checkpoint_sha256": checkpoint_sha256,
                "validation_seed_start": args.seed,
                "validation_trials_per_context": args.num_envs,
                "validation_all_contexts_passed": deployment_ready,
            })
            torch.save(
                validated, args.output / "validated_high_duty_walk_selector.pt")
            print("SELECTOR_VALIDATION_COMPLETE", json.dumps({
                "contexts": len(summaries),
                "deployment_ready": deployment_ready,
            }), flush=True)
        else:
            trial_path = args.output / "duty_grid_trials.csv"
            write_rows(trial_path, all_rows)
            record = completion_record(
                ("high_duty_walk_grid_exploratory_complete_v1"
                 if args.exploratory else "high_duty_walk_grid_complete_v1"),
                args.output / manifest_name, trial_path,
                [item["filename"] for item in manifest["conditions"]])
            (args.output / "GRID_COMPLETE").write_text(
                json.dumps(record, indent=2))
            print("DUTY_GRID_COMPLETE", len(conditions), "conditions", flush=True)
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
