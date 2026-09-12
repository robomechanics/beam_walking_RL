"""Proprioceptive gait-command Go2 policy on open flat ground; no map observations."""
import math

import torch
import isaaclab.sim as sim
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm, TerminationTermCfg as DoneTerm
from isaaclab.sim.utils import clone, create_prim
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from .protocol import (PERIOD, CYCLE_STEPS, PERIOD_TICKS,
                       STEP_WIDTH_RANGE, MIN_SWING_STEPS, leg_phase,
                       robust_contact_score,
                       discrete_stance_fraction, width_score, clearance_score,
                       duty_warped_phase, straight_motion_cost,
                       heading_stabilization_cost, planar_speed_score, world_to_body,
                       normalized_gait_command, sample_training_commands,
                       fore_aft_target, fore_aft_score, advance_phase_ticks)

LEG_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
TOP = 0.0
LENGTH = 3.0
# Opt-in (beam_experiment.py --lateral_heading_gain): add gain * lateral offset
# from the course line (env-frame y, +left) to the heading observation. The
# policy already steers heading to zero, so this makes it steer back onto the
# line without a new input. quad-sdk's beamwalking controller applies the same
# term (beamwalking.lateral_heading_gain) at deployment.
LATERAL_HEADING_GAIN = 0.0
# Opt-in (beam_experiment.py --heading_cost_weight): multiplier on the bounded
# heading cost. At 1.0 a 4 degree yaw costs ~0.2 per step against ~20 of
# reward, which lets a policy settle into a yawed crab-walk; 5.0 makes it ~1.
HEADING_COST_WEIGHT = 1.0
# Opt-in: apply the heading cost to the observed heading (yaw + gain * lateral
# offset) instead of raw yaw, so yawing toward the course line is free and
# only pointing away from it is penalized. Requires LATERAL_HEADING_GAIN.
HEADING_COST_ON_OBSERVATION = False
# Opt-in centering cost shape: cost = weight * .5 * (1 - exp(-(y/scale)^2)).
# Frozen protocol: scale .25 m, weight 1 (a 5 cm offset costs ~0.02/step).
CENTERING_SCALE = .25
CENTERING_WEIGHT = 1.0


@clone
def spawn_reference_line(prim_path, cfg, translation=None, orientation=None, **kwargs):
    root = create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    # Visual-only reference on the ground: no collision and no policy sensor.
    line = sim.CuboidCfg(size=(LENGTH + 2.4, .008, .002),
        visual_material=sim.PreviewSurfaceCfg(diffuse_color=(1., .85, .05)))
    line.func(f"{prim_path}/straight_reference", line,
              translation=(LENGTH / 2, 0., .002))
    return root


def command(env):
    return env.command_manager.get_term("gait")


class GaitCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.values = torch.zeros(self.num_envs, 5, device=self.device)
        # Commands: forward speed, duty factor, full step width, period, gait ID.
        self.values[:] = torch.tensor([0.3, 0.6, 0.26, PERIOD, 0.], device=self.device)
        self.phase_ticks = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.period_ticks = torch.full_like(self.phase_ticks, CYCLE_STEPS)
        self.contact_cache = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.nonfoot_cache = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stance_start = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.schedule_duty = discrete_stance_fraction(self.values[:, 1], self.period_ticks, self.values[:, 4].long())
        self.signs = torch.tensor([1., -1., 1., -1.], device=self.device)
        self.feet = env.scene["robot"].find_bodies(LEG_NAMES, preserve_order=True)[0]
        self.sensor_feet = env.scene["contact_forces"].find_bodies(LEG_NAMES, preserve_order=True)[0]
        self.sensor_other = [i for i in range(len(env.scene["contact_forces"].body_names)) if i not in self.sensor_feet]
        self.next_change = torch.zeros(self.num_envs, device=self.device)
        self.curriculum_stage = 0
        self.evaluation = None

    @property
    def command(self):
        return self.values

    @property
    def time(self):
        return self._env.episode_length_buf * self._env.step_dt

    @property
    def phase(self):
        return leg_phase(self.phase_ticks, self.period_ticks, self.values[:, 4].long())

    @property
    def desired(self):
        return self.phase < self.values[:, 1:2]

    def _resample_command(self, env_ids):
        # Internal timer disabled: commands change only at master-cycle boundaries.
        pass

    def _update_metrics(self):
        pass

    def _update_command(self):
        if self.evaluation is None:
            at_boundary = self.phase_ticks == 0
            ids = (at_boundary & (self.time + 1e-5 >= self.next_change)).nonzero().flatten()
            if len(ids):
                self.sample_values(ids)
                self.next_change[ids] = self.time[ids] + torch.randint(8, 13, (len(ids),), device=self.device) * self.values[ids, 3]

    def sample_values(self, ids):
        n = len(ids)
        sampled, ticks, stage = sample_training_commands(
            n, int(getattr(self._env, "common_step_counter", 0)), self.device)
        self.values[ids] = sampled
        self.period_ticks[ids] = ticks
        self.curriculum_stage = stage
        self.schedule_duty[ids] = discrete_stance_fraction(self.values[ids, 1], self.period_ticks[ids], self.values[ids, 4].long())

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.phase_ticks[env_ids] = 0
        self.contact_cache[env_ids] = False
        self.nonfoot_cache[env_ids] = False
        if hasattr(self._env, "substep_failure"):
            self._env.substep_failure[env_ids] = False
        self.next_change[env_ids] = torch.randint(8, 13, (len(env_ids),), device=self.device) * self.values[env_ids, 3]
        return super().reset(env_ids)


@configclass
class GaitCommandCfg(CommandTermCfg):
    class_type: type = GaitCommand
    resampling_time_range: tuple = (1e9, 1e9)


@configclass
class CommandsCfg:
    gait = GaitCommandCfg()


def reset_flat(env, env_ids):
    c = command(env)
    n = len(env_ids)
    if c.evaluation is None:
        # Flat-ground gait-command distribution throughout; no external pushes.
        c.sample_values(env_ids)
        c.stance_start[env_ids] = torch.rand(n, device=env.device) < env.cfg.stance_start_probability
        yaw = (torch.rand(n, device=env.device) - 0.5) * 0.10
        lateral = (torch.rand(n, device=env.device) - 0.5) * 0.03
    else:
        e = c.evaluation
        c.stance_start[env_ids] = e["stance_start"][env_ids]
        c.values[env_ids] = torch.tensor([e["speed"], e["df"], e["step_width"], e["period"], e["gait"]], device=env.device)
        c.period_ticks[env_ids] = round(e["period"] / env.step_dt)
        c.schedule_duty[env_ids] = discrete_stance_fraction(c.values[env_ids, 1], c.period_ticks[env_ids], c.values[env_ids, 4].long())
        yaw = e["yaw"][env_ids]
        lateral = e["lateral"][env_ids]
    robot = env.scene["robot"]
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] = env.scene.env_origins[env_ids]
    root[:, 0] -= 0.65
    root[:, 1] += lateral
    root[:, 2] += TOP + 0.36
    zero = torch.zeros(n, device=env.device)
    root[:, 3:7] = quat_from_euler_xyz(zero, zero, yaw)
    root[:, 7:] = 0
    stance = c.stance_start[env_ids]
    if stance.any():
        if not hasattr(env, "settled_stance"):
            raise RuntimeError("Grounded resets require validated stance calibration before the first reset")
        pose = env.settled_stance
        root[stance, 2] = env.scene.env_origins[env_ids[stance], 2] + pose["height"]
        root[stance, 3:7] = quat_mul(root[stance, 3:7], pose["quaternion"].expand(int(stance.sum()), -1))
        # Reset event order runs the default joint reset first; this replaces it
        # with a physically settled configuration for the selected 10%.
        robot.write_joint_state_to_sim(pose["joint_pos"].expand(int(stance.sum()), -1),
                                      torch.zeros(int(stance.sum()), robot.num_joints, device=env.device),
                                      env_ids=env_ids[stance])
    robot.write_root_state_to_sim(root, env_ids)


def local_body(env):
    return env.scene["robot"].data.root_pos_w - env.scene.env_origins


def local_feet(env):
    c = command(env)
    return env.scene["robot"].data.body_pos_w[:, c.feet] - env.scene.env_origins[:, None]


def contacts(env):
    # Never touch sensor.data from post-reset observations: PhysX still holds the
    # preceding episode's contact until the next physics tick. Cache only in step.
    return command(env).contact_cache


def gait_obs(env):
    c = command(env)
    # Rewards and contact targets use raw phase.  Only the policy encoding is
    # warped so every commanded stance-to-swing boundary appears at phase 0.5.
    observed_phase = duty_warped_phase(c.phase, c.values[:, 1])
    # Heading is an inertial proprioceptive signal relative to the fixed forward
    # command axis.  It is neither a yaw command nor a terrain/map observation.
    return torch.cat([torch.sin(2 * math.pi * observed_phase),
                      torch.cos(2 * math.pi * observed_phase),
                      normalized_gait_command(c.values), c.desired.float(),
                      heading_observation(env).unsqueeze(-1)], dim=-1)


def heading_observation(env):
    heading = heading_error(env)
    if LATERAL_HEADING_GAIN:
        heading = heading + LATERAL_HEADING_GAIN * local_body(env)[:, 1]
    return heading


def heading_error(env):
    q = env.scene["robot"].data.root_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    return yaw


def contact_obs(env):
    return contacts(env).float()


def instantaneous_failure(env):
    p = local_body(env)
    c = command(env)
    g = env.scene["robot"].data.projected_gravity_b
    return (p[:, 2] < .16) | (g[:, 2] > -.5) | c.nonfoot_cache


def support_failure(env):
    current = instantaneous_failure(env)
    return current | getattr(env, "substep_failure", torch.zeros_like(current))


def success(env):
    # Training is sustained flat-ground command execution for up to20 seconds.
    # The forward endpoint is only an evaluation stopping rule/metric.
    if command(env).evaluation is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    f = local_feet(env)
    return (local_body(env)[:, 0] >= LENGTH + .35) & (f[:, :, 0] > LENGTH).all(dim=1) & ~support_failure(env)


def task_reward(env):
    c = command(env)
    robot = env.scene["robot"]
    p, f, ct = local_body(env), local_feet(env), contacts(env)
    # The speed command and reference line use the fixed course frame. Rewarding
    # body-forward speed would permit a yawed robot to walk diagonally while
    # reporting perfect forward tracking.
    v = robot.data.root_lin_vel_w[:, 0]
    speed_reward = planar_speed_score(v - c.values[:, 0], robot.data.root_lin_vel_w[:, 1])
    desired = c.desired
    # Score every 5 ms physics sample.  The former last-sample-only score could
    # miss within-action contact chatter and lightly loaded stance feet.
    force_history = env.scene["contact_forces"].data.net_forces_w_history
    foot_force_norm = force_history[:, :, c.sensor_feet].norm(dim=-1)
    timing = robust_contact_score(
        foot_force_norm, desired, c.schedule_duty,
        contact_threshold=5., stance_floor=15., swing_ceiling=2.)
    substep_contact = foot_force_norm > 5.
    desired_substep = desired[:, None, :]
    target_y = c.values[:, 2:3] * c.signs / 2
    body_feet = world_to_body(f - p[:, None, :], robot.data.root_quat_w)
    lateral_error = body_feet[:, :, 1] - target_y
    world_lateral_error = f[:, :, 1] - p[:, 1:2] - target_y
    placement = width_score(lateral_error)
    target_x = fore_aft_target(c.phase, c.values[:, 1], c.values[:, 0], c.values[:, 3])
    fore_aft = fore_aft_score(body_feet[:, :, 0] - target_x)
    swing_phase = ((c.phase - c.values[:, 1:2]) / (1 - c.values[:, 1:2])).clamp(0, 1)
    target_z = TOP + .025 + .04 * torch.sin(math.pi * swing_phase)
    clearance = clearance_score(f[:, :, 2] - target_z, desired, c.schedule_duty)
    height = torch.exp(-(p[:, 2] - TOP - .32).square() / .0025)
    # Drift is unobserved absolute state. Bound its cost so long episodes do not
    # become worse than deliberately terminating early after a lateral excursion.
    centering_cost = CENTERING_WEIGHT * .5 * (
        1 - torch.exp(-(p[:, 1] / CENTERING_SCALE).square()))
    heading_cost = HEADING_COST_WEIGHT * heading_stabilization_cost(
        heading_observation(env) if HEADING_COST_ON_OBSERVATION else heading_error(env))
    velocity_cost = straight_motion_cost(robot.data.root_ang_vel_b[:, 2])
    # These instantaneous batch metrics help watchers diagnose progress. They do
    # not replace complete-cycle, held-out command-compliance evaluation.
    env.training_metrics = {
        "Gait/contact_accuracy": (ct == desired).float().mean(),
        "Gait/stance_recall": (ct & desired).sum().float() / desired.sum().clamp(min=1),
        "Gait/swing_recall": (~ct & ~desired).sum().float() / (~desired).sum().clamp(min=1),
        "Gait/substep_contact_accuracy":
            (substep_contact == desired_substep).float().mean(),
        "Gait/substep_stance_recall":
            (substep_contact & desired_substep).sum().float()
            / desired_substep.expand_as(substep_contact).sum().clamp(min=1),
        "Gait/substep_swing_recall":
            (~substep_contact & ~desired_substep).sum().float()
            / (~desired_substep).expand_as(substep_contact).sum().clamp(min=1),
        "Gait/stance_force_15_fraction":
            ((foot_force_norm >= 15.) & desired_substep).sum().float()
            / desired_substep.expand_as(foot_force_norm).sum().clamp(min=1),
        "Gait/swing_force_2_fraction":
            ((foot_force_norm <= 2.) & ~desired_substep).sum().float()
            / (~desired_substep).expand_as(foot_force_norm).sum().clamp(min=1),
        "Gait/substep_contact_margin_score": timing.mean(),
        "Gait/foot_lateral_rmse": world_lateral_error.square().mean().sqrt(),
        "Gait/body_foot_lateral_rmse": lateral_error.square().mean().sqrt(),
        "Gait/body_foot_fore_aft_rmse": (body_feet[:, :, 0] - target_x).square().mean().sqrt(),
        "Gait/heading_rmse": heading_error(env).square().mean().sqrt(),
        "Gait/forward_speed": robot.data.root_lin_vel_w[:, 0].mean(),
        "Gait/speed_mae": (v - c.values[:, 0]).abs().mean(),
        "Gait/lateral_rmse": p[:, 1].square().mean().sqrt(),
        "Gait/world_lateral_velocity_rmse": robot.data.root_lin_vel_w[:, 1].square().mean().sqrt(),
        "Gait/body_lateral_velocity_rmse": robot.data.root_lin_vel_b[:, 1].square().mean().sqrt(),
        "Gait/body_yaw_rate_rmse": robot.data.root_ang_vel_b[:, 2].square().mean().sqrt(),
        "Gait/curriculum_stage": torch.tensor(float(c.curriculum_stage), device=env.device),
    }
    # Capture pre-reset states, including the terminal step, for unbiased evaluation.
    if getattr(env, "capture", False):
        env.transition = {"time": c.time.clone(), "phase": c.phase.clone(), "contacts": ct.clone(),
            "desired": desired.clone(), "feet": f.clone(), "body": p.clone(),
            # Preserve the legacy body-forward field for historical comparisons;
            # forward_velocity below is the separately named world-course value.
            "speed": robot.data.root_lin_vel_b[:, 0].clone(),
            "root_quat": robot.data.root_quat_w.clone(), "feet_body": body_feet.clone(),
            "forward_velocity": robot.data.root_lin_vel_w[:, 0].clone(),
            "world_lateral_velocity": robot.data.root_lin_vel_w[:, 1].clone(),
            "body_lateral_velocity": robot.data.root_lin_vel_b[:, 1].clone(),
            "body_yaw_rate": robot.data.root_ang_vel_b[:, 2].clone(),
            "commands": c.values.clone(),
            "schedule_duty": c.schedule_duty.clone(),
            "stance_start": c.stance_start.clone(),
            "failure": support_failure(env).clone(),
            "success": success(env).clone()}
    return (4.0 * speed_reward + 8.0 * timing
            + 8.0 * placement + 2.0 * fore_aft + 1.0 * clearance + .5 * height
            - centering_cost - heading_cost - velocity_cost)


class BeamEnv(ManagerBasedRLEnv):
    def calibrate_stance(self):
        """Create and verify an all-feet-grounded reset pose under stock PD.

        This is deterministic reset calibration, not a learned-controller rollout.
        It runs before PPO creation; no hidden controller drives training actions.
        """
        robot = self.scene["robot"]
        c = command(self)
        print("STANCE_CALIBRATION_START", "playing", self.sim.is_playing(), flush=True)
        for tick in range(1200):
            robot.set_joint_position_target(robot.data.default_joint_pos)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            if tick % 300 == 0:
                print("STANCE_CALIBRATION_TICK", tick, flush=True)
        forces = self.scene["contact_forces"].data.net_forces_w.norm(dim=-1)
        print("STANCE_CALIBRATION_STATE", "sample_forces", forces[:4, c.sensor_feet].tolist(),
              "sample_height", robot.data.root_pos_w[:4, 2].tolist(),
              "sample_linear_speed", robot.data.root_lin_vel_w[:4].norm(dim=1).tolist(),
              "sample_angular_speed", robot.data.root_ang_vel_w[:4].norm(dim=1).tolist(),
              "sample_gravity_z", robot.data.projected_gravity_b[:4, 2].tolist(),
              "sample_nonfoot_max", forces[:4, c.sensor_other].amax(dim=1).tolist(), flush=True)
        valid = ((forces[:, c.sensor_feet] > 5.).all(dim=1)
                 & (forces[:, c.sensor_other] < 1.).all(dim=1)
                 & (robot.data.root_lin_vel_w.norm(dim=1) < .02)
                 & (robot.data.root_ang_vel_w.norm(dim=1) < .05)
                 # Stock Go2 PD posture settles with ~11 degrees pitch in this
                 # asset. Accept <14 degrees tilt while requiring quiet motion,
                 # four loaded feet, and zero non-foot support.
                 & (robot.data.projected_gravity_b[:, 2] < -math.cos(math.radians(14))))
        ids = valid.nonzero().flatten()
        if len(ids) == 0:
            raise RuntimeError("Stance calibration failed: no quiet upright pose with four supported feet")
        index = ids[0]
        self.settled_stance = {
            "height": (robot.data.root_pos_w[index, 2] - self.scene.env_origins[index, 2]).clone(),
            "quaternion": robot.data.root_quat_w[index].clone(),
            "joint_pos": robot.data.joint_pos[index].clone(),
        }
        print("STANCE_CALIBRATED", "height", self.settled_stance["height"].item(),
              "foot_forces", forces[index, c.sensor_feet].tolist(), flush=True)

    def step(self, action: torch.Tensor):
        """Isaac Lab manager step with per-substep contact and failure capture.

        Based on Isaac Lab ManagerBasedRLEnv.step (BSD-3-Clause, 2022-2026).
        Actions, physics, PPO interface, and reset ordering are unchanged.
        """
        c = command(self)
        self.substep_failure = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # Optional high-rate contact capture is enabled only by the stability
        # evaluator; ordinary training does not retain these tensors.
        if getattr(self, "capture_substeps", False):
            self.substep_contacts = []
        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
            forces = self.scene["contact_forces"].data.net_forces_w
            c.contact_cache[:] = forces[:, c.sensor_feet].norm(dim=-1) > 5.
            c.nonfoot_cache[:] = (forces[:, c.sensor_other].norm(dim=-1) > 1.).any(dim=1)
            if getattr(self, "capture_substeps", False):
                self.substep_contacts.append(c.contact_cache.clone())
            self.substep_failure |= instantaneous_failure(self)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        reset_mask = self.reset_buf.clone()
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            self.substep_failure[reset_env_ids] = False

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # The action, physics, reward, and captured transition above all use the
        # phase returned in the preceding observation. Advance only afterward so
        # the next observation exposes the next target. Fresh resets stay at
        # phase zero and continuing environments wrap before command resampling.
        c.phase_ticks[:] = advance_phase_ticks(c.phase_ticks, c.period_ticks, reset_mask)
        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.extras.setdefault("log", {}).update(getattr(self, "training_metrics", {}))

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras



@configclass
class BeamEnvCfg(UnitreeGo2FlatEnvCfg):
    stance_start_probability: float = .10

    def __post_init__(self):
        super().__post_init__()
        self.seed = 42
        self.scene.num_envs = 1024
        self.scene.env_spacing = 12.0
        # Inherit Isaac Lab's open flat ground plane; no beams or support bounds.
        self.scene.terrain.visual_material = sim.PreviewSurfaceCfg(diffuse_color=(.35, .38, .42))
        self.scene.reference_line = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/ReferenceLine", spawn=sim.SpawnerCfg(func=spawn_reference_line))
        self.scene.sky_light.spawn = sim.DomeLightCfg(intensity=1000, color=(.8, .85, 1.))
        self.commands = CommandsCfg()
        self.observations.policy.velocity_commands = None
        self.observations.policy.gait = ObsTerm(func=gait_obs)
        # Geometry is used by physics/rewards/metrics only, never actor or critic.
        self.observations.policy.beam = None
        self.observations.policy.contacts = ObsTerm(func=contact_obs)
        self.observations.policy.enable_corruption = False
        self.events.reset_base = EventTerm(func=reset_flat, mode="reset")
        # Ensure grounded joint pose is written after the stock joint reset.
        joint_reset = self.events.reset_robot_joints
        self.events.reset_robot_joints = None
        self.events.reset_flat_joints = joint_reset
        self.events.reset_flat_root = self.events.reset_base
        self.events.reset_base = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # Keep the paper-correlation experiment on the nominal Go2 plant.
        # Motor-gain domain randomization is postponed to a separate study.
        self.events.motor_gain_randomization = None
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.feet_air_time = None
        self.rewards.task = RewTerm(func=task_reward, weight=1.)
        # Isaac Lab multiplies reward terms by step_dt: failure is -3 per event.
        # No training endpoint bonus; commanded speed defines forward movement.
        self.rewards.failure = RewTerm(func=support_failure, weight=-3. / .02)
        self.rewards.crossing = None
        self.rewards.dof_pos_limits.weight = -1.
        self.terminations.support_failure = DoneTerm(func=support_failure)
        self.terminations.base_contact = None  # Included in reset-safe per-substep nonfoot cache.
        self.terminations.crossing = DoneTerm(func=success)
        self.episode_length_s = 20.
        self.sim.dt = .005
        self.decimation = 4
        self.sim.render_interval = 4
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.contact_forces.history_length = self.decimation
        self.viewer.eye = (5., -5., 3.)
        self.viewer.lookat = (1.5, 0., .6)


@configclass
class BeamPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_flat_gait_commands"
        self.obs_groups = {"policy": ["policy"], "critic": ["policy"]}
        self.max_iterations = 1800
        self.save_interval = 50
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
        self.policy.init_noise_std = .5
        self.algorithm.entropy_coef = .001
        # Fresh-training profile; incompatible historical checkpoints are rejected.
        self.num_steps_per_env = 48
        self.algorithm.gamma = .995
        self.algorithm.learning_rate = 1e-3
        self.algorithm.schedule = "adaptive"
        self.algorithm.clip_param = .2
        self.clip_actions = 5.
