"""Hardware-oriented randomization around the frozen 68D flat-ground task."""

import math

import torch

from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from .deployment import (
    CALIBRATION_STIFFNESS, CALIBRATION_DAMPING, CALIBRATION_QUIET_STEPS,
    CALIBRATION_MAX_STEPS,
    BASE_COM_RANGE_M,
    BASE_MASS_SCALE_RANGE,
    DYNAMIC_FRICTION_RANGE,
    JOINT_ARMATURE_RANGE,
    JOINT_FRICTION_RANGE,
    MOTOR_STRENGTH_SCALE_RANGE,
    DAMPING_SCALE_RANGE,
    STIFFNESS_SCALE_RANGE,
    STATIC_FRICTION_RANGE,
)
from .deployment_actuator import DeploymentDCMotor, DeploymentDCMotorCfg
from .task import (
    TOP, BeamEnv, BeamEnvCfg, BeamPPORunnerCfg, command,
)


def randomize_motor_strength(env, env_ids, scale_range):
    """Scale the DC-motor continuous and saturation torque together."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    for actuator in env.scene["robot"].actuators.values():
        if not isinstance(actuator, DeploymentDCMotor):
            raise TypeError("Deployment strength DR requires DeploymentDCMotor")
        scales = torch.empty(
            len(env_ids), actuator.num_joints, device=env.device).uniform_(
                *scale_range)
        actuator.effort_limit[env_ids] = (
            actuator.default_effort_limit[env_ids] * scales)
        actuator._saturation_effort[env_ids] = (
            actuator.default_saturation_effort[env_ids] * scales)
        actuator._vel_at_effort_lim[env_ids] = actuator.velocity_limit[env_ids] * (
            1 + actuator.effort_limit[env_ids]
            / actuator._saturation_effort[env_ids])
        actuator.strength_scale[env_ids] = scales


def randomize_deployment_com(env, env_ids, asset_cfg, com_range):
    """Randomize COM and retain the exact default/delta for runtime auditing."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    body_ids = (torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
                if asset_cfg.body_ids == slice(None)
                else torch.tensor(asset_cfg.body_ids, dtype=torch.int,
                                  device="cpu"))
    ranges = torch.tensor(
        [com_range.get(axis, (0.0, 0.0)) for axis in "xyz"], device="cpu")
    offsets = torch.empty((len(env_ids), 3), device="cpu").uniform_(0, 1)
    offsets = ranges[:, 0] + offsets * (ranges[:, 1] - ranges[:, 0])
    coms = asset.root_physx_view.get_coms().clone()
    if not hasattr(env, "deployment_default_com"):
        env.deployment_default_com = coms.clone()
        env.deployment_com_offsets = torch.zeros(
            asset.num_instances, asset.num_bodies, 3, device="cpu")
    coms[env_ids[:, None], body_ids, :3] = (
        env.deployment_default_com[env_ids[:, None], body_ids, :3]
        + offsets[:, None, :])
    env.deployment_com_offsets[env_ids[:, None], body_ids] = offsets[:, None]
    asset.root_physx_view.set_coms(coms, env_ids)


def reset_deployment(env, env_ids):
    """Reset with a stance pose calibrated for each randomized robot instance."""
    gait = command(env)
    count = len(env_ids)
    if gait.evaluation is None:
        gait.sample_values(env_ids)
        gait.stance_start[env_ids] = (
            torch.rand(count, device=env.device)
            < env.cfg.stance_start_probability)
        yaw = (torch.rand(count, device=env.device) - 0.5) * 0.10
        lateral = (torch.rand(count, device=env.device) - 0.5) * 0.03
    else:
        evaluation = gait.evaluation
        gait.stance_start[env_ids] = evaluation["stance_start"][env_ids]
        gait.values[env_ids] = torch.tensor([
            evaluation["speed"], evaluation["df"], evaluation["step_width"],
            evaluation["period"], evaluation["gait"],
        ], device=env.device)
        gait.period_ticks[env_ids] = round(evaluation["period"] / env.step_dt)
        from .protocol import discrete_stance_fraction
        gait.schedule_duty[env_ids] = discrete_stance_fraction(
            gait.values[env_ids, 1], gait.period_ticks[env_ids],
            gait.values[env_ids, 4].long())
        yaw = evaluation["yaw"][env_ids]
        lateral = evaluation["lateral"][env_ids]
    robot = env.scene["robot"]
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] = env.scene.env_origins[env_ids]
    root[:, 0] -= 0.65
    root[:, 1] += lateral
    root[:, 2] += TOP + 0.36
    zero = torch.zeros(count, device=env.device)
    root[:, 3:7] = quat_from_euler_xyz(zero, zero, yaw)
    root[:, 7:] = 0
    stance = gait.stance_start[env_ids]
    if stance.any():
        if not hasattr(env, "settled_stance"):
            raise RuntimeError("Deployment grounded resets require calibration")
        selected = env_ids[stance]
        pose = env.settled_stance
        root[stance, 2] = (
            env.scene.env_origins[selected, 2] + pose["height"][selected])
        root[stance, 3:7] = quat_mul(
            root[stance, 3:7], pose["quaternion"][selected])
        robot.write_joint_state_to_sim(
            pose["joint_pos"][selected],
            torch.zeros(len(selected), robot.num_joints, device=env.device),
            env_ids=selected)
    robot.write_root_state_to_sim(root, env_ids)


def motor_gain_event(kp_range=STIFFNESS_SCALE_RANGE,
                     kd_range=DAMPING_SCALE_RANGE):
    """Construct per-environment, per-joint multiplicative gain variation."""
    return EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": tuple(kp_range),
            "damping_distribution_params": tuple(kd_range),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class DeploymentBeamEnvCfg(BeamEnvCfg):
    """Randomize measurable deployment uncertainties without pushes or terrain."""

    def __post_init__(self):
        super().__post_init__()
        # Resolve contact velocity updates under the broad dynamics distribution.
        self.sim.physx.enable_external_forces_every_iteration = True
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 2
        self.scene.robot.actuators["base_legs"] = DeploymentDCMotorCfg(
            joint_names_expr=[
                ".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5, saturation_effort=23.5,
            velocity_limit=30.0, stiffness=25.0, damping=0.5,
            friction=0.0, min_delay=0, max_delay=4,
        )
        self.observations.policy.enable_corruption = True
        self.events.physics_material.params.update({
            "static_friction_range": STATIC_FRICTION_RANGE,
            "dynamic_friction_range": DYNAMIC_FRICTION_RANGE,
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        })
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": BASE_MASS_SCALE_RANGE,
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )
        self.events.base_com = EventTerm(
            func=randomize_deployment_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": BASE_COM_RANGE_M,
            },
        )
        self.events.motor_gain_randomization = motor_gain_event()
        self.events.motor_strength_randomization = EventTerm(
            func=randomize_motor_strength, mode="startup",
            params={"scale_range": MOTOR_STRENGTH_SCALE_RANGE})
        self.events.joint_parameter_randomization = EventTerm(
            func=mdp.randomize_joint_parameters, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "friction_distribution_params": JOINT_FRICTION_RANGE,
                "armature_distribution_params": JOINT_ARMATURE_RANGE,
                "operation": "abs", "distribution": "uniform",
            })
        self.events.reset_flat_root = EventTerm(
            func=reset_deployment, mode="reset")
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class DeploymentBeamPPORunnerCfg(BeamPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_flat_gait_commands_deployment_dr"


class DeploymentBeamEnv(BeamEnv):
    """BeamEnv mechanics with per-instance grounded-pose calibration."""

    def calibrate_stance(self):
        robot = self.scene["robot"]
        gait = command(self)
        # Reset-pose preparation uses a firm stand controller. Every instance
        # keeps its sampled mass/COM/material/strength/joint parameters. Restore
        # its exact randomized gains before any policy step, even on error.
        target = robot.data.default_joint_pos.clone()
        def state_checks():
            forces = self.scene["contact_forces"].data.net_forces_w.norm(dim=-1)
            metrics = {
                "minimum_foot_force_n":
                    forces[:, gait.sensor_feet].amin(dim=1),
                "maximum_nonfoot_force_n":
                    forces[:, gait.sensor_other].amax(dim=1),
                "linear_speed_m_s": robot.data.root_lin_vel_w.norm(dim=1),
                "angular_speed_rad_s": robot.data.root_ang_vel_w.norm(dim=1),
                "projected_gravity_z": robot.data.projected_gravity_b[:, 2],
            }
            criteria = {
                "four_feet_over_5n": metrics["minimum_foot_force_n"] > 5.,
                "nonfeet_under_1n": metrics["maximum_nonfoot_force_n"] < 1.,
                "linear_speed_under_0.02": metrics["linear_speed_m_s"] < .02,
                "angular_speed_under_0.05":
                    metrics["angular_speed_rad_s"] < .05,
                "tilt_under_14deg": metrics["projected_gravity_z"]
                    < -math.cos(math.radians(14)),
            }
            return forces, metrics, criteria

        quiet = torch.zeros(target.shape[0], dtype=torch.long, device=target.device)
        found = torch.zeros_like(quiet, dtype=torch.bool)
        sample_ticks = torch.full_like(quiet, -1)
        captured = None
        saved_gains = [(actuator, actuator.stiffness.clone(), actuator.damping.clone())
                       for actuator in robot.actuators.values()]
        print("DEPLOYMENT_STANCE_CALIBRATION_START", flush=True)
        try:
            for actuator, _, _ in saved_gains:
                actuator.stiffness.fill_(CALIBRATION_STIFFNESS)
                actuator.damping.fill_(CALIBRATION_DAMPING)
            for tick in range(CALIBRATION_MAX_STEPS):
                robot.set_joint_position_target(target)
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.physics_dt)
                if tick >= 1000:
                    forces, metrics, criteria = state_checks()
                    acceptable = torch.stack(list(criteria.values())).all(dim=0)
                    quiet = torch.where(acceptable, quiet + 1, 0)
                    selected = (quiet >= CALIBRATION_QUIET_STEPS) & ~found
                    current = {
                        "height": robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2],
                        "quaternion": robot.data.root_quat_w,
                        "joint_pos": robot.data.joint_pos,
                        "forces": forces,
                        **metrics,
                    }
                    if captured is None:
                        captured = {key: value.clone() for key, value in current.items()}
                    for key, value in current.items():
                        captured[key][~found] = value[~found]
                    sample_ticks[selected] = tick
                    found |= selected
                    if bool(found.all()):
                        break
                if tick % 300 == 0:
                    print("DEPLOYMENT_STANCE_CALIBRATION_TICK", tick, flush=True)
        finally:
            for actuator, kp, kd in saved_gains:
                actuator.stiffness.copy_(kp)
                actuator.damping.copy_(kd)
        forces = captured["forces"]
        metrics = {key: captured[key] for key in metrics}
        criteria = {
            "four_feet_over_5n": metrics["minimum_foot_force_n"] > 5.,
            "nonfeet_under_1n": metrics["maximum_nonfoot_force_n"] < 1.,
            "linear_speed_under_0.02": metrics["linear_speed_m_s"] < .02,
            "angular_speed_under_0.05":
                metrics["angular_speed_rad_s"] < .05,
            "tilt_under_14deg": metrics["projected_gravity_z"]
                < -math.cos(math.radians(14)),
        }
        valid = found & torch.stack(list(criteria.values())).all(dim=0)
        self.deployment_calibration_diagnostics = {
            "calibration_stiffness": CALIBRATION_STIFFNESS,
            "calibration_damping": CALIBRATION_DAMPING,
            "randomized_gains_restored": True,
            "required_consecutive_quiet_steps": CALIBRATION_QUIET_STEPS,
            "captured_physics_ticks": sample_ticks.cpu().tolist(),
            "joint_names": robot.joint_names,
            "joint_positions": captured["joint_pos"].detach().cpu().tolist(),
            "calibration_targets": target.detach().cpu().tolist(),
            "body_names": self.scene["contact_forces"].body_names,
            "body_force_norms": forces.detach().cpu().tolist(),
            "num_envs": self.num_envs,
            "valid_envs": int(valid.sum()),
            "failed_env_ids": (~valid).nonzero().flatten().cpu().tolist(),
            "criteria": {
                name: {
                    "passed": int(mask.sum()),
                    "failed_env_ids":
                        (~mask).nonzero().flatten().cpu().tolist(),
                }
                for name, mask in criteria.items()
            },
            "metrics": {
                name: {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                    "values": values.detach().cpu().tolist(),
                }
                for name, values in metrics.items()
            },
        }
        print("DEPLOYMENT_STANCE_CALIBRATION_DIAGNOSTICS",
              {"valid_envs": int(valid.sum()), "num_envs": self.num_envs,
               "criterion_pass_counts": {
                   name: int(mask.sum()) for name, mask in criteria.items()}},
              flush=True)
        if not bool(valid.all()):
            bad = (~valid).nonzero().flatten()
            raise RuntimeError(
                "Deployment stance calibration failed for randomized envs: "
                f"{bad[:32].cpu().tolist()} ({len(bad)}/{self.num_envs})")
        self.settled_stance = {key: captured[key] for key in
                               ("height", "quaternion", "joint_pos")}
        print("DEPLOYMENT_STANCE_CALIBRATED", self.num_envs, flush=True)


def deployment_runtime_summary(env):
    """Read back every randomized component and fail closed on misconfiguration."""
    robot = env.scene["robot"]
    base_ids, base_names = robot.find_bodies("base", preserve_order=True)
    if len(base_ids) != 1:
        raise RuntimeError(f"Expected one base body, got {base_names}")
    base_id = int(base_ids[0])
    masses = robot.root_physx_view.get_masses().cpu()
    default_mass = robot.data.default_mass.cpu()
    mass_ratio = masses[:, base_id] / default_mass[:, base_id]
    inertias = robot.root_physx_view.get_inertias().cpu()[:, base_id]
    default_inertia = robot.data.default_inertia.cpu()[:, base_id]
    expected_inertia = default_inertia * mass_ratio[:, None]
    inertia_error = ((inertias - expected_inertia).abs()
                     / default_inertia.abs().clamp(min=1e-6)).max()

    coms = robot.root_physx_view.get_coms().cpu()
    if not hasattr(env, "deployment_default_com"):
        raise RuntimeError("Deployment COM event did not retain its default")
    com_delta = (coms[:, base_id, :3]
                 - env.deployment_default_com[:, base_id, :3])
    recorded_com = env.deployment_com_offsets[:, base_id]
    com_readback_error = (com_delta - recorded_com).abs().max()

    materials = robot.root_physx_view.get_material_properties().cpu()
    static_friction = materials[..., 0].reshape(-1)
    dynamic_friction = materials[..., 1].reshape(-1)
    joint_friction = robot.data.joint_friction_coeff.detach().cpu().reshape(-1)
    joint_armature = robot.data.joint_armature.detach().cpu().reshape(-1)

    kp, kd, strength, effort_strength, saturation_strength = [], [], [], [], []
    delays, resolved_joints = [], []
    for actuator in robot.actuators.values():
        if not isinstance(actuator, DeploymentDCMotor):
            raise RuntimeError("Deployment actuator type was not installed")
        indices = actuator.joint_indices
        nominal_kp = robot.data.default_joint_stiffness[:, indices]
        nominal_kd = robot.data.default_joint_damping[:, indices]
        kp.append((actuator.stiffness / nominal_kp).detach().cpu().reshape(-1))
        kd.append((actuator.damping / nominal_kd).detach().cpu().reshape(-1))
        strength.append(actuator.strength_scale.detach().cpu().reshape(-1))
        effort_strength.append((
            actuator.effort_limit / actuator.default_effort_limit
        ).detach().cpu().reshape(-1))
        saturation_strength.append((
            actuator._saturation_effort / actuator.default_saturation_effort
        ).detach().cpu().reshape(-1))
        delays.append(
            actuator.positions_delay_buffer.time_lags.detach().cpu().reshape(-1))
        resolved_joints.extend(actuator.joint_names)
    kp = torch.cat(kp)
    kd = torch.cat(kd)
    strength = torch.cat(strength)
    effort_strength = torch.cat(effort_strength)
    saturation_strength = torch.cat(saturation_strength)
    delays = torch.cat(delays)
    active_events = sorted(
        name for names in env.event_manager.active_terms.values()
        for name in names)

    def bounded(values, bounds, name, minimum_std):
        if (not bool(torch.isfinite(values).all())
                or float(values.min()) < bounds[0] - 1e-6
                or float(values.max()) > bounds[1] + 1e-6
                or float(values.std()) <= minimum_std):
            raise RuntimeError(f"Invalid deployment {name} randomization")

    bounded(kp, STIFFNESS_SCALE_RANGE, "Kp", .01)
    bounded(kd, DAMPING_SCALE_RANGE, "Kd", .01)
    bounded(mass_ratio, BASE_MASS_SCALE_RANGE, "base mass", .005)
    bounded(static_friction, STATIC_FRICTION_RANGE, "static friction", .01)
    bounded(dynamic_friction, DYNAMIC_FRICTION_RANGE, "dynamic friction", .01)
    bounded(joint_friction, JOINT_FRICTION_RANGE, "joint friction", .002)
    bounded(joint_armature, JOINT_ARMATURE_RANGE, "joint armature", .0002)
    bounded(strength, MOTOR_STRENGTH_SCALE_RANGE, "motor strength", .005)
    if (not torch.allclose(strength, effort_strength, atol=1e-6, rtol=0.)
            or not torch.allclose(
                strength, saturation_strength, atol=1e-6, rtol=0.)):
        raise RuntimeError("Motor strength does not match both torque limits")
    for axis, bounds in enumerate(BASE_COM_RANGE_M.values()):
        bounded(com_delta[:, axis], bounds, f"COM {'xyz'[axis]}", .001)
    if float(inertia_error) > 1e-4:
        raise RuntimeError("Base inertia was not recomputed with mass scale")
    if float(com_readback_error) > 1e-6:
        raise RuntimeError("COM readback differs from applied offset")
    if bool((dynamic_friction > static_friction + 1e-6).any()):
        raise RuntimeError("Dynamic friction exceeds static friction")
    if int(delays.min()) < 0 or int(delays.max()) > 4 \
            or len(torch.unique(delays)) < 2:
        raise RuntimeError("Deployment actuator delay did not span its bounds")
    if {"base_external_force_torque", "push_robot"} & set(active_events):
        raise RuntimeError("Deployment runtime contains an external-force event")
    if not env.cfg.observations.policy.enable_corruption:
        raise RuntimeError("Deployment observation corruption is disabled")

    def statistics(values):
        return {
            "min": float(values.min()), "max": float(values.max()),
            "std": float(values.float().std()),
        }

    return {
        "base_body_id": base_id, "base_body_name": base_names[0],
        "resolved_joint_names": resolved_joints,
        "active_events": active_events,
        "kp_scale": statistics(kp), "kd_scale": statistics(kd),
        "motor_strength_scale": statistics(strength),
        "motor_effort_limit_scale": statistics(effort_strength),
        "motor_saturation_effort_scale": statistics(saturation_strength),
        "base_mass_scale": statistics(mass_ratio),
        "base_inertia_relative_error_max": float(inertia_error),
        "base_com_x_m": statistics(com_delta[:, 0]),
        "base_com_y_m": statistics(com_delta[:, 1]),
        "base_com_z_m": statistics(com_delta[:, 2]),
        "base_com_readback_error_max_m": float(com_readback_error),
        "static_friction": statistics(static_friction),
        "dynamic_friction": statistics(dynamic_friction),
        "joint_friction": statistics(joint_friction),
        "joint_armature": statistics(joint_armature),
        "actuator_delay_physics_steps": {
            **statistics(delays), "unique": sorted(set(delays.tolist()))},
        "observation_corruption": True,
        "external_force_events_absent": True,
    }
