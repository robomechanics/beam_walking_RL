"""DC-motor model with per-episode command latency for deployment training."""

from collections.abc import Sequence

import torch
from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer


class DeploymentDCMotor(DCMotor):
    """Keep Go2 torque-speed clipping while delaying actuator setpoints."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.positions_delay_buffer = DelayBuffer(
            cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(
            cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(
            cfg.max_delay, self._num_envs, device=self._device)
        self.default_effort_limit = self.effort_limit.clone()
        self._saturation_effort = torch.full_like(
            self.effort_limit, float(cfg.saturation_effort))
        self.default_saturation_effort = self._saturation_effort.clone()
        self.strength_scale = torch.ones_like(self.effort_limit)

    def reset(self, env_ids: Sequence[int] | None):
        if env_ids is None or env_ids == slice(None):
            count = self._num_envs
        else:
            count = len(env_ids)
        lags = torch.randint(
            self.cfg.min_delay, self.cfg.max_delay + 1, (count,),
            dtype=self.positions_delay_buffer.time_lags.dtype,
            device=self._device)
        for buffer in (
                self.positions_delay_buffer, self.velocities_delay_buffer,
                self.efforts_delay_buffer):
            buffer.set_time_lag(lags, env_ids)
            buffer.reset(env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        control_action.joint_positions = self.positions_delay_buffer.compute(
            control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(
            control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(
            control_action.joint_efforts)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DeploymentDCMotorCfg(DCMotorCfg):
    class_type: type = DeploymentDCMotor
    min_delay: int = 0
    max_delay: int = 4
