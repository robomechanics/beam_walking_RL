"""Flat-ground low-level controller task for high-duty walking."""

from isaaclab.utils import configclass

from .high_duty_walk import sample_high_duty_walk_commands
from .protocol import discrete_stance_fraction
from .task import (
    BeamEnv,
    BeamEnvCfg,
    BeamPPORunnerCfg,
    CommandsCfg,
    GaitCommand,
    GaitCommandCfg,
)


class HighDutyWalkGaitCommand(GaitCommand):
    """Sample only walk commands with duty factors from 0.75 to 0.90."""

    def sample_values(self, ids):
        sampled, ticks, stage = sample_high_duty_walk_commands(
            len(ids), int(getattr(self._env, "common_step_counter", 0)),
            self.device,
        )
        self.values[ids] = sampled
        self.period_ticks[ids] = ticks
        self.curriculum_stage = stage
        self.schedule_duty[ids] = discrete_stance_fraction(
            self.values[ids, 1], self.period_ticks[ids],
            self.values[ids, 4].long(),
        )


@configclass
class HighDutyWalkGaitCommandCfg(GaitCommandCfg):
    class_type: type = HighDutyWalkGaitCommand


@configclass
class HighDutyWalkCommandsCfg(CommandsCfg):
    gait = HighDutyWalkGaitCommandCfg()


@configclass
class HighDutyWalkEnvCfg(BeamEnvCfg):
    """Keep the plant and reward fixed; replace the command distribution."""

    def __post_init__(self):
        super().__post_init__()
        self.commands = HighDutyWalkCommandsCfg()


@configclass
class HighDutyWalkPPORunnerCfg(BeamPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_flat_high_duty_walk"


HighDutyWalkEnv = BeamEnv
