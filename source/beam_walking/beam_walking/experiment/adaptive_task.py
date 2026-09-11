"""Separate flat-ground locomotion task for adaptive duty-factor selection."""

from isaaclab.utils import configclass

from .adaptive_duty import sample_adaptive_training_commands
from .protocol import discrete_stance_fraction
from .task import (
    BeamEnv,
    BeamEnvCfg,
    BeamPPORunnerCfg,
    CommandsCfg,
    GaitCommand,
    GaitCommandCfg,
)


class AdaptiveGaitCommand(GaitCommand):
    """Train both walk and trot throughout the feasible duty-factor range."""

    def sample_values(self, ids):
        sampled, ticks, stage = sample_adaptive_training_commands(
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
class AdaptiveGaitCommandCfg(GaitCommandCfg):
    class_type: type = AdaptiveGaitCommand


@configclass
class AdaptiveCommandsCfg(CommandsCfg):
    gait = AdaptiveGaitCommandCfg()


@configclass
class AdaptiveBeamEnvCfg(BeamEnvCfg):
    """Keep the plant/reward fixed and replace only the command distribution."""

    def __post_init__(self):
        super().__post_init__()
        self.commands = AdaptiveCommandsCfg()


@configclass
class AdaptiveBeamPPORunnerCfg(BeamPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_flat_adaptive_duty"


AdaptiveBeamEnv = BeamEnv
