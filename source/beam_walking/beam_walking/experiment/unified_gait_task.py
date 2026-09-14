"""Flat-ground low-level controller task for the unified trot/walk policy."""

from isaaclab.utils import configclass

from .protocol import discrete_stance_fraction
from .task import (
    BeamEnv,
    BeamEnvCfg,
    BeamPPORunnerCfg,
    CommandsCfg,
    GaitCommand,
    GaitCommandCfg,
)
from .unified_gait import sample_unified_commands


class UnifiedGaitCommand(GaitCommand):
    """Sample both gaits with gait-specific duty ranges and a shared period range."""

    def sample_values(self, ids):
        sampled, ticks, stage = sample_unified_commands(
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
class UnifiedGaitCommandCfg(GaitCommandCfg):
    class_type: type = UnifiedGaitCommand


@configclass
class UnifiedCommandsCfg(CommandsCfg):
    gait = UnifiedGaitCommandCfg()


@configclass
class UnifiedEnvCfg(BeamEnvCfg):
    """Keep the plant and reward fixed; replace the command distribution."""

    def __post_init__(self):
        super().__post_init__()
        self.commands = UnifiedCommandsCfg()


@configclass
class UnifiedPPORunnerCfg(BeamPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_flat_unified_gait"


UnifiedEnv = BeamEnv
