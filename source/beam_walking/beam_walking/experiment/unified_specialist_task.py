"""Flat-ground low-level controller task for a single-gait specialist."""

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
from .unified_specialist import sample_specialist_commands


class SpecialistGaitCommand(GaitCommand):
    """Sample one gait with the unified per-gait distribution."""

    def sample_values(self, ids):
        sampled, ticks, stage = sample_specialist_commands(
            self.cfg.gait, len(ids),
            int(getattr(self._env, "common_step_counter", 0)), self.device,
        )
        self.values[ids] = sampled
        self.period_ticks[ids] = ticks
        self.curriculum_stage = stage
        self.schedule_duty[ids] = discrete_stance_fraction(
            self.values[ids, 1], self.period_ticks[ids],
            self.values[ids, 4].long(),
        )


@configclass
class SpecialistGaitCommandCfg(GaitCommandCfg):
    class_type: type = SpecialistGaitCommand
    gait: str = "trot"


@configclass
class SpecialistCommandsCfg(CommandsCfg):
    gait = SpecialistGaitCommandCfg()


@configclass
class SpecialistEnvCfg(BeamEnvCfg):
    """Keep the plant and reward fixed; replace the command distribution."""

    specialist_gait: str = "trot"

    def __post_init__(self):
        super().__post_init__()
        self.commands = SpecialistCommandsCfg()
        self.commands.gait.gait = self.specialist_gait


@configclass
class SpecialistPPORunnerCfg(BeamPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_flat_unified_specialist"


SpecialistEnv = BeamEnv
