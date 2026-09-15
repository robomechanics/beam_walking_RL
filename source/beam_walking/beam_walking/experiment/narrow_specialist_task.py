"""Flat-ground task for a narrow-stance single-gait specialist."""

from isaaclab.utils import configclass

from . import protocol as _protocol
from .narrow_specialist import NARROW_WIDTH_TOLERANCE_M, sample_narrow_commands
from .protocol import discrete_stance_fraction
from .task import (
    BeamEnv,
    BeamEnvCfg,
    BeamPPORunnerCfg,
    CommandsCfg,
    GaitCommand,
    GaitCommandCfg,
)

# The lateral placement reward reads this module global at call time. Only one
# task is built per process, so setting it here scopes it to this task.
_protocol.WIDTH_SCORE_VARIANCE = NARROW_WIDTH_TOLERANCE_M ** 2


class NarrowSpecialistGaitCommand(GaitCommand):
    """Sample one gait over narrow stance widths."""

    def sample_values(self, ids):
        sampled, ticks, stage = sample_narrow_commands(
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
class NarrowSpecialistGaitCommandCfg(GaitCommandCfg):
    class_type: type = NarrowSpecialistGaitCommand
    gait: str = "trot"


@configclass
class NarrowSpecialistCommandsCfg(CommandsCfg):
    gait = NarrowSpecialistGaitCommandCfg()


@configclass
class NarrowSpecialistEnvCfg(BeamEnvCfg):
    """Stock plant and reward, narrow command distribution, self-collision on."""

    specialist_gait: str = "trot"

    def __post_init__(self):
        super().__post_init__()
        self.commands = NarrowSpecialistCommandsCfg()
        self.commands.gait.gait = self.specialist_gait
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True


@configclass
class NarrowSpecialistPPORunnerCfg(BeamPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_flat_narrow_specialist"


NarrowSpecialistEnv = BeamEnv
