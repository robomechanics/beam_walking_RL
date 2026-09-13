"""Environment hook that applies the ramped base wrench from ``perturbation.py``.

Implemented as a mixin over ``BeamEnv``/``DeploymentBeamEnv`` rather than an
edit to ``task.py`` so the task source hash, and therefore every existing
checkpoint's evaluation eligibility, is unchanged. The wrench is written to
Isaac Lab's permanent wrench composer once per control step, so it persists
across the four physics substeps, and it is cleared by the articulation reset.
"""
import torch

from .perturbation import PerturbationCfg, RampedPerturbation
from .protocol import world_to_body
from .task import local_body


class PerturbedEnvMixin:
    """Applies ``self.perturbation`` before the inherited physics loop."""

    perturbation_cfg: PerturbationCfg = None

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if self.perturbation_cfg is None:
            raise RuntimeError("perturbed_env_class requires a perturbation profile")
        robot = self.scene["robot"]
        self.perturbation_body = robot.find_bodies("base")[0]
        if len(self.perturbation_body) != 1:
            raise RuntimeError("Expected exactly one Go2 base body for the perturbation")
        self.perturbation = RampedPerturbation(
            self.perturbation_cfg, self.num_envs, self.device, seed=int(cfg.seed))
        self.perturbation_start_x = torch.zeros(self.num_envs, device=self.device)
        self.applied_perturbation = None

    def perturbation_progress(self, time, fresh):
        cfg = self.perturbation_cfg
        if cfg.ramp_mode == "time":
            return time / cfg.ramp_time_s
        x = local_body(self)[:, 0]
        # Forward travel is measured from the first step of each episode.
        self.perturbation_start_x[fresh] = x[fresh]
        return (x - self.perturbation_start_x) / cfg.ramp_distance_m

    def step(self, action):
        time = self.episode_length_buf.float() * self.step_dt
        fresh = self.episode_length_buf == 0
        progress = self.perturbation_progress(time, fresh)
        force_w, torque_w, envelope = self.perturbation.step(time, progress, fresh)
        robot = self.scene["robot"]
        quat = robot.data.root_quat_w
        # The composer expects link-frame wrenches; rotate the world-frame
        # sample ourselves so no cached link pose inside the composer is used.
        force_b = world_to_body(force_w[:, None, :], quat).contiguous()
        torque_b = world_to_body(torque_w[:, None, :], quat).contiguous()
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=force_b, torques=torque_b, body_ids=self.perturbation_body)
        self.applied_perturbation = {
            "force_w": force_w, "torque_w": torque_w, "envelope": envelope}
        result = super().step(action)
        if getattr(self, "capture", False) and hasattr(self, "transition"):
            self.transition["perturbation_force_w"] = force_w.clone()
            self.transition["perturbation_torque_w"] = torque_w.clone()
            self.transition["perturbation_envelope"] = envelope.clone()
        return result


def perturbed_env_class(base, cfg: PerturbationCfg):
    """Subclass ``base`` (BeamEnv or DeploymentBeamEnv) with the perturbation hook."""
    return type(f"Perturbed{base.__name__}", (PerturbedEnvMixin, base),
                {"perturbation_cfg": cfg.validate()})
