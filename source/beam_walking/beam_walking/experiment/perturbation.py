"""Opt-in ramped external base perturbation for training and evaluation.

A random world-frame force and torque act on the Go2 base link. Each
environment holds one sampled direction/magnitude for a random duration and
then resamples, so the disturbance is a piecewise-constant random wrench rather
than 50 Hz white noise that would average out on a 15 kg body.

The magnitude envelope starts at ``start_fraction`` of the maximum at reset and
grows linearly with episode time (default) or forward travel until it reaches
the maximum, so every episode begins nearly undisturbed and is pushed harder
the further it goes. The same schedule is used by ``beam_experiment.py``
(train/smoke/evaluate) and ``evaluate_policy.py``; it is off unless
``--perturbation`` is passed, which keeps the paper protocol push-free.

This module is pure torch/numpy so it is unit-testable without Isaac Sim. The
environment hook that applies the wrench lives in ``perturbation_env.py``.
"""
from dataclasses import dataclass, asdict
import math

import numpy as np
import torch

PERTURBATION_SCHEMA = "ramped_base_wrench_v1"
RAMP_MODES = ("time", "distance")
FORCE_RANGE_N = (0., 100.)
TORQUE_RANGE_NM = (0., 20.)
RAMP_TIME_RANGE_S = (.5, 60.)
RAMP_DISTANCE_RANGE_M = (.25, 20.)
HOLD_RANGE_S = (.02, 2.)


@dataclass(frozen=True)
class PerturbationCfg:
    """Frozen per-run perturbation profile; recorded verbatim in provenance."""
    max_force_n: float = 25.
    max_torque_nm: float = 3.
    start_fraction: float = .10
    ramp_mode: str = "time"
    ramp_time_s: float = 10.
    ramp_distance_m: float = 3.
    hold_min_s: float = .10
    hold_max_s: float = .40
    vertical_force_fraction: float = .25

    def validate(self):
        if not FORCE_RANGE_N[0] <= self.max_force_n <= FORCE_RANGE_N[1]:
            raise ValueError(f"--perturbation_max_force must be in {FORCE_RANGE_N} N")
        if not TORQUE_RANGE_NM[0] <= self.max_torque_nm <= TORQUE_RANGE_NM[1]:
            raise ValueError(f"--perturbation_max_torque must be in {TORQUE_RANGE_NM} N m")
        if self.max_force_n == 0. and self.max_torque_nm == 0.:
            raise ValueError("Perturbation requires a nonzero maximum force or torque")
        if not 0. <= self.start_fraction <= 1.:
            raise ValueError("--perturbation_start_fraction must be in [0, 1]")
        if self.ramp_mode not in RAMP_MODES:
            raise ValueError(f"--perturbation_ramp_mode must be one of {RAMP_MODES}")
        if not RAMP_TIME_RANGE_S[0] <= self.ramp_time_s <= RAMP_TIME_RANGE_S[1]:
            raise ValueError(f"--perturbation_ramp_time must be in {RAMP_TIME_RANGE_S} s")
        if not RAMP_DISTANCE_RANGE_M[0] <= self.ramp_distance_m <= RAMP_DISTANCE_RANGE_M[1]:
            raise ValueError(f"--perturbation_ramp_distance must be in {RAMP_DISTANCE_RANGE_M} m")
        if not (HOLD_RANGE_S[0] <= self.hold_min_s <= self.hold_max_s <= HOLD_RANGE_S[1]):
            raise ValueError(
                f"--perturbation_hold needs min <= max within {HOLD_RANGE_S} s")
        if not 0. <= self.vertical_force_fraction <= 1.:
            raise ValueError("vertical_force_fraction must be in [0, 1]")
        return self

    def profile(self):
        """JSON-serializable description for provenance, manifests and archives."""
        return {"schema": PERTURBATION_SCHEMA, "frame": "world", "body": "base",
                "envelope": "start + (1 - start) * clip(progress, 0, 1)",
                "progress": ("episode_time_s / ramp_time_s" if self.ramp_mode == "time"
                             else "forward_travel_m / ramp_distance_m"),
                "sampling": "piecewise_constant_random_wrench", **asdict(self)}


def ramp_envelope(progress, start_fraction):
    """Linear envelope from ``start_fraction`` at progress 0 to 1 at progress >= 1."""
    progress = torch.as_tensor(progress, dtype=torch.float32)
    return start_fraction + (1. - start_fraction) * progress.clamp(0., 1.)


class RampedPerturbation:
    """Per-environment piecewise-constant random wrench under a ramped envelope.

    Sampling uses a private generator so evaluation conditions can share one
    disturbance sequence (``reseed``) independently of the simulator RNG.
    """

    def __init__(self, cfg: PerturbationCfg, num_envs: int, device, seed: int = 0):
        self.cfg = cfg.validate()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.force_direction = torch.zeros(self.num_envs, 3, device=self.device)
        self.torque_direction = torch.zeros(self.num_envs, 3, device=self.device)
        self.fraction = torch.zeros(self.num_envs, device=self.device)
        self.hold_until = torch.full((self.num_envs,), -1., device=self.device)
        self.generator = torch.Generator(device=self.device)
        self.reseed(seed)

    def reseed(self, seed: int):
        self.generator.manual_seed(int(seed))
        self.hold_until.fill_(-1.)

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.generator, device=self.device)

    def _resample(self, ids):
        n = len(ids)
        if n == 0:
            return
        cfg = self.cfg
        # Force: uniform horizontal heading, bounded vertical component.
        heading = self._rand(n) * 2 * math.pi
        vertical = (self._rand(n) * 2 - 1) * cfg.vertical_force_fraction
        force = torch.stack([torch.cos(heading), torch.sin(heading), vertical], dim=-1)
        self.force_direction[ids] = force / force.norm(dim=-1, keepdim=True)
        # Torque: uniform direction on the sphere.
        torque = torch.randn(n, 3, generator=self.generator, device=self.device)
        self.torque_direction[ids] = torque / torque.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.fraction[ids] = self._rand(n)
        self.hold_until[ids] = self.hold_until[ids] + cfg.hold_min_s + self._rand(n) * (
            cfg.hold_max_s - cfg.hold_min_s)

    def step(self, time, progress, fresh=None):
        """Return world-frame (force, torque, envelope) for the coming control step.

        ``time`` is seconds since reset, ``progress`` the ramp coordinate
        (time/ramp_time or travel/ramp_distance) and ``fresh`` marks
        environments whose episode starts now, forcing a fresh sample.
        """
        time = torch.as_tensor(time, dtype=torch.float32, device=self.device)
        if fresh is not None:
            self.hold_until[fresh] = time[fresh] - 1e-6
        due = time >= self.hold_until
        if bool(due.any()):
            # Holds are measured from the current time, not the stale deadline.
            self.hold_until[due] = time[due]
            self._resample(due.nonzero().flatten())
        envelope = ramp_envelope(progress, self.cfg.start_fraction).to(self.device)
        magnitude = self.fraction * envelope
        force = self.force_direction * (magnitude * self.cfg.max_force_n)[:, None]
        torque = self.torque_direction * (magnitude * self.cfg.max_torque_nm)[:, None]
        return force, torque, envelope


def add_perturbation_arguments(parser):
    group = parser.add_argument_group(
        "perturbation", "Opt-in ramped random base wrench (off by default)")
    group.add_argument("--perturbation", action="store_true",
                       help="Apply a random external base force/torque whose magnitude "
                            "starts small at reset and ramps up through the episode")
    group.add_argument("--perturbation_max_force", type=float, default=25.,
                       help="Peak force magnitude in N once the ramp completes")
    group.add_argument("--perturbation_max_torque", type=float, default=3.,
                       help="Peak torque magnitude in N m once the ramp completes")
    group.add_argument("--perturbation_start_fraction", type=float, default=.10,
                       help="Fraction of the peak applied at the start of an episode")
    group.add_argument("--perturbation_ramp_mode", choices=RAMP_MODES, default="time",
                       help="Ramp with episode time or with forward distance travelled")
    group.add_argument("--perturbation_ramp_time", type=float, default=10.,
                       help="Seconds from reset until the peak is reached (time mode)")
    group.add_argument("--perturbation_ramp_distance", type=float, default=3.,
                       help="Metres of forward travel until the peak is reached (distance mode)")
    group.add_argument("--perturbation_hold", type=float, nargs=2, default=(.10, .40),
                       metavar=("MIN_S", "MAX_S"),
                       help="Each sampled wrench is held for a random duration in this range")
    return group


def perturbation_from_args(args):
    """Build a validated profile from parsed flags, or None when disabled."""
    if not getattr(args, "perturbation", False):
        return None
    return PerturbationCfg(
        max_force_n=float(args.perturbation_max_force),
        max_torque_nm=float(args.perturbation_max_torque),
        start_fraction=float(args.perturbation_start_fraction),
        ramp_mode=str(args.perturbation_ramp_mode),
        ramp_time_s=float(args.perturbation_ramp_time),
        ramp_distance_m=float(args.perturbation_ramp_distance),
        hold_min_s=float(args.perturbation_hold[0]),
        hold_max_s=float(args.perturbation_hold[1]),
    ).validate()


def summarize_outcomes(traces):
    """Per-trial outcome counts and applied-wrench statistics for one condition.

    ``traces`` maps trace names to lists of per-step arrays as recorded by the
    evaluators; only the first (valid) episode of each trial is counted.
    """
    valid = np.stack(traces["valid"]).astype(bool)
    failure = np.stack(traces["failure"]).astype(bool)
    success = np.stack(traces["success"]).astype(bool)
    steps, trials = valid.shape
    last = np.array([int(np.flatnonzero(valid[:, i])[-1]) if valid[:, i].any() else 0
                     for i in range(trials)])
    index = (last, np.arange(trials))
    failed = failure[index]
    succeeded = success[index] & ~failed
    summary = {"trials": int(trials), "failure": int(failed.sum()),
               "success": int(succeeded.sum()),
               "timeout": int(trials - failed.sum() - succeeded.sum()),
               "mean_valid_steps": float(valid.sum(axis=0).mean())}
    if "perturbation_force_w" in traces:
        force = np.linalg.norm(np.stack(traces["perturbation_force_w"]), axis=-1)
        torque = np.linalg.norm(np.stack(traces["perturbation_torque_w"]), axis=-1)
        envelope = np.stack(traces["perturbation_envelope"])
        summary.update({
            "applied_force_n_mean": float(force[valid].mean()),
            "applied_force_n_max": float(force[valid].max()),
            "applied_torque_nm_mean": float(torque[valid].mean()),
            "applied_torque_nm_max": float(torque[valid].max()),
            "envelope_at_start": float(envelope[0].mean()),
            "envelope_at_end_mean": float(envelope[index].mean()),
            "failure_step_envelope_mean": (
                float(envelope[index][failed].mean()) if failed.any() else None),
        })
    return summary
