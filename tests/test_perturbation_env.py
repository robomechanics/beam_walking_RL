"""The perturbation env hook: wrench written before physics, body-frame rotation,
reset handling and trace capture, exercised against a fake Isaac-free base env."""
import math
import sys
import types
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/beam_walking"))

# ``perturbation_env`` imports ``task`` (which needs Isaac Sim) only for
# ``local_body``; provide that function without the simulator.
_fake_task = types.ModuleType("beam_walking.experiment.task")
_fake_task.local_body = lambda env: env.scene["robot"].data.root_pos_w - env.scene.env_origins
sys.modules.setdefault("beam_walking.experiment.task", _fake_task)

from beam_walking.experiment.perturbation import PerturbationCfg  # noqa: E402
from beam_walking.experiment.perturbation_env import perturbed_env_class  # noqa: E402


class FakeComposer:
    def __init__(self):
        self.calls = []
        self.active = False

    def set_forces_and_torques(self, forces=None, torques=None, body_ids=None, **kw):
        self.calls.append((forces.clone(), torques.clone(), list(body_ids)))
        self.active = True


class FakeRobot:
    def __init__(self, n, yaw):
        self.data = types.SimpleNamespace(
            root_pos_w=torch.zeros(n, 3),
            root_quat_w=torch.stack([torch.cos(yaw / 2), torch.zeros(n), torch.zeros(n),
                                     torch.sin(yaw / 2)], dim=-1))
        self.permanent_wrench_composer = FakeComposer()

    def find_bodies(self, name):
        return [7], [name]


class FakeScene:
    def __init__(self, robot):
        self.robot = robot
        self.env_origins = torch.zeros(robot.data.root_pos_w.shape[0], 3)

    def __getitem__(self, key):
        return self.robot


class FakeBase:
    """Minimal stand-in for BeamEnv: only what the mixin touches."""

    def __init__(self, cfg, render_mode=None):
        self.num_envs = cfg.num_envs
        self.device = torch.device("cpu")
        self.step_dt = .02
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.scene = FakeScene(FakeRobot(self.num_envs, cfg.yaw))
        self.capture = True
        self.wrench_at_step = []
        self.reset_next = None

    def step(self, action):
        composer = self.scene["robot"].permanent_wrench_composer
        self.wrench_at_step.append(composer.calls[-1] if composer.calls else None)
        self.transition = {"time": self.episode_length_buf.float() * self.step_dt}
        self.episode_length_buf += 1
        self.scene["robot"].data.root_pos_w[:, 0] += .05
        if self.reset_next is not None:
            self.episode_length_buf[self.reset_next] = 0
            self.scene["robot"].data.root_pos_w[self.reset_next, 0] = 0.
            self.reset_next = None
        return "obs", "reward", "terminated", "timeouts", {}


def make_env(n=4, yaw=0., **cfg_kwargs):
    cfg = types.SimpleNamespace(num_envs=n, seed=5, yaw=torch.full((n,), yaw))
    klass = perturbed_env_class(FakeBase, PerturbationCfg(**cfg_kwargs))
    return klass(cfg, render_mode=None)


class PerturbedEnvTest(unittest.TestCase):

    def test_wrench_is_set_before_physics_with_base_body_and_shapes(self):
        env = make_env(n=6, start_fraction=.2, ramp_time_s=1.)
        self.assertTrue(env.__class__.__name__.startswith("Perturbed"))
        env.step(None)
        forces, torques, body_ids = env.wrench_at_step[0]
        self.assertEqual(body_ids, [7])
        self.assertEqual(forces.shape, (6, 1, 3))
        self.assertEqual(torques.shape, (6, 1, 3))
        self.assertTrue(forces.is_contiguous())
        applied = env.applied_perturbation
        torch.testing.assert_close(applied["envelope"], torch.full((6,), .2))
        self.assertLessEqual(float(applied["force_w"].norm(dim=-1).max()), .2 * 25. + 1e-5)
        self.assertEqual(env.transition["perturbation_force_w"].shape, (6, 3))
        self.assertIn("perturbation_envelope", env.transition)

    def test_world_force_is_rotated_into_body_frame(self):
        env = make_env(n=3, yaw=math.pi / 2)
        env.step(None)
        forces, torques, _ = env.wrench_at_step[0]
        world = env.applied_perturbation["force_w"]
        # Body yawed +90 deg: world +x is body -y, world +y is body +x.
        expected = torch.stack([world[:, 1], -world[:, 0], world[:, 2]], dim=-1)
        torch.testing.assert_close(forces[:, 0], expected, atol=1e-5, rtol=0)
        world_t = env.applied_perturbation["torque_w"]
        expected_t = torch.stack([world_t[:, 1], -world_t[:, 0], world_t[:, 2]], dim=-1)
        torch.testing.assert_close(torques[:, 0], expected_t, atol=1e-5, rtol=0)

    def test_envelope_ramps_with_time_and_restarts_after_reset(self):
        env = make_env(n=4, start_fraction=.1, ramp_time_s=.5, hold_min_s=.02, hold_max_s=.02)
        envelopes = []
        for step in range(30):
            if step == 9:
                env.reset_next = torch.tensor([1])
            env.step(None)
            envelopes.append(env.applied_perturbation["envelope"].clone())
        envelopes = torch.stack(envelopes)
        # Env 0 never resets: 0.1 at t=0, saturates at 1 after 0.5 s (25 steps).
        self.assertAlmostEqual(float(envelopes[0, 0]), .1)
        self.assertAlmostEqual(float(envelopes[25, 0]), 1.)
        self.assertAlmostEqual(float(envelopes[29, 0]), 1.)
        # Env 1 reset after step 9, so its next step is back at the start value.
        self.assertAlmostEqual(float(envelopes[9, 1]), .1 + .9 * 9 * .02 / .5)
        self.assertAlmostEqual(float(envelopes[10, 1]), .1)
        self.assertAlmostEqual(float(envelopes[11, 1]), .1 + .9 * .02 / .5)

    def test_distance_mode_measures_travel_from_episode_start(self):
        env = make_env(n=2, start_fraction=0., ramp_mode="distance", ramp_distance_m=.25)
        for _ in range(6):
            env.step(None)
        # After five completed steps of 5 cm each the sixth step sees 25 cm of travel.
        torch.testing.assert_close(env.applied_perturbation["envelope"], torch.ones(2))
        env.reset_next = torch.tensor([0])
        env.step(None)
        env.step(None)
        self.assertAlmostEqual(float(env.applied_perturbation["envelope"][0]), 0.)
        self.assertAlmostEqual(float(env.applied_perturbation["envelope"][1]), 1.)

    def test_requires_profile(self):
        with self.assertRaises(RuntimeError):
            type("Broken", (perturbed_env_class(FakeBase, PerturbationCfg()).__mro__[1], FakeBase), {})(
                types.SimpleNamespace(num_envs=1, seed=0, yaw=torch.zeros(1)))


if __name__ == "__main__":
    unittest.main()
