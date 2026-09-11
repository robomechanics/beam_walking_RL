"""Exploration changes must be explicit and preserve the learned mean network."""
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/beam_walking"))
from beam_walking.experiment.checkpoint_utils import (
    validate_noise_reset, reset_action_std, migrate_heading_observation,
)


class NoiseResetTest(unittest.TestCase):
    def test_only_explicit_training_revision_accepts_reset(self):
        validate_noise_reset("train", True, "model.pt", .3)
        validate_noise_reset("evaluate", False, "model.pt", None)
        for args in [("evaluate", True, "model.pt", .3), ("train", False, "model.pt", .3),
                     ("train", True, None, .3)]:
            with self.assertRaises(ValueError):
                validate_noise_reset(*args)
        for value in [0., -1., float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                validate_noise_reset("train", True, "model.pt", value)

    def test_scalar_and_log_reset_preserve_actor_and_critic(self):
        for representation in ["scalar", "log"]:
            policy = torch.nn.Module()
            policy.actor = torch.nn.Linear(3, 12)
            policy.critic = torch.nn.Linear(3, 1)
            policy.noise_std_type = representation
            initial = torch.full((12,), .09)
            parameter_name = "std" if representation == "scalar" else "log_std"
            setattr(policy, parameter_name, torch.nn.Parameter(
                initial if representation == "scalar" else initial.log()))
            obs = torch.randn(4, 3)
            before_actor = policy.actor(obs).detach().clone()
            before_critic = policy.critic(obs).detach().clone()
            report = reset_action_std(policy, .3)
            self.assertTrue(report["non_noise_parameters_unchanged"])
            torch.testing.assert_close(torch.tensor(report["previous_std"]), torch.full((12,), .09))
            torch.testing.assert_close(torch.tensor(report["applied_std"]), torch.full((12,), .3))
            torch.testing.assert_close(policy.actor(obs), before_actor, rtol=0, atol=0)
            torch.testing.assert_close(policy.critic(obs), before_critic, rtol=0, atol=0)

    def test_heading_input_migration_preserves_old_controller(self):
        class Policy(torch.nn.Module):
            def __init__(self, inputs):
                super().__init__()
                self.actor = torch.nn.Sequential(torch.nn.Linear(inputs, 12), torch.nn.ELU())
                self.critic = torch.nn.Sequential(torch.nn.Linear(inputs, 1), torch.nn.ELU())
                self.std = torch.nn.Parameter(torch.full((12,), .07))

        class RslLikePolicy(Policy):
            def load_state_dict(self, state_dict, strict=True):
                super().load_state_dict(state_dict, strict=strict)
                return True

        old, new = Policy(62), RslLikePolicy(63)
        source = old.state_dict()
        old_obs = torch.randn(5, 62)
        heading = torch.randn(5, 1)
        new_obs = torch.cat([old_obs[:, :58], heading, old_obs[:, 58:]], dim=1)
        report = migrate_heading_observation(new, source)
        # GEMM accumulation order can change after inserting a zero column, so
        # outputs are equivalent to floating-point precision rather than bitwise.
        torch.testing.assert_close(new.actor(new_obs), old.actor(old_obs), rtol=1e-6, atol=2e-7)
        torch.testing.assert_close(new.critic(new_obs), old.critic(old_obs), rtol=1e-6, atol=2e-7)
        self.assertEqual(report["old_input_dim"], 62)
        self.assertEqual(report["new_input_dim"], 63)
        self.assertEqual(report["input_index"], 58)
        self.assertTrue(report["actor_and_critic_inserted_columns_zero"])
        self.assertTrue(report["old_input_columns_preserved"])
        self.assertTrue(report["non_input_parameters_unchanged"])

    def test_heading_input_migration_rejects_wrong_shape(self):
        policy = torch.nn.Module()
        policy.actor = torch.nn.Sequential(torch.nn.Linear(63, 12))
        policy.critic = torch.nn.Sequential(torch.nn.Linear(63, 1))
        source = {key: value.clone() for key, value in policy.state_dict().items()}
        with self.assertRaisesRegex(ValueError, "Cannot insert"):
            migrate_heading_observation(policy, source)
