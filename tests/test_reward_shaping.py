"""Regression checks for placement masking and finite-tick clearance incentives."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/beam_walking"))
from beam_walking.experiment.protocol import (
    width_score, clearance_score, leg_phase, discrete_stance_fraction, contact_score,
    duty_warped_phase, straight_motion_cost, heading_stabilization_cost,
    speed_score, planar_speed_score, world_to_body, normalized_gait_command,
    sample_training_commands, fore_aft_target, FOUNDATION_CONTROL_STEPS,
    CORE_CONTROL_STEPS, SPEED_ANCHORS, DUTY_ANCHORS, STEP_WIDTH_ANCHORS,
    GAIT_OFFSETS, WALK_TOUCHDOWN_ORDER, validate_scientific_gait_duties,
)


class RewardShapingTest(unittest.TestCase):


    def test_paper_walk_offsets_touchdown_order_and_support_count(self):
        self.assertEqual(GAIT_OFFSETS[1], (0., .75, .50, .25))
        ticks = torch.arange(24)
        period = torch.full_like(ticks, 24)
        gait = torch.ones_like(ticks)
        phase = leg_phase(ticks, period, gait)
        self.assertTrue(torch.all((phase < .75).sum(dim=1) == 3))
        wrap_ticks = []
        for leg in range(4):
            wraps = ((phase[1:, leg] < phase[:-1, leg]).nonzero().flatten() + 1)
            wrap_ticks.append(int(wraps[0]) if len(wraps) else 24)
        after_phase_zero = tuple(
            ("FL", "FR", "RL", "RR")[index]
            for index in sorted(range(4), key=lambda index: wrap_ticks[index])
            if index != 0)
        self.assertEqual(("FL",) + after_phase_zero, WALK_TOUCHDOWN_ORDER)

    def test_scientific_walk_rejects_low_duty_factor(self):
        validate_scientific_gait_duties("trot", [.5, .625, .75])
        validate_scientific_gait_duties("walk", [.75])
        with self.assertRaisesRegex(ValueError, "0.75"):
            validate_scientific_gait_duties("walk", [.625, .75])

    def test_normalized_commands_and_explicit_gait_encoding(self):
        values = torch.tensor([[.25, .50, .10, .36, 0.], [.40, .75, .50, .54, 1.]])
        encoded = normalized_gait_command(values)
        self.assertEqual(encoded.shape, (2, 6))
        torch.testing.assert_close(encoded[0, :4], -torch.ones(4))
        torch.testing.assert_close(encoded[1, :4], torch.ones(4))
        torch.testing.assert_close(encoded[:, 4:], torch.tensor([[1., 0.], [0., 1.]]))

    def test_training_sampler_stages_and_anchor_coverage(self):
        torch.manual_seed(7)
        foundation, ticks, stage = sample_training_commands(120, FOUNDATION_CONTROL_STEPS - 1)
        self.assertEqual(stage, 0)
        torch.testing.assert_close(
            foundation[:, [0, 2, 3]],
            torch.tensor([.30, .30, .48]).expand(120, -1))
        self.assertEqual(torch.bincount(foundation[:, 4].long()).tolist(), [60, 60])
        self.assertTrue(torch.all(foundation[foundation[:, 4] == 0, 1] == .625))
        self.assertTrue(torch.all(foundation[foundation[:, 4] == 1, 1] == .75))
        core, ticks, stage = sample_training_commands(120, FOUNDATION_CONTROL_STEPS)
        self.assertEqual(stage, 1)
        self.assertEqual({round(value, 3) for value in core[:, 0].tolist()}, set(SPEED_ANCHORS))
        self.assertEqual({round(value, 3) for value in core[:, 1].tolist()}, set(DUTY_ANCHORS))
        self.assertEqual({round(value, 3) for value in core[:, 2].tolist()}, set(STEP_WIDTH_ANCHORS))
        self.assertEqual(torch.bincount(core[:, 4].long()).tolist(), [60, 60])
        self.assertTrue(torch.all(core[core[:, 4] == 1, 1] == .75))
        self.assertEqual(
            {round(value, 3) for value in core[core[:, 4] == 0, 1].tolist()},
            set(DUTY_ANCHORS))
        self.assertTrue(torch.all(ticks == 24))
        coverage, ticks, stage = sample_training_commands(400, CORE_CONTROL_STEPS)
        self.assertEqual(stage, 2)
        self.assertTrue(torch.all((coverage[:, 0] >= .25) & (coverage[:, 0] <= .40)))
        self.assertTrue(torch.all((coverage[:, 1] >= .50) & (coverage[:, 1] <= .75)))
        self.assertTrue(torch.all((coverage[:, 2] >= .10) & (coverage[:, 2] <= .50)))
        self.assertTrue(torch.all((ticks >= 18) & (ticks <= 27)))
        walk = coverage[:, 4] == 1
        self.assertTrue(torch.all(coverage[walk, 1] == .75))
        self.assertTrue(torch.all(ticks[walk] >= 20))

    def test_fore_aft_target_advances_one_step_length(self):
        phase = torch.tensor([[0., .25, .50, .75]])
        duty = torch.tensor([.50])
        target = fore_aft_target(phase, duty, torch.tensor([.30]), torch.tensor([.50]))
        hip = torch.tensor([.1934, .1934, -.1934, -.1934])
        expected = hip + torch.tensor([.0375, 0., -.0375, 0.])
        torch.testing.assert_close(target[0], expected)
    def test_straight_cost_is_bounded_symmetric_and_rewards_zero_drift(self):
        yaw = torch.tensor([0., .1, .5, 5., 1e10])
        costs = straight_motion_cost(yaw)
        self.assertEqual(costs[0].item(), 0.)
        self.assertTrue(torch.all(costs[1:] >= costs[:-1]))
        self.assertTrue(torch.isfinite(costs).all())
        self.assertLessEqual(costs.max().item(), 1.0)
        torch.testing.assert_close(costs, straight_motion_cost(-yaw))

    def test_straight_cost_resolves_measured_slow_turn(self):
        # R5 accumulated about 0.03 rad/s of signed yaw.  The revised cost must
        # make that persistent bias visible while remaining bounded at extremes.
        zero = straight_motion_cost(torch.tensor([0.]))
        slow_turn = straight_motion_cost(torch.tensor([.03]))
        self.assertGreater(slow_turn.item() - zero.item(), .02)

    def test_heading_cost_is_bounded_symmetric_and_resolves_visible_turn(self):
        headings = torch.tensor([0., .0873, .2, 1., 1e10])
        costs = heading_stabilization_cost(headings)
        self.assertEqual(costs[0].item(), 0.)
        self.assertGreater(costs[1].item(), .3)
        self.assertTrue(torch.all(costs[1:] >= costs[:-1]))
        self.assertLessEqual(costs.max().item(), 2.)
        torch.testing.assert_close(costs, heading_stabilization_cost(-headings))

    def test_world_lateral_motion_is_penalized_even_with_zero_yaw_rate(self):
        # A yawed controller can have zero body-lateral velocity while its
        # forward motion carries it sideways in the fixed course frame.
        speed = torch.tensor([.3])
        heading = torch.tensor([.1])
        world_lateral = speed * torch.sin(heading)
        straight = planar_speed_score(torch.zeros(1), torch.zeros(1))
        diagonal = planar_speed_score(torch.zeros(1), world_lateral)
        self.assertGreater(straight.item() - diagonal.item(), .08)

    def test_planar_speed_score_prefers_world_forward_course_velocity(self):
        command = .3
        correct = planar_speed_score(torch.tensor([command - command]), torch.tensor([0.]))
        diagonal = planar_speed_score(torch.tensor([.28 - command]), torch.tensor([.08]))
        stopped = planar_speed_score(torch.tensor([0. - command]), torch.tensor([0.]))
        self.assertEqual(correct.item(), 1.)
        self.assertGreater(correct.item(), diagonal.item())
        self.assertGreater(diagonal.item(), stopped.item())

    def test_duty_warped_phase_aligns_liftoff_and_is_invertible(self):
        duties = torch.tensor([.50, .625, .75])
        phase = torch.stack([torch.tensor([0., .25, .50, .75]),
                             torch.tensor([0., .3125, .625, .8125]),
                             torch.tensor([0., .375, .75, .875])])
        warped = duty_warped_phase(phase, duties)
        torch.testing.assert_close(warped[:, 0], torch.zeros(3))
        torch.testing.assert_close(warped[:, 1], torch.full((3,), .25))
        torch.testing.assert_close(warped[:, 2], torch.full((3,), .5))
        torch.testing.assert_close(warped[:, 3], torch.full((3,), .75))
        duty = duties[:, None]
        recovered = torch.where(warped < .5, 2 * warped * duty,
                                duty + 2 * (warped - .5) * (1 - duty))
        torch.testing.assert_close(recovered, phase)

    def test_duty_warp_does_not_change_raw_contact_schedule(self):
        ticks = torch.arange(24)
        periods = torch.full_like(ticks, 24)
        for gid in [0, 1]:
            phase = leg_phase(ticks, periods, torch.full_like(ticks, gid))
            for df in [.5, .625, .75]:
                duty = torch.full((24,), df)
                desired_before = phase < duty[:, None]
                duty_warped_phase(phase, duty)
                desired_after = phase < duty[:, None]
                torch.testing.assert_close(desired_after, desired_before)

    def test_each_foot_contributes_and_width_error_is_discriminated(self):
        # A nominal .35 m stance must score below correct tucking to .10 m.
        errors = torch.tensor([[0., 0., 0., 0.], [.125, -.125, .125, -.125],
                               [.025, -.025, .025, -.025]])
        scores = width_score(errors)
        self.assertEqual(scores[0].item(), 1.)
        self.assertLess(scores[1].item(), scores[2].item())
        for leg in range(4):
            single_bad_foot = torch.zeros(1, 4)
            single_bad_foot[0, leg] = .5
            self.assertLess(width_score(single_bad_foot).item(), .01)
        # Losing support cannot erase errors: this reward has no contact mask.
        self.assertLess(scores[1].item(), .25)

    def test_width_reward_does_not_favor_sacrificing_two_feet(self):
        balanced = torch.full((1, 4), .10)
        asymmetric = torch.tensor([[0., 0., .20, .20]])
        # Same mean absolute error; two badly misplaced feet must not improve
        # the score. Mean-of-per-foot-Gaussians violates this inequality.
        self.assertLess(width_score(asymmetric).item(), width_score(balanced).item())

    def test_body_width_reward_is_invariant_to_world_rotation(self):
        feet = torch.tensor([[[.1934, .15, -.295], [.1934, -.15, -.295],
                              [-.1934, .15, -.295], [-.1934, -.15, -.295]]])
        target = torch.tensor([[.05, -.05, .05, -.05]])
        reference = width_score(feet[:, :, 1] - target)
        for raw in [[1., 0., 0., 0.], [.9848, 0., 0., .1736],
                    [.7, .2, -.1, .6], [0., 0., 0., 1.]]:
            quat = torch.tensor([raw])
            quat /= quat.norm(dim=1, keepdim=True)
            inverse = quat.clone()
            inverse[:, 1:] *= -1
            world = world_to_body(feet, inverse)
            recovered = world_to_body(world, quat)
            torch.testing.assert_close(recovered, feet)
            torch.testing.assert_close(width_score(recovered[:, :, 1] - target), reference)

    def test_clearance_has_equal_cycle_scale_for_both_gaits(self):
        for count in [18, 24, 27]:
            for gid in [0, 1]:
                ticks = torch.arange(count)
                periods = torch.full_like(ticks, count)
                gait = torch.full_like(ticks, gid)
                duty = torch.full((count,), min(.75, 1 - 5 / count))
                desired = leg_phase(ticks, periods, gait) < duty[:, None]
                realized = discrete_stance_fraction(duty, periods, gait)
                perfect = clearance_score(torch.zeros(count, 4), desired, realized)
                poor = clearance_score(torch.full((count, 4), .08), desired, realized)
                self.assertAlmostEqual(perfect.mean().item(), 1., places=6)
                self.assertLess(poor.mean().item(), .02)
                self.assertTrue(torch.isfinite(perfect).all())
                self.assertGreaterEqual(poor.min().item(), 0.)

    def test_speed_reward_prefers_command_over_overspeed_and_standing(self):
        velocities = torch.tensor([0., .2, .3, .4, .6])
        scores = speed_score(velocities - .3)
        self.assertEqual(scores.argmax().item(), 2)
        self.assertLess(scores[0].item(), .001)
        self.assertLess(scores[3].item(), .4)
        torch.testing.assert_close(scores[1], scores[3])

    def test_following_df_commands_beats_a_fixed_median_schedule(self):
        for count in range(18, 28):
            for gid in [0, 1]:
                ticks = torch.arange(count)
                periods = torch.full_like(ticks, count)
                gait = torch.full_like(ticks, gid)
                phase = leg_phase(ticks, periods, gait)
                median_contact = phase < .625
                for df in [.5, min(.75, 1 - 5 / count)]:
                    duty = torch.full((count,), df)
                    desired = phase < duty[:, None]
                    realized = discrete_stance_fraction(duty, periods, gait)
                    ideal = contact_score(desired, desired, realized).mean()
                    compromise = contact_score(median_contact, desired, realized).mean()
                    self.assertGreater(ideal.item(), compromise.item())


if __name__ == "__main__":
    unittest.main()
