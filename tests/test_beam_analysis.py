"""Synthetic offline tests only; fixtures never enter measured results."""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/beam_walking"))
from beam_walking.experiment.analysis import (
    LEGACY_NOMINAL_SCHEMA, NOMINAL_SCHEMA, V3_NOMINAL_SCHEMA, analyze,
    combined_cell_acceptance, combined_cell_acceptance_v4, complete_cycle_df,
    include_post_step_video_frame, nominal_archive_payload,
    oldest_first_force_history, trial_rows,
    schedule_centered_topology_fraction, topology_cycle_fraction,
    validate_manifest, validate_matched_study_resets,
    wilson,
)
from beam_walking.experiment.protocol import leg_phase, contact_score, discrete_stance_fraction, GAIT_OFFSETS


def synthetic_run(directory, period=.48, gait="trot", widths=(.1, .5), dfs=(.5, .625)):
    """Known straight motion/contact signals, not physical robot predictions."""
    gid = ("trot", "walk").index(gait)
    ticks = np.arange(1, 151)
    n, steps = 3, len(ticks)
    phase = leg_phase(torch.tensor(ticks), torch.full((steps,), round(period / .02)),
                      torch.full((steps,), gid)).numpy()
    seeds = np.array([10000, 10001, 10002])
    manifest = {"seeds": seeds.tolist(), "period": period, "gait": gait,
                "source_sha256": "synthetic-source", "checkpoint_sha256": "synthetic-checkpoint",
                "expected_conditions": []}
    for width in widths:
        for df in dfs:
            name = f"trial_s{width:.3f}_d{df:.3f}.npz"
            desired = np.broadcast_to((phase < df)[:, None, :], (steps, n, 4)).copy()
            body = np.zeros((steps, n, 3))
            body[:, :, 0] = -.65 + ticks[:, None] * .02 * .3
            body[:, :, 2] = .32
            feet = np.repeat(body[:, :, None, :], 4, axis=2)
            feet[:, :, :, 1] += np.array([1, -1, 1, -1]) * width / 2
            commands = np.broadcast_to([.3, df, width, period, gid], (steps, n, 5)).copy()
            flags = np.zeros((steps, n), dtype=bool)
            done, failed, success = flags.copy(), flags.copy(), flags.copy()
            done[-1] = True
            failed[-1, 1] = True
            success[-1, 0] = True
            initial_root = np.zeros((n, 13))
            initial_root[:, 0] = -.65
            plan = np.zeros((n, 6))
            plan[:, 2] = 5.
            plan[:, 5] = .2
            np.savez_compressed(directory / name, period=period, gait=gait, step_width=width,
                df=df, disturbed=False, control_dt=.02, phase_offsets=GAIT_OFFSETS[gid],
                source_sha256=manifest["source_sha256"], checkpoint_sha256=manifest["checkpoint_sha256"],
                valid=np.ones((steps, n), dtype=bool), done=done, failure=failed, success=success,
                time=np.broadcast_to(ticks[:, None] * .02, (steps, n)), phase=np.repeat(phase[:, None], n, axis=1),
                contacts=desired, desired=desired, schedule_duty=np.mean(desired[:round(period/.02)], axis=0),
                commands=commands, body=body, feet=feet, speed=np.full((steps, n), .3),
                forward_velocity=np.full((steps, n), .3), force=np.zeros((steps, n, 3)),
                root_quat=np.broadcast_to(
                    [1., 0., 0., 0.], (steps, n, 4)).copy(),
                world_lateral_velocity=np.full((steps, n), -.04),
                body_lateral_velocity=np.full((steps, n), .02),
                body_yaw_rate=np.full((steps, n), -.03),
                seeds=seeds, push_plan=plan, planned_force=np.zeros((n, 3)),
                initial_root=initial_root, initial_joints=np.zeros((n, 12)))
            manifest["expected_conditions"].append({"filename": name, "step_width": width,
                                                     "df": df, "disturbed": False})
    (directory / "evaluation_manifest.json").write_text(json.dumps(manifest))
    return manifest


def declare_body_run(directory, manifest, yaw=.0, roll=.0):
    """Rotate the same synthetic body-relative stance into world coordinates."""
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_euler("zyx", [yaw, 0., roll])
    xyzw = rotation.as_quat()
    for cell in manifest["expected_conditions"]:
        path = directory / cell["filename"]
        with np.load(path) as data:
            payload = {key: data[key] for key in data.files}
        relative = payload["feet"] - payload["body"][..., None, :]
        relative[..., 0] = np.array([.1934, .1934, -.1934, -.1934])
        relative[..., 2] = -.295
        payload["feet_body"] = relative
        payload["feet"] = np.einsum("ij,...j->...i", rotation.as_matrix(), relative) + payload["body"][..., None, :]
        payload["root_quat"] = np.broadcast_to(xyzw[[3, 0, 1, 2]], payload["valid"].shape + (4,)).copy()
        payload["step_width_frame"] = "body"
        np.savez_compressed(path, **payload)
    manifest["step_width_frame"] = "body"
    (directory / "evaluation_manifest.json").write_text(json.dumps(manifest))


class MetricsTest(unittest.TestCase):
    def test_force_history_is_archived_oldest_first(self):
        newest_first = np.asarray([[[4.], [3.], [2.], [1.]]])
        np.testing.assert_array_equal(
            oldest_first_force_history(newest_first),
            np.asarray([[[1.], [2.], [3.], [4.]]]))
        with self.assertRaisesRegex(ValueError, "history axis"):
            oldest_first_force_history(np.ones(4))

    def test_study_reset_matching_rejects_stance_or_joint_changes(self):
        identity = {
            "reset_plan": np.zeros((3, 2)),
            "initial_root": np.zeros((3, 13)),
            "initial_joints": np.zeros((3, 12)),
            "stance_start": np.asarray([True, False, False]),
        }
        validate_matched_study_resets(identity, {
            key: value.copy() for key, value in identity.items()})
        for key in ("initial_joints", "stance_start"):
            changed = {name: value.copy() for name, value in identity.items()}
            changed[key].flat[0] = not changed[key].flat[0] if key == "stance_start" else 1.
            with self.assertRaisesRegex(ValueError, key):
                validate_matched_study_resets(identity, changed)

    def test_combined_cell_rejects_sub90_pooled_topology(self):
        record = {
            "compliance_screen_pass": True,
            "force_df_span_trial_fraction": .95,
            "force_robust_trial_fraction": .85,
            **{f"force{int(value)}n_pooled_topology_fraction": .95
               for value in (2., 5., 10.)},
        }
        self.assertTrue(combined_cell_acceptance(record, [True, True, True]))
        record["force10n_pooled_topology_fraction"] = .89
        self.assertFalse(combined_cell_acceptance(record, [True, True, True]))

    def test_v4_primary_gate_does_not_multiply_sensitivity_thresholds(self):
        record = {
            "compliance_screen_pass": True,
            "force5n_compliance_screen_pass": True,
            "force5n_topology_screen_pass": True,
            # These diagnostics deliberately fail.  Five newtons is the
            # policy's operational contact definition and the V4 primary gate.
            "force2n_threshold_screen_pass": False,
            "force10n_threshold_screen_pass": False,
        }
        self.assertTrue(combined_cell_acceptance_v4(record))
        for key in (
                "compliance_screen_pass", "force5n_compliance_screen_pass",
                "force5n_topology_screen_pass"):
            changed = dict(record)
            changed[key] = False
            self.assertFalse(combined_cell_acceptance_v4(changed))

    def test_nominal_archive_separates_command_and_measured_speed(self):
        trace = np.asarray([.27, .31], dtype=np.float32)
        payload = nominal_archive_payload(
            {"speed": [trace], "valid": [np.ones(2, dtype=bool)]},
            {"command_speed": .30, "period": .48})
        self.assertNotIn("speed", payload)
        np.testing.assert_array_equal(
            payload["body_forward_velocity"], trace[None])
        self.assertEqual(payload["command_speed"], .30)
        with self.assertRaisesRegex(ValueError, "collision"):
            nominal_archive_payload(
                {"speed": [trace]}, {"body_forward_velocity": trace})

    def test_v2_manifest_provenance_and_speed_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = synthetic_run(directory, widths=(.3,), dfs=(.625,))
            training_bytes = b'{"synthetic": true}'
            (directory / "training_provenance.json").write_bytes(training_bytes)
            training_hash = hashlib.sha256(training_bytes).hexdigest()
            manifest.update({
                "schema": LEGACY_NOMINAL_SCHEMA, "speed": .30,
                "terrain": "flat_ground", "external_pushes": False,
                "split": "validation", "condition_reset_seed": 3010000,
                "evaluator_sha256": "synthetic-evaluator",
                "training_provenance_sha256": training_hash,
            })
            path = directory / manifest["expected_conditions"][0]["filename"]
            with np.load(path) as data:
                payload = {key: data[key] for key in data.files}
            payload["body_forward_velocity"] = payload.pop("speed")
            payload["reset_plan"] = payload.pop("push_plan")
            for key in ("force", "planned_force"):
                payload.pop(key)
            payload.update({
                "schema": LEGACY_NOMINAL_SCHEMA, "command_speed": .30,
                "terrain": "flat_ground", "external_pushes": False,
                "split": "validation", "condition_reset_seed": 3010000,
                "evaluator_sha256": "synthetic-evaluator",
                "training_provenance_sha256": training_hash,
            })
            np.savez_compressed(path, **payload)
            complete = {
                "complete": True, "schema": LEGACY_NOMINAL_SCHEMA,
                "conditions": 1, "trials_per_condition": 3,
                "source_sha256": manifest["source_sha256"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "evaluator_sha256": manifest["evaluator_sha256"],
                "training_provenance_sha256": training_hash,
            }
            (directory / "evaluation_manifest.json").write_text(json.dumps(manifest))
            (directory / "evaluation_complete.json").write_text(json.dumps(complete))
            self.assertEqual(validate_manifest(directory), [path])
            rows = trial_rows(path)
            self.assertTrue(all(np.isclose(row["mean_speed"], .30) for row in rows))

            payload["command_speed"] = .31
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, "speed"):
                validate_manifest(directory)
            payload["command_speed"] = .30

            payload["force"] = np.zeros((150, 3, 3))
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, "push fields"):
                validate_manifest(directory)
            payload.pop("force")
            np.savez_compressed(path, **payload)

            manifest["evaluator_sha256"] = "tampered"
            (directory / "evaluation_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "completion marker"):
                validate_manifest(directory)
            manifest["evaluator_sha256"] = "synthetic-evaluator"
            (directory / "evaluation_manifest.json").write_text(json.dumps(manifest))

            complete["conditions"] = 2
            (directory / "evaluation_complete.json").write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError, "completion marker"):
                validate_manifest(directory)

    def test_v4_force_archive_identity_and_terminal_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = synthetic_run(directory, widths=(.3,), dfs=(.625,))
            path = directory / manifest["expected_conditions"][0]["filename"]
            training_bytes = b'{"synthetic": true}'
            (directory / "training_provenance.json").write_bytes(training_bytes)
            training_hash = hashlib.sha256(training_bytes).hexdigest()
            force_identity = {
                "force_instrumented": True,
                "force_field": "foot_force_norm_200hz", "force_units": "N",
                "force_sample_dt": .005, "force_samples_per_control": 4,
                "force_history_order": "oldest_to_newest",
                "force_leg_order": ["FL", "FR", "RL", "RR"],
                "contact_thresholds_n": [2., 5., 10.],
                "policy_contact_threshold_n": 5.,
                "terminal_force_capture": "recorder_pre_reset",
                "scientific_role": "command_fidelity_prerequisite",
                "not_closed_loop_chi_evidence": True,
                "topology_definition":
                    "schedule_centered_unique_circular_events_v1",
                "topology_event_tolerance_s": .02,
                "topology_primary_threshold_n": 5.,
                "topology_gate_aggregation":
                    "mean_trial_cycle_fraction",
                "topology_sensitivity_thresholds_are_diagnostic": True,
            }
            manifest.update({
                "schema": NOMINAL_SCHEMA, "speed": .30,
                "terrain": "flat_ground", "external_pushes": False,
                "split": "validation", "condition_reset_seed": 3010000,
                "evaluator_sha256": "synthetic-evaluator",
                "training_provenance_sha256": training_hash,
                **force_identity,
            })
            with np.load(path) as data:
                payload = {key: data[key] for key in data.files}
            payload["body_forward_velocity"] = payload.pop("speed")
            payload["reset_plan"] = payload.pop("push_plan")
            payload.pop("force")
            payload.pop("planned_force")
            valid, done = payload["valid"], payload["done"]
            payload["foot_force_norm_200hz"] = (
                np.repeat(payload["contacts"][:, :, None, :], 4, axis=2)
                * np.asarray([3., 7., 12., 12.])[None, None, :, None]
            ).astype(np.float32)
            payload["terminal_force_preserved"] = valid & done
            payload.update({
                "schema": NOMINAL_SCHEMA, "command_speed": .30,
                "terrain": "flat_ground", "external_pushes": False,
                "split": "validation", "condition_reset_seed": 3010000,
                "evaluator_sha256": "synthetic-evaluator",
                "training_provenance_sha256": training_hash,
                **{key: np.asarray(value) for key, value in force_identity.items()},
            })
            np.savez_compressed(path, **payload)
            complete = {
                "complete": True, "schema": NOMINAL_SCHEMA,
                "conditions": 1, "trials_per_condition": 3,
                "source_sha256": manifest["source_sha256"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "evaluator_sha256": manifest["evaluator_sha256"],
                "training_provenance_sha256": training_hash,
            }
            (directory / "evaluation_manifest.json").write_text(json.dumps(manifest))
            (directory / "evaluation_complete.json").write_text(json.dumps(complete))
            self.assertEqual(validate_manifest(directory), [path])
            rows = trial_rows(path)
            for row in rows:
                self.assertGreater(row["force2n_df_FL"], row["force10n_df_FL"])
            with contextlib.redirect_stdout(io.StringIO()):
                summary = analyze(directory)
            self.assertIn("combined_cell_pass", summary)
            self.assertFalse(bool(summary.combined_cell_pass.iloc[0]))
            self.assertTrue((directory / "force_threshold_summary.csv").is_file())
            self.assertTrue((directory / "endpoint_success_diagnostics.json").is_file())

            # V3 archives retain their original fixed-bin interpretation and
            # remain readable; V4 fields are neither required nor inferred.
            topology_fields = [
                "topology_definition", "topology_event_tolerance_s",
                "topology_primary_threshold_n", "topology_gate_aggregation",
                "topology_sensitivity_thresholds_are_diagnostic",
            ]
            manifest["schema"] = V3_NOMINAL_SCHEMA
            payload["schema"] = np.asarray(V3_NOMINAL_SCHEMA)
            complete["schema"] = V3_NOMINAL_SCHEMA
            for key in topology_fields:
                manifest.pop(key)
                payload.pop(key)
            np.savez_compressed(path, **payload)
            (directory / "evaluation_manifest.json").write_text(
                json.dumps(manifest))
            (directory / "evaluation_complete.json").write_text(
                json.dumps(complete))
            self.assertEqual(validate_manifest(directory), [path])
            self.assertEqual(len(trial_rows(path)), 3)

            payload["terminal_force_preserved"][-1, 0] = False
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, "Terminal force preservation"):
                validate_manifest(directory)
            payload["terminal_force_preserved"][-1, 0] = True
            payload["foot_force_norm_200hz"][-1, 0, -1, 0] = 0.
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, "reconstruction"):
                validate_manifest(directory)

    def test_video_frame_gate_excludes_terminal_reset_frame(self):
        self.assertTrue(include_post_step_video_frame(0, True, False))
        self.assertFalse(include_post_step_video_frame(1, True, False))
        self.assertFalse(include_post_step_video_frame(2, True, True))
        self.assertFalse(include_post_step_video_frame(2, False, False))
        with self.assertRaisesRegex(ValueError, "stride"):
            include_post_step_video_frame(0, True, False, stride=0)

    def test_complete_cycles_both_gaits_and_short_periods(self):
        for period in [.36, .48, .54]:
            count = round(period / .02)
            ticks = torch.arange(1, count * 8)
            for gait in [0, 1]:
                phase = leg_phase(ticks, torch.full_like(ticks, count), torch.full_like(ticks, gait)).numpy()
                for df in [.5, .625, .75]:
                    desired = phase < df
                    for leg, offset in enumerate(GAIT_OFFSETS[gait]):
                        values = complete_cycle_df(ticks.numpy() * .02, desired[:, leg], offset, period=period)
                        self.assertGreater(len(values), 4)
                        expected = desired[:count, leg].mean()
                        np.testing.assert_allclose(values, expected, atol=1e-7)

    def test_high_rate_topology_rejects_contact_chatter(self):
        count = round(.48 / .005)
        ticks = torch.arange(1, count * 6 + 1)
        for gait_id, gait in enumerate(("trot", "walk")):
            phase = leg_phase(
                ticks, torch.full_like(ticks, count),
                torch.full_like(ticks, gait_id)).numpy()
            reference = phase < (.625 if gait == "trot" else .75)
            fraction, good, eligible = topology_cycle_fraction(
                ticks.numpy() * .005, reference, reference, gait)
            self.assertEqual(fraction, 1.)
            self.assertEqual(good, eligible)
            chatter = reference.copy()
            chatter[count * 2 + 10, 0] = ~chatter[count * 2 + 10, 0]
            fraction, _, _ = topology_cycle_fraction(
                ticks.numpy() * .005, chatter, reference, gait)
            self.assertLess(fraction, 1.)

    def test_schedule_centered_topology_owns_boundary_jitter(self):
        count = 96
        ticks = torch.arange(1, count * 6 + 1)
        phase = leg_phase(
            ticks, torch.full_like(ticks, count), torch.zeros_like(ticks)
        ).numpy()
        desired = phase < .5
        candidate = desired.copy()
        # Advance one scheduled FL rise from tick 385 to tick 381.  It belongs
        # to that scheduled event, even though it crosses the fixed cycle bin.
        candidate[380:384, 0] = True
        fraction, good, eligible = schedule_centered_topology_fraction(
            ticks.numpy() * .005, candidate, desired, "trot")
        self.assertEqual(fraction, 1.)
        self.assertEqual(good, eligible)

    def test_schedule_centered_topology_rejects_duplicate_and_trot_desync(self):
        count = 96
        ticks = torch.arange(1, count * 6 + 1)
        desired = leg_phase(
            ticks, torch.full_like(ticks, count), torch.zeros_like(ticks)
        ).numpy() < .5
        chatter = desired.copy()
        chatter[count * 3 + 20, 0] = ~chatter[count * 3 + 20, 0]
        self.assertLess(schedule_centered_topology_fraction(
            ticks.numpy() * .005, chatter, desired, "trot")[0], 1.)
        desynchronized = desired.copy()
        # Each diagonal touchdown is within 20 ms of schedule, while the two
        # legs are 40 ms apart and must fail trot synchrony.
        desynchronized[count * 3 - 4:count * 3, 0] = True
        desynchronized[count * 3:count * 3 + 4, 3] = False
        self.assertLess(schedule_centered_topology_fraction(
            ticks.numpy() * .005, desynchronized, desired, "trot")[0], 1.)

    def test_schedule_centered_topology_walk_order_and_circular_edges(self):
        count = 96
        ticks = torch.arange(1, count + 1)
        desired_walk = leg_phase(
            ticks, torch.full_like(ticks, count), torch.ones_like(ticks)
        ).numpy() < .75
        circular = desired_walk.copy()
        circular[:, 0] = np.roll(circular[:, 0], -4)
        fraction, good, eligible = schedule_centered_topology_fraction(
            ticks.numpy() * .005, circular, desired_walk, "walk",
            initial_contacts=circular[-1], initial_desired=desired_walk[-1],
            circular=True)
        self.assertEqual((fraction, good, eligible), (1., 1, 1))
        swapped = desired_walk[:, [1, 0, 2, 3]]
        fraction, _, _ = schedule_centered_topology_fraction(
            ticks.numpy() * .005, swapped, desired_walk, "walk",
            initial_contacts=swapped[-1], initial_desired=desired_walk[-1],
            circular=True)
        self.assertEqual(fraction, 0.)

    def test_schedule_centered_topology_rejects_partial_and_missing_ticks(self):
        ticks = np.arange(1, 96 * 2)
        desired = leg_phase(
            torch.tensor(ticks), torch.full((len(ticks),), 96),
            torch.zeros(len(ticks), dtype=torch.long)).numpy() < .5
        self.assertTrue(np.isnan(schedule_centered_topology_fraction(
            ticks * .005, desired, desired, "trot")[0]))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            schedule_centered_topology_fraction(
                np.delete(ticks, 20) * .005, np.delete(desired, 20, axis=0),
                np.delete(desired, 20, axis=0), "trot")

    def test_terminal_boundary_and_missing_tick(self):
        ticks = np.arange(25, 48)
        self.assertEqual(complete_cycle_df(ticks * .02, np.ones(len(ticks)), 0), [])
        ticks = np.arange(25, 49)
        self.assertEqual(complete_cycle_df(ticks * .02, np.ones(len(ticks)), 0), [1.])
        ticks = np.delete(ticks, 12)
        self.assertEqual(complete_cycle_df(ticks * .02, np.ones(len(ticks)), 0), [])

    def test_fractional_walk_cycle_needs_closing_boundary(self):
        # At 27 ticks and quarter offset, first settled cycle is (47.25,74.25].
        ticks = np.arange(28, 75)
        self.assertEqual(complete_cycle_df(ticks * .02, np.ones(len(ticks)), .25, period=.54), [])
        ticks = np.arange(28, 76)
        self.assertEqual(complete_cycle_df(ticks * .02, np.ones(len(ticks)), .25, period=.54), [1.])

    def test_reward_neutrality_with_discrete_schedules(self):
        for count in [18, 24, 27]:
            ticks = torch.arange(count)
            for gait in [0, 1]:
                ids, periods = torch.full_like(ticks, gait), torch.full_like(ticks, count)
                for df in [.5, .625, .75]:
                    duty = torch.full((count,), df)
                    phase = leg_phase(ticks, periods, ids)
                    desired = phase < duty[:, None]
                    realized = discrete_stance_fraction(duty, periods, ids)
                    self.assertAlmostEqual(contact_score(torch.ones_like(desired), desired, realized).mean().item(), .5, places=6)
                    self.assertAlmostEqual(contact_score(desired, desired, realized).mean().item(), 1., places=6)

    def test_wilson_extremes(self):
        self.assertAlmostEqual(wilson(0, 64)[0], 0.)
        self.assertGreater(wilson(0, 64)[1], .05)
        self.assertAlmostEqual(wilson(64, 64)[1], 1.)
        self.assertLess(wilson(64, 64)[0], .95)

    def test_raw_metrics_both_gaits_and_periods(self):
        for period in [.36, .48, .54]:
            for gait in ["trot", "walk"]:
                with tempfile.TemporaryDirectory(prefix="synthetic_flat_metrics_") as tmp:
                    directory = Path(tmp)
                    synthetic_run(directory, period, gait, widths=(.5,), dfs=(.5,))
                    rows = trial_rows(validate_manifest(directory)[0])
                    self.assertEqual([r["success"] for r in rows], [1, 0, 0])
                    self.assertEqual([r["failure"] for r in rows], [0, 1, 0])
                    self.assertEqual([r["timeout"] for r in rows], [0, 0, 1])
                    for row in rows:
                        self.assertAlmostEqual(row["achieved_width"], .5)
                        self.assertEqual(row["step_width_frame"], "world")
                        self.assertEqual(row["achieved_width"], row["world_achieved_width"])
                        self.assertAlmostEqual(row["body_achieved_width"], .5)
                        self.assertAlmostEqual(row["forward_speed"], .3)
                        self.assertAlmostEqual(row["settled_body_forward_speed"], .3)
                        self.assertEqual(row["lateral_rmse"], 0.)
                        self.assertAlmostEqual(row["world_lateral_velocity_mean"], -.04)
                        self.assertAlmostEqual(row["world_lateral_velocity_rmse"], .04)
                        self.assertAlmostEqual(row["body_lateral_velocity_mean"], .02)
                        self.assertAlmostEqual(row["body_lateral_velocity_rmse"], .02)
                        self.assertAlmostEqual(row["body_yaw_rate_mean"], -.03)
                        self.assertAlmostEqual(row["body_yaw_rate_rmse"], .03)
                        self.assertGreater(row["cycles_RL"], 1)
                        self.assertEqual(row["swing_recall_RL"], 1.)

    def test_body_and_world_velocity_fields_remain_distinct(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_velocity_frames_") as tmp:
            directory = Path(tmp)
            manifest = synthetic_run(directory, widths=(.3,), dfs=(.625,))
            path = directory / manifest["expected_conditions"][0]["filename"]
            with np.load(path) as data:
                payload = {key: data[key] for key in data.files}
            payload["speed"][:] = .27
            payload["forward_velocity"][:] = .31
            payload["world_lateral_velocity"][:] = -.06
            np.savez_compressed(path, **payload)
            row = trial_rows(path)[0]
            self.assertAlmostEqual(row["mean_speed"], .27)
            self.assertAlmostEqual(row["settled_body_forward_speed"], .27)
            self.assertAlmostEqual(row["forward_speed"], .31)
            self.assertAlmostEqual(row["world_lateral_velocity_mean"], -.06)
            self.assertAlmostEqual(row["world_lateral_velocity_rmse"], .06)

    def test_nominal_only_schema_needs_no_push_fields(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_nominal_schema_") as tmp:
            directory = Path(tmp)
            manifest = synthetic_run(directory, widths=(.3,), dfs=(.625,))
            path = directory / manifest["expected_conditions"][0]["filename"]
            with np.load(path) as data:
                payload = {key: data[key] for key in data.files}
            payload["reset_plan"] = payload.pop("push_plan")[:, :2]
            del payload["planned_force"]
            del payload["force"]
            np.savez_compressed(path, **payload)
            self.assertEqual(validate_manifest(directory), [path])
            row = trial_rows(path)[0]
            self.assertEqual(row["applied_impulse"], 0.)
            self.assertEqual(row["planned_force"], 0.)
            self.assertEqual(row["planned_push_duration"], 0.)
            self.assertTrue(np.isnan(row["planned_push_time"]))

    def test_manifest_rejects_missing_files_and_identity(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_flat_manifest_") as tmp:
            directory = Path(tmp)
            manifest = synthetic_run(directory, widths=(.1,), dfs=(.5,))
            path = directory / manifest["expected_conditions"][0]["filename"]
            with np.load(path) as data:
                payload = {key: data[key] for key in data.files}
            payload["checkpoint_sha256"] = "wrong"
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_manifest(directory)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "exactly match"):
                validate_manifest(directory)

    def test_manifest_rejects_unmatched_seed_and_reset(self):
        for key in ["seeds", "initial_joints"]:
            with tempfile.TemporaryDirectory(prefix="synthetic_flat_pair_") as tmp:
                directory = Path(tmp)
                manifest = synthetic_run(directory, widths=(.1,), dfs=(.5, .625))
                path = directory / manifest["expected_conditions"][-1]["filename"]
                with np.load(path) as data:
                    payload = {field: data[field] for field in data.files}
                payload[key] = payload[key] + 1
                np.savez_compressed(path, **payload)
                with self.assertRaises(ValueError):
                    validate_manifest(directory)

    def test_end_to_end_analysis_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_flat_artifacts_") as tmp:
            directory = Path(tmp)
            synthetic_run(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = analyze(directory)
            self.assertEqual(len(summary), 4)
            self.assertTrue(summary.compliance_screen_pass.all())
            self.assertTrue((summary.successes == 1).all())
            self.assertFalse(any("beam" in col for col in summary.columns))
            for name in ["trials.csv", "summary.csv", "paired_contrasts.json", "success_vs_df.pdf",
                         "success_heatmap.png", "command_compliance.pdf", "per_leg_df.pdf",
                         "straight_path_tracking.png"]:
                self.assertTrue((directory / name).is_file(), name)

    def test_rejects_reset_contamination_and_command_mismatch(self):
        for key in ["done", "commands"]:
            with tempfile.TemporaryDirectory(prefix="synthetic_flat_invalid_") as tmp:
                directory = Path(tmp)
                manifest = synthetic_run(directory, widths=(.1,), dfs=(.5,))
                path = directory / manifest["expected_conditions"][0]["filename"]
                with np.load(path) as data:
                    payload = {field: data[field] for field in data.files}
                if key == "done":
                    payload[key][5, 0] = True
                else:
                    payload[key][5, 0, 1] = .7
                np.savez_compressed(path, **payload)
                with self.assertRaises(ValueError):
                    trial_rows(path)

    def test_body_width_invariant_under_yaw(self):
        for angle in [0., np.pi / 3]:
            with tempfile.TemporaryDirectory(prefix="synthetic_body_axes_") as tmp:
                directory = Path(tmp)
                manifest = synthetic_run(directory, widths=(.3,), dfs=(.5,))
                declare_body_run(directory, manifest, yaw=angle)
                rows = trial_rows(validate_manifest(directory)[0])
                for row in rows:
                    self.assertEqual(row["step_width_frame"], "body")
                    self.assertAlmostEqual(row["achieved_width"], .3)
                    self.assertAlmostEqual(row["body_achieved_width"], .3)
                    self.assertAlmostEqual(row["world_achieved_width"], .3 * np.cos(angle))
                    self.assertAlmostEqual(row["foot_lateral_mae"], 0.)
                    self.assertAlmostEqual(row["heading_rmse_rad"], angle)
                    self.assertAlmostEqual(row["max_abs_heading_rad"], angle)
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = analyze(directory)
                self.assertEqual(
                    bool(summary.compliance_screen_pass.all()),
                    angle == 0.)
                # Width remains body-frame correct, but a 60-degree yaw fails
                # the independent straight-path compliance gate.
                self.assertTrue(summary.step_width_frame.eq("body").all())

    def test_body_axes_require_full_rotation_and_verified_capture(self):
        for corruption in ["missing_quat", "missing_feet", "bad_shape", "nonunit", "wrong_frame", "wrong_coordinates"]:
            with tempfile.TemporaryDirectory(prefix="synthetic_bad_body_axes_") as tmp:
                directory = Path(tmp)
                manifest = synthetic_run(directory, widths=(.3,), dfs=(.5,))
                declare_body_run(directory, manifest, yaw=.5, roll=.2)
                path = validate_manifest(directory)[0]
                self.assertAlmostEqual(trial_rows(path)[0]["body_achieved_width"], .3)
                with np.load(path) as data:
                    payload = {key: data[key] for key in data.files}
                if corruption == "missing_quat":
                    del payload["root_quat"]
                elif corruption == "missing_feet":
                    del payload["feet_body"]
                elif corruption == "bad_shape":
                    payload["root_quat"] = payload["root_quat"][..., :3]
                elif corruption == "nonunit":
                    payload["root_quat"] *= 2
                elif corruption == "wrong_frame":
                    payload["step_width_frame"] = "world"
                else:
                    payload["feet_body"][..., 1] += .02
                np.savez_compressed(path, **payload)
                with self.assertRaises(ValueError):
                    validate_manifest(directory)

    def test_manifest_rejects_mixed_legacy_and_body_frames(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_mixed_axes_") as tmp:
            directory = Path(tmp)
            manifest = synthetic_run(directory, widths=(.1, .3), dfs=(.5,))
            declare_body_run(directory, manifest)
            path = directory / manifest["expected_conditions"][0]["filename"]
            with np.load(path) as data:
                payload = {key: data[key] for key in data.files if key != "step_width_frame"}
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, "Width frame mismatch"):
                analyze(directory)


if __name__ == "__main__":
    unittest.main()
