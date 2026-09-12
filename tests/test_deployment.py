"""CPU-only checks for the hardware deployment randomization contract."""

import ast
import json
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from collections.abc import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))

from beam_walking.experiment.deployment import (
    BASE_COM_RANGE_M,
    BASE_MASS_SCALE_RANGE,
    ACTUATOR_DELAY_PHYSICS_STEPS,
    DYNAMIC_FRICTION_RANGE,
    DAMPING_EVALUATION_SCALES,
    DAMPING_SCALE_RANGE,
    STIFFNESS_EVALUATION_SCALES,
    STIFFNESS_SCALE_RANGE,
    STATIC_FRICTION_RANGE,
    JOINT_ARMATURE_RANGE, JOINT_FRICTION_RANGE,
    MOTOR_STRENGTH_SCALE_RANGE,
    STANCE_START_PROBABILITY, TRAINING_ITERATIONS, TRAINING_NUM_ENVS,
    deployment_profile, deployment_profile_sha256,
    deployment_training_source_hash,
    validate_gain_scales, initialize_deployment_policy,
    deployment_finetune_lineage_valid, DEPLOYMENT_PARENT_SHA256,
)


class DeploymentProfileTest(unittest.TestCase):
    def test_bounded_hardware_profile(self):
        self.assertEqual(STIFFNESS_SCALE_RANGE, (.6, 1.4))
        self.assertEqual(DAMPING_SCALE_RANGE, (.5, 1.5))
        self.assertEqual(BASE_MASS_SCALE_RANGE, (.8, 1.2))
        self.assertEqual(
            STIFFNESS_EVALUATION_SCALES, (.6, .8, 1., 1.2, 1.4))
        self.assertEqual(
            DAMPING_EVALUATION_SCALES, (.5, .75, 1., 1.25, 1.5))
        self.assertLessEqual(DYNAMIC_FRICTION_RANGE[1],
                             STATIC_FRICTION_RANGE[1])
        self.assertEqual(BASE_COM_RANGE_M["x"], (-.04, .04))
        self.assertEqual(MOTOR_STRENGTH_SCALE_RANGE, (.7, 1.1))
        self.assertEqual(JOINT_FRICTION_RANGE, (0., .25))
        self.assertEqual(JOINT_ARMATURE_RANGE, (0., .02))
        self.assertEqual(ACTUATOR_DELAY_PHYSICS_STEPS, (0, 4))
        self.assertEqual(TRAINING_NUM_ENVS, 3072)
        self.assertEqual(TRAINING_ITERATIONS, 1800)
        self.assertEqual(STANCE_START_PROBABILITY, .10)
        self.assertFalse(deployment_profile()["external_pushes"])

    def test_profile_and_training_hashes_bind_frozen_content(self):
        profile_hash = deployment_profile_sha256()
        self.assertEqual(len(profile_hash), 64)
        self.assertEqual(
            json.loads(json.dumps(deployment_profile())), deployment_profile())
        source_hash = deployment_training_source_hash(ROOT, "00" * 32)
        self.assertEqual(len(source_hash), 64)
        self.assertNotEqual(source_hash, deployment_training_source_hash(
            ROOT, "01" * 32))

    def test_gain_scale_validation(self):
        self.assertEqual(validate_gain_scales(.8, 1.2), (.8, 1.2))
        for values in ((0, 1), (-1, 1), (1, 2.1)):
            with self.assertRaises(ValueError):
                validate_gain_scales(*values)

    def test_task_keeps_pushes_disabled_and_enables_sensor_noise(self):
        source = (ROOT / "source/beam_walking/beam_walking/experiment/"
                  "deployment_task.py").read_text()
        tree = ast.parse(source)
        assignments = {
            ast.unparse(node.targets[0]): ast.unparse(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and len(node.targets) == 1
        }
        self.assertEqual(assignments["self.events.push_robot"], "None")
        self.assertEqual(assignments["self.events.base_external_force_torque"],
                         "None")
        self.assertEqual(
            assignments["self.observations.policy.enable_corruption"], "True")
        for name in (
                "self.events.add_base_mass", "self.events.base_com",
                "self.events.motor_gain_randomization",
                "self.events.motor_strength_randomization",
                "self.events.joint_parameter_randomization"):
            self.assertIn(name, assignments)
        self.assertIn("valid.all()", source)
        self.assertIn("deployment_runtime_summary", source)

    def test_evaluation_matrix_contains_joint_gain_cross_product(self):
        source = (ROOT / "scripts/run_deployment_evaluation.py").read_text()
        tree = ast.parse(source)
        self.assertIn("for kp in STIFFNESS_EVALUATION_SCALES", source)
        self.assertIn("for kd in DAMPING_EVALUATION_SCALES", source)
        self.assertIn('yield "randomized", None, None', source)
        self.assertIn("TEST_SEED = 1_500_000", source)
        self.assertIn("VALIDATION_SEED = 20_000", source)
        self.assertIn("orchestrator_sha256", source)
        self.assertIn("Archived deployment selector source mismatch", source)
        self.assertIn("validate_completed", source)
        self.assertIsNotNone(tree)

    def test_fixed_gain_sweep_keeps_other_deployment_randomization(self):
        source = (ROOT / "scripts/evaluate_policy.py").read_text()
        self.assertIn(
            'cfg = (BeamEnvCfg() if args.deployment_profile == "nominal"\n'
            '           else DeploymentBeamEnvCfg())', source)
        self.assertIn('saved.get("deployment_profile_schema")', source)
        self.assertIn('saved.get("training_source_sha256")', source)
        self.assertIn("DeploymentBeamEnv)", source)
        self.assertIn("deployment_profile_sha256", source)

    def test_delay_reset_uses_buffer_dtype_for_indexed_envs(self):
        path = ROOT / "source/beam_walking/beam_walking/experiment/deployment_actuator.py"
        tree = ast.parse(path.read_text())
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "reset")
        namespace = {"torch": torch, "Sequence": Sequence}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)

        class Buffer:
            def __init__(self):
                self.time_lags = torch.full((16,), -1, dtype=torch.int32)
                self.reset_ids = None

            def set_time_lag(self, lags, env_ids):
                self.time_lags[env_ids] = lags

            def reset(self, env_ids):
                self.reset_ids = env_ids.clone()

        buffers = [Buffer() for _ in range(3)]
        actuator = SimpleNamespace(
            _num_envs=16, _device="cpu", cfg=SimpleNamespace(min_delay=0, max_delay=4),
            positions_delay_buffer=buffers[0], velocities_delay_buffer=buffers[1],
            efforts_delay_buffer=buffers[2])
        ids = torch.tensor([1, 4, 7])
        namespace["reset"](actuator, ids)
        for buffer in buffers:
            self.assertTrue(bool(((buffer.time_lags[ids] >= 0)
                                 & (buffer.time_lags[ids] <= 4)).all()))
            self.assertEqual(int(buffer.time_lags[0]), -1)
            torch.testing.assert_close(buffer.time_lags, buffers[0].time_lags)
            torch.testing.assert_close(buffer.reset_ids, ids)

    def test_calibration_restores_sampled_gains_on_simulation_error(self):
        path = ROOT / "source/beam_walking/beam_walking/experiment/deployment_task.py"
        tree = ast.parse(path.read_text())
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "calibrate_stance")
        namespace = {"torch": torch, "command": lambda env: None,
                     "CALIBRATION_STIFFNESS": 80., "CALIBRATION_DAMPING": 3.,
                     "CALIBRATION_MAX_STEPS": 4000}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        kp = torch.tensor([[15., 35.]])
        kd = torch.tensor([[.25, .75]])
        actuator = SimpleNamespace(stiffness=kp.clone(), damping=kd.clone())
        robot = SimpleNamespace(
            data=SimpleNamespace(default_joint_pos=torch.zeros(1, 2)),
            actuators={"legs": actuator}, set_joint_position_target=lambda target: None)

        class Scene(dict):
            def write_data_to_sim(self):
                raise RuntimeError("injected simulation failure")

        env = SimpleNamespace(scene=Scene(robot=robot))
        with self.assertRaisesRegex(RuntimeError, "injected"):
            namespace["calibrate_stance"](env)
        torch.testing.assert_close(actuator.stiffness, kp)
        torch.testing.assert_close(actuator.damping, kd)

    def test_each_robot_must_have_a_contiguous_quiet_window(self):
        path = ROOT / "source/beam_walking/beam_walking/experiment/deployment_task.py"
        method = next(node for node in ast.walk(ast.parse(path.read_text()))
                      if isinstance(node, ast.FunctionDef) and node.name == "calibrate_stance")
        gait = SimpleNamespace(sensor_feet=[0, 1, 2, 3], sensor_other=[4])
        ns = {"torch": torch, "math": math, "command": lambda env: gait,
              "CALIBRATION_STIFFNESS": 80., "CALIBRATION_DAMPING": 3.,
              "CALIBRATION_QUIET_STEPS": 20, "CALIBRATION_MAX_STEPS": 4000}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        for never_quiet in (False, True):
            with self.subTest(never_quiet=never_quiet):
                data = SimpleNamespace(
                    default_joint_pos=torch.zeros(2, 12), joint_pos=torch.zeros(2, 12),
                    root_pos_w=torch.tensor([[0., 0., .3]] * 2),
                    root_quat_w=torch.tensor([[1., 0., 0., 0.]] * 2),
                    root_lin_vel_w=torch.zeros(2, 3), root_ang_vel_w=torch.zeros(2, 3),
                    projected_gravity_b=torch.tensor([[0., 0., -1.]] * 2))
                motor = SimpleNamespace(stiffness=torch.full((2, 12), 20.),
                                        damping=torch.full((2, 12), .4))
                robot = SimpleNamespace(data=data, actuators={"legs": motor},
                                        joint_names=[str(i) for i in range(12)],
                                        set_joint_position_target=lambda target: None)
                forces = torch.zeros(2, 5, 3)
                forces[:, :4, 2] = 20.
                sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces),
                                         body_names=[str(i) for i in range(5)])

                class Scene(dict):
                    env_origins = torch.zeros(2, 3)
                    def write_data_to_sim(self):
                        pass
                    def update(self, dt):
                        pass

                class Sim:
                    tick = -1
                    def step(self, render):
                        self.tick += 1
                        data.joint_pos.fill_(self.tick / 10000)
                        data.root_ang_vel_w.zero_()
                        if self.tick < 1030 or never_quiet:
                            data.root_ang_vel_w[1, 0] = .1
                        if self.tick >= 1150:
                            data.root_ang_vel_w[:, 0] = .1

                env = SimpleNamespace(scene=Scene(robot=robot, contact_forces=sensor),
                                      sim=Sim(), physics_dt=.005, num_envs=2)
                if never_quiet:
                    with self.assertRaisesRegex(RuntimeError, "calibration failed"):
                        ns["calibrate_stance"](env)
                else:
                    ns["calibrate_stance"](env)
                    self.assertEqual(env.deployment_calibration_diagnostics[
                        "captured_physics_ticks"], [1019, 1049])
                    torch.testing.assert_close(env.settled_stance["joint_pos"][:, 0],
                                               torch.tensor([.1019, .1049]))
                torch.testing.assert_close(motor.stiffness, torch.full((2, 12), 20.))

    def test_finetune_loads_parent_weights_without_resuming_optimizer_or_iteration(self):
        policy = torch.nn.Linear(3, 2)
        parent = torch.nn.Linear(3, 2).state_dict()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        runner = SimpleNamespace(alg=SimpleNamespace(policy=policy, optimizer=optimizer),
                                 current_learning_iteration=1799)
        module = "beam_walking.experiment.deployment"
        with patch(module + ".deployment_parent_metadata", return_value={}), patch(
                "torch.load", return_value={"model_state_dict": parent, "iter": 1799}):
            result = initialize_deployment_policy(runner, "parent.pt", "hash")
        self.assertTrue(result["initial_weights_match_parent"])
        self.assertEqual(runner.current_learning_iteration, 0)
        self.assertFalse(optimizer.state)
        for key, value in policy.state_dict().items():
            self.assertTrue(torch.equal(value, parent[key]))

    def test_finetune_evaluation_binds_parent_in_both_provenance_and_checkpoint(self):
        training = {"training_kind": "deployment_finetune", "fresh_training": False,
                    "parent_checkpoint_sha256": DEPLOYMENT_PARENT_SHA256,
                    "parent_training_seed": 2, "parent_checkpoint_iteration": 1799,
                    "initial_weights_match_parent": True}
        saved = {"training_kind": "deployment_finetune",
                 "parent_checkpoint_sha256": DEPLOYMENT_PARENT_SHA256}
        self.assertTrue(deployment_finetune_lineage_valid(training, saved))
        for record in (training, saved):
            original = record["parent_checkpoint_sha256"]
            record["parent_checkpoint_sha256"] = "wrong"
            self.assertFalse(deployment_finetune_lineage_valid(training, saved))
            record["parent_checkpoint_sha256"] = original
        training["initial_weights_match_parent"] = False
        self.assertFalse(deployment_finetune_lineage_valid(training, saved))

    def test_training_entrypoint_enforces_frozen_scale(self):
        source = (ROOT / "scripts/beam_experiment.py").read_text()
        for token in (
                "TRAINING_NUM_ENVS", "TRAINING_ITERATIONS",
                "STANCE_START_PROBABILITY", "args.no_watcher",
                "deployment_runtime_summary",
                'env.cfg.stance_start_probability = 1.0'):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
