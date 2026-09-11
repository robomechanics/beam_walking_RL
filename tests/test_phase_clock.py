"""Regression tests for action/reward gait-phase chronology."""
import ast
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))
from beam_walking.experiment.protocol import advance_phase_ticks


class PhaseClockTest(unittest.TestCase):

    def test_continuing_envs_advance_wrap_and_resets_stay_at_zero(self):
        reward_phase = torch.tensor([0, 23, 7])
        periods = torch.tensor([24, 24, 18])
        reset = torch.tensor([False, False, True])
        next_phase = advance_phase_ticks(reward_phase, periods, reset)
        torch.testing.assert_close(next_phase, torch.tensor([1, 0, 0]))
        # Computing the next clock must not mutate the phase used by the action,
        # reward, and captured transition in the current interval.
        torch.testing.assert_close(reward_phase, torch.tensor([0, 23, 7]))

    def test_invalid_clock_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            advance_phase_ticks(torch.zeros(2, dtype=torch.long),
                                torch.ones(3, dtype=torch.long),
                                torch.zeros(2, dtype=torch.bool))

    def test_smoke_imports_runtime_phase_symbols(self):
        tree = ast.parse((ROOT / "scripts/beam_experiment.py").read_text())
        imported = {
            alias.asname or alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "beam_walking.experiment.protocol"
            for alias in node.names
        }
        self.assertTrue({"CYCLE_STEPS", "advance_phase_ticks"} <= imported)

    def test_step_orders_action_reward_phase_advance_and_next_observation(self):
        """Keep the reference clock fixed until the current reward is complete."""
        tree = ast.parse(
            (ROOT / "source/beam_walking/beam_walking/experiment/task.py").read_text())
        beam = next(node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == "BeamEnv")
        step = next(node for node in beam.body
                    if isinstance(node, ast.FunctionDef) and node.name == "step")
        calls = []
        mutation_lines = []
        for node in ast.walk(step):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name in {"self.action_manager.process_action",
                            "self.reward_manager.compute",
                            "advance_phase_ticks",
                            "self.command_manager.compute",
                            "self.observation_manager.compute"}:
                    calls.append((name, node.lineno))
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any("phase_ticks" in ast.unparse(target) for target in targets):
                    mutation_lines.append(node.lineno)
        by_name = {}
        for name, line in calls:
            by_name.setdefault(name, []).append(line)
        self.assertEqual(len(mutation_lines), 1)
        phase_line = by_name["advance_phase_ticks"][0]
        self.assertLess(by_name["self.action_manager.process_action"][0],
                        by_name["self.reward_manager.compute"][0])
        self.assertLess(by_name["self.reward_manager.compute"][0], phase_line)
        self.assertLess(phase_line, by_name["self.command_manager.compute"][0])
        self.assertLess(by_name["self.command_manager.compute"][0],
                        max(by_name["self.observation_manager.compute"]))


if __name__ == "__main__":
    unittest.main()
