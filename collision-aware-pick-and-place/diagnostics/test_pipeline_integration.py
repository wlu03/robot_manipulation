"""Opt-in MuJoCo integration matrix for the complete MP2 pipeline.

Plan-only matrix (roughly one minute):

    MP2_RUN_SLOW_INTEGRATION=1 python -m unittest \
        diagnostics.test_pipeline_integration -v

Physical-style default-scene trials (additional time):

    MP2_RUN_PHYSICAL_INTEGRATION=1 python -m unittest \
        diagnostics.test_pipeline_integration -v
"""

from __future__ import annotations

import os
import pathlib
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

STUDENT = pathlib.Path(__file__).resolve().parents[1]
if str(STUDENT) not in sys.path:
    sys.path.insert(0, str(STUDENT))


RUN_SLOW = os.environ.get("MP2_RUN_SLOW_INTEGRATION") == "1"
RUN_PHYSICAL = os.environ.get("MP2_RUN_PHYSICAL_INTEGRATION") == "1"


def _add_local_x_variants(candidates, maximum_candidates):
    """Interleave 0/+45/-45-degree approaches while preserving each closing axis."""
    result = []
    for candidate in candidates:
        for theta in (0.0, np.pi / 4, -np.pi / 4):
            if theta == 0.0:
                variant = candidate
            else:
                c, s = np.cos(theta), np.sin(theta)
                R_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
                T = candidate.T_object_grasp.copy()
                T[:3, :3] = T[:3, :3] @ R_x
                variant = replace(candidate, T_object_grasp=T)
            result.append(variant)
            if len(result) >= maximum_candidates:
                return result
    return result


@unittest.skipUnless(RUN_SLOW, "set MP2_RUN_SLOW_INTEGRATION=1")
class PlanOnlyPipelineTests(unittest.TestCase):
    def _assert_case(self, seed, target, obstacle):
        from run_sim import run

        outcome = run(
            seed,
            headless=True,
            plan_only=True,
            with_obstacle=True,
            target_name=target,
            obstacle_name=obstacle,
            capture_outcome=True,
        )
        self.assertTrue(
            outcome.success,
            f"seed={seed}, target={target}, obstacle={obstacle}: "
            f"{outcome.failure_stage}: {outcome.error}",
        )

    def test_spherical_targets(self):
        for seed in range(3):
            for target in ("plum", "lemon"):
                with self.subTest(seed=seed, target=target):
                    self._assert_case(seed, target, "mustard_bottle")

    def test_tall_targets(self):
        for seed in range(3):
            for target in ("tall_can", "tall_box"):
                with self.subTest(seed=seed, target=target):
                    self._assert_case(seed, target, "mustard_bottle")

    def test_tall_targets_recover_with_approach_variants(self):
        import run_sim

        original_generate = run_sim._generate_grasps

        def generate_with_variants(task, seed):
            candidates = original_generate(task, seed)
            return _add_local_x_variants(candidates, len(candidates))

        with patch.object(
            run_sim, "_generate_grasps", side_effect=generate_with_variants
        ):
            for target in ("tall_can", "tall_box"):
                with self.subTest(target=target):
                    outcome = run_sim.run(
                        0,
                        headless=True,
                        plan_only=True,
                        with_obstacle=True,
                        target_name=target,
                        obstacle_name="mustard_bottle",
                        capture_outcome=True,
                    )
                    self.assertTrue(
                        outcome.success,
                        f"target={target}: {outcome.failure_stage}: {outcome.error}",
                    )


@unittest.skipUnless(RUN_PHYSICAL, "set MP2_RUN_PHYSICAL_INTEGRATION=1")
class PhysicalPipelineTests(unittest.TestCase):
    def test_default_scene_five_seeds(self):
        from run_sim import run

        for seed in range(5):
            with self.subTest(seed=seed):
                outcome = run(
                    seed,
                    headless=True,
                    plan_only=False,
                    with_obstacle=True,
                    capture_outcome=True,
                )
                self.assertTrue(
                    outcome.success,
                    f"seed={seed}: {outcome.failure_stage}: {outcome.error}",
                )
                self.assertIsNotNone(outcome.final_position_error_m)
                self.assertLessEqual(outcome.final_position_error_m, 0.02)

    def test_tall_can_with_approach_variants(self):
        import run_sim

        original_generate = run_sim._generate_grasps

        def generate_with_variants(task, seed):
            candidates = original_generate(task, seed)
            return _add_local_x_variants(candidates, len(candidates))

        with patch.object(
            run_sim, "_generate_grasps", side_effect=generate_with_variants
        ):
            outcome = run_sim.run(
                0,
                headless=True,
                plan_only=False,
                with_obstacle=True,
                target_name="tall_can",
                obstacle_name="mustard_bottle",
                capture_outcome=True,
            )
        self.assertTrue(
            outcome.success,
            f"{outcome.failure_stage}: {outcome.error}",
        )
        self.assertIsNotNone(outcome.final_position_error_m)
        self.assertLessEqual(outcome.final_position_error_m, 0.02)


if __name__ == "__main__":
    unittest.main()
