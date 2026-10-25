from __future__ import annotations
import pathlib
import sys
import unittest

import numpy as np

STUDENT = pathlib.Path(__file__).resolve().parents[1]
if str(STUDENT) not in sys.path:
    sys.path.insert(0, str(STUDENT))
from grasping import GraspCandidate, is_antipodal, make_grasp_frame
from planner import plan_pick_and_place
from manipulation import CandidateContext, ManipulationProblem
from rrt_connect import edge_is_valid, rrt_connect


class _Problem(ManipulationProblem):
    def __init__(self, first, second):
        self.q_home = np.array([0.0])
        self.joint_limits = np.array([[-1.0, 1.0]])
        self.first = first
        self.second = second

    def context_for(self, grasp):
        approach = None if grasp is self.first else [np.array([0.10]), np.array([0.20])]
        if approach is None:
            return None
        return CandidateContext(
            grasp=grasp,
            q_pregrasp=np.array([0.10]),
            q_lift=np.array([0.30]),
            q_preplace=np.array([0.55]),
            gripper_position_m=0.02,
            transit_valid=lambda q: True,
            carry_valid=lambda q: True,
            approach=approach,
            lift=[np.array([0.20]), np.array([0.30])],
            place=[np.array([0.55]), np.array([0.65])],
        )


class MP2PublicTests(unittest.TestCase):
    def test_basic_antipodal_pair(self):
        self.assertTrue(
            is_antipodal(
                [-0.02, 0, 0],
                [-1, 0, 0],
                [0.02, 0, 0],
                [1, 0, 0],
                friction_coefficient=0.5,
            )
        )

    def test_grasp_frame_is_proper(self):
        T = make_grasp_frame([-0.02, 0, 0], [0.02, 0, 0])
        np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-10)
        self.assertAlmostEqual(float(np.linalg.det(T[:3, :3])), 1.0, places=10)

    def test_edge_validation_checks_interior(self):
        self.assertFalse(
            edge_is_valid(
                [0.0], [1.0], lambda q: not (0.45 < q[0] < 0.55), edge_step=0.1
            )
        )

    def test_rrt_connect_returns_start_to_goal(self):
        path = rrt_connect(
            [0.0],
            [0.8],
            [[-1.0, 1.0]],
            lambda q: True,
            rng=np.random.default_rng(5),
            step_size=0.15,
            edge_step=0.05,
            max_iterations=100,
        )
        self.assertIsNotNone(path)
        np.testing.assert_allclose(path[0], [0.0])
        np.testing.assert_allclose(path[-1], [0.8])

    def test_complete_planner_retries_after_downstream_failure(self):
        T = np.eye(4)
        first = GraspCandidate(np.zeros(3), np.ones(3), T, 0.03, 2.0)
        second = GraspCandidate(np.zeros(3), np.ones(3), T, 0.03, 1.0)
        result = plan_pick_and_place(
            [first, second],
            _Problem(first, second),
            rng=np.random.default_rng(9),
            max_iterations=200,
            shortcut_attempts=5,
        )
        self.assertIs(result.grasp, second)
        self.assertEqual(
            set(result.phases),
            {"transit", "approach", "lift", "transfer", "place"},
        )


if __name__ == "__main__":
    unittest.main()
