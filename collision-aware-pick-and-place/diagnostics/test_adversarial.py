"""Adversarial, rubric-oriented tests for the MP2 student implementation.

Run separately from the supplied public suite:

    python -m unittest discover -s diagnostics -p 'test_adversarial.py' -v

These tests intentionally express the required behavior, so a confirmed bug
appears as a normal test failure rather than being hidden by expectedFailure.
"""

from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np

STUDENT = pathlib.Path(__file__).resolve().parents[1]
if str(STUDENT) not in sys.path:
    sys.path.insert(0, str(STUDENT))

from grasping import (  # noqa: E402
    GraspCandidate,
    generate_grasp_candidates,
    is_antipodal,
    make_grasp_frame,
)
from manipulation import CandidateContext, ManipulationProblem  # noqa: E402
from planner import plan_pick_and_place  # noqa: E402
from provided.rrt_helpers import Tree  # noqa: E402
from rrt_connect import (  # noqa: E402
    _connect,
    _extend,
    edge_is_valid,
    rrt_connect,
)


def _simple_candidate(score: float = 1.0) -> GraspCandidate:
    return GraspCandidate(
        contact_a_object=np.array([-0.02, 0.0, 0.0]),
        contact_b_object=np.array([0.02, 0.0, 0.0]),
        T_object_grasp=np.eye(4),
        jaw_width_m=0.04,
        score=score,
    )


class AntipodalAdversarialTests(unittest.TestCase):
    def test_checks_each_friction_cone_independently(self):
        a = [-0.02, 0.0, 0.0]
        b = [0.02, 0.0, 0.0]
        self.assertTrue(is_antipodal(a, [-1, 0, 0], b, [1, 0, 0]))
        self.assertFalse(is_antipodal(a, [1, 0, 0], b, [1, 0, 0]))
        self.assertFalse(is_antipodal(a, [-1, 0, 0], b, [-1, 0, 0]))

    def test_normal_scaling_does_not_change_result(self):
        for scale_a in (1e-3, 1.0, 1e3):
            for scale_b in (1e-3, 1.0, 1e3):
                self.assertTrue(
                    is_antipodal(
                        [-0.02, 0, 0],
                        np.array([-1.0, 0, 0]) * scale_a,
                        [0.02, 0, 0],
                        np.array([1.0, 0, 0]) * scale_b,
                    )
                )

    def test_width_limits_are_inclusive(self):
        minimum = 0.01
        maximum = 0.05

        def check(width):
            return is_antipodal(
                [0, 0, 0],
                [-1, 0, 0],
                [width, 0, 0],
                [1, 0, 0],
                minimum_width_m=minimum,
                maximum_width_m=maximum,
            )

        self.assertTrue(check(minimum))
        self.assertTrue(check(maximum))
        self.assertFalse(check(np.nextafter(minimum, 0.0)))
        self.assertFalse(check(np.nextafter(maximum, np.inf)))

    def test_friction_cone_boundary_and_just_outside(self):
        mu = 0.5
        alpha = float(np.arctan(mu))

        def normal_a(angle):
            inward = np.array([np.cos(angle), np.sin(angle), 0.0])
            return -inward

        args = ([0, 0, 0], [0.04, 0, 0])
        self.assertTrue(
            is_antipodal(
                args[0], normal_a(alpha), args[1], [1, 0, 0],
                friction_coefficient=mu,
            )
        )
        self.assertFalse(
            is_antipodal(
                args[0], normal_a(alpha + 1e-7), args[1], [1, 0, 0],
                friction_coefficient=mu,
            )
        )

    def test_degenerate_inputs_are_rejected(self):
        self.assertFalse(is_antipodal([0, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0]))
        self.assertFalse(is_antipodal([0, 0, 0], [0, 0, 0], [0.04, 0, 0], [1, 0, 0]))
        self.assertFalse(is_antipodal([0, 0, 0], [-1, 0, 0], [0.04, 0, 0], [np.nan, 0, 0]))


class GraspFrameAdversarialTests(unittest.TestCase):
    def test_random_frames_are_proper_and_follow_the_convention(self):
        rng = np.random.default_rng(4803)
        for _ in range(1000):
            a = rng.normal(size=3)
            b = rng.normal(size=3)
            preferred = rng.normal(size=3)
            T = make_grasp_frame(a, b, preferred_approach=preferred)
            R = T[:3, :3]
            closing = (b - a) / np.linalg.norm(b - a)
            np.testing.assert_allclose(T[:3, 3], (a + b) / 2, atol=1e-12)
            np.testing.assert_allclose(R[:, 0], closing, atol=1e-12)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)
            projected = preferred - np.dot(preferred, closing) * closing
            if np.linalg.norm(projected) > 1e-10:
                np.testing.assert_allclose(
                    R[:, 2], projected / np.linalg.norm(projected), atol=1e-10
                )

    def test_parallel_approach_uses_a_valid_fallback(self):
        T = make_grasp_frame([0, 0, 0], [1, 0, 0], preferred_approach=[2, 0, 0])
        np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(T[:3, :3])), 1.0, places=12)
        self.assertAlmostEqual(float(np.dot(T[:3, 0], T[:3, 2])), 0.0, places=12)

    def test_coincident_contacts_raise(self):
        with self.assertRaises(ValueError):
            make_grasp_frame([1, 2, 3], [1, 2, 3])

    def test_rotation_about_local_x_preserves_contact_geometry(self):
        T = make_grasp_frame([-0.02, 0, 0], [0.02, 0, 0])
        theta = -np.pi / 12
        c, s = np.cos(theta), np.sin(theta)
        R_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        variant = T.copy()
        variant[:3, :3] = T[:3, :3] @ R_x
        np.testing.assert_allclose(variant[:3, 0], T[:3, 0], atol=1e-12)
        np.testing.assert_allclose(variant[:3, 3], T[:3, 3], atol=1e-12)
        np.testing.assert_allclose(variant[:3, :3].T @ variant[:3, :3], np.eye(3), atol=1e-12)


class CandidateGenerationAdversarialTests(unittest.TestCase):
    def test_two_point_candidate_has_exact_geometry_and_copies_inputs(self):
        points = np.array([[-0.02, 0, 0], [0.02, 0, 0]], dtype=float)
        normals = np.array([[-1, 0, 0], [1, 0, 0]], dtype=float)
        candidates = generate_grasp_candidates(
            points,
            normals,
            rng=np.random.default_rng(7),
            sample_count=1,
            maximum_candidates=4,
        )
        self.assertGreaterEqual(len(candidates), 1)
        candidate = candidates[0]
        original_a = candidate.contact_a_object.copy()
        original_b = candidate.contact_b_object.copy()
        self.assertTrue(any(np.array_equal(candidate.contact_a_object, row) for row in points))
        self.assertTrue(any(np.array_equal(candidate.contact_b_object, row) for row in points))
        self.assertAlmostEqual(candidate.jaw_width_m, 0.04)
        np.testing.assert_allclose(candidate.T_object_grasp[:3, 3], [0, 0, 0])
        np.testing.assert_allclose(
            candidate.T_object_grasp[:3, 0],
            (candidate.contact_b_object - candidate.contact_a_object) / 0.04,
        )
        points[:] = 99.0
        np.testing.assert_array_equal(candidate.contact_a_object, original_a)
        np.testing.assert_array_equal(candidate.contact_b_object, original_b)

    def test_cap_sorting_and_seed_reproducibility(self):
        rng = np.random.default_rng(3)
        directions = rng.normal(size=(40, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points = np.vstack([-0.025 * directions, 0.025 * directions])
        normals = np.vstack([-directions, directions])

        def generate(seed):
            return generate_grasp_candidates(
                points,
                normals,
                rng=np.random.default_rng(seed),
                sample_count=4000,
                maximum_candidates=7,
            )

        first, second = generate(19), generate(19)
        self.assertLessEqual(len(first), 7)
        self.assertGreater(len(first), 0)
        self.assertTrue(all(np.isfinite(candidate.score) for candidate in first))
        self.assertEqual(
            [candidate.score for candidate in first],
            sorted((candidate.score for candidate in first), reverse=True),
        )
        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.contact_a_object, right.contact_a_object)
            np.testing.assert_array_equal(left.contact_b_object, right.contact_b_object)
            np.testing.assert_array_equal(left.T_object_grasp, right.T_object_grasp)
            self.assertEqual(left.jaw_width_m, right.jaw_width_m)
            self.assertEqual(left.score, right.score)

    def test_one_contact_pair_offers_approach_diversity(self):
        """A closing axis should not force one potentially infeasible approach."""
        points = np.array([[-0.02, 0, 0], [0.02, 0, 0]], dtype=float)
        normals = np.array([[-1, 0, 0], [1, 0, 0]], dtype=float)
        candidates = generate_grasp_candidates(
            points,
            normals,
            rng=np.random.default_rng(11),
            sample_count=16,
            maximum_candidates=8,
        )
        approaches = {
            tuple(np.round(candidate.T_object_grasp[:3, 2], decimals=12))
            for candidate in candidates
        }
        self.assertGreaterEqual(
            len(approaches),
            2,
            "only one approach was proposed for the contact pair",
        )


class EdgeAndExtendAdversarialTests(unittest.TestCase):
    def test_exact_endpoint_is_checked_not_a_reconstruction(self):
        start = np.array([9.953789526047839])
        goal = np.array([0.4925940698449227])
        reconstructed = start + (goal - start)
        self.assertGreater(reconstructed[0], goal[0])
        predicate = lambda q: bool(q[0] > goal[0])
        self.assertFalse(edge_is_valid(start, goal, predicate, edge_step=2.0))

    def test_exact_invalid_target_is_not_added_by_extend(self):
        start = np.array([9.953789526047839])
        goal = np.array([0.4925940698449227])
        predicate = lambda q: bool(q[0] > goal[0])
        tree = Tree(start, "start")
        self.assertEqual(
            _extend(
                tree,
                goal,
                predicate,
                step_size=100.0,
                edge_step=2.0,
            ),
            ("trapped", 0),
        )
        self.assertEqual(len(tree.nodes), 1)

    def test_spacing_endpoints_and_zero_length(self):
        samples = []
        start = np.array([-0.8, 0.4, 1.2])
        goal = np.array([0.7, -0.3, -0.2])
        step = 0.13
        self.assertTrue(
            edge_is_valid(
                start,
                goal,
                lambda q: samples.append(q.copy()) or True,
                edge_step=step,
            )
        )
        np.testing.assert_array_equal(samples[0], start)
        np.testing.assert_allclose(samples[-1], goal, atol=1e-15, rtol=0.0)
        self.assertLessEqual(
            max(np.linalg.norm(b - a) for a, b in zip(samples, samples[1:])),
            step * (1 + 1e-12),
        )
        self.assertTrue(edge_is_valid(start, start, lambda q: True, edge_step=step))
        self.assertFalse(edge_is_valid(start, start, lambda q: False, edge_step=step))

    def test_extend_status_parent_and_trapped_mutation(self):
        tree = Tree([0.0], "start")
        tree.add([0.8], 0)
        status, index = _extend(
            tree, [0.65], lambda q: True, step_size=0.1, edge_step=0.02
        )
        self.assertEqual(status, "advanced")
        self.assertEqual(tree.parent[index], 1)
        np.testing.assert_allclose(tree.nodes[index], [0.7])

        blocked = Tree([0.0], "start")
        before_nodes = [node.copy() for node in blocked.nodes]
        before_parents = blocked.parent.copy()
        self.assertEqual(
            _extend(
                blocked,
                [0.3],
                lambda q: not (0.04 < q[0] < 0.06),
                step_size=0.1,
                edge_step=0.02,
            ),
            ("trapped", 0),
        )
        self.assertEqual(blocked.parent, before_parents)
        for before, after in zip(before_nodes, blocked.nodes):
            np.testing.assert_array_equal(before, after)

    def test_connect_repeats_and_stops_at_obstacle(self):
        tree = Tree([0.0], "start")
        status, index = _connect(
            tree, [0.55], lambda q: True, step_size=0.2, edge_step=0.04
        )
        self.assertEqual(status, "reached")
        np.testing.assert_array_equal(tree.nodes[index], [0.55])
        self.assertEqual(len(tree.nodes), 4)

        blocked = Tree([0.0], "start")
        status, index = _connect(
            blocked,
            [1.0],
            lambda q: not (0.45 < q[0] < 0.55),
            step_size=0.2,
            edge_step=0.02,
        )
        self.assertEqual(status, "trapped")
        self.assertLess(blocked.nodes[index][0], 0.55)


class RRTAdversarialTests(unittest.TestCase):
    @staticmethod
    def _valid(q):
        x, y = np.asarray(q)
        return not (0.35 < x < 0.65 and 0.0 < y < 0.82)

    def test_obstacle_route_order_edges_and_reproducibility(self):
        kwargs = dict(
            q_start=[0.1, 0.1],
            q_goal=[0.9, 0.1],
            joint_limits=[[0.0, 1.0], [0.0, 1.0]],
            state_is_valid=self._valid,
            step_size=0.10,
            edge_step=0.025,
            max_iterations=4000,
            goal_bias=0.10,
        )
        first = rrt_connect(rng=np.random.default_rng(37), **kwargs)
        second = rrt_connect(rng=np.random.default_rng(37), **kwargs)
        self.assertIsNotNone(first)
        self.assertEqual(len(first), len(second))
        np.testing.assert_array_equal(first[0], kwargs["q_start"])
        np.testing.assert_array_equal(first[-1], kwargs["q_goal"])
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(first, first[1:]):
            self.assertTrue(edge_is_valid(left, right, self._valid, edge_step=0.025))

    def test_solid_wall_and_invalid_endpoints_return_none(self):
        wall = lambda q: not (0.45 < q[0] < 0.55)
        result = rrt_connect(
            [0.1, 0.5],
            [0.9, 0.5],
            [[0, 1], [0, 1]],
            wall,
            rng=np.random.default_rng(4),
            max_iterations=400,
        )
        self.assertIsNone(result)
        self.assertIsNone(
            rrt_connect(
                [-0.1],
                [0.5],
                [[0, 1]],
                lambda q: True,
                rng=np.random.default_rng(4),
            )
        )
        self.assertIsNone(
            rrt_connect(
                [0.1],
                [0.5],
                [[0, 1]],
                lambda q: q[0] != 0.1,
                rng=np.random.default_rng(4),
            )
        )


class _PlannerProblem(ManipulationProblem):
    def __init__(self, contexts):
        self.q_home = np.array([0.0])
        self.joint_limits = np.array([[-1.0, 1.0]])
        self.contexts = contexts

    def context_for(self, grasp):
        return self.contexts.get(id(grasp))


def _context(grasp):
    return CandidateContext(
        grasp=grasp,
        q_pregrasp=np.array([0.1]),
        q_lift=np.array([0.3]),
        q_preplace=np.array([0.5]),
        gripper_position_m=0.02,
        transit_valid=lambda q: True,
        carry_valid=lambda q: True,
        approach=[np.array([0.1]), np.array([0.2])],
        lift=[np.array([0.2]), np.array([0.3])],
        place=[np.array([0.5]), np.array([0.6])],
    )


class PlannerAdversarialTests(unittest.TestCase):
    def test_retries_each_failure_stage_and_uses_supplied_rng(self):
        missing, transit_fail, transfer_fail, complete = [
            _simple_candidate(score) for score in (4.0, 3.0, 2.0, 1.0)
        ]
        contexts = {
            id(transit_fail): _context(transit_fail),
            id(transfer_fail): _context(transfer_fail),
            id(complete): _context(complete),
        }
        problem = _PlannerProblem(contexts)
        rng = np.random.default_rng(9)
        transit = [np.array([0.0]), np.array([0.1])]
        transfer = [np.array([0.3]), np.array([0.5])]
        rrt_results = [None, transit, None, transit, transfer]

        with patch("planner.rrt_connect", side_effect=rrt_results) as mocked_rrt:
            with patch("planner.shortcut_path", side_effect=lambda path, *_a, **_k: path) as mocked_shortcut:
                plan = plan_pick_and_place(
                    [missing, transit_fail, transfer_fail, complete],
                    problem,
                    rng=rng,
                    step_size=0.17,
                    edge_step=0.03,
                    max_iterations=321,
                    shortcut_attempts=17,
                )

        self.assertIs(plan.grasp, complete)
        self.assertEqual(mocked_rrt.call_count, 5)
        self.assertEqual(mocked_shortcut.call_count, 2)
        for call in mocked_rrt.call_args_list:
            self.assertIs(call.kwargs["rng"], rng)
            self.assertEqual(call.kwargs["step_size"], 0.17)
            self.assertEqual(call.kwargs["edge_step"], 0.03)
            self.assertEqual(call.kwargs["max_iterations"], 321)
        for call in mocked_shortcut.call_args_list:
            self.assertIs(call.kwargs["rng"], rng)
            self.assertEqual(call.kwargs["attempts"], 17)

    def test_raises_when_no_candidate_is_complete(self):
        candidate = _simple_candidate()
        problem = _PlannerProblem({id(candidate): _context(candidate)})
        with patch("planner.rrt_connect", return_value=None):
            with self.assertRaisesRegex(ValueError, "complete collision-free plan"):
                plan_pick_and_place(
                    [candidate], problem, rng=np.random.default_rng(2)
                )


if __name__ == "__main__":
    unittest.main()
