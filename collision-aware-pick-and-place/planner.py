"""Structured grasp-conditioned pick-and-place planner for MP2."""

from __future__ import annotations
import numpy as np
from grasping import GraspCandidate
from manipulation import CandidatePlan, ManipulationProblem
from rrt_connect import rrt_connect, shortcut_path


def plan_pick_and_place(
    candidates: list[GraspCandidate],
    problem: ManipulationProblem,
    *,
    rng: np.random.Generator,
    step_size: float = 0.12,
    edge_step: float = 0.04,
    max_iterations: int = 4000,
    shortcut_attempts: int = 80,
) -> CandidatePlan:
    """Return the first ranked feasible plan; raise ValueError if none exists."""
    q_home = problem.q_home
    joint_limits = problem.joint_limits

    for grasp_candidate in candidates:
        # get the local approach, lift, and place motions for this grasp
        candidate_context = problem.context_for(grasp_candidate)
        if candidate_context is None:
            continue

        q_pregrasp = candidate_context.q_pregrasp
        transit_valid = candidate_context.transit_valid

        # plan from home to pregrasp with an empty gripper
        transit_path = rrt_connect(
            q_home,
            q_pregrasp,
            joint_limits,
            transit_valid,
            rng=rng,
            step_size=step_size,
            edge_step=edge_step,
            max_iterations=max_iterations,
        )
        if transit_path is None:
            continue

        q_lift = candidate_context.q_lift
        q_preplace = candidate_context.q_preplace
        carry_valid = candidate_context.carry_valid

        # plan from lift to preplace while carrying the object
        transfer_path = rrt_connect(
            q_lift,
            q_preplace,
            joint_limits,
            carry_valid,
            rng=rng,
            step_size=step_size,
            edge_step=edge_step,
            max_iterations=max_iterations,
        )
        if transfer_path is None:
            continue

        # shorten the transit path using the empty gripper collision rules
        shortened_transit_path = shortcut_path(
            transit_path,
            transit_valid,
            rng=rng,
            edge_step=edge_step,
            attempts=shortcut_attempts,
        )

        # shorten the transfer path using the carried object collision rules
        shortened_transfer_path = shortcut_path(
            transfer_path,
            carry_valid,
            rng=rng,
            edge_step=edge_step,
            attempts=shortcut_attempts,
        )

        complete_plan = candidate_context.make_plan(
            shortened_transit_path,
            shortened_transfer_path,
        )
        return complete_plan

    raise ValueError("no cand. admits a complete collision-free plan")
