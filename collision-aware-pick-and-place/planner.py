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
    # TODO(MP2)
    raise NotImplementedError
