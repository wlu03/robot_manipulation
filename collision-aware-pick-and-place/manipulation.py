"""Small public interface between the supplied MP2 fixture and student planner.

Students may read this file to understand what `planner.py` receives. Simulator,
collision, IK, and execution machinery lives under `provided/`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from grasping import GraspCandidate

StatePredicate = Callable[[NDArray[np.float64]], bool]
REQUIRED_PHASES = ("transit", "approach", "lift", "transfer", "place")


@dataclass(frozen=True)
class CandidatePlan:
    """Complete plan returned by planner.py; normally built with context.make_plan."""

    grasp: GraspCandidate
    gripper_position_m: float
    phases: dict[str, list[NDArray[np.float64]]]


@dataclass(frozen=True)
class CandidateContext:
    """A grasp whose local approach/lift/place motions are already feasible.

    gripper_position_m predicts the one-finger position at object contact for
    collision geometry. Execution always commands fully closed (zero).
    make_plan preserves this supplied prediction without student tuning.
    """

    grasp: GraspCandidate
    q_pregrasp: NDArray[np.float64]  # transit goal; approach starts here
    q_lift: NDArray[np.float64]  # end of lift; transfer starts here
    q_preplace: NDArray[np.float64]  # transfer goal; place starts here
    gripper_position_m: float
    transit_valid: StatePredicate  # empty gripper, object resting
    carry_valid: StatePredicate  # robot plus carried object, for transfer
    approach: list[NDArray[np.float64]]
    lift: list[NDArray[np.float64]]
    place: list[NDArray[np.float64]]

    def make_plan(
        self,
        transit: list[ArrayLike],
        transfer: list[ArrayLike],
    ) -> CandidatePlan:
        """Combine the two global paths with the supplied local motions.

        Copies waypoints and preserves the grasp and contact geometry.
        The runner independently validates the complete plan before execution.
        """
        paths = {
            "transit": transit,
            "approach": self.approach,
            "lift": self.lift,
            "transfer": transfer,
            "place": self.place,
        }
        phases = {
            name: [np.asarray(q, dtype=np.float64).copy() for q in path]
            for name, path in paths.items()
        }
        if any(not path for path in phases.values()):
            raise ValueError("plan paths must be nonempty")
        for left, right in zip(REQUIRED_PHASES[:-1], REQUIRED_PHASES[1:]):
            a, b = phases[left][-1], phases[right][0]
            if a.shape != b.shape or not np.allclose(a, b, atol=1e-6, rtol=0.0):
                raise ValueError("plan phases must join continuously")
        return CandidatePlan(
            self.grasp,
            self.gripper_position_m,
            phases,
        )


class ManipulationProblem:
    """Minimal problem interface intentionally exposed to submitted planning code."""

    q_home: NDArray[np.float64]  # transit start
    joint_limits: NDArray[np.float64]  # (6, 2) arm limits in radians, for RRT

    def context_for(
        self, grasp: GraspCandidate
    ) -> CandidateContext | None:  # pragma: no cover
        raise NotImplementedError
