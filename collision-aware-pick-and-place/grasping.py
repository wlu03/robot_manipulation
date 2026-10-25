"""Geometric parallel-jaw grasp synthesis for MP2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class GraspCandidate:
    contact_a_object: NDArray[np.float64]
    contact_b_object: NDArray[np.float64]
    T_object_grasp: NDArray[np.float64]
    jaw_width_m: float
    score: float


def is_antipodal(
    contact_a: ArrayLike,
    normal_a: ArrayLike,
    contact_b: ArrayLike,
    normal_b: ArrayLike,
    *,
    friction_coefficient: float = 0.5,
    minimum_width_m: float = 0.005,
    maximum_width_m: float = 0.070,
) -> bool:
    """Test jaw width and both inward friction cones using outward normals."""
    # contact_a, contact_b: (3,), [x, y, z] coordinates in meters
    # normal_a, normal_b: (3,), [nx, ny, nz] components
    print("contact_a:", contact_a)
    print("normal_a:", normal_a)
    print("contact_b:", contact_b)
    print("normal_b:", normal_b)
    print("friction_coefficient:", friction_coefficient)
    print("minimum_width_m:", minimum_width_m)
    print("maximum_width_m:", maximum_width_m)


def make_grasp_frame(
    contact_a: ArrayLike,
    contact_b: ArrayLike,
    *,
    preferred_approach: ArrayLike = (0.0, 0.0, -1.0),
) -> NDArray[np.float64]:
    """Return T_object_grasp with +x closing and +z approach."""
    # TODO(MP2)
    raise NotImplementedError


def generate_grasp_candidates(
    points_object: ArrayLike,
    normals_object: ArrayLike,
    *,
    rng: np.random.Generator,
    sample_count: int = 6000,
    friction_coefficient: float = 0.5,
    minimum_width_m: float = 0.005,
    maximum_width_m: float = 0.070,
    preferred_approach: ArrayLike = (0.0, 0.0, -1.0),
    maximum_candidates: int = 100,
) -> list[GraspCandidate]:
    """Sample pairs from (N,3) points/normals and rank valid object-frame grasps.

    Each frame may be rotated about its +x (closing) axis; the robot uses it as
    returned, so the approach directions you propose are part of your design.
    """
    # TODO(MP2)
    raise NotImplementedError
