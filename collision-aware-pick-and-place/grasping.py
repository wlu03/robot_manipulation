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
    p_a = np.asarray(contact_a, dtype=np.float64)
    p_b = np.asarray(contact_b, dtype=np.float64)
    n_a = np.asarray(normal_a, dtype=np.float64)
    n_b = np.asarray(normal_b, dtype=np.float64)

    length_a = np.linalg.norm(n_a)
    length_b = np.linalg.norm(n_b)
    if (
        not np.isfinite(length_a)
        or not np.isfinite(length_b)
        or length_a == 0.0
        or length_b == 0.0
    ):
        return False

    n_a = n_a / length_a
    n_b = n_b / length_b

    separation = p_b - p_a
    width = np.linalg.norm(separation)
    if not np.isfinite(width) or width == 0.0:
        return False

    if width < minimum_width_m or width > maximum_width_m:
        return False

    if not np.isfinite(friction_coefficient) or friction_coefficient < 0.0:
        return False

    d = separation / width
    alpha = np.arctan(friction_coefficient)

    cos_theta_a = np.clip(np.dot(d, -n_a), -1.0, 1.0)
    cos_theta_b = np.clip(np.dot(-d, -n_b), -1.0, 1.0)
    theta_a = np.arccos(cos_theta_a)
    theta_b = np.arccos(cos_theta_b)

    return bool(theta_a <= alpha and theta_b <= alpha)



def make_grasp_frame(
    contact_a: ArrayLike,
    contact_b: ArrayLike,
    *,
    preferred_approach: ArrayLike = (0.0, 0.0, -1.0),
) -> NDArray[np.float64]:
    """Return T_object_grasp with +x closing and +z approach."""
    p_a = np.asarray(contact_a, dtype=np.float64)
    p_b = np.asarray(contact_b, dtype=np.float64)
    approach = np.asarray(preferred_approach, dtype=np.float64)

    separation = p_b - p_a
    width = np.linalg.norm(separation)
    if not np.isfinite(width) or width == 0.0:
        raise ValueError("contacts must be distinct finite points")

    center = (p_a + p_b) / 2.0
    x_G = separation / width

    # Project the requested approach onto the plane perpendicular to +x.
    z_G = approach - np.dot(approach, x_G) * x_G
    z_norm = np.linalg.norm(z_G)

    # A parallel (or zero) approach has no perpendicular projection. Choose the
    # coordinate axis least aligned with +x, then project that axis instead.
    if not np.isfinite(z_norm) or z_norm <= 1e-12:
        fallback = np.zeros(3, dtype=np.float64)
        fallback[np.argmin(np.abs(x_G))] = 1.0
        z_G = fallback - np.dot(fallback, x_G) * x_G
        z_norm = np.linalg.norm(z_G)

    z_G = z_G / z_norm

    # +y = +z x +x produces a right-handed frame: +x x +y = +z.
    y_G = np.cross(z_G, x_G)
    y_G = y_G / np.linalg.norm(y_G)

    # Recompute +z to remove the small numerical error from projection.
    z_G = np.cross(x_G, y_G)

    T_object_grasp = np.eye(4)
    T_object_grasp[:3, 0] = x_G
    T_object_grasp[:3, 1] = y_G
    T_object_grasp[:3, 2] = z_G
    T_object_grasp[:3, 3] = center

    return T_object_grasp


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
    points = np.asarray(points_object, dtype=np.float64)
    normals = np.asarray(normals_object, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3 or points.shape != normals.shape:
        raise ValueError(
            "points_object and normals_object must both have shape (N, 3)"
        )

    point_count = points.shape[0]
    if point_count < 2 or sample_count <= 0 or maximum_candidates <= 0:
        return []

    candidates: list[GraspCandidate] = []
    sampled_pairs: set[tuple[int, int]] = set()

    for _ in range(sample_count):
        index_a, index_b = rng.choice(point_count, size=2, replace=False)
        index_a = int(index_a)
        index_b = int(index_b)

        # Swapping the two indices describes the same physical contact pair.
        pair_key = (min(index_a, index_b), max(index_a, index_b))
        if pair_key in sampled_pairs:
            continue
        sampled_pairs.add(pair_key)

        contact_a = points[index_a]
        contact_b = points[index_b]
        normal_a = normals[index_a]
        normal_b = normals[index_b]

        if not is_antipodal(
            contact_a,
            normal_a,
            contact_b,
            normal_b,
            friction_coefficient=friction_coefficient,
            minimum_width_m=minimum_width_m,
            maximum_width_m=maximum_width_m,
        ):
            continue

        separation = contact_b - contact_a
        jaw_width = np.linalg.norm(separation)
        d = separation / jaw_width

        norm_a = np.linalg.norm(normal_a)
        norm_b = np.linalg.norm(normal_b)
        normal_a_unit = normal_a / norm_a
        normal_b_unit = normal_b / norm_b

        a_aligned = np.clip(np.dot(d, -normal_a_unit), -1.0, 1.0)
        b_aligned = np.clip(np.dot(-d, -normal_b_unit), -1.0, 1.0)
        score = float(min(a_aligned, b_aligned))

        if not np.isfinite(score):
            continue

        T_object_grasp = make_grasp_frame(
            contact_a, contact_b, preferred_approach=preferred_approach
        )

        candidates.append(
            GraspCandidate(
                contact_a_object=contact_a.copy(),
                contact_b_object=contact_b.copy(),
                T_object_grasp=T_object_grasp,
                jaw_width_m=float(jaw_width),
                score=score,
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates[:maximum_candidates]
