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
    # contact_a, contact_b, (3,), [x,y,z] coordinates in meters 
    # normal_a, normal_a, (3,), [nx, ny, nz] components
    p_a = np.asarray(contact_a, dtype=np.float64)
    p_b = np.asarray(contact_b, dtype=np.float64)
    n_a = np.asarray(normal_a, dtype=np.float64)
    n_b = np.asarray(normal_b, dtype=np.float64)
    norm_na = n_a / np.linalg.norm(n_a)
    norm_nb = n_b / np.linalg.norm(n_b)
    L = np.linalg.norm(p_b - p_a)
    if not np.isfinite(L) or L == 0.0:
        return False
    if L < minimum_width_m or L > maximum_width_m: 
        return False 


    d = (p_b - p_a) / L # distance vector

    # since the normals are pointed outwards
    # the inwards normals are just the -(n)

    mu = friction_coefficient
    alpha = np.arctan(mu)

    # d for A
    # -d for B
    theta_a = np.arccos(np.dot(d, -norm_na) / np.linalg.norm(-norm_na) * np.linalg.norm(d))
    theta_b = np.arccos(np.dot(-d, -norm_nb) / np.linalg.norm(-norm_nb) * np.linalg.norm(-d))
    max_theta = max(theta_a, theta_b)
    if max_theta > alpha: 
        return False
    return True



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
    