"""Grasp-conditioned TCP targets, expressed with explicit transform frame names."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .robot import grasp_center_to_tcp, offset_grasp_center


@dataclass(frozen=True)
class GraspMotionTargets:
    T_base_tcp_pregrasp: NDArray[np.float64]
    T_base_tcp_grasp: NDArray[np.float64]
    T_base_tcp_lift: NDArray[np.float64]
    T_base_tcp_preplace: NDArray[np.float64]
    T_base_tcp_place: NDArray[np.float64]
    T_tcp_object: NDArray[np.float64]


def grasp_motion_targets(
    grasp,
    T_base_object_pick,
    T_base_object_place,
    calibration,
    *,
    pregrasp_distance_m,
    lift_height_m,
    place_clearance_m,
):
    """Construct approach/lift/place targets while preserving the grasped relation.

    T_A_B maps B coordinates into A. Once grasped,
    T_tcp_object = inv(T_base_tcp_grasp) @ T_base_object_pick, so a desired
    object pose requires T_base_tcp = T_base_object @ inv(T_tcp_object).
    """
    T_base_grasp = T_base_object_pick @ np.asarray(
        grasp.T_object_grasp, dtype=np.float64
    )
    T_base_pregrasp = offset_grasp_center(T_base_grasp, pregrasp_distance_m)
    T_base_lift = T_base_grasp.copy()
    T_base_lift[:3, 3] += [0.0, 0.0, lift_height_m]

    T_base_tcp_grasp = grasp_center_to_tcp(T_base_grasp, calibration.T_tcp_grasp)
    T_base_tcp_pregrasp = grasp_center_to_tcp(T_base_pregrasp, calibration.T_tcp_grasp)
    T_base_tcp_lift = grasp_center_to_tcp(T_base_lift, calibration.T_tcp_grasp)
    T_tcp_object = np.linalg.inv(T_base_tcp_grasp) @ T_base_object_pick

    T_base_object_preplace = T_base_object_place.copy()
    T_base_object_preplace[:3, 3] += [0.0, 0.0, place_clearance_m]
    T_base_tcp_place = T_base_object_place @ np.linalg.inv(T_tcp_object)
    T_base_tcp_preplace = T_base_object_preplace @ np.linalg.inv(T_tcp_object)
    return GraspMotionTargets(
        T_base_tcp_pregrasp,
        T_base_tcp_grasp,
        T_base_tcp_lift,
        T_base_tcp_preplace,
        T_base_tcp_place,
        T_tcp_object,
    )
