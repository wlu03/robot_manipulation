"""Execute a completed MP2 plan in the live MuJoCo scene and check the outcome.

Simulation-only: hardware execution needs its own backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from .contacts import CollisionMode, target_environment_contacts, target_finger_contacts
from .motion_control import close_on_object, follow_path, hold, move_gripper
from .scene import object_pose

PLACEMENT_TOLERANCE_M = 0.02  # same tolerance as hardware grading


@dataclass(frozen=True)
class ExecutionMetrics:
    final_position_error_m: float
    final_rotation_error_rad: float


def execute_pick_and_place(
    robot,
    plan,
    problem,
    calibration,
    *,
    T_place,
    open_gripper: float,
    phase_check: Callable[[str], None] | None = None,
) -> ExecutionMetrics:
    """Execute using initial/final perception; optional diagnostics check each phase.

    Simulation diagnostics do not provide object poses to the controller.
    """
    groups = problem.groups
    target = problem.target_object_name
    phases = plan.phases

    # Pick with open fingers, then close on the object.
    move_gripper(robot, open_gripper, groups=groups, mode=CollisionMode.FREE)
    follow_path(
        robot, phases["transit"], open_gripper, groups=groups, mode=CollisionMode.FREE
    )
    follow_path(
        robot, phases["approach"], open_gripper, groups=groups, mode=CollisionMode.FREE
    )
    close_on_object(robot, groups, calibration.finger_geom_ids)
    hold(robot, 0.4, groups=groups, mode=CollisionMode.CLOSURE)
    if not all(
        target_finger_contacts(
            robot.model, robot.data, groups, calibration.finger_geom_ids
        )
    ):
        raise RuntimeError("two-sided contact was lost after closing the gripper")
    if phase_check is not None:
        phase_check("closure")

    # Lift and carry with the gripper commanded fully closed.
    follow_path(
        robot,
        phases["lift"],
        0.0,
        groups=groups,
        mode=CollisionMode.LIFT,
        final_mode=CollisionMode.CARRY,
    )
    if phase_check is not None:
        phase_check("lift")
    follow_path(robot, phases["transfer"], 0.0, groups=groups, mode=CollisionMode.CARRY)
    if phase_check is not None:
        phase_check("transfer")

    # Descend until the object touches the support, release, and back out.
    descent = follow_path(
        robot,
        phases["place"],
        0.0,
        groups=groups,
        mode=CollisionMode.PLACE,
        stop_on_support=True,
    )
    if phase_check is not None:
        phase_check("placement")
    move_gripper(robot, open_gripper, groups=groups, mode=CollisionMode.CLOSURE)
    hold(robot, 0.8, groups=groups, mode=CollisionMode.CLOSURE)
    follow_path(
        robot, descent[::-1], open_gripper, groups=groups, mode=CollisionMode.FREE
    )
    hold(robot, 0.9, groups=groups, mode=CollisionMode.FREE)
    return _check_placement(robot, groups, target, T_place)


def _check_placement(robot, groups, target, T_place) -> ExecutionMetrics:
    T_final = object_pose(robot, target)
    on_support, on_obstacle = target_environment_contacts(
        robot.model, robot.data, groups
    )
    if not on_support:
        raise RuntimeError("placed object is not resting on the support")
    if on_obstacle:
        raise RuntimeError("placed object is touching an obstacle")
    position_error = float(np.linalg.norm(T_final[:3, 3] - T_place[:3, 3]))
    rotation_error = float(
        Rotation.from_matrix(T_final[:3, :3] @ T_place[:3, :3].T).magnitude()
    )
    if position_error > PLACEMENT_TOLERANCE_M:
        raise RuntimeError(
            f"placed object is {1000 * position_error:.1f} mm from the goal "
            f"(tolerance {1000 * PLACEMENT_TOLERANCE_M:.0f} mm)"
        )
    print(
        f"PASS: placed {target}; position error={1000 * position_error:.1f} mm, "
        f"rotation error={np.rad2deg(rotation_error):.1f} deg"
    )
    return ExecutionMetrics(position_error, rotation_error)
