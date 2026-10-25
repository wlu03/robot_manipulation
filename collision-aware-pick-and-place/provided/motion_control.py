"""Execute joint paths through the AMS position interface and enforce contact rules.

Simulation-only: hardware execution needs its own backend.

Paths are executed as constant-speed straight lines in joint space: every
control step commands the next point on the line, so the arm follows the edges
the planner validated. Contacts are checked against the phase's allowed set.
"""

from __future__ import annotations

import numpy as np

from .contacts import (
    ALLOWED_CONTACTS,
    COLLISION_PENETRATION_TOLERANCE_M,
    CollisionMode,
    contact_label,
    contact_labels,
    read_contacts,
    target_environment_contacts,
    target_finger_contacts,
)

JOINT_SPEED_RAD_S = 0.25  # well below the PiPER joint speed limits
GRIPPER_SPEED_M_S = 0.07  # per finger


def require_allowed_contacts(robot, groups, mode: CollisionMode) -> None:
    """Raise if the live scene has a contact that this phase does not allow."""
    tolerance = -COLLISION_PENETRATION_TOLERANCE_M
    labels = contact_labels(robot.model, robot.data, groups, clearance_m=tolerance)
    forbidden = labels - ALLOWED_CONTACTS[CollisionMode(mode)]
    if not forbidden:
        return
    pairs = [
        f"{robot.model.geom(c.geom1).name or c.geom1}<->"
        f"{robot.model.geom(c.geom2).name or c.geom2} ({1000 * c.distance_m:.2f} mm)"
        for c in read_contacts(robot.model, robot.data, maximum_distance_m=tolerance)
        if contact_label(c.geom1, c.geom2, groups) in forbidden
    ]
    raise RuntimeError(
        f"forbidden contact during {CollisionMode(mode).value}: "
        f"{', '.join(sorted(forbidden))}; pairs=[{', '.join(pairs)}]"
    )


def _step(robot, groups, mode) -> None:
    """Advance through AMS and read the resulting contacts."""
    robot.step()
    if groups is not None and mode is not None:
        require_allowed_contacts(robot, groups, mode)


def hold(
    robot, seconds: float, *, groups=None, mode: CollisionMode | None = None
) -> None:
    """Keep the current command for a fixed time."""
    for _ in range(int(np.ceil(seconds / robot.TIME_STEP))):
        _step(robot, groups, mode)


def settle(robot, q_arm, gripper_m: float, *, groups=None, mode=None) -> None:
    """Use AMS settling and its 10 mrad arm-position tolerance."""
    from ams.utilities.misc import at_position

    target = np.r_[np.asarray(q_arm, dtype=np.float64), float(gripper_m)]
    robot.set_joint_positions(target)
    robot.wait_until_stationary()
    if groups is not None and mode is not None:
        require_allowed_contacts(robot, groups, mode)
    q = np.asarray(robot.get_joint_positions(), dtype=np.float64)
    # A held object prevents the fingers from reaching their zero command.
    if not at_position(q[:6], target[:6]):
        error = np.array2string(q[:6] - target[:6], precision=4)
        raise RuntimeError(f"arm did not reach its commanded configuration; error={error}")


def follow_path(
    robot,
    path,
    gripper_m: float,
    *,
    groups,
    mode: CollisionMode,
    final_mode: CollisionMode | None = None,
    stop_on_support: bool = False,
) -> list[np.ndarray]:
    """Move along a joint path at constant speed and return the visited waypoints.

    With stop_on_support, motion stops as soon as the held object touches the
    support. final_mode applies while settling at the last waypoint.
    """
    dt = float(robot.TIME_STEP)
    visited = [np.asarray(robot.get_joint_positions(), dtype=np.float64)[:6].copy()]
    for q0, q1 in zip(path[:-1], path[1:]):
        q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
        steps = max(1, int(np.ceil(np.max(np.abs(q1 - q0)) / (JOINT_SPEED_RAD_S * dt))))
        for k in range(1, steps + 1):
            robot.set_joint_positions(np.r_[q0 + (k / steps) * (q1 - q0), gripper_m])
            _step(robot, groups, mode)
            if (
                stop_on_support
                and target_environment_contacts(robot.model, robot.data, groups)[0]
            ):
                q_now = np.asarray(robot.get_joint_positions(), dtype=np.float64)[:6]
                robot.set_joint_positions(np.r_[q_now, gripper_m])
                visited.append(q_now.copy())
                return visited
        visited.append(q1.copy())
    settle(robot, path[-1], gripper_m, groups=groups, mode=final_mode or mode)
    return visited


def move_gripper(robot, target_m: float, *, groups=None, mode=None) -> None:
    """Ramp the finger command to target_m with the arm held, then let it settle."""
    q = np.asarray(robot.get_joint_positions(), dtype=np.float64)
    q_arm, start = q[:6].copy(), float(q[6])
    dt = float(robot.TIME_STEP)
    steps = max(1, int(np.ceil(abs(target_m - start) / (GRIPPER_SPEED_M_S * dt))))
    for k in range(1, steps + 1):
        robot.set_joint_positions(
            np.r_[q_arm, start + (k / steps) * (target_m - start)]
        )
        _step(robot, groups, mode)
    hold(robot, 0.2, groups=groups, mode=mode)


def close_on_object(robot, groups, finger_groups) -> None:
    """Close the fingers fully; the object should stop them with two-sided contact."""
    move_gripper(robot, 0.0, groups=groups, mode=CollisionMode.CLOSURE)
    if not all(target_finger_contacts(robot.model, robot.data, groups, finger_groups)):
        raise RuntimeError("the gripper closed without two-sided contact on the object")
