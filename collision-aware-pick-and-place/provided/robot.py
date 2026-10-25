"""Trusted PiPER kinematics, gripper conventions, and transform helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from .transforms import rigid_transform

INITIAL_Q = np.array([0.0, 0.65, -1.30, 0.0, 0.65, 0.0, 0.0], dtype=np.float64)

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
FINGER_JOINT_NAMES = ("joint7", "joint8")
BASE_BODY_NAMES = ("base_link",)
ARM_BODY_NAMES = (*(f"link{i}" for i in range(1, 7)), "gripper_base")
FINGER_BODY_NAMES = ("link7", "link8")
# Scene geometry the resting target may touch, matched exactly against geom and
# body names. Setup fails if the settled target touches none of it.
SUPPORT_NAMES = ("floor", "breadboard", "breadboard_collision")


def transform_to_api_pose(T_base_tcp: ArrayLike) -> NDArray[np.float64]:
    T = rigid_transform(T_base_tcp, "T_base_tcp")
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_euler("xyz")]


def tcp_transform_from_kinematics(kinematics, q_arm: ArrayLike) -> NDArray[np.float64]:
    q = np.asarray(q_arm, dtype=np.float64)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("q_arm must be a finite six-vector")
    pose = np.asarray(kinematics.computeFK(q)[-1], dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise RuntimeError("computeFK returned an invalid TCP pose")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = kinematics.euler_to_rotation_matrix(np.deg2rad(pose[3:]))
    T[:3, 3] = pose[:3] * 1e-3
    return T


class TrustedPiperKinematics:
    """Supplied Jacobian and damped least-squares IK using the AMS PiPER model.

    Lazy loading keeps geometric helpers usable without AMS installed.
    """

    def __init__(self):
        try:
            from ams.utilities.piper_kinematics import PiperKinematics
        except ImportError as error:
            raise RuntimeError(
                "TrustedPiperKinematics requires the armlab_mujoco_simulator package"
            ) from error
        self._base = PiperKinematics()

        # Use AMS geometry and limits; solve with a bounded course IK budget.
        self._ik_max_iterations = 120
        self._ik_damping_factor = 0.05
        self._ik_step_limit_rad = 0.40

    def __getattr__(self, name):
        return getattr(self._base, name)

    def computeFK(self, q):
        return self._base.computeFK(q)

    def euler_to_rotation_matrix(self, angles):
        return self._base.euler_to_rotation_matrix(angles)

    def _link_transforms(self, q):
        """Return cumulative modified-DH transforms in metres.

        Direct matrix composition avoids Euler-angle conversions during IK.
        """
        alpha = np.asarray(self._base._alpha)
        theta = np.asarray(q) + np.asarray(self._base._theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        a = 1e-3 * np.asarray(self._base._a)
        d = 1e-3 * np.asarray(self._base._d)
        links = np.zeros((6, 4, 4))
        links[:, 0, 0] = ct
        links[:, 0, 1] = -st
        links[:, 0, 3] = a
        links[:, 1, 0] = st * ca
        links[:, 1, 1] = ct * ca
        links[:, 1, 2] = -sa
        links[:, 1, 3] = -sa * d
        links[:, 2, 0] = st * sa
        links[:, 2, 1] = ct * sa
        links[:, 2, 2] = ca
        links[:, 2, 3] = ca * d
        links[:, 3, 3] = 1.0
        for index in range(1, 6):
            links[index] = links[index - 1] @ links[index]
        return links

    @staticmethod
    def _jacobian_from_transforms(links):
        """Stack linear z_i x (p_tcp - p_i) and angular z_i Jacobian columns.

        z_i is the joint axis; p_i and p_tcp are joint and TCP origins in the base.
        """
        axes = links[:, :3, 2]
        offsets = links[-1, :3, 3] - links[:, :3, 3]
        return np.vstack((np.cross(axes, offsets).T, axes.T))

    def computeJacobian_ee(self, cur_j):
        """Return the base-frame 6x6 geometric Jacobian, linear rows first."""
        q = np.asarray(cur_j, dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("cur_j must be a finite six-vector")
        return self._jacobian_from_transforms(self._link_transforms(q))

    def computeIK(self, target_pose, initial_guess):
        """Solve for xyz (metres) and xyz Euler angles (radians); retain the gripper.

        Use dq = J.T @ solve(J @ J.T + damping**2 * I, pose_error), where I is
        the 6x6 identity. Orientation error is the rotation vector of
        R_target @ R_current.T. Bound each increment and project onto joint limits.
        """
        target = np.asarray(target_pose, dtype=np.float64)
        initial = np.asarray(initial_guess, dtype=np.float64)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError("target_pose must be a finite six-vector")
        if initial.shape != (7,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_guess must be a finite seven-vector")
        q = initial.copy()
        limits = np.asarray(self.joint_position_limits[:6], dtype=np.float64)
        q[:6] = np.clip(q[:6], limits[:, 0], limits[:, 1])
        R_target = self.euler_to_rotation_matrix(target[3:])
        for _ in range(self._ik_max_iterations):
            links = self._link_transforms(q[:6])
            current_position = links[-1, :3, 3]
            position_error = target[:3] - current_position
            R_current = links[-1, :3, :3]
            orientation_error = Rotation.from_matrix(R_target @ R_current.T).as_rotvec()
            if (
                np.linalg.norm(position_error) <= self._ik_position_tolerance
                and np.linalg.norm(orientation_error) <= self._ik_orientation_tolerance
            ):
                return q
            error = np.concatenate((position_error, orientation_error))
            J = self._jacobian_from_transforms(links)
            damping = float(self._ik_damping_factor)
            delta_q = J.T @ np.linalg.solve(
                J @ J.T + damping**2 * np.eye(6),
                error,
            )
            delta_q = np.clip(
                delta_q,
                -self._ik_step_limit_rad,
                self._ik_step_limit_rad,
            )
            q[:6] = np.clip(q[:6] + delta_q, limits[:, 0], limits[:, 1])
        raise ValueError("IK did not converge within the maximum number of iterations")


class GripperPositionAdapter:
    """Keep MP2's one-finger coordinates identical across AMS backends.

    For q[6], AMS simulation uses half the jaw opening; hardware uses the full
    opening. Hardware commands are doubled and readings/limits are halved.
    Shadow-state planning uses one-finger coordinates; zero closes both backends.
    """

    def __init__(self, robot, *, run_mode: str):
        if run_mode not in ("sim", "real"):
            raise ValueError("run_mode must be 'sim' or 'real'")
        self.robot = robot
        self.gripper_position_scale = 1.0 if run_mode == "sim" else 2.0

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def set_joint_positions(self, positions) -> None:
        command = np.asarray(positions, dtype=np.float64).copy()
        if command.shape != (7,) or not np.all(np.isfinite(command)):
            raise ValueError("positions must be a finite seven-vector")
        command[6] *= self.gripper_position_scale
        self.robot.set_joint_positions(command)

    def get_joint_positions(self) -> NDArray[np.float64]:
        positions = np.asarray(
            self.robot.get_joint_positions(), dtype=np.float64
        ).copy()
        positions[6] /= self.gripper_position_scale
        return positions


def runtime_finger_limit(robot) -> float:
    limits = np.asarray(
        robot.piper_kinematics.GRIPPER_POSITION_LIMITS, dtype=np.float64
    )
    if (
        limits.shape != (2,)
        or not np.all(np.isfinite(limits))
        or limits[1] <= limits[0]
    ):
        raise RuntimeError("invalid runtime gripper limits")
    return float(limits[1]) / getattr(robot, "gripper_position_scale", 1.0)


# The PiPER kinematic TCP lies midway between the finger pads. The grasp frame
# (+x closing, +z approach) is the TCP frame rotated +90 degrees about its z-axis.
T_TCP_GRASP = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
# The finger pads meet when the gripper is fully closed.
CLOSED_JAW_WIDTH_M = 0.0


@dataclass(frozen=True)
class GripperCalibration:
    """Fixed PiPER grasp-frame constants plus the finger collision geoms."""

    T_tcp_grasp: NDArray[np.float64]
    finger_geom_ids: tuple[frozenset[int], frozenset[int]]
    closed_jaw_width_m: float

    def maximum_jaw_width(self, finger_limit_m: float) -> float:
        return float(self.closed_jaw_width_m + 2.0 * float(finger_limit_m))


def gripper_calibration(model) -> GripperCalibration:
    """Return the fixed grasp-frame constants and each finger's collision geoms."""
    from .contacts import geom_ids_for_bodies

    fingers = tuple(geom_ids_for_bodies(model, (name,)) for name in FINGER_BODY_NAMES)
    if any(not group for group in fingers):
        raise RuntimeError("finger collision geometry could not be identified")
    return GripperCalibration(T_TCP_GRASP.copy(), fingers, CLOSED_JAW_WIDTH_M)


def contact_opening_for_width(
    width_m: float, *, finger_limit_m: float, closed_jaw_width_m: float
) -> float | None:
    """One-finger opening at which both pads touch contacts width_m apart.

    Returns None when the gripper cannot open or close to that width.
    """
    opening = 0.5 * (float(width_m) - float(closed_jaw_width_m))
    if not np.isfinite(opening) or opening < 0.0 or opening > float(finger_limit_m):
        return None
    return opening


def grasp_center_to_tcp(
    T_base_grasp: ArrayLike, T_tcp_grasp: ArrayLike
) -> NDArray[np.float64]:
    return rigid_transform(T_base_grasp, "T_base_grasp") @ np.linalg.inv(
        rigid_transform(T_tcp_grasp, "T_tcp_grasp")
    )


def offset_grasp_center(
    T_base_grasp: ArrayLike, distance_m: float
) -> NDArray[np.float64]:
    T = rigid_transform(T_base_grasp, "T_base_grasp").copy()
    distance_m = float(distance_m)
    if not np.isfinite(distance_m) or distance_m < 0.0:
        raise ValueError("distance_m must be finite and nonnegative")
    T[:3, 3] -= distance_m * T[:3, 2]
    return T
