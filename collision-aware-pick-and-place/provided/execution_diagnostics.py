"""Read-only MuJoCo grasp diagnostics, separate from task perception.

These checks observe the free payload without supplying poses to the controller.
They are simulation instrumentation, not a hardware perception implementation.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .robot import tcp_transform_from_kinematics
from .scene import base_pose_in_world

SLIP_TOLERANCE_M = 0.03
SLIP_TOLERANCE_RAD = np.deg2rad(30.0)
MINIMUM_LIFT_M = 0.025


def _require_held(current, held, phase: str) -> None:
    """Reject excessive translation or rotation relative to the closure pose."""
    position = float(np.linalg.norm(current[:3, 3] - held[:3, 3]))
    rotation = float(Rotation.from_matrix(current[:3, :3] @ held[:3, :3].T).magnitude())
    if position > SLIP_TOLERANCE_M or rotation > SLIP_TOLERANCE_RAD:
        raise RuntimeError(
            f"object slipped during {phase}: {1000 * position:.1f} mm, "
            f"{np.rad2deg(rotation):.1f} deg"
        )


class SimulationGraspMonitor:
    """Measure lift and slip from simulator state without changing live state."""

    def __init__(self, robot, kinematics, target_name: str):
        self.robot = robot
        self.kinematics = kinematics
        self.target_id = int(robot.model.body(target_name).id)
        self.T_base_world = np.linalg.inv(base_pose_in_world(robot.model))
        self.held = None
        self.z_before = None

    def object_in_base(self) -> np.ndarray:
        """Read the actual payload pose and convert world coordinates to base."""
        T_world_object = np.eye(4)
        T_world_object[:3, :3] = self.robot.data.xmat[self.target_id].reshape(3, 3)
        T_world_object[:3, 3] = self.robot.data.xpos[self.target_id]
        return self.T_base_world @ T_world_object

    def __call__(self, phase: str) -> None:
        T_base_object = self.object_in_base()
        q_arm = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)[:6]
        T_base_tcp = tcp_transform_from_kinematics(self.kinematics, q_arm)
        current = np.linalg.inv(T_base_tcp) @ T_base_object
        if phase == "closure":
            self.held = current
            self.z_before = float(T_base_object[2, 3])
            return
        if self.held is None:
            raise RuntimeError("grasp diagnostics require a closure measurement")
        if phase == "lift" and T_base_object[2, 3] < self.z_before + MINIMUM_LIFT_M:
            raise RuntimeError("object did not lift with the gripper")
        _require_held(current, self.held, phase)
