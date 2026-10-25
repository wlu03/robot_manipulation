"""Scene configuration, observations, YCB geometry, and shadow-state helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from .surface import load_ycb_surface as load_ycb_surface
from .transforms import rigid_transform


@dataclass(frozen=True)
class SceneConfig:
    """Staff-level object choice and nominal simulation layout."""

    target_name: str = "plum"
    obstacle_name: str | None = "banana"
    # Spawns are (x, y, yaw in degrees); AMS rests each object on the board.
    target_spawn: tuple[float, float, float] = (0.40, -0.12, 0.0)
    # Keep the convex banana parts clear under the grading perturbations.
    obstacle_spawn: tuple[float, float, float] = (0.47, 0.015, 0.0)
    place_y_m: float = 0.145

    def with_objects(
        self,
        *,
        target_name: str | None = None,
        obstacle_name: str | None = None,
    ) -> "SceneConfig":
        return replace(
            self,
            target_name=self.target_name if target_name is None else str(target_name),
            obstacle_name=(
                self.obstacle_name if obstacle_name is None else str(obstacle_name)
            ),
        )

    def object_specs(
        self, *, with_obstacle: bool = True
    ) -> dict[str, tuple[float, float, float]]:
        if with_obstacle and self.obstacle_name == self.target_name:
            raise ValueError("target and obstacle must be different YCB objects")
        objects = {self.target_name: self.target_spawn}
        if with_obstacle and self.obstacle_name is not None:
            objects[self.obstacle_name] = self.obstacle_spawn
        return objects

    def place_pose(self, T_base_target: ArrayLike) -> NDArray[np.float64]:
        """Keep the observed orientation/height and move the target across the board."""
        T = rigid_transform(T_base_target, "T_base_target").copy()
        T[1, 3] = float(self.place_y_m)
        return T


DEFAULT_SCENE = SceneConfig()


@dataclass(frozen=True)
class SceneObservation:
    """Perception/planning boundary: known object names mapped to base-frame poses."""

    poses: Mapping[str, NDArray[np.float64]]

    def pose(self, object_name: str) -> NDArray[np.float64]:
        try:
            T = self.poses[str(object_name)]
        except KeyError as error:
            raise KeyError(f"no pose observation for {object_name!r}") from error
        return rigid_transform(T, f"pose[{object_name!r}]").copy()


def object_pose(robot, object_name: str) -> NDArray[np.float64]:
    """Read a settled object's pose in the robot base frame through AMS."""
    robot.wait_until_stationary()
    poses = robot.get_object_poses(frame="base")
    return SceneObservation(poses).pose(object_name)


def observe_sim_scene(robot, object_names: Iterable[str]) -> SceneObservation:
    robot.wait_until_stationary()
    poses = robot.get_object_poses(frame="base")
    return SceneObservation({str(name): poses[str(name)].copy() for name in object_names})


def base_pose_in_world(model) -> NDArray[np.float64]:
    """Fixed robot-base transform for collision queries and simulator display."""
    body = model.body("base_link")
    if int(model.body_parentid[body.id]) != 0:
        raise ValueError("the query model must have a base fixed to the world")
    T = np.eye(4)
    wxyz = np.asarray(model.body_quat[body.id])
    T[:3, :3] = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = model.body_pos[body.id]
    return T


def validate_ycb_objects(robot, config: SceneConfig, *, with_obstacle: bool) -> None:
    available = set(robot.scene_builder.registry)
    for name in config.object_specs(with_obstacle=with_obstacle):
        if name not in available:
            raise ValueError(
                f"unknown YCB object {name!r}; available: {sorted(available)}"
            )


def _joint_id(model, name: str) -> int:
    try:
        identifier = int(model.joint(str(name)).id)
    except Exception as error:
        raise ValueError(f"unknown MuJoCo joint {name!r}") from error
    if identifier < 0:
        raise ValueError(f"unknown MuJoCo joint {name!r}")
    return identifier


def scalar_joint_qpos_addresses(model, joint_names: Sequence[str]) -> NDArray[np.int64]:
    addresses = []
    for name in joint_names:
        joint_id = _joint_id(model, name)
        address = int(model.jnt_qposadr[joint_id])
        next_address = (
            int(model.jnt_qposadr[joint_id + 1])
            if joint_id + 1 < int(model.njnt)
            else int(model.nq)
        )
        if next_address - address != 1:
            raise ValueError(f"joint {name!r} is not a scalar hinge/slide coordinate")
        addresses.append(address)
    return np.asarray(addresses, dtype=np.int64)


def set_scalar_joint_positions(data, addresses: ArrayLike, values: ArrayLike) -> None:
    addresses = np.asarray(addresses, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if addresses.ndim != 1 or values.shape != addresses.shape:
        raise ValueError("addresses and values must be matching vectors")
    if not np.all(np.isfinite(values)):
        raise ValueError("joint values must be finite")
    data.qpos[addresses] = values


def free_qpos_from_transform(T_world_body: ArrayLike) -> NDArray[np.float64]:
    T = rigid_transform(T_world_body, "T_world_body")
    xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.r_[T[:3, 3], xyzw[[3, 0, 1, 2]]]


@dataclass(frozen=True)
class FreeBodyHandle:
    body_name: str
    joint_id: int
    qpos_address: int

    @classmethod
    def from_body_name(cls, model, body_name: str) -> "FreeBodyHandle":
        try:
            body = model.body(str(body_name))
        except Exception as error:
            raise ValueError(f"unknown MuJoCo body {body_name!r}") from error
        body_id = int(body.id)
        joint_count = int(model.body_jntnum[body_id])
        first_joint = int(model.body_jntadr[body_id])
        if joint_count != 1 or first_joint < 0:
            raise ValueError(f"body {body_name!r} must have exactly one free joint")
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError("MuJoCo is required to inspect free joints") from error
        if int(model.jnt_type[first_joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"body {body_name!r} does not use a free joint")
        return cls(str(body_name), first_joint, int(model.jnt_qposadr[first_joint]))

    def read_qpos(self, data) -> NDArray[np.float64]:
        return np.asarray(
            data.qpos[self.qpos_address : self.qpos_address + 7],
            dtype=np.float64,
        ).copy()

    def write_qpos(self, data, qpos: ArrayLike) -> None:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (7,) or not np.all(np.isfinite(qpos)):
            raise ValueError("free-body qpos must be a finite seven-vector")
        data.qpos[self.qpos_address : self.qpos_address + 7] = qpos

    def write_transform(self, data, T_world_body: ArrayLike) -> None:
        self.write_qpos(data, free_qpos_from_transform(T_world_body))


class ShadowState:
    """Write one planning state into query data: arm, fingers, and target.

    The target is either held, moving rigidly with the TCP (T_tcp_object), or
    resting at a fixed pose (T_base_object_rest). T_world_base maps the robot
    base into the simulator world.
    """

    def __init__(
        self,
        model,
        *,
        arm_joint_names: Sequence[str],
        finger_joint_names: Sequence[str],
        target_body_name: str,
        tcp_transform,
        finger_qpos: ArrayLike,
        T_tcp_object: ArrayLike | None = None,
        T_base_object_rest: ArrayLike | None = None,
        T_world_base: ArrayLike | None = None,
    ):
        if (T_tcp_object is None) == (T_base_object_rest is None):
            raise ValueError("give exactly one of T_tcp_object or T_base_object_rest")
        self.arm_addresses = scalar_joint_qpos_addresses(model, arm_joint_names)
        self.finger_addresses = scalar_joint_qpos_addresses(model, finger_joint_names)
        self.target = FreeBodyHandle.from_body_name(model, target_body_name)
        self.tcp_transform = tcp_transform
        self.T_world_base = rigid_transform(
            np.eye(4) if T_world_base is None else T_world_base, "T_world_base"
        )
        self.finger_qpos = np.asarray(finger_qpos, dtype=np.float64)
        if self.finger_qpos.shape != (self.finger_addresses.size,) or not np.all(
            np.isfinite(self.finger_qpos)
        ):
            raise ValueError("finger_qpos does not match the finger joints")
        self.T_tcp_object = (
            None if T_tcp_object is None else rigid_transform(T_tcp_object, "T_tcp_object")
        )
        self.rest_qpos = (
            None
            if T_base_object_rest is None
            else free_qpos_from_transform(self.T_world_base @ T_base_object_rest)
        )

    def __call__(self, model, data, q_arm: NDArray[np.float64]) -> None:
        q_arm = np.asarray(q_arm, dtype=np.float64)
        if q_arm.shape != (self.arm_addresses.size,) or not np.all(np.isfinite(q_arm)):
            raise ValueError("q_arm does not match the configured arm joints")
        set_scalar_joint_positions(data, self.arm_addresses, q_arm)
        set_scalar_joint_positions(data, self.finger_addresses, self.finger_qpos)
        if self.T_tcp_object is not None:
            T_base_tcp = rigid_transform(self.tcp_transform(q_arm), "tcp_transform(q)")
            self.target.write_transform(data, self.T_world_base @ T_base_tcp @ self.T_tcp_object)
        else:
            self.target.write_qpos(data, self.rest_qpos)


def clone_query_data(model, source_data):
    """Allocate independent MuJoCo data initialized from the live scene."""

    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo is required to allocate collision-query data"
        ) from error
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(source_data.qpos, dtype=np.float64)
    if data.qvel.shape == source_data.qvel.shape:
        data.qvel[:] = np.asarray(source_data.qvel, dtype=np.float64)
    if (
        hasattr(data, "act")
        and hasattr(source_data, "act")
        and data.act.shape == source_data.act.shape
    ):
        data.act[:] = np.asarray(source_data.act, dtype=np.float64)
    if (
        hasattr(data, "mocap_pos")
        and hasattr(source_data, "mocap_pos")
        and data.mocap_pos.shape == source_data.mocap_pos.shape
    ):
        data.mocap_pos[:] = np.asarray(source_data.mocap_pos, dtype=np.float64)
    if (
        hasattr(data, "mocap_quat")
        and hasattr(source_data, "mocap_quat")
        and data.mocap_quat.shape == source_data.mocap_quat.shape
    ):
        data.mocap_quat[:] = np.asarray(source_data.mocap_quat, dtype=np.float64)
    mujoco.mj_forward(model, data)
    return data
