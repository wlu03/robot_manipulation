"""Fast student-facing visualization helpers for MP2.

Grasp inspection and plan preview update poses and call mj_forward() without
stepping dynamics. Live-viewer throttling limits rendering calls independently
of the physics and control trajectory.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .paths import interpolate_joint_edge
from .robot import (
    ARM_JOINT_NAMES,
    FINGER_JOINT_NAMES,
    grasp_center_to_tcp,
    tcp_transform_from_kinematics,
)
from .scene import (
    FreeBodyHandle,
    base_pose_in_world,
    scalar_joint_qpos_addresses,
    set_scalar_joint_positions,
)


class ThrottledViewer:
    """Proxy a passive MuJoCo viewer while limiting expensive sync calls.

    ARMLab requests a sync every physics step. Limit regular syncs to one per
    frame period in simulation time; force_sync() displays immediately.
    """

    def __init__(self, viewer, data, *, fps: float = 30.0):
        fps = float(fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("viewer fps must be finite and positive")
        self._viewer = viewer
        self._data = data
        self._period_s = 1.0 / fps
        self._next_sync_time = -np.inf

    def sync(self) -> None:
        now = float(self._data.time)
        if now + 1e-12 < self._next_sync_time:
            return
        self._viewer.sync()
        self._next_sync_time = now + self._period_s

    def force_sync(self) -> None:
        self._viewer.sync()
        self._next_sync_time = float(self._data.time) + self._period_s

    def __getattr__(self, name):
        return getattr(self._viewer, name)


def install_viewer_throttle(robot, *, fps: float = 30.0) -> None:
    """Throttle passive-viewer rendering without altering simulation stepping."""

    viewer = getattr(robot, "viewer", None)
    if viewer is None or isinstance(viewer, ThrottledViewer):
        return
    robot.viewer = ThrottledViewer(viewer, robot.data, fps=fps)


def _sync(robot, *, force: bool = False) -> None:
    viewer = getattr(robot, "viewer", None)
    if viewer is None:
        return
    if force and hasattr(viewer, "force_sync"):
        viewer.force_sync()
    else:
        viewer.sync()


def _add_sphere(scene, position: ArrayLike, radius: float, rgba: ArrayLike) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([float(radius), 0.0, 0.0], dtype=np.float64),
        pos=np.asarray(position, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_arrow(
    scene,
    start: ArrayLike,
    end: ArrayLike,
    width: float,
    rgba: ArrayLike,
) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([width, width, width], dtype=np.float64),
        pos=np.asarray(start, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        float(width),
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def _add_box(
    scene,
    center: ArrayLike,
    half_size: ArrayLike,
    rotation: ArrayLike,
    rgba: ArrayLike,
) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=np.asarray(half_size, dtype=np.float64),
        pos=np.asarray(center, dtype=np.float64),
        mat=np.asarray(rotation, dtype=np.float64).reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def draw_grasp_candidates(
    robot,
    T_base_object: ArrayLike,
    candidates: Sequence,
    *,
    selected_index: int = 0,
    top_k: int = 10,
    surface_points_object: ArrayLike | None = None,
    surface_normals_object: ArrayLike | None = None,
    maximum_surface_points: int = 90,
    maximum_normal_arrows: int = 24,
) -> None:
    """Draw ranked antipodal pairs and one selected abstract parallel-jaw grasp.

    Transform proposed object-frame contacts and axes into the base for display,
    before IK/collision checks.
    """

    viewer = getattr(robot, "viewer", None)
    if viewer is None or getattr(viewer, "user_scn", None) is None:
        raise RuntimeError("grasp visualization requires the interactive MuJoCo viewer")
    if not candidates:
        raise ValueError("at least one grasp candidate is required")

    T = np.asarray(T_base_object, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("T_base_object must be 4x4")
    T = base_pose_in_world(robot.model) @ T
    R, t = T[:3, :3], T[:3, 3]
    selected_index = int(np.clip(selected_index, 0, len(candidates) - 1))
    top_k = max(1, min(int(top_k), len(candidates)))

    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0

        _draw_surface(
            scene,
            R,
            t,
            surface_points_object,
            surface_normals_object,
            maximum_surface_points,
            maximum_normal_arrows,
        )
        _draw_ranked_pairs(scene, R, t, candidates, top_k)
        _draw_selected_grasp(scene, T, R, t, candidates[selected_index])

    _sync(robot, force=True)


def _draw_surface(
    scene,
    R,
    t,
    surface_points_object,
    surface_normals_object,
    maximum_surface_points,
    maximum_normal_arrows,
):
    # Sparse surface samples/normals provide context without hiding the object.
    if surface_points_object is not None:
        points = np.asarray(surface_points_object, dtype=np.float64)
        if points.ndim == 2 and points.shape[1:] == (3,) and len(points):
            ids = np.linspace(
                0,
                len(points) - 1,
                min(int(maximum_surface_points), len(points)),
                dtype=int,
            )
            world = points[ids] @ R.T + t
            for p in world:
                _add_sphere(scene, p, 0.0012, [0.55, 0.58, 0.62, 0.35])

            if surface_normals_object is not None:
                normals = np.asarray(surface_normals_object, dtype=np.float64)
                if normals.shape == points.shape:
                    normal_ids = ids[
                        np.linspace(
                            0,
                            len(ids) - 1,
                            min(int(maximum_normal_arrows), len(ids)),
                            dtype=int,
                        )
                    ]
                    for idx in normal_ids:
                        p = R @ points[idx] + t
                        n = R @ normals[idx]
                        n /= max(np.linalg.norm(n), 1e-12)
                        _add_arrow(
                            scene,
                            p,
                            p + 0.012 * n,
                            0.0007,
                            [0.55, 0.58, 0.62, 0.38],
                        )


def _draw_ranked_pairs(scene, R, t, candidates, top_k):
    # Top-ranked accepted pairs.
    for rank, grasp in enumerate(candidates[:top_k]):
        p_a = R @ np.asarray(grasp.contact_a_object, dtype=np.float64) + t
        p_b = R @ np.asarray(grasp.contact_b_object, dtype=np.float64) + t
        alpha = 0.28 + 0.42 * (1.0 - rank / max(1, top_k - 1))
        _add_arrow(
            scene,
            p_a,
            p_b,
            0.0009,
            [0.10, 0.65, 0.92, alpha],
        )


def _draw_selected_grasp(scene, T, R, t, grasp):
    # Selected candidate: contacts, closing axis, approach, and ghost jaws.
    p_a = R @ np.asarray(grasp.contact_a_object, dtype=np.float64) + t
    p_b = R @ np.asarray(grasp.contact_b_object, dtype=np.float64) + t
    T_base_grasp = T @ np.asarray(grasp.T_object_grasp, dtype=np.float64)
    R_g = T_base_grasp[:3, :3]
    center = T_base_grasp[:3, 3]
    x_axis, z_axis = R_g[:, 0], R_g[:, 2]

    _add_sphere(scene, p_a, 0.0040, [0.08, 0.42, 0.95, 1.0])
    _add_sphere(scene, p_b, 0.0040, [0.08, 0.42, 0.95, 1.0])
    _add_arrow(scene, p_a, p_b, 0.0020, [0.08, 0.42, 0.95, 1.0])
    _add_arrow(
        scene,
        center - 0.050 * z_axis,
        center,
        0.0022,
        [0.55, 0.24, 0.78, 0.95],
    )

    # Ghost pads show the geometric grasp; feasibility checks use PiPER meshes.
    pad_offset = 0.0045
    half_size = np.array([0.0035, 0.011, 0.021], dtype=np.float64)
    _add_box(
        scene,
        p_a - pad_offset * x_axis,
        half_size,
        R_g,
        [0.12, 0.48, 0.95, 0.26],
    )
    _add_box(
        scene,
        p_b + pad_offset * x_axis,
        half_size,
        R_g,
        [0.12, 0.48, 0.95, 0.26],
    )


def _densify_path(
    path: Sequence[ArrayLike],
    *,
    max_joint_step: float,
) -> list[NDArray[np.float64]]:
    points = [np.asarray(q, dtype=np.float64).copy() for q in path]
    if not points:
        return []
    if len(points) == 1:
        return points
    dense = [points[0]]
    for q0, q1 in zip(points[:-1], points[1:]):
        dense.extend(interpolate_joint_edge(q0, q1, max_joint_step)[1:])
    return dense


class _PreviewPlayer:
    """Write and display kinematic states; never advance contact dynamics."""

    def __init__(self, robot, target_name, kinematics, T_tcp_object, frame_delay):
        self.robot = robot
        self.T_world_base = base_pose_in_world(robot.model)
        self.kinematics = kinematics
        self.T_tcp_object = T_tcp_object
        self.frame_delay = frame_delay
        self.arm_addresses = scalar_joint_qpos_addresses(robot.model, ARM_JOINT_NAMES)
        self.finger_addresses = scalar_joint_qpos_addresses(
            robot.model, FINGER_JOINT_NAMES
        )
        self.target = FreeBodyHandle.from_body_name(robot.model, str(target_name))

    def carried_pose(self, q_arm):
        return tcp_transform_from_kinematics(self.kinematics, q_arm) @ self.T_tcp_object

    def show(self, q_arm, finger_m, T_object):
        robot = self.robot
        set_scalar_joint_positions(robot.data, self.arm_addresses, q_arm)
        set_scalar_joint_positions(
            robot.data,
            self.finger_addresses,
            np.array([finger_m, -finger_m], dtype=np.float64),
        )
        self.target.write_transform(robot.data, self.T_world_base @ T_object)
        robot.data.qvel[:] = 0.0
        mujoco.mj_forward(robot.model, robot.data)
        _sync(robot, force=True)
        time.sleep(self.frame_delay)

    def animate_gripper(self, q_arm, start, end, T_object):
        for alpha in np.linspace(0.0, 1.0, 18):
            finger_m = (1.0 - alpha) * float(start) + alpha * float(end)
            self.show(q_arm, finger_m, T_object)


def preview_pick_and_place(
    robot,
    plan,
    calibration,
    kinematics,
    *,
    target_name: str,
    T_pick: ArrayLike,
    T_place: ArrayLike,
    open_gripper_m: float,
    fps: float = 30.0,
    playback_speed: float = 1.5,
    max_joint_step: float = 0.035,
) -> None:
    """Fast kinematic preview of the five returned manipulation phases.

    During lift/transfer/place, move the object with planning's fixed TCP-object
    transform. No dynamics are stepped; physical grasp success is checked during
    execution.
    """

    if getattr(robot, "viewer", None) is None:
        raise RuntimeError("plan preview requires the interactive MuJoCo viewer")
    fps = float(fps)
    playback_speed = float(playback_speed)
    if (
        not np.isfinite(fps)
        or fps <= 0.0
        or not np.isfinite(playback_speed)
        or playback_speed <= 0.0
    ):
        raise ValueError("preview fps and playback speed must be positive")
    frame_delay = 1.0 / (fps * playback_speed)

    T_pick = np.asarray(T_pick, dtype=np.float64)
    T_place = np.asarray(T_place, dtype=np.float64)
    T_base_grasp = T_pick @ np.asarray(plan.grasp.T_object_grasp, dtype=np.float64)
    T_base_tcp_grasp = grasp_center_to_tcp(
        T_base_grasp,
        calibration.T_tcp_grasp,
    )
    T_tcp_object = np.linalg.inv(T_base_tcp_grasp) @ T_pick

    player = _PreviewPlayer(robot, target_name, kinematics, T_tcp_object, frame_delay)

    phase_spec = (
        ("transit", False),
        ("approach", False),
        ("lift", True),
        ("transfer", True),
        ("place", True),
    )
    for phase_name, carrying in phase_spec:
        print(f"[preview] {phase_name}")
        for q in _densify_path(
            plan.phases[phase_name],
            max_joint_step=max_joint_step,
        ):
            T_object = player.carried_pose(q) if carrying else T_pick
            finger = (
                float(plan.gripper_position_m) if carrying else float(open_gripper_m)
            )
            player.show(q, finger, T_object)

        if phase_name == "approach":
            print("[preview] closure")
            q = np.asarray(plan.phases["approach"][-1], dtype=np.float64)
            player.animate_gripper(q, open_gripper_m, plan.gripper_position_m, T_pick)

    # Freeze the payload where the nominal placement phase ended, then release.
    q_place = np.asarray(plan.phases["place"][-1], dtype=np.float64)
    T_released = player.carried_pose(q_place)
    print("[preview] release")
    player.animate_gripper(q_place, plan.gripper_position_m, open_gripper_m, T_released)

    print("[preview] retreat")
    for q in _densify_path(
        plan.phases["place"][::-1],
        max_joint_step=max_joint_step,
    ):
        player.show(q, float(open_gripper_m), T_released)

    print(
        "Preview complete. This was a kinematic plan replay; "
        "run the full simulator for physical grasp validation."
    )
