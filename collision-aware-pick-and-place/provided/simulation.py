"""Prepare the AMS/MuJoCo fixture and expose the geometry needed for planning.

Simulation-only: hardware execution needs its own backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .collision import planning_model
from .contacts import GeomGroups, contact_labels, scene_geom_groups
from .local_motions import PLANNING_CLEARANCE_M
from .motion_control import hold, settle
from .robot import (
    INITIAL_Q,
    ARM_BODY_NAMES,
    BASE_BODY_NAMES,
    FINGER_BODY_NAMES,
    SUPPORT_NAMES,
    GripperCalibration,
    GripperPositionAdapter,
    TrustedPiperKinematics,
    gripper_calibration,
    runtime_finger_limit,
)
from .scene import (
    SceneConfig,
    clone_query_data,
    observe_sim_scene,
    validate_ycb_objects,
)
from .surface import load_ycb_surface
from .visualization import install_viewer_throttle


@dataclass(frozen=True)
class SimulationTask:
    scene: SceneConfig
    q_home: NDArray[np.float64]
    finger_limit_m: float
    kinematics: TrustedPiperKinematics
    calibration: GripperCalibration
    groups: GeomGroups
    T_base_object_pick: NDArray[np.float64]
    T_base_object_place: NDArray[np.float64]
    points_object: NDArray[np.float64]
    normals_object: NDArray[np.float64]


def configure_robot(robot, *, run_mode: str, preview_plan: bool):
    """Adapt gripper coordinates and viewer rendering; preserve AMS settings."""
    if run_mode != "sim":
        raise NotImplementedError(
            "MP2's supplied runner is simulation-only; a hardware runner may reuse "
            "the planning and plan-validation code but not this backend"
        )
    if not hasattr(robot, "viewer"):
        robot.viewer = None
    if robot.viewer is not None:
        install_viewer_throttle(robot)
    return GripperPositionAdapter(robot, run_mode=run_mode)


def prepare_simulation(robot, scene, objects, *, seed, with_obstacle, preview_plan):
    """Settle the scene, calibrate the hand, then observe poses and known geometry."""
    validate_ycb_objects(robot, scene, with_obstacle=with_obstacle)
    robot.initialize()
    print(f"Target object: {scene.target_name}")
    if with_obstacle and scene.obstacle_name is not None:
        print(f"Obstacle object: {scene.obstacle_name}")
    robot.set_joint_positions(INITIAL_Q.copy())
    hold(robot, 1.25)

    finger_limit = runtime_finger_limit(robot)
    q_home = _home_configuration(robot, finger_limit)
    kinematics = TrustedPiperKinematics()
    calibration = gripper_calibration(robot.model)
    groups = scene_geom_groups(
        robot.model,
        target_body_name=scene.target_name,
        base_body_names=BASE_BODY_NAMES,
        arm_body_names=ARM_BODY_NAMES,
        finger_body_names=FINGER_BODY_NAMES,
        support_names=SUPPORT_NAMES,
    )
    _require_initial_clearance(robot, groups)

    observation = observe_sim_scene(robot, objects)
    T_base_object_pick = observation.pose(scene.target_name)
    T_base_object_place = scene.place_pose(T_base_object_pick)
    points, normals = load_ycb_surface(
        robot,
        scene.target_name,
        sample_count=4500,
        rng=np.random.default_rng(seed + 101),
    )
    return SimulationTask(
        scene,
        q_home,
        finger_limit,
        kinematics,
        calibration,
        groups,
        T_base_object_pick,
        T_base_object_place,
        points,
        normals,
    )


def _home_configuration(robot, finger_limit):
    settle(robot, INITIAL_Q[:6], finger_limit)
    hold(robot, 0.4)
    return np.asarray(robot.get_joint_positions(), dtype=np.float64)[:6].copy()


def _require_initial_clearance(robot, groups):
    import mujoco

    model = planning_model(robot.model, PLANNING_CLEARANCE_M)
    data = clone_query_data(model, robot.data)
    mujoco.mj_forward(model, data)
    labels = contact_labels(model, data, groups, clearance_m=PLANNING_CLEARANCE_M)
    if "target-support" not in labels:
        raise RuntimeError(
            "invalid initial scene: the settled target touches no geometry named "
            f"in SUPPORT_NAMES {SUPPORT_NAMES!r}"
        )
    if "target-obstacle" in labels:
        raise RuntimeError(
            "invalid initial scene: target and obstacle require at least "
            f"{1000 * PLANNING_CLEARANCE_M:.1f} mm clearance"
        )
