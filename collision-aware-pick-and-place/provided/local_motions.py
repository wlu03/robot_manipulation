"""Construct grasp-conditioned local motions and collision predicates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from grasping import GraspCandidate
from manipulation import CandidateContext, ManipulationProblem, REQUIRED_PHASES

from .collision import CollisionQuery, planning_model
from .contacts import (
    CollisionMode,
    GeomGroups,
    allowed_in,
    read_contacts,
)
from .robot import (
    ARM_JOINT_NAMES,
    FINGER_JOINT_NAMES,
    GripperCalibration,
    contact_opening_for_width,
    runtime_finger_limit,
    tcp_transform_from_kinematics,
    transform_to_api_pose,
)
from .scene import ShadowState, base_pose_in_world
from .transforms import rigid_transform

from .task_geometry import grasp_motion_targets
from .paths import (
    interpolate_joint_edge as interpolate_joint_edge,
    joint_path_is_valid,
    interpolate_transform_path,
)

StatePredicate = Callable[[NDArray[np.float64]], bool]

# Plan with a small clearance to allow for controller tracking error.
PLANNING_CLEARANCE_M = 5e-3
# Recheck physical edge interiors more finely than the student planning samples.
VALIDATION_EDGE_STEP_RAD = 0.005
# Planned finger closure beyond first contact. A pad deeper than this plus the
# tolerance means a wider part of the object would block the planned contacts.
GRASP_SQUEEZE_M = 1e-3
GRASP_PENETRATION_TOLERANCE_M = 2e-3


@dataclass(frozen=True)
class _PickMotion:
    q_pregrasp: NDArray[np.float64]
    approach: list[NDArray[np.float64]]
    lift: list[NDArray[np.float64]]
    finger_qpos: NDArray[np.float64]
    carry_valid: StatePredicate
    place_valid: StatePredicate


@dataclass(frozen=True)
class _PlaceMotion:
    q_preplace: NDArray[np.float64]
    place: list[NDArray[np.float64]]


class SimulationManipulationProblem(ManipulationProblem):
    """Resolve one grasp into local motions and RRT collision predicates."""

    def __init__(
        self,
        robot,
        *,
        target_object_name: str,
        T_base_object_pick: ArrayLike,
        T_base_object_place: ArrayLike,
        q_home: ArrayLike,
        kinematics,
        calibration: GripperCalibration,
        seed: int = 19,
        pregrasp_distance_m: float = 0.040,
        lift_height_m: float = 0.040,
        place_clearance_m: float = 0.040,
        cartesian_step_m: float = 0.012,
        rotation_step_rad: float = np.deg2rad(7.5),
        local_edge_step: float = 0.035,
        ik_seed_count: int = 10,
        geom_groups: GeomGroups,
    ):
        self.robot = robot
        self.model = robot.model
        self.T_world_base = base_pose_in_world(self.model)
        self.source_data = robot.data
        self.target_object_name = str(target_object_name)
        self.kinematics = kinematics
        self.calibration = calibration

        self.q_home = np.asarray(q_home, dtype=np.float64).copy()
        if self.q_home.shape != (6,) or not np.all(np.isfinite(self.q_home)):
            raise ValueError("q_home must be a finite six-vector")
        self.joint_limits = np.asarray(
            kinematics.joint_position_limits[:6],
            dtype=np.float64,
        ).copy()
        if self.joint_limits.shape != (6, 2):
            raise RuntimeError("trusted PiPER joint limits are invalid")

        self.T_base_object_pick = rigid_transform(
            T_base_object_pick,
            "T_base_object_pick",
        ).copy()
        self.T_base_object_place = rigid_transform(
            T_base_object_place,
            "T_base_object_place",
        ).copy()
        self.groups = geom_groups
        self.planning_model = planning_model(self.model, PLANNING_CLEARANCE_M)

        self.finger_limit_m = runtime_finger_limit(robot)
        self.open_finger_qpos = np.array(
            [self.finger_limit_m, -self.finger_limit_m],
            dtype=np.float64,
        )
        self.pregrasp_distance_m = float(pregrasp_distance_m)
        self.lift_height_m = float(lift_height_m)
        self.place_clearance_m = float(place_clearance_m)
        self.cartesian_step_m = float(cartesian_step_m)
        self.rotation_step_rad = float(rotation_step_rad)
        self.local_edge_step = float(local_edge_step)
        self._validate_parameters()

        rng = np.random.default_rng(seed)
        low, high = self.joint_limits[:, 0], self.joint_limits[:, 1]
        span = high - low
        self._ik_seeds = [self.q_home.copy(), 0.5 * (low + high)]
        for _ in range(max(0, int(ik_seed_count))):
            self._ik_seeds.append(rng.uniform(low + 0.08 * span, high - 0.08 * span))

    def _validate_parameters(self) -> None:
        """Reject invalid local-motion distances and steps."""
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in (
                self.pregrasp_distance_m,
                self.lift_height_m,
                self.place_clearance_m,
                self.cartesian_step_m,
                self.rotation_step_rad,
                self.local_edge_step,
            )
        ):
            raise ValueError("planning distances/steps must be finite and positive")

    def _tcp_transform(self, q_arm: NDArray[np.float64]) -> NDArray[np.float64]:
        return tcp_transform_from_kinematics(self.kinematics, q_arm)

    def _ik_solutions(
        self,
        T_base_tcp: NDArray[np.float64],
        preferred: NDArray[np.float64] | None = None,
        maximum_solutions: int | None = None,
    ) -> list[NDArray[np.float64]]:
        seeds = []
        if preferred is not None:
            seeds.append(np.asarray(preferred, dtype=np.float64))
        seeds.extend(self._ik_seeds)
        target_pose = transform_to_api_pose(T_base_tcp)
        solutions = []
        for seed in seeds:
            if seed.shape != (6,):
                continue
            initial = np.r_[seed, self.finger_limit_m]
            try:
                q = np.asarray(
                    self.kinematics.computeIK(target_pose, initial),
                    dtype=np.float64,
                )[:6]
            except (ValueError, np.linalg.LinAlgError):
                continue
            if q.shape != (6,) or not np.all(np.isfinite(q)):
                continue
            if any(np.linalg.norm(q - existing) < 0.035 for existing in solutions):
                continue
            solutions.append(q.copy())
            if maximum_solutions is not None and len(solutions) >= maximum_solutions:
                break
        return solutions

    def _solve_transform_path(
        self,
        transforms: Sequence[NDArray[np.float64]],
        q_start: NDArray[np.float64],
    ) -> list[NDArray[np.float64]] | None:
        if not transforms:
            return None
        path = [np.asarray(q_start, dtype=np.float64).copy()]
        for T in transforms[1:]:
            solutions = self._ik_solutions(T, preferred=path[-1], maximum_solutions=1)
            if not solutions:
                return None
            path.append(
                min(
                    solutions,
                    key=lambda candidate: float(np.linalg.norm(candidate - path[-1])),
                ).copy()
            )
        return path

    def _query(
        self,
        finger_qpos: NDArray[np.float64],
        *,
        T_tcp_object: NDArray[np.float64] | None = None,
        T_base_object_rest: NDArray[np.float64] | None = None,
        clearance_m: float = PLANNING_CLEARANCE_M,
    ) -> CollisionQuery:
        """Collision checks with these fingers and the target held or resting."""
        state = ShadowState(
            self.model,
            arm_joint_names=ARM_JOINT_NAMES,
            finger_joint_names=FINGER_JOINT_NAMES,
            target_body_name=self.target_object_name,
            tcp_transform=self._tcp_transform,
            finger_qpos=finger_qpos,
            T_tcp_object=T_tcp_object,
            T_base_object_rest=T_base_object_rest,
            T_world_base=self.T_world_base,
        )
        return CollisionQuery(
            self.planning_model,
            self.groups,
            state,
            source_data=self.source_data,
            clearance_m=clearance_m,
            configuration_is_valid=self._configuration_is_valid,
        )

    def _configuration_is_valid(self, q_arm) -> bool:
        """Apply joint bounds and the native AMS workspace limits before collisions."""
        q = np.asarray(q_arm, dtype=np.float64)
        if (
            q.shape != self.q_home.shape
            or not np.all(np.isfinite(q))
            or np.any(q < self.joint_limits[:, 0])
            or np.any(q > self.joint_limits[:, 1])
        ):
            return False
        pose = self.robot.compute_ee_pose(np.r_[q, self.finger_limit_m])
        return bool(self.robot.check_safety_limits(pose))

    def validate_plan(self, plan, *, edge_step: float = 0.04) -> None:
        """Independently check returned student paths before granting success."""

        def same(a, b):
            a, b = np.asarray(a), np.asarray(b)
            return a.shape == b.shape and np.allclose(a, b, atol=1e-6, rtol=0.0)

        # Recompute the supplied local motions from the returned grasp; the
        # computation is deterministic, so a valid plan reproduces them exactly.
        context = self.context_for(plan.grasp)
        if context is None or not same(
            plan.gripper_position_m, context.gripper_position_m
        ):
            raise ValueError("plan grasp does not admit the supplied local motions")

        order = REQUIRED_PHASES
        if set(plan.phases) != set(order):
            raise ValueError("plan must contain all five required phases")
        paths = {
            name: np.asarray(plan.phases[name], dtype=np.float64) for name in order
        }
        for name, path in paths.items():
            if (
                path.ndim != 2
                or not len(path)
                or path.shape[1] != len(self.q_home)
                or not np.all(np.isfinite(path))
                or np.any(path < self.joint_limits[:, 0] - 1e-9)
                or np.any(path > self.joint_limits[:, 1] + 1e-9)
            ):
                raise ValueError(f"invalid configurations in {name}")
        if any(
            not same(paths[a][-1], paths[b][0]) for a, b in zip(order[:-1], order[1:])
        ):
            raise ValueError("plan phases must join continuously")
        for name in ("approach", "lift", "place"):
            if not same(paths[name], getattr(context, name)):
                raise ValueError(f"{name} must preserve the supplied local motion")
        physical_predicates = self._physical_path_predicates(context)
        for name, start, goal, predicate in (
            ("transit", self.q_home, context.q_pregrasp, context.transit_valid),
            ("transfer", context.q_lift, context.q_preplace, context.carry_valid),
        ):
            path = paths[name]
            if not same(path[0], start) or not same(path[-1], goal):
                raise ValueError(f"{name} has incorrect endpoints")
            # Keep the planning buffer at waypoints. Recheck edge interiors for
            # physical collisions without requiring the planner's sampling grid.
            if not all(predicate(q) for q in path):
                raise ValueError(f"{name} waypoint violates planning clearance")
            if not joint_path_is_valid(
                path,
                physical_predicates[name],
                min(edge_step, VALIDATION_EDGE_STEP_RAD),
            ):
                raise ValueError(f"{name} contains a forbidden swept collision")

    def _physical_path_predicates(self, context):
        """Transit/transfer collision rules without the extra planning buffer."""
        targets = self._grasp_targets(context.grasp)
        opening = context.gripper_position_m
        transit = self._query(
            self.open_finger_qpos,
            T_base_object_rest=self.T_base_object_pick,
            clearance_m=0.0,
        )
        transfer = self._query(
            np.array([opening, -opening], dtype=np.float64),
            T_tcp_object=targets.T_tcp_object,
            clearance_m=0.0,
        )
        return {
            "transit": transit.predicate(CollisionMode.FREE),
            "transfer": transfer.predicate(CollisionMode.CARRY),
        }

    def _grasp_targets(self, grasp):
        """TCP poses and held-object transform for this grasp."""
        return grasp_motion_targets(
            grasp,
            self.T_base_object_pick,
            self.T_base_object_place,
            self.calibration,
            pregrasp_distance_m=self.pregrasp_distance_m,
            lift_height_m=self.lift_height_m,
            place_clearance_m=self.place_clearance_m,
        )

    def context_for(self, grasp: GraspCandidate) -> CandidateContext | None:
        """Local motions for exactly this grasp frame, or None if any step fails."""
        opening = contact_opening_for_width(
            grasp.jaw_width_m,
            finger_limit_m=self.finger_limit_m,
            closed_jaw_width_m=self.calibration.closed_jaw_width_m,
        )
        if opening is None:
            return None
        # Model the closed grasp with the pads pressed slightly into the object.
        closed = max(0.0, opening - GRASP_SQUEEZE_M)
        finger_qpos = np.array([closed, -closed], dtype=np.float64)
        targets = self._grasp_targets(grasp)
        # Transit and approach: open fingers, target resting where it was observed.
        transit_valid = self._query(
            self.open_finger_qpos, T_base_object_rest=self.T_base_object_pick
        ).predicate(CollisionMode.FREE)
        pick = self._pick_motion(targets, finger_qpos, transit_valid)
        if pick is None:
            return None
        placement = self._place_motion(targets, pick)
        if placement is None:
            return None
        return CandidateContext(
            grasp=grasp,
            q_pregrasp=pick.q_pregrasp.copy(),
            q_lift=pick.lift[-1].copy(),
            q_preplace=placement.q_preplace.copy(),
            gripper_position_m=float(pick.finger_qpos[0]),
            transit_valid=transit_valid,
            carry_valid=pick.carry_valid,
            approach=[q.copy() for q in pick.approach],
            lift=[q.copy() for q in pick.lift],
            place=[q.copy() for q in placement.place],
        )

    def _transform_path(self, T_start, T_goal):
        return interpolate_transform_path(
            T_start,
            T_goal,
            translation_step_m=self.cartesian_step_m,
            rotation_step_rad=self.rotation_step_rad,
        )

    def _pick_motion(self, targets, finger_qpos, transit_valid):
        """Try pregrasp IK branches until approach, closure, and lift all work."""
        approach_targets = self._transform_path(
            targets.T_base_tcp_pregrasp,
            targets.T_base_tcp_grasp,
        )
        lift_targets = self._transform_path(
            targets.T_base_tcp_grasp, targets.T_base_tcp_lift
        )
        for q_pregrasp in self._ik_solutions(
            targets.T_base_tcp_pregrasp, preferred=self.q_home
        ):
            approach = self._solve_transform_path(approach_targets, q_pregrasp)
            if approach is None or not joint_path_is_valid(
                approach,
                transit_valid,
                self.local_edge_step,
            ):
                continue
            q_grasp = approach[-1]
            predicates = self._loaded_motion_predicates(
                q_grasp, targets.T_tcp_object, finger_qpos
            )
            if predicates is None:
                continue
            lift_valid, carry_valid, place_valid = predicates
            lift = self._solve_transform_path(lift_targets, q_grasp)
            if lift is None or not joint_path_is_valid(
                lift,
                lift_valid,
                self.local_edge_step,
                skip_first=True,
                terminal_predicate=carry_valid,
            ):
                continue
            return _PickMotion(
                q_pregrasp, approach, lift, finger_qpos, carry_valid, place_valid
            )
        return None

    def _closing_sweep_is_valid(self, closure, q_grasp, finger_qpos) -> bool:
        """Check the fingers from open to the planned contact opening.

        Steps equal the planning clearance, so anything a pad passes between two
        samples lies within the clearance of one of them. One query is reused and
        only its finger opening changes.
        """
        state = closure.write_state
        start, stop = float(self.open_finger_qpos[0]), float(finger_qpos[0])
        count = max(1, int(np.ceil((start - stop) / PLANNING_CLEARANCE_M)))
        try:
            for opening in np.linspace(start, stop, count + 1)[:-1]:
                state.finger_qpos = np.array([opening, -opening], dtype=np.float64)
                if not allowed_in(closure.labels(q_grasp), CollisionMode.CLOSURE):
                    return False
        finally:
            state.finger_qpos = np.asarray(finger_qpos, dtype=np.float64).copy()
        return True

    def _loaded_motion_predicates(self, q_grasp, T_tcp_object, finger_qpos):
        """Check closure on the resting target; return lift/carry/place rules."""
        closure = self._query(finger_qpos, T_base_object_rest=self.T_base_object_pick)
        if not closure.is_valid(q_grasp, CollisionMode.CLOSURE):
            return None
        if not self._closing_sweep_is_valid(closure, q_grasp, finger_qpos):
            return None
        closure.labels(q_grasp)  # restore the closed state for the contact checks
        # Both pads must touch the object, and neither may be pressed much
        # deeper than the planned squeeze. Touching is geometric (dist <= 0):
        # the planning copy's inflated margins make near misses solver contacts.
        fingers, target = self.groups.fingers, self.groups.target
        touching = [False, False]
        for c in read_contacts(closure.model, closure.data):
            for index, pad in enumerate(self.calibration.finger_geom_ids):
                if (c.geom1 in pad and c.geom2 in target) or (
                    c.geom2 in pad and c.geom1 in target
                ):
                    touching[index] = True
        if not all(touching):
            return None
        deepest = min(
            (
                c.distance_m
                for c in read_contacts(closure.model, closure.data)
                if (c.geom1 in fingers and c.geom2 in target)
                or (c.geom2 in fingers and c.geom1 in target)
            ),
            default=0.0,
        )
        if deepest < -(GRASP_SQUEEZE_M + GRASP_PENETRATION_TOLERANCE_M):
            return None
        held = self._query(finger_qpos, T_tcp_object=T_tcp_object)
        return (
            held.predicate(CollisionMode.LIFT),
            held.predicate(CollisionMode.CARRY),
            held.predicate(CollisionMode.PLACE),
        )

    def _place_motion(self, targets, pick):
        """Try preplace IK branches, then validate descent and open-gripper retreat."""
        # Retreat: open fingers, target resting at its placed pose.
        retreat_valid = self._query(
            self.open_finger_qpos, T_base_object_rest=self.T_base_object_place
        ).predicate(CollisionMode.FREE)
        place_targets = self._transform_path(
            targets.T_base_tcp_preplace, targets.T_base_tcp_place
        )
        for q_preplace in self._ik_solutions(
            targets.T_base_tcp_preplace, preferred=pick.lift[-1]
        ):
            if not pick.carry_valid(q_preplace):
                continue
            place = self._solve_transform_path(place_targets, q_preplace)
            if place is None or not joint_path_is_valid(
                place,
                pick.carry_valid,
                self.local_edge_step,
                terminal_predicate=pick.place_valid,
            ):
                continue
            # After release the gripper retreats by reversing the descent.
            if not joint_path_is_valid(place[::-1], retreat_valid, self.local_edge_step):
                continue
            return _PlaceMotion(q_preplace, place)
        return None
