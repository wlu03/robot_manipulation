"""Run the MP2 collision-aware YCB pick-and-place episode in ARMLab/MuJoCo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from ams.armlab import ARMLab

from grasping import generate_grasp_candidates
from planner import plan_pick_and_place
from provided.custom_objects import install_custom_objects
from provided.execution import execute_pick_and_place
from provided.execution_diagnostics import SimulationGraspMonitor
from provided.local_motions import SimulationManipulationProblem
from provided.robot import INITIAL_Q
from provided.scene import DEFAULT_SCENE, SceneConfig
from provided.simulation import configure_robot, prepare_simulation
from provided.visualization import preview_pick_and_place

# This runner uses the MuJoCo scene and observations.
RUN_MODE = "sim"


@dataclass(frozen=True)
class SimulationOutcome:
    """Structured result from one simulation trial."""

    success: bool
    plan_only: bool
    failure_stage: str | None = None
    error: str | None = None
    infrastructure_error: bool = False
    generated_candidates: int = 0
    selected_grasp_score: float | None = None
    final_position_error_m: float | None = None
    final_rotation_error_rad: float | None = None


_INFRASTRUCTURE_STAGES = {
    "simulator_setup",
    "scene_setup",
    "problem_setup",
}


def _generate_grasps(task, seed):
    """Express the world-down approach in the object's frame, then sample grasps."""
    preferred_approach_object = task.T_base_object_pick[:3, :3].T @ np.array(
        [0.0, 0.0, -1.0]
    )
    return generate_grasp_candidates(
        task.points_object,
        task.normals_object,
        rng=np.random.default_rng(seed),
        sample_count=12000,
        friction_coefficient=0.5,
        minimum_width_m=max(0.010, task.calibration.closed_jaw_width_m + 0.004),
        maximum_width_m=task.calibration.maximum_jaw_width(task.finger_limit_m),
        preferred_approach=preferred_approach_object,
        maximum_candidates=140,
    )


def _make_problem(robot, task, seed):
    return SimulationManipulationProblem(
        robot,
        target_object_name=task.scene.target_name,
        T_base_object_pick=task.T_base_object_pick,
        T_base_object_place=task.T_base_object_place,
        q_home=task.q_home,
        kinematics=task.kinematics,
        calibration=task.calibration,
        seed=seed + 701,
        geom_groups=task.groups,
    )


def _print_plan(plan, diagnostics):
    print(
        "Planned phases:",
        ", ".join(f"{name}={len(path)}" for name, path in plan.phases.items()),
    )
    print(
        f"Selected grasp score: {diagnostics['selected_grasp_score']:.4f}; "
        f"predicted contact: {1000.0 * plan.gripper_position_m:.2f} mm/finger"
    )


def run(
    seed: int,
    headless: bool,
    plan_only: bool,
    with_obstacle: bool,
    obstacle_name: str | None = None,
    target_name: str | None = None,
    *,
    scene: SceneConfig = DEFAULT_SCENE,
    capture_outcome: bool = False,
    preview_plan: bool = False,
) -> SimulationOutcome:
    """Observe → generate grasps → plan → independently validate → execute."""
    scene = scene.with_objects(target_name=target_name, obstacle_name=obstacle_name)
    objects = scene.object_specs(with_obstacle=with_obstacle)
    diagnostics = dict(generated_candidates=0, selected_grasp_score=None)
    stage = "simulator_setup"
    robot = None
    try:
        install_custom_objects()
        robot = ARMLab(
            RunMode=RUN_MODE,
            ControlMode="position",
            HeadlessMode=headless,
            InitialConditions=INITIAL_Q.copy(),
            Objects=objects,
        )
        robot = configure_robot(robot, run_mode=RUN_MODE, preview_plan=preview_plan)
        stage = "scene_setup"
        task = prepare_simulation(
            robot,
            scene,
            objects,
            seed=seed,
            with_obstacle=with_obstacle,
            preview_plan=preview_plan,
        )
        stage = "grasp_generation"
        candidates = _generate_grasps(task, seed)
        diagnostics["generated_candidates"] = len(candidates)
        if not candidates:
            raise RuntimeError(
                f"no antipodal grasps were generated for {scene.target_name!r}"
            )
        print(f"Generated {len(candidates)} ranked antipodal candidates.")
        stage = "problem_setup"
        problem = _make_problem(robot, task, seed)
        stage = "planning"
        plan = plan_pick_and_place(
            candidates,
            problem,
            rng=np.random.default_rng(seed + 1701),
            step_size=0.12,
            edge_step=0.04,
            max_iterations=5000,
            shortcut_attempts=100,
        )
        problem.validate_plan(plan, edge_step=0.04)
        diagnostics.update(
            selected_grasp_score=float(plan.grasp.score),
        )
        _print_plan(plan, diagnostics)
        if preview_plan:
            stage = "preview"
            preview_pick_and_place(
                robot,
                plan,
                task.calibration,
                task.kinematics,
                target_name=scene.target_name,
                T_pick=task.T_base_object_pick,
                T_place=task.T_base_object_place,
                open_gripper_m=task.finger_limit_m,
            )
            print("PASS: MP2 complete plan previewed kinematically.")
        elif plan_only:
            print("PASS: MP2 complete simulation plan found (plan-only).")
        if preview_plan or plan_only:
            return SimulationOutcome(success=True, plan_only=True, **diagnostics)
        stage = "execution"
        execution = execute_pick_and_place(
            robot,
            plan,
            problem,
            task.calibration,
            T_place=task.T_base_object_place,
            open_gripper=task.finger_limit_m,
            phase_check=SimulationGraspMonitor(
                robot, task.kinematics, scene.target_name
            ),
        )
        return SimulationOutcome(
            success=True,
            plan_only=False,
            **diagnostics,
            final_position_error_m=float(execution.final_position_error_m),
            final_rotation_error_rad=float(execution.final_rotation_error_rad),
        )
    except Exception as error:
        if not capture_outcome:
            raise
        return SimulationOutcome(
            success=False,
            plan_only=bool(plan_only or preview_plan),
            failure_stage=stage,
            error=f"{type(error).__name__}: {error}",
            infrastructure_error=stage in _INFRASTRUCTURE_STAGES,
            **diagnostics,
        )
    finally:
        if robot is not None:
            robot.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-obstacle", action="store_true")
    parser.add_argument(
        "--target-name",
        help=f"YCB target object (default: {DEFAULT_SCENE.target_name})",
    )
    parser.add_argument(
        "--obstacle-name",
        help=f"YCB obstacle object (default: {DEFAULT_SCENE.obstacle_name})",
    )
    parser.add_argument(
        "--preview-plan",
        action="store_true",
        help="fast kinematic replay of the complete plan",
    )
    args = parser.parse_args()

    if args.preview_plan and args.headless:
        parser.error("--preview-plan requires the interactive viewer; omit --headless")

    run(
        seed=args.seed,
        headless=args.headless,
        plan_only=args.plan_only,
        with_obstacle=not args.no_obstacle,
        obstacle_name=args.obstacle_name,
        target_name=args.target_name,
        preview_plan=args.preview_plan,
    )


if __name__ == "__main__":
    main()
