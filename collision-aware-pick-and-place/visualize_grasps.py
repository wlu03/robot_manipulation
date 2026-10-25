#!/usr/bin/env python3
"""Inspect the output of your MP2 grasp-synthesis code in the MuJoCo scene.

This script depends only on grasping.py. You can use it before RRT-Connect or
planner.py are implemented.
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np
from ams.armlab import ARMLab

from grasping import generate_grasp_candidates
from provided.custom_objects import install_custom_objects
from provided.robot import INITIAL_Q, FINGER_JOINT_NAMES, runtime_finger_limit
from provided.scene import (
    DEFAULT_SCENE,
    load_ycb_surface,
    object_pose,
    scalar_joint_qpos_addresses,
    set_scalar_joint_positions,
    validate_ycb_objects,
)
from provided.visualization import draw_grasp_candidates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-name", default=DEFAULT_SCENE.target_name)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="1-indexed candidate to highlight (default: 1)",
    )
    parser.add_argument(
        "--browse",
        action="store_true",
        help="use terminal n/p/rank commands to browse candidates",
    )
    parser.add_argument(
        "--surface-samples",
        type=int,
        default=2500,
        help="triangle-area surface samples passed to your grasp generator",
    )
    args = parser.parse_args()
    if args.top_k <= 0 or args.rank <= 0 or args.surface_samples < 100:
        parser.error("--top-k/--rank must be positive and --surface-samples >= 100")

    return args


def _browse_candidates(robot, candidate_count, selected, render):
    """Navigate ranked candidates without touching planning or execution."""
    while True:
        command = (
            input("\nBrowse [n]ext, [p]revious, rank number, or [q]uit: ")
            .strip()
            .lower()
        )
        if command in {"q", "quit"}:
            if robot.viewer is not None:
                robot.viewer.close()
            break
        if command in {"n", "next", ""}:
            selected = (selected + 1) % candidate_count
        elif command in {"p", "prev", "previous"}:
            selected = (selected - 1) % candidate_count
        else:
            try:
                requested = int(command)
            except ValueError:
                print("Enter n, p, q, or a 1-indexed candidate rank.")
                continue
            if not 1 <= requested <= candidate_count:
                print(f"Rank must lie in [1, {candidate_count}].")
                continue
            selected = requested - 1
        render(selected)


def _generate_candidates(robot, scene, args, finger_limit):
    """Sample the known surface and call only the student's grasp generator."""
    T_pick = object_pose(robot, scene.target_name)
    surface_rng = np.random.default_rng(args.seed + 101)
    points, normals = load_ycb_surface(
        robot,
        scene.target_name,
        sample_count=args.surface_samples,
        rng=surface_rng,
    )
    preferred_approach_object = T_pick[:3, :3].T @ np.array([0.0, 0.0, -1.0])
    try:
        candidates = generate_grasp_candidates(
            points,
            normals,
            rng=np.random.default_rng(args.seed),
            sample_count=12000,
            friction_coefficient=0.5,
            minimum_width_m=0.010,
            maximum_width_m=2.0 * finger_limit,
            preferred_approach=preferred_approach_object,
            maximum_candidates=140,
        )
    except NotImplementedError as error:
        if robot.viewer is not None:
            robot.viewer.close()
        raise SystemExit(
            "Your grasp-synthesis TODOs are not complete yet. "
            "Implement is_antipodal, make_grasp_frame, and "
            "generate_grasp_candidates, then rerun visualize_grasps.py."
        ) from error

    if not candidates:
        if robot.viewer is not None:
            robot.viewer.close()
        raise SystemExit(
            "Your generator returned no grasp candidates. "
            "Check the antipodal test, width limits, and grasp-frame convention."
        )

    return T_pick, points, normals, candidates


def _inspect_candidates(robot, args, T_pick, points, normals, candidates):
    top_k = min(args.top_k, len(candidates))
    selected = min(args.rank - 1, len(candidates) - 1)

    def render(index: int) -> None:
        draw_grasp_candidates(
            robot,
            T_pick,
            candidates,
            selected_index=index,
            top_k=top_k,
            surface_points_object=points,
            surface_normals_object=normals,
        )
        grasp = candidates[index]
        print(
            f"Selected rank {index + 1}/{len(candidates)}: "
            f"score={float(grasp.score):.4f}, "
            f"jaw_width={1000.0 * float(grasp.jaw_width_m):.1f} mm"
        )

    print(f"Generated {len(candidates)} ranked candidate grasps.")
    print("Top candidates:")
    for rank, grasp in enumerate(candidates[:top_k], start=1):
        print(
            f"  {rank:2d}: score={float(grasp.score):.4f}, "
            f"width={1000.0 * float(grasp.jaw_width_m):.1f} mm"
        )
    print(
        "\nViewer key: gray = sampled surface/normals; cyan = top accepted pairs; "
        "blue = selected contacts/closing axis; purple = approach direction."
    )
    print(
        "The translucent jaws are an abstract parallel-jaw visualization, "
        "not a claim that the PiPER can reach that grasp."
    )
    render(selected)

    if args.browse:
        _browse_candidates(robot, len(candidates), selected, render)
    else:
        print(
            "\nClose the MuJoCo viewer window when you are finished inspecting the grasps."
        )


def main() -> None:
    args = _parse_args()
    install_custom_objects()
    scene = DEFAULT_SCENE.with_objects(target_name=args.target_name)
    objects = scene.object_specs(with_obstacle=False)
    robot = ARMLab(
        RunMode="sim",
        ControlMode="position",
        HeadlessMode=False,
        InitialConditions=INITIAL_Q.copy(),
        Objects=objects,
    )
    try:
        validate_ycb_objects(robot, scene, with_obstacle=False)
        robot.initialize()

        # Keep this checkpoint purely kinematic and fast. Open the fingers
        # directly for a clear view; grasp visualization does not step dynamics.
        finger_limit = runtime_finger_limit(robot)
        finger_addresses = scalar_joint_qpos_addresses(
            robot.model,
            FINGER_JOINT_NAMES,
        )
        set_scalar_joint_positions(
            robot.data,
            finger_addresses,
            np.array([finger_limit, -finger_limit], dtype=np.float64),
        )
        robot.data.qvel[:] = 0.0
        mujoco.mj_forward(robot.model, robot.data)

        T_pick, points, normals, candidates = _generate_candidates(
            robot, scene, args, finger_limit
        )

        _inspect_candidates(robot, args, T_pick, points, normals, candidates)
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
