"""Bidirectional RRT-Connect for MP2.

The generic tree container, sampling, and steering helpers are provided.
Students implement swept edge validation, EXTEND/CONNECT, and bidirectional tree
growth. Path joining and collision-safe shortcutting are provided.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


from provided.rrt_helpers import (
    StatePredicate,
    Tree,
    _configuration,
    join_tree_paths,
    sample_configuration,
    steer,
    shortcut_path as _shortcut_path,
)


def edge_is_valid(
    q_a: ArrayLike,
    q_b: ArrayLike,
    state_is_valid: StatePredicate,
    *,
    edge_step: float = 0.04,
) -> bool:
    """Return whether every interpolated configuration on the edge is valid."""
    start = _configuration(q_a, "q_a")
    goal = _configuration(q_b, "q_b")
    edge_step = float(edge_step)
    if start.shape != goal.shape or not np.isfinite(edge_step) or edge_step <= 0.0:
        raise ValueError("edge endpoints must match and edge_step must be positive")

    delta = goal - start
    total_length = float(np.linalg.norm(delta))
    segment_count = max(1, int(np.ceil(total_length / edge_step)))

    if not state_is_valid(start.copy()):
        return False
    if total_length == 0.0:
        return True

    for sample_index in range(1, segment_count):
        alpha = sample_index / segment_count
        if not state_is_valid(start + alpha * delta):
            return False

    return bool(state_is_valid(goal.copy()))


def _extend(
    tree: Tree,
    target: ArrayLike,
    state_is_valid: StatePredicate,
    *,
    step_size: float,
    edge_step: float,
) -> tuple[str, int]:
    """Perform one RRT-Connect EXTEND operation.

    Return ("trapped"|"advanced"|"reached", node_index): the nearest node's
    index when trapped, otherwise the index of the node at the new configuration.
    """
    # steer from the nearest node by at most step_size
    # validate the edge before adding the new node
    target = _configuration(target, "target")
    nearest_index = tree.nearest(target)
    q_near = tree.nodes[nearest_index]
    distance_to_target = float(np.linalg.norm(target - q_near))

    if distance_to_target == 0.0:
        if not state_is_valid(q_near.copy()):
            return ("trapped", nearest_index)
        return ("reached", nearest_index)

    q_new = steer(q_near, target, step_size=step_size)
    if np.array_equal(q_new, q_near):
        return ("trapped", nearest_index)

    if not edge_is_valid(q_near, q_new, state_is_valid, edge_step=edge_step):
        return ("trapped", nearest_index)
    new_index = tree.add(q_new, nearest_index)
    if distance_to_target <= float(step_size):
        return ("reached", new_index)
    return ("advanced", new_index)


def _connect(
    tree: Tree,
    target: ArrayLike,
    state_is_valid: StatePredicate,
    *,
    step_size: float,
    edge_step: float,
) -> tuple[str, int]:
    """Repeatedly EXTEND until trapped or the target is reached."""
    target = _configuration(target, "target")

    # keep extending toward the same target until progress stops
    while True:
        status, index = _extend(
            tree,
            target,
            state_is_valid,
            step_size=step_size,
            edge_step=edge_step,
        )
        if status != "advanced":
            return status, index


def rrt_connect(
    q_start: ArrayLike,
    q_goal: ArrayLike,
    joint_limits: ArrayLike,
    state_is_valid: StatePredicate,
    *,
    rng: np.random.Generator,
    step_size: float = 0.12,
    edge_step: float = 0.04,
    max_iterations: int = 4_000,
    goal_bias: float = 0.10,
) -> list[NDArray[np.float64]] | None:
    """Plan a collision-free path with bidirectional RRT-Connect.

    Return None for invalid or out-of-bounds endpoints, or if no path is found
    within max_iterations. With probability goal_bias, sample the other tree's root.
    """
    # validate the endpoints and joint limits
    try:
        start = _configuration(q_start, "q_start")
        goal = _configuration(q_goal, "q_goal")
    except ValueError:
        return None

    limits = np.asarray(joint_limits, dtype=np.float64)
    if start.shape != goal.shape or limits.shape != (start.size, 2):
        return None
    if not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
        return None

    # validate the RRT parameters
    step_size = float(step_size)
    edge_step = float(edge_step)
    goal_bias = float(goal_bias)
    if not np.isfinite(step_size) or step_size <= 0.0 or not np.isfinite(edge_step) or edge_step <= 0.0 or not np.isfinite(goal_bias) or not 0.0 <= goal_bias <= 1.0:
        raise ValueError("invalid RRT-Connect parameters")

    # both endpoints must be inside the limits and collision-free
    lower = limits[:, 0]
    upper = limits[:, 1]
    if np.any(start < lower) or np.any(start > upper) or np.any(goal < lower) or np.any(goal > upper):
        return None
    if not state_is_valid(start.copy()) or not state_is_valid(goal.copy()):
        return None

    if np.array_equal(start, goal):
        return [start.copy()]

    # initialize one tree from each endpoint
    active = Tree(start, "start")
    other = Tree(goal, "goal")

    for _ in range(max(0, int(max_iterations))):
        # sample a target for the active tree
        if rng.random() < goal_bias:
            target = other.nodes[0].copy()
        else:
            target = sample_configuration(limits, rng)

        active_status, active_index = _extend(
            active,
            target,
            state_is_valid,
            step_size=step_size,
            edge_step=edge_step,
        )

        # try to connect the other tree to the new active node
        if active_status != "trapped":
            other_status, other_index = _connect(
                other,
                active.nodes[active_index],
                state_is_valid,
                step_size=step_size,
                edge_step=edge_step,
            )
            if other_status == "reached":
                return join_tree_paths(
                    active,
                    active_index,
                    other,
                    other_index,
                )

        # alternate which tree grows first
        active, other = other, active

    return None


def shortcut_path(
    path: list[ArrayLike],
    state_is_valid: StatePredicate,
    *,
    rng: np.random.Generator,
    edge_step: float = 0.04,
    attempts: int = 100,
) -> list[NDArray[np.float64]]:
    """Provided shortcutting; uses the student's swept edge validator."""
    return _shortcut_path(
        path,
        state_is_valid,
        rng=rng,
        edge_step=edge_step,
        attempts=attempts,
        edge_validator=edge_is_valid,
    )
