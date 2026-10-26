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
    if not state_is_valid(start.copy()):
        return False
    if total_length == 0.0:
        return True

    segment_count = max(1, int(np.ceil(total_length / edge_step)))

    # Keep the original equal-length segments, then progressively refine each
    # one at 1/2, 1/2 + 1/4, 1/2 + 1/4 + 1/8, and through 1/16.
    local_fractions = (0.5, 0.75, 0.875, 0.9375, 1.0)
    for segment_index in range(segment_count):
        for local_fraction in local_fractions:
            alpha = (segment_index + local_fraction) / segment_count
            if not state_is_valid(start + alpha * delta):
                return False

    return True


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

    # TODO(MP2)
    raise NotImplementedError


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

    # TODO(MP2)
    raise NotImplementedError


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
