"""Read-only RRT storage, steering, path joining, and shortcutting helpers."""

from __future__ import annotations
from collections.abc import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray

StatePredicate = Callable[[NDArray[np.float64]], bool]


def _configuration(value: ArrayLike, name: str) -> NDArray[np.float64]:
    q = np.asarray(value, dtype=np.float64)
    if q.ndim != 1 or q.size < 1 or not np.all(np.isfinite(q)):
        raise ValueError(f"{name} must be a finite vector")
    return q


class Tree:
    """Provided nearest-neighbor tree storage.

    `nodes[i]` is a configuration and `parent[i]` its parent index (-1 at the root).
    """

    def __init__(self, root: ArrayLike, label: str):
        self.nodes = [_configuration(root, "root").copy()]
        self.parent = [-1]
        self.label = str(label)

    def nearest(self, query: ArrayLike) -> int:
        """Return the index of the node closest to `query` in Euclidean distance."""
        q = _configuration(query, "query")
        return int(np.argmin([np.linalg.norm(node - q) for node in self.nodes]))

    def add(self, node: ArrayLike, parent: int) -> int:
        """Append `node` as a child of node index `parent`; return its index."""
        q = _configuration(node, "node")
        parent = int(parent)
        if not 0 <= parent < len(self.nodes):
            raise ValueError("invalid parent")
        self.nodes.append(q.copy())
        self.parent.append(parent)
        return len(self.nodes) - 1

    def path_to(self, index: int) -> list[NDArray[np.float64]]:
        """Return copies of the configurations from the root to node `index`."""
        index = int(index)
        if not 0 <= index < len(self.nodes):
            raise ValueError("invalid node index")
        path: list[NDArray[np.float64]] = []
        while index >= 0:
            path.append(self.nodes[index].copy())
            index = self.parent[index]
        return path[::-1]


def sample_configuration(
    joint_limits: ArrayLike,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Provided uniform joint-space sampler."""

    limits = np.asarray(joint_limits, dtype=np.float64)
    if (
        limits.ndim != 2
        or limits.shape[1:] != (2,)
        or not np.all(np.isfinite(limits))
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("joint_limits must be a finite (D,2) array")
    return rng.uniform(limits[:, 0], limits[:, 1])


def steer(
    q_from: ArrayLike,
    q_toward: ArrayLike,
    *,
    step_size: float,
) -> NDArray[np.float64]:
    """Provided one-step steering helper."""

    start = _configuration(q_from, "q_from")
    target = _configuration(q_toward, "q_toward")
    step_size = float(step_size)
    if start.shape != target.shape or not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("invalid steering inputs")
    delta = target - start
    distance = float(np.linalg.norm(delta))
    if distance <= step_size:
        return target.copy()
    return start + delta * (step_size / distance)


def join_tree_paths(
    active: Tree,
    active_index: int,
    other: Tree,
    other_index: int,
) -> list[NDArray[np.float64]]:
    """Provided start-to-goal reconstruction after the two trees meet.

    Trees must have labels "start" and "goal"; pass their meeting-node indices.
    The helper handles either tree order and removes the shared meeting node.
    """
    if {active.label, other.label} != {"start", "goal"}:
        raise ValueError("trees must have start/goal labels")
    if not np.allclose(
        active.nodes[active_index], other.nodes[other_index], atol=1e-9, rtol=0.0
    ):
        raise ValueError("tree endpoints have not met")
    if active.label == "start":
        start_path, goal_path = active.path_to(active_index), other.path_to(other_index)
    else:
        start_path, goal_path = other.path_to(other_index), active.path_to(active_index)
    return start_path + goal_path[-2::-1]


def shortcut_path(
    path: list[ArrayLike],
    state_is_valid: StatePredicate,
    *,
    rng: np.random.Generator,
    edge_step: float = 0.04,
    attempts: int = 100,
    edge_validator: Callable[..., bool],
) -> list[NDArray[np.float64]]:
    """Provided shortcutting; uses the supplied edge validator."""
    result = [_configuration(node, "path element").copy() for node in path]
    for _ in range(max(0, int(attempts))):
        if len(result) < 3:
            break
        first, second = sorted(rng.choice(len(result), size=2, replace=False))
        if second > first + 1 and edge_validator(
            result[first],
            result[second],
            state_is_valid,
            edge_step=edge_step,
        ):
            result = result[: first + 1] + result[second:]
    return result
