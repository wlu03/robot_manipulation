"""Pure joint-space and SE(3) path interpolation; no simulator dependency."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation, Slerp

from .transforms import rigid_transform

StatePredicate = Callable[[NDArray[np.float64]], bool]


def interpolate_joint_edge(
    q0: ArrayLike,
    q1: ArrayLike,
    max_step: float,
) -> list[NDArray[np.float64]]:
    """Sample q(t) = (1-t) q0 + t q1 at bounded joint-space spacing."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if (
        q0.ndim != 1
        or q1.shape != q0.shape
        or not np.all(np.isfinite(q0))
        or not np.all(np.isfinite(q1))
    ):
        raise ValueError("joint configurations must be matching finite vectors")
    max_step = float(max_step)
    if not np.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("max_step must be finite and positive")
    count = max(1, int(np.ceil(np.linalg.norm(q1 - q0) / max_step)))
    return [q0 + (i / count) * (q1 - q0) for i in range(count + 1)]


def joint_path_is_valid(
    path: Sequence[ArrayLike],
    predicate: StatePredicate,
    max_step: float,
    *,
    skip_first: bool = False,
    terminal_predicate: StatePredicate | None = None,
) -> bool:
    """Check every swept edge, with optional start/end contact exceptions."""
    points = [np.asarray(q, dtype=np.float64).copy() for q in path]
    if not points:
        return False
    if len(points) == 1:
        if terminal_predicate is not None:
            return bool(terminal_predicate(points[0]))
        return True if skip_first else bool(predicate(points[0]))

    for edge_index, (q0, q1) in enumerate(zip(points[:-1], points[1:])):
        samples = interpolate_joint_edge(q0, q1, max_step)
        start_index = 1 if (skip_first and edge_index == 0) else 0
        last_edge = edge_index == len(points) - 2
        stop = (
            len(samples) - 1
            if last_edge and terminal_predicate is not None
            else len(samples)
        )
        for q in samples[start_index:stop]:
            if not predicate(q):
                return False
    if terminal_predicate is not None and not terminal_predicate(points[-1]):
        return False
    return True


def interpolate_transform_path(
    T_start: ArrayLike,
    T_goal: ArrayLike,
    *,
    translation_step_m: float = 0.012,
    rotation_step_rad: float = np.deg2rad(7.5),
) -> list[NDArray[np.float64]]:
    """Interpolate translation linearly and rotation along its shortest SO(3) arc."""
    start = rigid_transform(T_start, "T_start")
    goal = rigid_transform(T_goal, "T_goal")
    if any(
        not np.isfinite(step) or step <= 0.0
        for step in (translation_step_m, rotation_step_rad)
    ):
        raise ValueError("interpolation steps must be finite and positive")

    distance = float(np.linalg.norm(goal[:3, 3] - start[:3, 3]))
    angle = float(Rotation.from_matrix(goal[:3, :3] @ start[:3, :3].T).magnitude())
    count = max(
        1,
        int(np.ceil(distance / float(translation_step_m))),
        int(np.ceil(angle / float(rotation_step_rad))),
    )
    times = np.linspace(0.0, 1.0, count + 1)
    slerp = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(np.stack((start[:3, :3], goal[:3, :3]))),
    )
    rotations = slerp(times).as_matrix()
    result = []
    for index, alpha in enumerate(times):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotations[index]
        T[:3, 3] = start[:3, 3] + alpha * (goal[:3, 3] - start[:3, 3])
        result.append(T)
    return result
