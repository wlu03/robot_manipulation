"""Validate rigid transforms before frame composition or simulator state writes."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def rigid_transform(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return a finite SE(3) matrix with a proper rotation and homogeneous last row."""
    T = np.asarray(value, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    R = T[:3, :3]
    if np.linalg.norm(R.T @ R - np.eye(3)) > 1e-6 or np.linalg.det(R) <= 0.0:
        raise ValueError(f"{name} rotation must be proper")
    return T
