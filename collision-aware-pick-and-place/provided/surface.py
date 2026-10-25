"""Known-mesh surface sampling with triangle-area probabilities and face normals."""

from __future__ import annotations

import pathlib
import numpy as np
from numpy.typing import NDArray


def _visual_mesh_path(robot, object_name: str) -> pathlib.Path:
    builder = robot.scene_builder
    if object_name not in builder.registry:
        raise ValueError(
            f"unknown YCB object {object_name!r}; available: {builder.available}"
        )
    return pathlib.Path(builder.registry[object_name]["visual"])


def load_ycb_surface(
    robot,
    object_name: str,
    *,
    sample_count: int = 4500,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return object-frame points/normals, sampled uniformly over surface area."""
    path = _visual_mesh_path(robot, object_name)
    points, triangles = _read_obj_triangles(path)
    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if rng is None:
        rng = np.random.default_rng(0)
    triangle_points, face_normals, double_area = _triangle_geometry(
        points, triangles, path
    )
    return _sample_triangles(
        triangle_points, face_normals, double_area, sample_count, rng
    )


def _read_obj_triangles(path: pathlib.Path):
    """Read vertices and triangulate OBJ polygon faces without changing their order."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as mesh:
        for raw in mesh:
            if raw.startswith("v "):
                fields = raw.split()
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif raw.startswith("f "):
                polygon: list[int] = []
                for token in raw.split()[1:]:
                    raw_index = int(token.split("/")[0])
                    polygon.append(
                        raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                    )
                if len(polygon) >= 3:
                    faces.append(polygon)

    points = np.asarray(vertices, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or points.shape[0] < 4
        or not np.all(np.isfinite(points))
    ):
        raise RuntimeError(f"could not read a usable OBJ surface from {path}")

    triangles: list[tuple[int, int, int]] = []
    for polygon in faces:
        anchor = polygon[0]
        for offset in range(1, len(polygon) - 1):
            tri = (anchor, polygon[offset], polygon[offset + 1])
            if all(0 <= index < len(points) for index in tri):
                triangles.append(tri)

    return points, triangles


def _triangle_geometry(points, triangles, path):
    """Compute cross-product normals and twice each triangle's area."""
    if not triangles:
        raise RuntimeError(f"mesh {path} has no usable triangles")
    triangle_indices = np.asarray(triangles, dtype=np.int64)
    triangle_points = points[triangle_indices]
    cross = np.cross(
        triangle_points[:, 1] - triangle_points[:, 0],
        triangle_points[:, 2] - triangle_points[:, 0],
    )
    double_area = np.linalg.norm(cross, axis=1)
    keep = double_area > 1e-14
    if not np.any(keep):
        raise RuntimeError(f"mesh {path} contains only degenerate triangles")

    triangle_points = triangle_points[keep]
    cross = cross[keep]
    double_area = double_area[keep]
    face_normals = cross / double_area[:, None]

    mesh_centroid = points.mean(axis=0)
    triangle_centers = triangle_points.mean(axis=1)
    inward = (
        np.einsum(
            "ij,ij->i",
            face_normals,
            triangle_centers - mesh_centroid,
        )
        < 0.0
    )
    face_normals[inward] *= -1.0

    return triangle_points, face_normals, double_area


def _sample_triangles(triangle_points, face_normals, double_area, sample_count, rng):
    """Choose triangles by area, then use uniform barycentric coordinates."""
    probabilities = double_area / np.sum(double_area)
    chosen = rng.choice(
        triangle_points.shape[0],
        size=sample_count,
        replace=True,
        p=probabilities,
    )
    chosen_triangles = triangle_points[chosen]

    # Uniform sampling over triangle area via square-root barycentric sampling.
    u = rng.random(sample_count)
    v = rng.random(sample_count)
    root_u = np.sqrt(u)
    w0 = 1.0 - root_u
    w1 = root_u * (1.0 - v)
    w2 = root_u * v
    sampled_points = (
        w0[:, None] * chosen_triangles[:, 0]
        + w1[:, None] * chosen_triangles[:, 1]
        + w2[:, None] * chosen_triangles[:, 2]
    )
    sampled_normals = face_normals[chosen].copy()
    return sampled_points, sampled_normals
