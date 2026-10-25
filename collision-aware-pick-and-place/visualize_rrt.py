#!/usr/bin/env python3
"""Visualize your MP2 RRT-Connect implementation on a small 2-D C-space task.

This checkpoint depends only on rrt_connect.py; grasping.py and planner.py may
still be unfinished.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import webbrowser

import numpy as np

from rrt_connect import rrt_connect, shortcut_path
from provided.paths import joint_path_is_valid


START = np.array([0.10, 0.18], dtype=np.float64)
GOAL = np.array([0.90, 0.82], dtype=np.float64)
JOINT_LIMITS = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)

# A vertical C-space wall. The straight start-goal segment crosses it, so the
# planner has to grow around the upper or lower end rather than "passing" a
# collision at unsampled edge interiors.
OBSTACLE = (0.42, 0.58, 0.12, 0.88)  # xmin, xmax, ymin, ymax

# Line styles shared by the plot and its legend.
STRAIGHT = {"stroke": "#888", "stroke_width": 1.5, "opacity": 0.8, "dash": "8,6"}
RAW = {"stroke": "#5f6f7a", "stroke_width": 2.0, "opacity": 0.65}
SHORTCUT = {"stroke": "#1565c0", "stroke_width": 4.0, "opacity": 0.95}


def state_is_valid(q) -> bool:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (2,) or not np.all(np.isfinite(q)):
        return False
    if np.any(q < JOINT_LIMITS[:, 0]) or np.any(q > JOINT_LIMITS[:, 1]):
        return False
    xmin, xmax, ymin, ymax = OBSTACLE
    inside = xmin <= q[0] <= xmax and ymin <= q[1] <= ymax
    return not inside


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--step-size", type=float, default=0.08)
    parser.add_argument("--edge-step", type=float, default=0.015)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument(
        "--no-shortcut",
        action="store_true",
        help="show only the raw RRT-Connect path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("mp2_rrt_checkpoint.svg"),
        help="SVG output path (default: mp2_rrt_checkpoint.svg)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="write the SVG but do not ask the system browser to open it",
    )
    args = parser.parse_args()
    if (
        not np.isfinite(args.step_size)
        or not np.isfinite(args.edge_step)
        or args.step_size <= 0.0
        or args.edge_step <= 0.0
        or args.max_iterations <= 0
    ):
        parser.error("planning steps and iteration count must be finite and positive")
    return args


def _require_valid_path(path, edge_step):
    """Check checkpoint paths independently of the student's edge validator."""
    nodes = np.asarray(path, dtype=np.float64)
    if (
        nodes.ndim != 2
        or nodes.shape[1:] != (2,)
        or len(nodes) < 2
        or not np.all(np.isfinite(nodes))
        or not np.allclose(nodes[0], START, atol=1e-9, rtol=0.0)
        or not np.allclose(nodes[-1], GOAL, atol=1e-9, rtol=0.0)
        or not joint_path_is_valid(nodes, state_is_valid, edge_step)
    ):
        raise SystemExit(
            "Invalid checkpoint path: check endpoints, limits, and swept collisions."
        )


def _plan_checkpoint(args):
    """Run the two-dimensional planner independently of grasping/simulation."""
    try:
        raw_path = rrt_connect(
            START,
            GOAL,
            JOINT_LIMITS,
            state_is_valid,
            rng=np.random.default_rng(args.seed),
            step_size=args.step_size,
            edge_step=args.edge_step,
            max_iterations=args.max_iterations,
            goal_bias=0.15,
        )
    except NotImplementedError as error:
        raise SystemExit(
            "Your RRT-Connect TODOs are not complete yet. Implement edge_is_valid, "
            "EXTEND/CONNECT, and rrt_connect, then rerun visualize_rrt.py."
        ) from error

    if raw_path is None:
        raise SystemExit(
            "RRT-Connect returned no path on the supplied checkpoint problem. "
            "Check edge interpolation, tree swapping/path order, and CONNECT."
        )

    _require_valid_path(raw_path, args.edge_step)
    displayed_path = [np.asarray(q, dtype=np.float64) for q in raw_path]
    shortcut_complete = False
    if not args.no_shortcut:
        try:
            displayed_path = shortcut_path(
                displayed_path,
                state_is_valid,
                rng=np.random.default_rng(args.seed + 1),
                edge_step=args.edge_step,
                attempts=120,
            )
            _require_valid_path(displayed_path, args.edge_step)
            shortcut_complete = True
        except NotImplementedError:
            print(
                "The edge checker is unfinished; showing the raw RRT-Connect "
                "path so you can debug the planner incrementally."
            )

    return (
        np.asarray(raw_path, dtype=np.float64),
        np.asarray(displayed_path, dtype=np.float64),
        shortcut_complete,
    )


def _checkpoint_svg(raw, shown, shortcut_complete):
    """Render the configuration-space obstacle and raw/shortened paths."""
    # Write a dependency-free SVG so this checkpoint works in the same minimal
    # environment as the rest of MP2. No matplotlib installation is required.
    width = 720
    margin = 70
    span = width - 2 * margin
    plot_bottom = margin + span
    height = plot_bottom + 140  # axis label and legend below the plot

    def px(q):
        q = np.asarray(q, dtype=np.float64)
        return (
            margin + span * float(q[0]),
            plot_bottom - span * float(q[1]),
        )

    def polyline(points, *, stroke, stroke_width, opacity=1.0, dash=None):
        coords = " ".join(f"{px(q)[0]:.1f},{px(q)[1]:.1f}" for q in points)
        dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
        return (
            f'<polyline points="{coords}" fill="none" stroke="{stroke}" '
            f'stroke-width="{stroke_width}" opacity="{opacity}"'
            f'{dash_attr} stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def text(x, y, label, *, size=14, fill="#222", anchor="start"):
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-family="sans-serif" font-size="{size}" fill="{fill}">{label}</text>'
        )

    xmin, xmax, ymin, ymax = OBSTACLE
    ox, oy_top = px((xmin, ymax))
    ox2, oy_bottom = px((xmax, ymin))
    obstacle_width = ox2 - ox
    obstacle_height = oy_bottom - oy_top

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{margin}" y="{margin}" width="{span}" height="{span}" '
        'fill="none" stroke="#222" stroke-width="1.5"/>',
        f'<rect x="{ox:.1f}" y="{oy_top:.1f}" width="{obstacle_width:.1f}" '
        f'height="{obstacle_height:.1f}" fill="#999" opacity="0.30" '
        'stroke="#666" stroke-width="1.2"/>',
        polyline(np.stack([START, GOAL]), **STRAIGHT),
        polyline(raw, **RAW),
    ]
    if shortcut_complete:
        parts.append(polyline(shown, **SHORTCUT))

    sx, sy = px(START)
    gx, gy = px(GOAL)
    parts += [
        f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="7" fill="#2e7d32"/>',
        f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="8" fill="#c62828"/>',
        text(sx + 12, sy + 22, "q_start", fill="#2e7d32"),
        text(gx - 12, gy - 14, "q_goal", fill="#c62828", anchor="end"),
        text(width / 2, 34, "MP2 RRT-Connect checkpoint", size=22, anchor="middle"),
        text(
            width / 2,
            plot_bottom + 34,
            "configuration coordinate q1",
            size=16,
            anchor="middle",
        ),
        f'<text x="20" y="{margin + span / 2:.1f}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="16" '
        f'transform="rotate(-90 20 {margin + span / 2:.1f})">'
        "configuration coordinate q2</text>",
    ]

    # Legend: one row per mark, so the figure is readable without the console.
    entries = [
        ("box", "C-space obstacle: state_is_valid(q) is False"),
        (STRAIGHT, "straight start-goal edge (crosses the obstacle)"),
        (RAW, f"raw RRT-Connect path ({len(raw)} waypoints)"),
    ]
    if shortcut_complete:
        entries.append(
            (SHORTCUT, f"after supplied shortcutting ({len(shown)} waypoints)")
        )
    for row, (style, label) in enumerate(entries):
        y = plot_bottom + 62 + 20 * row
        x = margin
        if style == "box":
            parts.append(
                f'<rect x="{x}" y="{y - 7}" width="36" height="12" fill="#999" '
                'opacity="0.30" stroke="#666" stroke-width="1.2"/>'
            )
        else:
            dash = style.get("dash")
            dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
            parts.append(
                f'<line x1="{x}" y1="{y - 1}" x2="{x + 36}" y2="{y - 1}" '
                f'stroke="{style["stroke"]}" stroke-width="{style["stroke_width"]}" '
                f'opacity="{style["opacity"]}"{dash_attr}/>'
            )
        parts.append(text(x + 48, y + 4, label))
    parts.append("</svg>")

    return "\n".join(parts)


def main() -> None:
    args = _parse_args()
    raw, shown, shortcut_complete = _plan_checkpoint(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        _checkpoint_svg(raw, shown, shortcut_complete), encoding="utf-8"
    )
    print(
        f"PASS: RRT-Connect returned {len(raw)} waypoints"
        + (
            f"; shortcutting reduced this to {len(shown)}."
            if shortcut_complete
            else "."
        )
    )
    print(
        "The dashed start-goal segment intersects the obstacle; a valid returned "
        "path must go around it and remain ordered start -> goal."
    )
    print(f"Wrote visualization to: {args.out.resolve()}")

    if not args.no_open:
        opened = False
        try:
            opened = bool(webbrowser.open(args.out.resolve().as_uri()))
        except Exception:
            opened = False
        if not opened:
            print(
                "Could not open a browser automatically; open the SVG path above "
                "manually."
            )


if __name__ == "__main__":
    main()
