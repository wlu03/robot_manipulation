# MP2: Grasping and Collision-Aware Pick-and-Place

**CS 4803: Advanced Robotic Manipulation**

Choose an antipodal grasp that the PiPER can reach, lift, carry around an obstacle,
and place. Implement **eight functions in three files**. IK, object poses/geometry,
collision queries, local motions, gripper control, and simulation are supplied.

Use the MP0 course environment and run commands from this folder. Keep the
function signatures; edit only `grasping.py`, `rrt_connect.py`, `planner.py`, and
`AI_DISCLOSURE.txt`.

## 1. Grasping — `grasping.py`

All contacts and normals are in the **object frame**; normals point outward.
`T_A_B` maps coordinates from B into A. Distances are metres.

- **`is_antipodal`:** normalize the normals. Return `False` for coincident contacts
  or a separation outside the width limits. With `d = (p_b - p_a) / ||p_b - p_a||`
  and `alpha = atan(mu)`, require both:
  `angle(d, -n_a) <= alpha` and `angle(-d, -n_b) <= alpha`.
- **`make_grasp_frame`:** return a proper 4×4 `T_object_grasp` with origin at the
  contact midpoint; `+x` from A toward B; `+z` the normalized preferred approach
  projected perpendicular to `+x`; and `+y = +z × +x`. If the preferred approach
  is parallel to `+x`, choose any perpendicular `+z`. Raise `ValueError` for
  coincident contacts.
- **`generate_grasp_candidates`:** sample contact pairs from the given points with
  the supplied `rng`, keep those passing the antipodal test with the passed width
  limits, and return at most `maximum_candidates` with finite scores in descending
  order. Contacts must be input points, `jaw_width_m` their distance, and each
  frame centered between them with `+x` from A toward B. Equal seeds must
  reproduce the output.

  A pair fixes the closing axis but not the approach: rotating the frame about
  `+x` gives a different grasp. `make_grasp_frame` gives one choice, closest to
  `preferred_approach`. You decide which approaches to propose and how to rank
  them; the robot uses each frame exactly as returned. No particular scheme is
  required. A single default approach may be infeasible; consider reachability and
  finger clearance when choosing frames.

Inspect candidates without the planner:

```bash
python visualize_grasps.py --browse
```

The jaws show geometry only; reachability is checked later.

## 2. Motion planning — `rrt_connect.py`

Configurations are joint vectors (arm angles in radians). Implement:

- **`edge_is_valid`:** check both endpoints and equally spaced interior samples;
  adjacent samples must be at most `edge_step` apart in Euclidean joint distance.
- **`_extend`:** steer from the nearest node by at most `step_size`, validate the
  edge, and add a valid node. Return `(status, node_index)` with status `"trapped"`
  (index of the nearest node, tree unchanged), `"advanced"`, or `"reached"` (index
  of the node at the new configuration).
- **`_connect`:** repeatedly extend toward the target until trapped or reached.
- **`rrt_connect`:** return `None` if an endpoint is outside `joint_limits` or
  fails `state_is_valid`. Otherwise alternate start/goal trees using the supplied
  RNG: sample a configuration (with probability `goal_bias`, the other tree's root),
  extend the active tree toward it, and connect the other tree to the new node.
  Return a start-to-goal path, or `None` after `max_iterations`.

Tree storage, nearest neighbors, sampling, steering, path joining, and shortcutting
are provided. Label trees `"start"` and `"goal"`; when they meet, return
`join_tree_paths(active, active_index, other, other_index)`, which handles swaps.
`shortcut_path` uses your edge checker.

Inspect a path in 2-D (no simulator needed):

```bash
python visualize_rrt.py
```

## 3. Complete planning — `planner.py`

Implement **`plan_pick_and_place`** to find the first ranked grasp with a complete
collision-free plan; raise `ValueError` if none works. **Transit** connects home to
pregrasp with an empty gripper; **transfer** connects lift to preplace with the
robot **plus carried object**. Use the supplied shortcutting and try another
candidate if local motions or either path are infeasible.

`problem.context_for(grasp)` returns a feasible `CandidateContext` or `None`.
See [manipulation.py](manipulation.py) for endpoints, collision predicates, and
`make_plan`, which combines your paths with the supplied approach, lift, and place
motions into five phases. Predicates allow intended contacts; the runner re-checks
the plan. Planning rigidly attaches the carried object; execution holds a free
object by friction and controls the gripper.

## 4. Check and submit

```bash
python -m unittest discover -s tests -v    # public checks; not exhaustive
python run_sim.py --headless --plan-only # complete plan validation
python run_sim.py --headless             # physical pick-and-place
```

Try other objects, e.g. `--target-name tall_can --obstacle-name mustard_bottle`
(targets: `plum`, `lemon`, `tall_can`, `tall_box`).

`--preview-plan` shows a fast **kinematic** replay (it does not test holding or
placement); omit `--headless` to watch execution. Try `--seed 12`.

Complete `AI_DISCLOSURE.txt` (write "None" if no AI was used). Upload
**`grasping.py`, `rrt_connect.py`, `planner.py`, and `AI_DISCLOSURE.txt`** to MP2 on
Gradescope, or a ZIP with those four files at its root. Do not include AMS or
`provided/`.

## Grading — 100 points

| Criterion | Points |
|---|---:|
| Antipodality: both friction cones / normal scaling / width and coincident contacts | 4 / 2 / 4 |
| `make_grasp_frame`: midpoint and proper transform / closing and approach axes | 5 / 5 |
| Generated candidate geometry (contacts, width, midpoint, closing axis) | 10 |
| Candidates: finite descending scores / cap / supplied RNG / reproducibility | 2 / 1 / 1 / 1 |
| Edges: endpoints / interior collision / requested spacing | 3 / 4 / 3 |
| EXTEND: nearest-node steering / reached and trapped outcomes | 4 / 4 |
| CONNECT: repeated extension / obstacle stop | 3 / 2 |
| RRT: obstacle route / wall gap / supplied RNG / reproducibility | 3 / 3 / 1 / 1 |
| RRT: no path / invalid endpoints | 2 / 2 |
| Simulation: 10 plan-only trials + 5 physical trials, 1 point per success | 15 |
| Hardware performance using the same submission | 15 |

Deterministic checks award independent partial credit; `planner.py` earns credit
through simulation and hardware. Simulation scenes use spherical and tall targets
and tall obstacles that force a carry detour. Physical success requires two-sided
contact, lift, a collision-free carry, placement within 2 cm, release, and retreat
without disturbing the object.

Hardware uses your same final submission, with no extra code or upload. Details
will be announced before hardware testing.
