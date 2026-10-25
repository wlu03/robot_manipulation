# Reading the supplied robotics code

Start with `run_sim.run`: observe → generate grasps → plan → validate → execute.
Each stage has one responsibility. Pure geometry is separated from simulation state
and physical control. You can read this code without implementing its internals.

**Notation:** `T_A_B` maps coordinates from B to A; `R` is a rotation, `p` a
position, and `q` a joint configuration. Distances are metres; arm angles are radians.
Names such as `T_base_tcp_place` identify both frames and the task phase.

| Lecture concept | Code to read |
|---|---|
| Antipodal contacts and the grasp frame | Your `../grasping.py` |
| Bidirectional tree growth and swept edges | Your `../rrt_connect.py`; supplied `rrt_helpers.py` |
| Grasp selection conditioned on downstream feasibility | Your `../planner.py`; `local_motions.py:context_for` |
| Composition of object, grasp, and TCP frames | `task_geometry.py:grasp_motion_targets` |
| Proper SE(3) matrices and frame-boundary validation | `transforms.py:rigid_transform` |
| Joint interpolation and shortest-arc rotation interpolation | `paths.py` |
| Triangle-area sampling and surface normals | `surface.py` |
| Intentional contacts versus collisions: one allowed-contact table per phase | `contacts.py:ALLOWED_CONTACTS` |
| Collision checks for the robot plus a carried object, with clearance | `collision.py:CollisionQuery` |
| Predicted finger opening from the contact width | `robot.py:contact_opening_for_width` |
| Physical pick, lift, transfer, place, release, and verification | `execution.py:execute_pick_and_place` |
| Independent simulation lift/slip measurements | `execution_diagnostics.py:SimulationGraspMonitor` |

The key carried-object equations appear directly in `task_geometry.py`:

```python
T_tcp_object = inv(T_base_tcp_grasp) @ T_base_object_pick
T_base_tcp_place = T_base_object_place @ inv(T_tcp_object)
```

The first fixes the grasped relation. The second places the TCP so that this same
relation produces the requested object pose. Local feasibility supplies approach,
lift, and placement; your RRT plans the two global motions between them. The
supplied grasp-frame convention is a constant: `robot.py:T_TCP_GRASP`.

`scene.py` reads base-frame poses through AMS and owns planning state;
`simulation.py` prepares the fixture,
`robot.py` adapts AMS kinematics and gripper conventions, and `motion_control.py`
moves along joint paths at constant speed and enforces the contact table.
AMS supplies stepping, settling, and controller settings. Visualization is isolated
in `visualization.py`.

Planning checks collisions in a separate copy of the scene, so it never moves the
live simulation. Collision predicates include a 5 mm planning buffer. The final
check keeps that buffer at waypoints and samples edge interiors more finely for
physical collisions; your edge checker can use any grid meeting the handout's
spacing rule. Physical execution leaves the object free and measures whether
contact holds it. Student functions use the same planning
interfaces across backends. Submit only the four files listed in the handout.

Task perception takes stationary snapshots before planning and after release and
retreat. Control does not request poses of a held object. Separate read-only
simulation diagnostics measure lift and slip from the actual free payload;
these measurements are not hardware perception. A pose predicted from arm FK
can guide planning, but cannot independently verify that a grasp held.
