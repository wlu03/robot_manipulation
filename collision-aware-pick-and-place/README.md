# MP2: Grasping and Collision-Aware Pick-and-Place

Start with the [handout](HANDOUT.md): it contains the tasks,
checkpoints, rubric, and Gradescope instructions.

Use the updated MP0 course AMS/MuJoCo environment, including
`get_object_poses()` and `wait_until_stationary()`. From this folder:

```bash
python visualize_grasps.py --browse      # after grasping.py
python visualize_rrt.py                 # after rrt_connect.py
python -m unittest discover -s tests -v
python run_sim.py --headless --plan-only # after planner.py
python run_sim.py --headless             # physical validation
```

Edit and submit only `grasping.py`, `rrt_connect.py`, `planner.py`, and
`AI_DISCLOSURE.txt`. Read `manipulation.py` for the supplied planning interface.
Optional viewer/scene settings are listed by each script's `--help`.

For an optional math-to-code tour, see the [supplied code guide](provided/README.md).
