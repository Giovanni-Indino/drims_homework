# DRIMS Homework 🚧

This ROS 2 package serves as the starting point for the summer school challenge.  
The repository is **work in progress** — more components and examples will be added soon.

---

## 📚 Useful resources

- **ROS 2 Cheat Sheet (Jazzy, mostly compatible with Humble)**  
  [Download PDF](https://s3.amazonaws.com/assets.clearpathrobotics.com/wp-content/uploads/2025/02/06151220/ROS-2-Cheat-Sheet_Jazzy_FINAL.pdf)

- **Git Cheat Sheet (GitHub)**  
  [Download PDF](https://education.github.com/git-cheat-sheet-education.pdf)

- **Git Cheat Sheet (GitLab)**  
  [Download PDF](https://about.gitlab.com/images/press/git-cheat-sheet.pdf)

---

## 🚀 Next steps

- Fork this repository  
- Work together in teams  
- Start experimenting, learning, and **have fun!** 😃

---

## 🧩 Project architecture

Four small pieces coordinated by an explicit state machine — see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full picture
(diagram, frames, bring-up order):

* **`dice_common.py`** — the perception *contract* (`/dice_identification`):
  the simulator today, a real camera node tomorrow, same interface either way.
* **`dice_manipulation_node`** — the pick/roll/place *skill*: identify → pick
  (grasp aligned with live `dice_tf`, yaw pre-chosen for the upcoming roll
  *before* contact) → lift → roll a quarter turn about a fixed **world** axis
  (`x +90°` or `y -90°`, never about world **Z**) near the table → release →
  report `face X -> face Y`. Doesn't decide *which* roll — that's the next
  node.
* **`dice_face_map.py`** — pure-Python, closed-form model of *this* die.
  Real perception can only ever report *which number* is up, never the die's
  precise yaw (a top face is a square; several pip patterns are themselves
  rotationally symmetric) — so `from_two_faces()` fixes the whole six-face
  layout exactly from one *known, commanded* roll and the two face numbers
  around it, no yaw reading needed at all (no ROS, unit-tested in
  `test/test_dice_face_map.py`). From there `plan_min_sequence()` finds the
  *provably minimal* roll sequence to any target face by BFS over the
  reachable orientation graph — at most 3 more rolls, any face from any
  already-known orientation.
* **`dice_task_orchestrator`** — the "smart" layer, an explicit state machine
  (`IDENTIFY → CALIBRATE (if needed) → CHECK_TARGET → PLAN → ROLL → VERIFY →
  …`, see its module docstring): given a `target_face` (1-6), spends at most
  one roll calibrating the die's layout (only when it isn't already tracked),
  then loops until reached or `max_attempts` is hit, driving
  `dice_manipulation_node` and re-deriving the die's orientation exactly
  (never a mere prediction) after every further roll.

```bash
# both nodes together
ros2 launch drims_homework dice_task_start.launch.py

# ask for a face
ros2 param set /dice_task_orchestrator target_face 6
ros2 service call /dice_task_orchestrator/reach_target_face std_srvs/srv/Trigger "{}"
```

To test `dice_manipulation_node` on its own (single configured roll, no
orchestrator):
```bash
ros2 launch drims_homework dice_manipulation_start.launch.py
ros2 service call /dice_manipulation_node/pick_rotate_place std_srvs/srv/Trigger "{}"
```

Config: [`config/dice_manipulation_config.yaml`](config/dice_manipulation_config.yaml),
[`config/dice_task_orchestrator_config.yaml`](config/dice_task_orchestrator_config.yaml).

## Yellow-die localization from the OAK RGB camera

`yellow_dice_localizer` detects the yellow die in
`/oak/rgb/image_raw/compressed`, uses `/oak/rgb/camera_info` and square PnP
to publish the centre of its upper face in **camera optical-frame coordinates**
(metres: X right, Y down, Z forward):

```bash
ros2 launch drims_homework yellow_dice_localizer_start.launch.py
ros2 topic echo /yellow_dice/pose
```

The same node works directly on the supplied recordings; no physical camera
is required. Replay one bag together with the localizer:

```bash
ros2 launch drims_homework yellow_dice_bag.launch.py \
  bag_path:=/home/drims/bags/setup_1/rosbag2_2026_09_01-09_14_02 rate:=1.0
```

While it runs, inspect the estimated position with `ros2 topic echo
/yellow_dice/pose` or open `/yellow_dice/debug_image` in RViz/rqt_image_view.
The result topics retain their last value; after replay use
`ros2 topic echo --qos-durability transient_local /yellow_dice/face_number`.

It opens an OpenCV window by default and publishes `/yellow_dice/position`
(`PointStamped`), the annotated `/yellow_dice/debug_image`, rectified
`/yellow_dice/top_face`, the counted `/yellow_dice/face_number` (`Int32`),
and `/yellow_dice/face_orientation_deg` (`Float64`, modulo 90 degrees).
The separate **Yellow die masks** window shows the exact binary masks used for
the yellow die, bright upper face, and locally high-contrast pips; they are also published as
`/yellow_dice/die_mask`, `/yellow_dice/top_face_mask`, and
`/yellow_dice/pip_mask`.
Set `die_size_m` in
[`config/yellow_dice_localizer_config.yaml`](config/yellow_dice_localizer_config.yaml)
to the measured edge length of the physical die: this is required to obtain a
metric 3D position from an RGB camera. The node intentionally only localizes
the die; exposing it through the package's `dice_identification` service and
recognizing the upper face are the next pipeline steps.
