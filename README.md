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