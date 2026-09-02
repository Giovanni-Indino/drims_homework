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
  (grasp aligned with live `dice_tf`) → lift → align the gripper's opening
  axis to the roll's axis (so release is always exactly horizontal) → roll a
  quarter turn about a fixed **world** axis, near the table → release → report
  `face X -> face Y`. Doesn't decide *which* roll — that's the next node.
* **`dice_face_map.py`** — pure-Python model of *this* die: learns online
  which face a given roll produces from a given face (no ROS, unit-tested in
  `test/test_dice_face_map.py`), using an opposite-faces-sum-geometry shortcut
  plus breadth-first search over whatever has actually been observed.
* **`dice_task_orchestrator`** — the "smart" layer, an explicit state machine
  (`IDENTIFY → CHECK_TARGET → PLAN → ROLL → VERIFY → …`, see its module
  docstring): given a `target_face` (1-6), loops until reached or
  `max_attempts` is hit, driving `dice_manipulation_node` and feeding
  `dice_face_map` with what actually happened.

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