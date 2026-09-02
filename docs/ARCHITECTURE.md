# DRIMS3 Homework — Project Architecture

## Goal

A camera localizes a dice on the table (position + orientation) and
reads the number on the top face. If it is already the desired one,
stop. Otherwise pick the dice, roll it, put it back, and repeat until
the desired number is up. Everything runs on **ROS 2 (Humble)**.

---

## Four pieces, one state machine

```
                     ┌───────────────────────────────────────────┐
                     │        dice_task_orchestrator              │  "smart" layer:
                     │  explicit state machine (State enum)        │  an explicit
                     │  IDENTIFY → CHECK_TARGET → PLAN → ROLL →    │  state machine
                     │  VERIFY → (loop) → DONE / FAILED            │  coordinating
                     └───────┬─────────────────────┬───────────────┘  everything else
                             │                      │
                    plan_next()/record()   set_parameters + ~/pick_rotate_place
                             │                      │
              ┌──────────────▼──────────┐  ┌────────▼───────────────────────┐
              │      dice_face_map        │  │      dice_manipulation_node    │
              │  pure-Python model of     │  │  the pick/roll/place SKILL:    │
              │  THIS die's face layout,  │  │  identify→pick→lift→align→     │
              │  no ROS, no robot         │  │  roll→release→report           │
              └────────────────────────────┘  └───────┬─────────────────────┘
                             ▲                          │ easy_motion (MoveIt 2)
                             │ face_number                │ move_to_pose/joint,
                    ┌────────┴──────────┐                 │ gripper, attach/detach
                    │   dice_common      │                 │
                    │ /dice_identification│                ▼
                    │ contract (perception)│    ┌──────────────────────────────┐
                    └────────┬───────────┘    │  easy_motion + drims_cells     │
                             │                  │  (MotionServer / MoveIt 2 /   │
                    simulator today,            │  UR5e + hand-e description)  │
                    real camera tomorrow         └──────────────────────────────┘
```

Four small pieces, each doing one job:

### 1. `dice_common.py` — the perception *contract*

Every node that needs "where is the dice / which face is up" goes
through this, never anything simulator-specific:

* `create_dice_identification_client()` / `identify_dice()` — thin
  wrappers around the service `/dice_identification`
  (`easy_motion_msgs/srv/DiceIdentification`): request nothing, get back
  `face_number` + `PoseStamped` + `success`.

**Today** it's served by the simulator (`drims2_dice_simulator`'s
`dice_spawner_node`), which also fakes "gravity" (snaps the dice flat and
re-computes the up face after every release) and broadcasts the TF
frames used for grasping: `dice_com_tf`, `face1_tf…face6_tf`, `dice_tf`
(aligned with whichever face is currently up — the pre-grasp frame).

**Tomorrow**, a real vision node replaces it: detect the cube, estimate
its 6-DoF pose, count the pips on the top face, publish the same
service + the same TF frames. Nothing downstream changes.

### 2. `dice_manipulation_node.py` — the pick/roll/place *skill*

One service, `~/pick_rotate_place` (`std_srvs/srv/Trigger`): identify →
pick (grasp aligned with live `dice_tf`) → lift → **align the gripper's
opening axis to the upcoming roll's world axis** → roll a quarter turn
about that fixed world axis → release right there → retreat → report
`face X -> face Y`. It does **not** decide which roll to use — `roll_axis`
/ `roll_angle_deg` are parameters, normally set by the orchestrator via
`set_parameters` right before each call.

Everything non-obvious about *why* it is built this way (near-the-table
rolls instead of a round trip through `home_joints`, a world-fixed
instead of body-relative roll axis, the grip-axis alignment step, why
release never returns to the exact pick spot, why nothing re-reads TF
mid-sequence) is explained once, in the module's own docstring — read it
before changing the motion sequence.

**The grip-axis alignment, concretely** (this is what makes the release
horizontal): a flush top-down grasp must follow the die's actual,
arbitrary live yaw — real physical grasping, not just a scripted attach.
A world-fixed roll axis is what live testing found reliable for this
arm's IK, but combined naively that leaves the jaws tilted at release by
however much they happened to be off from that axis. Right after lifting
(safely clear of the table), the node now yaws the tool — a plain,
well-conditioned wrist rotation — until the jaws line up exactly with
the roll's axis; from then on they stay exactly aligned through the roll
(a rotation about an axis leaves that axis fixed), so release is always
exactly horizontal, exactly, not just approximately.

### 3. `dice_face_map.py` — the die's own "map" (pure Python, no ROS)

What `dice_manipulation_node` cannot know in advance — which physical
face a given roll produces from a given starting face — this module
*learns online* and answers `plan_next(current, target)` with the best
next roll to try. No ROS, no ``rclpy`` — a small graph-learning class
over `record()`/`mark_infeasible()` observations, testable on its own
(`test/test_dice_face_map.py`) without touching a robot or simulator.

Uses one purely geometric shortcut, true for any die numbering: two
applications of the *same* roll are a single 180° turn about a fixed
horizontal axis, which always swaps top and bottom — so two consecutive
identical rolls immediately reveal an opposite-face pair, and once two
of the three pairs are known the third is forced by elimination.
Everything else is breadth-first search over whatever has actually been
observed so far.

### 4. `dice_task_orchestrator.py` — the state machine

`DiceTaskOrchestrator.reach_target_face(target_face)` runs an explicit
`State` enum machine (`IDENTIFY → CHECK_TARGET → PLAN → ROLL → VERIFY →
… → DONE/FAILED`, see the module docstring for the full diagram). It
only ever talks to two contracts: `/dice_identification` (perception)
and `dice_manipulation_node`'s `set_parameters` + `~/pick_rotate_place`
(the skill) — never MoveIt/TF/the gripper directly, and never assumes
the simulator's internal geometry: everything about face transitions
comes from `dice_face_map`, fed by what the robot actually observed.

Exposed as service `~/reach_target_face` (`std_srvs/srv/Trigger`) +
parameter `target_face` (1-6).

---

## External stack (not part of this package)

| Package | Role |
| --- | --- |
| `easy_motion` | `MotionServer` (MoveIt 2 via pymoveit2) + `MotionClient` (Py) + C++/BT clients. Actions: `move_to_pose`, `move_to_joint`, `plan_to_*`, `execute_trajectory`. Services: `attach_object`, `detach_object`, `get_ik`, `get_fk`. |
| `easy_motion_behavior_tree` | BT leaves (`MoveToPose`, `GripperCommand`, `DiceIdentification`, `CheckDiceFace`, `AttachObject`…) + `bt_executer_node` — an alternative to this package's plain-Python orchestrator, not currently used by it. |
| `drims_cells` | UR5e + Robotiq hand-e description, MoveIt config, controllers. Frames: `world` → `table_top` (MoveIt base) → … → `tool0` → `tip` (virtual EE, `tool0` + 0.15 m Z). |
| `drims2_dice_simulator` | Dice spawner + fake `/dice_identification` + `/reset_dice`. |

### Key frames

| Frame | Meaning |
| --- | --- |
| `world` | fixed root, motion reference used here |
| `table_top` | MoveIt planning base (`base_link_name`) |
| `tip` | virtual end-effector, planning target (`end_effector_name`) |
| `dice_com_tf` | dice centre of mass |
| `faceN_tf` | centre of face `N`, +Z out of the face |
| `dice_tf` | == the up-facing `faceN_tf`; pre-grasp frame |

---

## Bring-up order

```bash
# 1. robot cell: MoveIt + controllers + easy_motion's motion_server, all
#    in one (ur5e_1_start.launch.py already includes ur5e_motion_server.launch.py)
ros2 launch drims_description ur5e_1_start.launch.py fake:=true
# 2. dice simulator (perception stand-in)
ros2 launch drims_dice_simulator spawn_dice.launch.py face_up:=2
# 3. the skill + the state machine (this package)
ros2 launch drims_homework dice_task_start.launch.py
# 4. ask for a face
ros2 param set /dice_task_orchestrator target_face 6
ros2 service call /dice_task_orchestrator/reach_target_face std_srvs/srv/Trigger "{}"
```

Or drive `dice_manipulation_node` on its own for a single-roll smoke
test: `ros2 launch drims_homework dice_manipulation_start.launch.py`
(`run_on_start: true` in its config does one roll immediately).

## Status

- [x] Perception contract (`dice_common.py`, simulated)
- [x] Manipulation skill (`dice_manipulation_node.py`): pick, lift, grip-axis
      alignment, world-axis roll, release, report
- [x] Die-geometry model (`dice_face_map.py`): online learning + opposite-pair
      shortcut + BFS planning, unit-tested
- [x] Orchestrating state machine (`dice_task_orchestrator.py`)
- [ ] Empirical validation of the new grip-axis alignment step on the real
      arm/simulator (everything IK-related here has needed at least one
      round of "live testing found X actually fails" — treat this the same
      way until confirmed)
- [ ] Real vision node replacing the simulator behind `dice_common.py`
