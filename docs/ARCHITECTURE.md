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
                     │  IDENTIFY → CALIBRATE(if needed) →           │  state machine
                     │  CHECK_TARGET → PLAN → ROLL → VERIFY →      │  coordinating
                     │  (loop) → DONE / FAILED                      │  everything else
                     └───────┬─────────────────────┬───────────────┘
                             │                      │
              from_two_faces()/plan_min_sequence()  set_parameters + ~/pick_rotate_place
                             │                      │
              ┌──────────────▼──────────┐  ┌────────▼───────────────────────┐
              │      dice_face_map        │  │      dice_manipulation_node    │
              │  closed-form model of     │  │  the pick/roll/place SKILL:    │
              │  THIS die's face layout,  │  │  identify→pick(yaw-chosen)→    │
              │  no ROS, no robot         │  │  lift→roll→release→report      │
              └────────────────────────────┘  └───────┬─────────────────────┘
                             ▲                          │ easy_motion (MoveIt 2)
                             │ face_number (only)          │ move_to_pose/joint,
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
pick (grasp aligned with live `dice_tf`, yaw **pre-chosen** for the
upcoming roll before contact) → lift → roll a quarter turn about a fixed
world axis (**`x +90°` or `y -90°` only — never `z`**) → release right
there → retreat → report `face X -> face Y`. It does **not** decide which
roll to use — `roll_axis` / `roll_angle_deg` are parameters, normally set
by the orchestrator via `set_parameters` right before each call.

Everything non-obvious about *why* it is built this way (near-the-table
rolls instead of a round trip through `home_joints`, a world-fixed
instead of body-relative roll axis, the grasp-yaw pre-selection, why
release never returns to the exact pick spot, why nothing re-reads TF
mid-sequence) is explained once, in the module's own docstring — read it
before changing the motion sequence.

**The grasp-yaw pre-selection, concretely** (this is what makes the
release horizontal, without ever rotating the die itself): a flush
top-down grasp must follow the die's actual live yaw — real physical
grasping, not a scripted attach. Because the die is *never* yawed about
world Z by anything in this sequence, it stays axis-aligned forever, so
`dice_tf`'s own X/Y axes are always exactly parallel to world X/Y. That
means the jaws can be brought parallel to the upcoming roll's axis with
an exact 0° or 90° choice, decided from the already-known live `dice_tf`
orientation and the already-known `roll_axis` — entirely on paper, before
the gripper ever touches the die. `pick_dice()` folds this straight into
the grasp approach pose itself (the gripper is open and clear of the die
throughout approach/descent), and — because a rotation about an axis
leaves that axis fixed — the jaws stay exactly aligned with the roll
axis all the way through the roll to release, exactly, not just
approximately, at zero extra motion cost.

### 3. `dice_face_map.py` — the die's own "map" (pure Python, no ROS)

Because the die is never yawed about world Z, its orientation is exact,
closed-form geometry, not something that has to be learned by trial —
but a single `/dice_identification` reading is *not* enough on its own:
real perception can reliably report *which number* is up, never the
die's precise yaw (a top face is a square, and several pip patterns are
themselves rotationally symmetric — no vision quality fixes that). What
**is** always exact: the face number before a roll, the face number
after it, and the roll itself, since it was *commanded*, not measured.
Two known, perpendicular body-face normals pinned to two known,
perpendicular world directions fix a rigid body's entire remaining
orientation — `DiceOrientation.from_two_faces(face_before, move,
face_after)` builds exactly that (cross/dot products, `STANDARD_BODY_
NORMALS` for this die's own numbering), and it is both necessary and
sufficient: nothing less determines the layout, nothing more is needed.
From there, `plan_min_sequence(orientation, target_face)` is a plain
breadth-first search over the reachable orientation graph (at most the
24 rotations of a cube — the two candidate rolls generate the full
group, verified in `test/test_dice_face_map.py`) for the *provably
shortest* roll sequence: worst case 3 more rolls, any face from any
already-known orientation. No ROS, no `rclpy` — pure vector arithmetic,
no external dependency, testable entirely on its own.

A roll can still be *kinematically infeasible* at a given orientation (a
motion-layer planning failure, distinct from "wrong face"): callers pass
a `blocked` set of already-failed `(orientation, move)` pairs, and BFS
simply routes around them.

### 4. `dice_task_orchestrator.py` — the state machine

`DiceTaskOrchestrator.reach_target_face(target_face)` runs an explicit
`State` enum machine (`IDENTIFY → CALIBRATE (if needed) → CHECK_TARGET →
PLAN → ROLL → VERIFY → … → DONE/FAILED`, see the module docstring for
the full diagram). It only ever talks to two contracts:
`/dice_identification` (perception, and only ever its `face_number` —
never its orientation, see point 3 above) and `dice_manipulation_node`'s
`set_parameters` + `~/pick_rotate_place` (the skill) — never
MoveIt/TF/the gripper directly. `CALIBRATE` is only entered when the
tracked orientation is missing or no longer matches the live face (the
very first call, or after anything that could have desynced it) and
costs exactly one physical roll; every `VERIFY` step afterwards
re-derives the full orientation again from `(face, move, new face)` via
`dice_face_map.DiceOrientation.from_two_faces()` — exact by
construction, never a prediction that could turn out wrong.

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
- [x] Manipulation skill (`dice_manipulation_node.py`): pick with pre-chosen
      grasp yaw, lift, world-`x`/`y`-only roll (never `z`), release, report
- [x] Die-geometry model (`dice_face_map.py`): closed-form orientation from
      one known roll + two face numbers (`from_two_faces()` -- no reliance
      on a vision-measured yaw) + BFS-proven-minimal planning, unit-tested
      (worst case 1 calibration roll + 3 planning rolls, any face from any
      starting orientation)
- [x] Orchestrating state machine (`dice_task_orchestrator.py`): calibrates
      with at most one roll when the tracked layout is missing/stale, then
      re-derives it exactly (not a prediction) after every further roll
- [ ] Empirical validation of the new pre-grasp yaw-selection step on the real
      arm/simulator (everything IK-related here has needed at least one
      round of "live testing found X actually fails" — treat this the same
      way until confirmed)
- [ ] Real vision node replacing the simulator behind `dice_common.py`
