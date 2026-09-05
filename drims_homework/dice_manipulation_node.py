# Copyright 2026 DRIMS3 Summer School
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dice manipulation node — the reusable pick / roll / place *skill*.

Given the dice on the table (localized through ``/dice_identification``,
see ``dice_common.py``), a single ``~/pick_rotate_place`` call:

    1. moves home, opens the gripper, identifies the dice;
    2. picks it from above -- grasp yaw aligned with the live ``dice_tf``
       *and* pre-rotated (before ever touching the die, see "Why the
       grasp yaw is picked before grasping" below) so the jaws already
       line up with the world axis the roll is about to turn around;
    3. lifts it straight up (still attached);
    4. rolls it a quarter turn about that fixed world axis while, in the
       same move, carrying it to a fixed, known-safe table spot (see "Why
       release always happens at a fixed spot");
    5. descends and releases right there, retreats;
    6. re-identifies the dice to report the new face-up number.

Throughout all of this the die is never rotated about world Z, at any
point, by any amount -- not during the grasp (the yaw that lines the
jaws up happens before contact, see below), not during the roll (which
is always exactly about world X or world Y), not at release. This is a
deliberate invariant, not an incidental property: it is what keeps a
roll's outcome exact, closed-form, integer arithmetic instead of an
empirical guess -- see ``dice_face_map``'s module docstring for the
planning consequence (the die's full six-face layout is known exactly
from one *executed* roll and the two face numbers around it -- real
perception cannot promise more than that, see "Why two faces, not one"
there -- and every subsequent minimal roll sequence is a proven BFS
minimum, not a discovered one).

The *smart* logic that decides **which** roll gets the die to a desired
number is intentionally NOT here — that is ``dice_task_orchestrator``,
using ``dice_face_map``. This node only executes one roll at a time.

Design notes (each earned by an actual failure while building this)
---------------------------------------------------------------------
**Why roll near the table, not at a distant "home" configuration.**
Carrying the dice all the way to ``home_joints`` and back for every
single quarter turn is a long, unrelated excursion for what should be a
small, local action. Rolling "in place" (pinning the exact pick-time
Cartesian *XY* while swinging the wrist ~90 deg) regularly hit
``NO_IK_SOLUTION`` — too demanding for a 6-DoF arm. ``roll_dice()``
instead combines the rotation with a translation to ``release_position``
(see below) in the very same move, at lift height: the solver gets to
choose a sensible path through both changes together, rather than being
pinned to the pick-time XY while also swinging the wrist.

**Why the roll is about a FIXED WORLD axis** (``roll_axis`` = ``'x'`` or
``'y'``), not the die's own body axis. An earlier, die-body-relative
version made the wrist sweep a *different* physical motion every time,
depending on the die's landing yaw — occasionally near a wrist
singularity, unpredictably. A world-fixed axis means a given roll is
always the same physical wrist motion, so a direction that turns out to
over-rotate the wrist can be permanently excluded once (see
``dice_face_map.CANDIDATE_ROLLS``) instead of rediscovered per face.

**Why the grasp yaw is picked before grasping, not after.** The gripper
must open with its jaws horizontal at release, so the dice actually
falls/settles instead of being released at a tilted angle -- and since a
world-axis roll leaves that axis's own line fixed, jaws that start out
parallel to the *upcoming* roll axis stay exactly parallel to it all the
way to release. The jaws close along the tool's local X axis, and the
grasp itself must stay flush with the die's *actual* live yaw (real
physical grasping, not a scripted attach) -- but because the die is
always axis-aligned at pick time (see the module docstring's invariant:
nothing here ever yaws it), ``dice_tf``'s own X/Y axes are *always*
exactly parallel to world X/Y, never at some arbitrary in-between angle.
That means the jaws can be brought parallel to either world axis with an
exact 0 or 90 deg choice, decided from the already-known live ``dice_tf``
orientation and the already-known upcoming ``roll_axis`` (the
orchestrator sets it via ``set_parameters`` before this call even
starts) -- entirely on paper, before the gripper ever moves.
``pick_dice()`` folds this straight into the grasp approach pose itself:
the gripper is open and clear of the die throughout approach and
descent, so choosing its yaw there costs no extra motion and, unlike the
old post-lift alignment step this replaces, never touches the die's own
orientation at all -- only the still-open gripper's pose changes.

Because the die is consequently *never* yawed about world Z by anything
in this sequence -- not at grasp, not during the roll, not at release --
a given ``(orientation, roll_axis)`` roll's resulting face is exact,
closed-form geometry, not something that has to be learned by trial —
see ``dice_face_map``'s module docstring for the planning consequence
(the full six-face layout is known after a single
``/dice_identification`` reading, and every face is reachable from every
orientation, worst case 3 rolls).

**Why release always happens at a fixed spot** (``release_position``,
a world (X, Y) configured per cell), never wherever the dice happened to
be picked up from. The dice can be found anywhere within the perception
layer's spawn/detection area, which may be close to a table edge or a
cell barrier (see ``drims_cells``' URDF safety barriers); opening the
gripper there risks the gripper body itself colliding with something
right at the moment of release. Picking is unavoidably wherever the dice
actually is — but nothing requires releasing there too. ``roll_dice()``
folds the trip to ``release_position`` into the roll itself (one
combined rotate+translate move, not two), so this costs no extra motion
over the earlier "roll near the pick spot" design, while removing the
edge-collision risk entirely. Where exactly the dice lands within the
next perception cycle does not matter for correctness either way — the
*next* ``pick_rotate_place()`` call always re-reads the dice's actual
position fresh — so centralizing it is a pure safety win.

**Why position/orientation come from captured or just-commanded values,
never a fresh TF lookup mid-sequence.** While the dice is away from the
table, the simulator only refreshes ``dice_tf``/``dice_com_tf`` on its
own ~0.5s "gravity" timer, so a lookup shortly after a move can return a
stale value indistinguishable from "still resting on the table". This
node instead only ever uses: a value read once via a synchronous
request/response (``/dice_identification`` at the very start of the
call), or a value it just commanded itself (if ``move_to_pose`` reports
success, the tool really is there, within tolerance — no read-back
needed).

Interfaces
----------
* Service ``~/pick_rotate_place`` (``std_srvs/srv/Trigger``): runs the
  sequence once with the currently configured ``roll_axis`` /
  ``roll_angle_deg`` and returns a short report (old face -> new face).
* Parameter ``run_on_start`` (bool): also run once at start-up.
"""

import math
import queue
import threading
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from moveit_msgs.msg import MoveItErrorCodes
from tf_transformations import quaternion_multiply, quaternion_about_axis

from tf2_ros import Buffer, TransformListener

from easy_motion.motion_client import MotionClient

from drims_homework.dice_common import create_dice_identification_client, identify_dice

# Quaternion (x, y, z, w) that flips the tool so its +Z axis points "into"
# the frame it is expressed in. Used to grasp the dice from above:
# expressed in ``dice_tf`` (whose +Z points up, away from the top face) it
# makes the gripper approach vertically downwards.
GRASP_DOWN_QUAT = (1.0, 0.0, 0.0, 0.0)

Quat = Tuple[float, float, float, float]
Xyz = Tuple[float, float, float]


def _rotate_vector(v: Xyz, q: Quat) -> Xyz:
    """
    Rotate a 3-vector by a quaternion (x, y, z, w).

    Same convention used by the simulator's own realignment math
    (drims_dice_simulator); kept dependency-free (no numpy) on purpose.
    """
    v_q = (v[0], v[1], v[2], 0.0)
    q_conj = (-q[0], -q[1], -q[2], q[3])
    x, y, z, _ = quaternion_multiply(quaternion_multiply(q, v_q), q_conj)
    return (x, y, z)


class DiceManipulator:
    """Stateless-ish helper that turns high level steps into motion calls."""

    def __init__(self, node: Node, motion: MotionClient, client_node: Node):
        self._node = node
        self._motion = motion
        self._log = node.get_logger()

        # Dedicated node for every blocking call this class makes
        # (/dice_identification). Must NOT be `node`: `node` is added to
        # its own executor once main() starts serving ~/pick_rotate_place
        # (see main()'s comment on why that executor is private, not
        # rclpy's implicit global one), so spin_until_future_complete(node,
        # ...) from inside that very callback would reenter that executor
        # and deadlock. client_node is never spun persistently,
        # so each blocking call gets its own private, non-reentrant spin
        # (same pattern MotionClient itself uses internally).
        self._client_node = client_node

        gp = node.get_parameter

        self.object_id = gp('object_id').value
        self.world_frame = gp('world_frame').value
        self.dice_grasp_frame = gp('dice_grasp_frame').value
        self.attach_frame = gp('attach_frame').value

        self.home_joints = list(gp('home_joints').value)
        self.gripper_open = gp('gripper_open').value
        self.gripper_close = gp('gripper_close').value
        self.gripper_max_effort = gp('gripper_max_effort').value

        self.approach_distance = gp('approach_distance').value
        self.grasp_offset = gp('grasp_offset').value
        self.lift_distance = gp('lift_distance').value
        self.place_safety_height = gp('place_safety_height').value
        self.gripper_settle_time = gp('gripper_settle_time').value

        # Fixed (X, Y) every roll+release targets -- see roll_dice()'s
        # docstring for why this must be a fixed, known-safe spot rather
        # than wherever the dice happened to be picked from. Configured in
        # ``release_frame`` (default ``base_link``, the natural frame for
        # "somewhere on the table in front of the robot", same convention
        # as drims_dice_simulator's spawn ``position``); resolved to
        # ``world_frame`` once here via a static TF lookup, since every
        # downstream pose (and the pick-time table height) is world-frame.
        release_xy = gp('release_position').value
        release_frame = gp('release_frame').value
        self.release_position = self._resolve_release_xy(
            float(release_xy[0]), float(release_xy[1]), release_frame)

        # roll_axis/roll_angle_deg are deliberately NOT cached: they are
        # the two knobs dice_task_orchestrator changes at runtime (via
        # set_parameters) before each call. See _current_roll().
        self.identify_after = gp('identify_after').value

        self._dice_identification_client = create_dice_identification_client(
            self._client_node)

    def _resolve_release_xy(self, x: float, y: float, frame: str) -> Tuple[float, float]:
        """
        Convert the configured release (X, Y) from ``frame`` to ``world_frame``.

        ``frame == world_frame`` (or empty) is a no-op. Otherwise a one-off
        static TF lookup (``world_frame`` <- ``frame``) is done here at
        start-up -- ``base_link`` -> ``world`` on this cell carries a real
        yaw (see configuration_cell_1.yaml: ``rpy "0 0 -1.57"``), so a bare
        X/Y copy would be wrong. Unlike the die pose (which goes stale, see
        the module docstring), this is a fixed robot transform, valid for
        the node's whole life. On failure the raw numbers are kept and a
        loud warning is logged -- check them against RViz before trusting
        them near a barrier.
        """
        if not frame or frame == self.world_frame:
            return (x, y)

        buf = Buffer()
        TransformListener(buf, self._client_node)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                tf = buf.lookup_transform(self.world_frame, frame, rclpy.time.Time())
                t = tf.transform.translation
                q = (tf.transform.rotation.x, tf.transform.rotation.y,
                     tf.transform.rotation.z, tf.transform.rotation.w)
                rx, ry, _ = _rotate_vector((x, y, 0.0), q)
                world_xy = (rx + t.x, ry + t.y)
                self._log.info(
                    f'release_position [{x:.3f}, {y:.3f}] in {frame!r} -> '
                    f'[{world_xy[0]:.3f}, {world_xy[1]:.3f}] in {self.world_frame!r}')
                return world_xy
            except Exception:  # noqa: BLE001 -- TF not ready yet, keep waiting
                rclpy.spin_once(self._client_node, timeout_sec=0.1)

        self._log.error(
            f"Could not look up {self.world_frame!r} <- {frame!r} to resolve "
            f"release_position; using [{x:.3f}, {y:.3f}] as-is (frame {frame!r} "
            f"values in a {self.world_frame!r} field -- verify against RViz!).")
        return (x, y)

    # ------------------------------------------------------------------ #
    # Small utilities                                                     #
    # ------------------------------------------------------------------ #
    def _ok(self, result: MoveItErrorCodes, what: str) -> bool:
        if result is not None and result.val == MoveItErrorCodes.SUCCESS:
            self._log.info(f'{what}: OK')
            return True
        self._log.error(f'{what}: FAILED (MoveItErrorCode={getattr(result, "val", None)})')
        return False

    def _pose(self, frame_id: str, xyz: Xyz, quat_xyzw: Quat) -> PoseStamped:
        p = PoseStamped()
        p.header.frame_id = frame_id
        p.header.stamp = self._node.get_clock().now().to_msg()
        p.pose.position.x, p.pose.position.y, p.pose.position.z = xyz
        (p.pose.orientation.x, p.pose.orientation.y,
         p.pose.orientation.z, p.pose.orientation.w) = quat_xyzw
        return p

    def _failed_moveit_result(self) -> MoveItErrorCodes:
        result = MoveItErrorCodes()
        result.val = MoveItErrorCodes.FAILURE
        return result

    def _safe_call(self, description: str, fn, timeout: float = 50.0,
                   retries: int = 3, retry_delay: float = 0.5, default=None):
        """
        Call a zero-argument MotionClient wrapper with two safety nets.

        Nothing here can hang the ``~/pick_rotate_place`` callback
        forever (both observed in practice):

        1. **Hard wall-clock timeout**, via a background thread: several
           ``easy_motion.MotionClient`` methods call
           ``rclpy.spin_until_future_complete()`` with no timeout at all
           — if the server is ever slow/unresponsive, that blocks
           forever with no way to bound it from here. Waiting on the
           background thread with a timeout frees the caller regardless
           (the thread itself cannot be killed, only abandoned).
        2. **Retry on a raised ``RuntimeError``**: ``move_to_*()`` raise
           this if the action server *rejects* the goal — usually a
           transient hiccup right after an aborted goal.

        Returns ``default`` instead of ever propagating an exception or
        hanging, keeping every ``_ok()``/truthiness check downstream
        working unchanged.
        """
        last_exc = None
        for attempt in range(retries):
            result_queue: 'queue.Queue' = queue.Queue(maxsize=1)

            def _run():
                try:
                    result_queue.put(('ok', fn()))
                except Exception as exc:  # noqa: BLE001 -- deliberately broad
                    result_queue.put(('error', exc))

            threading.Thread(target=_run, daemon=True).start()
            try:
                status, value = result_queue.get(timeout=timeout)
            except queue.Empty:
                self._log.error(
                    f'{description}: no response after {timeout:.0f}s; giving up '
                    f'and moving on (background thread left running, best-effort).')
                return default

            if status == 'ok':
                return value

            last_exc = value
            if isinstance(value, RuntimeError):
                self._log.warn(
                    f'{description}: goal rejected ({value}); '
                    f'retrying ({attempt + 1}/{retries})...')
                time.sleep(retry_delay)
                continue

            self._log.error(f'{description}: raised {value!r}')
            return default

        self._log.error(
            f'{description}: goal kept being rejected after {retries} attempts '
            f'({last_exc!r}); giving up on this call.')
        return default

    def _current_roll(self) -> Tuple[str, float]:
        """
        Read roll_axis/roll_angle_deg live.

        dice_task_orchestrator overwrites these via set_parameters before
        each call, so caching them at __init__ time would silently
        ignore that.
        """
        axis = self._node.get_parameter('roll_axis').value
        angle_deg = self._node.get_parameter('roll_angle_deg').value
        return axis, angle_deg

    # ------------------------------------------------------------------ #
    # Primitives                                                          #
    # ------------------------------------------------------------------ #
    def go_home(self) -> bool:
        self._log.info('Moving to home configuration...')
        res = self._safe_call(
            'move_to_joint(home)',
            lambda: self._motion.move_to_joint(self.home_joints),
            default=self._failed_moveit_result())
        return self._ok(res, 'move_to_joint(home)')

    def open_gripper(self) -> bool:
        reached, _ = self._safe_call(
            'gripper_command(open)',
            lambda: self._motion.gripper_command(
                position=self.gripper_open, max_effort=self.gripper_max_effort),
            default=(False, False))
        self._log.info(f'Gripper opened (reached_goal={reached})')
        if self.gripper_settle_time > 0.0:
            time.sleep(self.gripper_settle_time)
        return True

    def close_gripper(self) -> bool:
        reached, stalled = self._safe_call(
            'gripper_command(close)',
            lambda: self._motion.gripper_command(
                position=self.gripper_close, max_effort=self.gripper_max_effort),
            default=(False, False))
        self._log.info(f'Gripper closed (reached_goal={reached}, stalled={stalled})')
        if self.gripper_settle_time > 0.0:
            time.sleep(self.gripper_settle_time)
        return True

    def identify_dice(self) -> Tuple[Optional[int], Optional[PoseStamped]]:
        return identify_dice(self._client_node, self._dice_identification_client)

    def grasp_orientation(self, dice_quat_at_pick: Quat, roll_axis: str) -> Quat:
        """
        Pick a top-down grasp yaw whose jaws end up parallel to ``roll_axis``.

        Decided in ``dice_grasp_frame``, entirely *before* the gripper
        ever touches the die. See the module docstring's "Why the grasp
        yaw is picked before grasping, not after". Because the die is
        always axis-aligned at pick time (nothing in this sequence
        ever yaws it about world Z),
        ``dice_tf``'s X/Y axes are always exactly parallel to world X/Y,
        so the two candidates below (``GRASP_DOWN_QUAT`` with 0 or 90 deg
        of extra yaw about ``dice_tf``'s own Z -- which *is* world Z
        here) are the only two that can possibly matter; picking by exact
        dot product is not an approximation.

        ``roll_axis == 'z'`` is the one exception: ``dice_task_orchestrator``
        uses it, once, to straighten a die found with residual yaw before
        it ever calibrates (see that module's docstring) -- there is no
        roll axis to line the jaws up with yet, so this returns the flush
        0 deg grasp directly rather than running the scoring below, which
        also means the straightening roll lands the die at *exactly*
        world yaw 0 (not merely some other axis-aligned multiple of 90).
        """
        if roll_axis == 'z':
            return GRASP_DOWN_QUAT
        target = (1.0, 0.0, 0.0) if roll_axis == 'x' else (0.0, 1.0, 0.0)
        best_quat, best_score = GRASP_DOWN_QUAT, -1.0
        for psi in (0.0, math.pi / 2.0):
            yaw = quaternion_about_axis(psi, (0.0, 0.0, 1.0))
            candidate = quaternion_multiply(yaw, GRASP_DOWN_QUAT)
            grip_dir = _rotate_vector(
                (1.0, 0.0, 0.0), quaternion_multiply(dice_quat_at_pick, candidate))
            score = abs(grip_dir[0] * target[0] + grip_dir[1] * target[1])
            if score > best_score:
                best_quat, best_score = candidate, score
        return best_quat

    def pick_dice(self, grasp_quat: Quat) -> bool:
        """
        Approach from above and grasp on live ``dice_tf``, then attach.

        ``grasp_quat`` (from ``grasp_orientation()``) is ``dice_tf``-flush
        *and* already yawed for the upcoming roll -- see the module
        docstring.
        """
        approach = self._pose(self.dice_grasp_frame,
                              (0.0, 0.0, self.approach_distance),
                              grasp_quat)
        self._log.info('Moving to pre-grasp pose...')
        res = self._safe_call(
            'move_to_pose(pre-grasp)',
            lambda: self._motion.move_to_pose(approach, cartesian_motion=False),
            default=self._failed_moveit_result())
        if not self._ok(res, 'move_to_pose(pre-grasp)'):
            return False

        grasp = self._pose(self.dice_grasp_frame,
                           (0.0, 0.0, self.grasp_offset),
                           grasp_quat)
        self._log.info('Descending to grasp pose...')
        res = self._safe_call(
            'move_to_pose(grasp)',
            lambda: self._motion.move_to_pose(grasp, cartesian_motion=True),
            default=self._failed_moveit_result())
        if not self._ok(res, 'move_to_pose(grasp)'):
            return False

        self.close_gripper()

        # attached = self._safe_call(
        #     'attach_object',
        #     lambda: self._motion.attach_object(self.object_id, self.attach_frame),
        #     default=False)
        # self._log.info(f'attach_object({self.object_id}) -> {attached}')
        # return bool(attached)
        return True 

    def lift(self) -> bool:
        up = self._pose(self.world_frame, (0.0, 0.0, self.lift_distance),
                        (0.0, 0.0, 0.0, 1.0))
        self._log.info(f'Lifting {self.lift_distance:.3f} m...')
        res = self._safe_call(
            'move_to_pose(lift)',
            lambda: self._motion.move_to_pose(up, cartesian_motion=True,
                                              relative_motion=True),
            default=self._failed_moveit_result())
        return self._ok(res, 'move_to_pose(lift)')

    def roll_dice(self, table_z: float, current_quat: Quat) -> Optional[Quat]:
        """
        Roll the held dice one quarter turn, translating to ``release_position``.

        Applies ``roll_axis``/``roll_angle_deg`` on top of
        ``current_quat`` (the tool's actual current orientation --
        already grip-axis-aligned at grasp time, see
        ``grasp_orientation()``) while, in the very same move, carrying
        the dice to the fixed world (X, Y) in ``self.release_position``
        (``release_position``/``release_frame`` resolved to world at
        start-up, see ``_resolve_release_xy()``) -- see the module
        docstring's "Why release always happens at a fixed spot" for why
        every roll ends there rather than near wherever the dice was
        picked up.
        ``table_z`` is the table-height reference read from
        ``/dice_identification`` at pick time (valid for any face, see
        ``pick_rotate_place()``). Returns the resulting world-frame tool
        orientation, or None on failure.

        ``roll_axis == 'z'`` is accepted too, purely so
        ``dice_task_orchestrator`` can drive its one-time pre-calibration
        de-yaw straightening through this exact same
        set_parameters/~pick_rotate_place path -- see that module's
        docstring. ``CANDIDATE_ROLLS``/planning never produce ``'z'``, so
        this never happens as part of an actual face-changing roll; the
        module docstring's "never yawed about world Z" invariant is about
        that roll-execution path, not this narrow, explicit exception.

        No further yaw check happens here on every ``'x'``/``'y'`` roll
        (an earlier version re-checked and, if needed, cancelled residual
        yaw on every single roll): in practice, once
        ``dice_task_orchestrator``'s one-time de-yaw has lined the die up,
        it does not drift roll after roll, so paying for that check every
        time was pure overhead with no observed benefit -- removed.
        """
        axis, angle_deg = self._current_roll()
        if axis not in ('x', 'y', 'z'):
            self._log.error(f"roll_axis={axis!r} must be 'x', 'y' or 'z'")
            return None

        axis_vec = {'x': (1.0, 0.0, 0.0),
                    'y': (0.0, 1.0, 0.0),
                    'z': (0.0, 0.0, 1.0)}[axis]
        roll_delta = quaternion_about_axis(math.radians(angle_deg), axis_vec)
        # Extrinsic (world-frame) rotation on top of the current
        # orientation -- left-multiplied, unlike a body-relative roll.
        world_quat = quaternion_multiply(roll_delta, current_quat)

        roll_pose = self._pose(
            self.world_frame,
            (self.release_position[0], self.release_position[1],
             table_z + self.lift_distance),
            world_quat)

        self._log.info(
            f'Rolling held dice: axis={axis} {angle_deg:+.0f} deg, '
            f'moving to release_position={self.release_position}...')
        res = self._safe_call(
            'move_to_pose(roll)',
            lambda: self._motion.move_to_pose(roll_pose, cartesian_motion=False),
            default=self._failed_moveit_result())
        if not self._ok(res, 'move_to_pose(roll)'):
            return None
        return world_quat

    def release_after_roll(self, quat: Quat) -> bool:
        """
        Release the dice at ``release_position``, where ``roll_dice()`` left it.

        A straight-down relative Cartesian descent (orientation
        unchanged -- kept exactly as ``grasp_orientation()``/``roll_dice()``
        left it, i.e. jaws horizontal, see module docstring), open, detach,
        retreat. Always opens+detaches even if the descent itself fails --
        releasing from wherever the arm is beats leaving the dice attached,
        which would break every subsequent call.
        """
        descend = self._pose(
            self.world_frame,
            (0.0, 0.0, -(self.lift_distance - self.place_safety_height)),
            (0.0, 0.0, 0.0, 1.0))
        self._log.info('Descending to release height...')
        res = self._safe_call(
            'move_to_pose(release-descend)',
            lambda: self._motion.move_to_pose(descend, cartesian_motion=True,
                                              relative_motion=True),
            default=self._failed_moveit_result())
        descended = self._ok(res, 'move_to_pose(release-descend)')
        if not descended:
            self._log.warn(
                'Could not descend to release height; releasing from '
                'wherever the dice currently is instead of leaving it attached.')

        self.open_gripper()
        detached = self._safe_call(
            'detach_object', lambda: self._motion.detach_object(self.object_id),
            default=False)
        self._log.info(f'detach_object({self.object_id}) -> {detached}')

        if descended:
            retreat = self._pose(
                self.world_frame, (0.0, 0.0, self.lift_distance - self.place_safety_height),
                (0.0, 0.0, 0.0, 1.0))
            res = self._safe_call(
                'move_to_pose(retreat)',
                lambda: self._motion.move_to_pose(retreat, cartesian_motion=True,
                                                  relative_motion=True),
                default=self._failed_moveit_result())
            self._ok(res, 'move_to_pose(retreat)')
        return descended

    def place_dice(self, xyz: Xyz, quat: Quat) -> bool:
        """
        Approach and descend to a specific ``xyz``/``quat``, then release.

        Only used for **recovery** (``_recover()``): the normal path uses
        ``release_after_roll()`` instead (see its docstring). Stops
        ``place_safety_height`` above ``xyz`` rather than touching down
        exactly -- a deliberate margin so the wrist/gripper never drives
        into the table. Always opens+detaches even on a failed
        approach/descent, same reasoning as ``release_after_roll()``.
        """
        x, y, z = xyz
        approach = self._pose(self.world_frame, (x, y, z + self.approach_distance), quat)
        self._log.info('Moving to pre-place pose...')
        res = self._safe_call(
            'move_to_pose(pre-place)',
            lambda: self._motion.move_to_pose(approach, cartesian_motion=False),
            default=self._failed_moveit_result())
        approach_ok = self._ok(res, 'move_to_pose(pre-place)')

        descended = False
        if approach_ok:
            descend = self._pose(self.world_frame, (x, y, z + self.place_safety_height), quat)
            self._log.info('Descending to place pose...')
            res = self._safe_call(
                'move_to_pose(place)',
                lambda: self._motion.move_to_pose(descend, cartesian_motion=True),
                default=self._failed_moveit_result())
            descended = self._ok(res, 'move_to_pose(place)')

        if not descended:
            self._log.warn(
                'Could not reach the expected place pose; releasing from '
                'wherever the dice currently is instead of leaving it attached.')

        self.open_gripper()
        detached = self._safe_call(
            'detach_object', lambda: self._motion.detach_object(self.object_id),
            default=False)
        self._log.info(f'detach_object({self.object_id}) -> {detached}')

        if approach_ok:
            retreat = self._pose(self.world_frame, (x, y, z + self.approach_distance), quat)
            res = self._safe_call(
                'move_to_pose(retreat)',
                lambda: self._motion.move_to_pose(retreat, cartesian_motion=False),
                default=self._failed_moveit_result())
            self._ok(res, 'move_to_pose(retreat)')
        return descended

    def _recover(self, xyz: Xyz, current_quat: Quat) -> None:
        """
        Best-effort: get a still-attached dice back down safely.

        Used after a failure, so the next call starts clean instead of
        trying to pick_dice() an object that is already attached. Always
        targets ``release_position`` (never the original, possibly
        near-an-edge pick spot) -- the dice gets released either way, so
        the same collision-safety reasoning as the normal path applies
        (see the module docstring's "Why release always happens at a
        fixed spot"). Two tiers:

        1. Place at ``release_position``, using ``current_quat`` (the
           last orientation a motion actually reported success at -- so
           known reachable from here). Usually fast.
        2. Only if that fails too: ``home_joints`` (pure joint-space,
           proven reachable at the very start of this call) then place
           at ``release_position`` with a neutral, yaw-free straight-down
           orientation (broadly reachable, at the cost of not caring
           which face ends up up -- fine for best-effort recovery,
           always re-checked by the caller afterwards).
        """
        safe_xyz = (self.release_position[0], self.release_position[1], xyz[2])
        self._log.warn(
            'Attempting recovery: placing the dice back down at '
            'release_position after a failure...')
        if self.place_dice(safe_xyz, current_quat):
            return
        self._log.warn('Direct recovery failed too; falling back to home + neutral orientation...')
        self.go_home()
        self.place_dice(safe_xyz, GRASP_DOWN_QUAT)

    # ------------------------------------------------------------------ #
    # Full task                                                           #
    # ------------------------------------------------------------------ #
    def pick_rotate_place(self) -> Tuple[bool, str]:
        if not self.go_home():
            return False, 'failed to reach home'
        self.open_gripper()

        face_before, pick_pose = self.identify_dice()
        if face_before is None or pick_pose is None:
            return False, 'dice identification failed'

        # Only place_xyz[2] (table height) is reused later, as the Z
        # reference for the roll/release target -- valid for any face:
        # every face sits at the same "table height + dice edge length"
        # when resting flat. X/Y are NOT reused: release always happens
        # at the fixed release_position instead, see module docstring.
        place_xyz = (pick_pose.pose.position.x, pick_pose.pose.position.y,
                     pick_pose.pose.position.z)
        dice_quat_at_pick = (pick_pose.pose.orientation.x, pick_pose.pose.orientation.y,
                             pick_pose.pose.orientation.z, pick_pose.pose.orientation.w)

        # Known before contact -- dice_task_orchestrator always sets this
        # via set_parameters before calling this service -- so the grasp
        # yaw can be chosen for the upcoming roll from the very first
        # move, see grasp_orientation()/the module docstring.
        axis, _ = self._current_roll()
        grasp_quat = self.grasp_orientation(dice_quat_at_pick, axis)

        if not self.pick_dice(grasp_quat):
            return False, 'pick failed'

        # From here on the dice is rigidly attached: any failure below
        # must recover before reporting, so the next call starts clean.
        # current_quat tracks the tool's actual orientation through the
        # sequence -- always a value a motion just reported success at,
        # never re-derived from a (possibly stale) TF lookup, see module
        # docstring.
        current_quat = quaternion_multiply(dice_quat_at_pick, grasp_quat)

        if not self.lift():
            self._recover(place_xyz, current_quat)
            return False, 'lift failed'

        roll_quat = self.roll_dice(place_xyz[2], current_quat)
        if roll_quat is None:
            self._recover(place_xyz, current_quat)
            return False, 'rotation failed'

        place_ok = self.release_after_roll(roll_quat)
        self.go_home()  # safe now: the dice was already released above
        if not place_ok:
            return False, 'place failed'

        face_after = None
        if self.identify_after:
            time.sleep(0.6)
            face_after, _ = self.identify_dice()

        msg = f'pick-rotate-place done: face {face_before} -> {face_after}'
        self._log.info(msg)
        return True, msg


def _declare_parameters(node: Node) -> None:
    node.declare_parameter('object_id', 'dice')
    node.declare_parameter('world_frame', 'world')
    node.declare_parameter('dice_grasp_frame', 'dice_tf')
    node.declare_parameter('attach_frame', 'tip')

    node.declare_parameter('home_joints', [0.0, -1.97, 2.13, -1.83, -1.50, 0.0])
    node.declare_parameter('gripper_open', 0.045)
    node.declare_parameter('gripper_close', 0.029)
    node.declare_parameter('gripper_max_effort', 10.0)

    # Pre-grasp height above the die, in dice_grasp_frame -- kept at 10 cm,
    # it's a fine hover height on its own.
    node.declare_parameter('approach_distance', 0.10)
    # Stop grasp_offset short of the die surface (dice_tf origin), not
    # flush with it: descending the full approach_distance all the way to
    # the die was hitting a collision. Net cartesian descent onto the die
    # in pick_dice() is approach_distance - grasp_offset = 8.5 cm.
    node.declare_parameter('grasp_offset', 0.015)
    # See dice_manipulation_config.yaml for why these two defaults were
    # bumped up from 0.15/0.02 (diagnosing why 'y' rolls were never
    # observed to succeed).
    node.declare_parameter('lift_distance', 0.20)
    # Clearance kept above the table when releasing -- never touch down
    # exactly (see release_after_roll()/place_dice()). Lowered by 1 cm
    # again (0.03 -> 0.02): release wanted 1 cm closer to the table.
    node.declare_parameter('place_safety_height', 0.02)
    # Seconds to hold still after a gripper open/close command, so the
    # jaws physically settle / the die actually falls before the next move.
    node.declare_parameter('gripper_settle_time', 1.0)
    # Fixed (X, Y) every roll carries the dice to before releasing -- see
    # roll_dice()'s and the module docstring's "Why release always happens
    # at a fixed spot". Default is the ur5e_1 table top's centre in
    # `world` (table_length/2, table_width/2 -- see
    # dice_manipulation_config.yaml for the derivation from
    # ur5e_cell.urdf.xacro / configuration_cell_1.yaml), Y nudged +0.05.
    # Given in ``release_frame`` and resolved to ``world_frame`` at
    # start-up if different, see _resolve_release_xy(). Tune per cell.
    node.declare_parameter('release_position', [0.6, 0.47])
    node.declare_parameter('release_frame', 'world')

    # Which fixed world axis the roll turns about ('x' or 'y') and by how
    # much (deg). Normally overridden at runtime (via set_parameters) by
    # dice_task_orchestrator; the defaults below are only used for a
    # standalone/manual test.
    node.declare_parameter('roll_axis', 'x')
    # 60, not 90: released ~30 deg early, gravity finishes the tip -- see
    # dice_face_map.CANDIDATE_ROLLS. Overridden per-call by the orchestrator.
    node.declare_parameter('roll_angle_deg', 60.0)

    node.declare_parameter('identify_after', True)
    node.declare_parameter('run_on_start', False)

    node.declare_parameter('gripper_action_name', '/gripper_action_controller/gripper_cmd')


def main(args=None):
    rclpy.init(args=args)

    # use_global_arguments stays at its default (True): this node's name
    # must match dice_manipulation_config.yaml so the launch --params-file
    # applies.
    node = rclpy.create_node('dice_manipulation_node')
    _declare_parameters(node)

    # Never spun in a persistent loop -- see the comment in
    # DiceManipulator.__init__ for why this must be a separate node from
    # `node` (which gets its own private executor below).
    client_node = rclpy.create_node('dice_manipulation_node_clients')

    motion = MotionClient(
        gripper_action_name=node.get_parameter('gripper_action_name').value)

    manipulator = DiceManipulator(node, motion, client_node)

    def _srv_cb(request, response):
        # An unforeseen exception here must never leave this Trigger
        # response unsent -- that hangs whoever called the service
        # (e.g. dice_task_orchestrator, blocked in
        # spin_until_future_complete()) forever. Observed once in
        # practice before this was added.
        try:
            ok, message = manipulator.pick_rotate_place()
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see above
            node.get_logger().error(f'pick_rotate_place() raised: {exc!r}')
            ok, message = False, f'unhandled exception: {exc!r}'
        response.success = ok
        response.message = message
        return response

    node.create_service(Trigger, '~/pick_rotate_place', _srv_cb)
    node.get_logger().info(
        'dice_manipulation_node ready (service: ~/pick_rotate_place, std_srvs/Trigger)')

    # `node` gets its OWN executor here instead of the bare rclpy.spin(node)
    # this used to be -- that implicitly uses rclpy's process-wide global
    # executor (rclpy.get_global_executor()), same as every bare
    # rclpy.spin_until_future_complete(...) call with no executor= given.
    # `easy_motion.MotionClient` (== `motion` above) IS a Node and calls
    # exactly that, with no executor, from the *background thread*
    # `_safe_call()` spawns for every motion command. Normally the main
    # thread is just blocked on `queue.get()` while that runs, so there is
    # no conflict -- but if a call is ever abandoned on `_safe_call()`'s
    # timeout (motion server slow/stuck: "no response after Ns; giving up
    # ... background thread left running, best-effort"), that background
    # thread stays inside its own spin on the global executor. The main
    # thread then returns control to its own spin loop -- if that loop is
    # ALSO the global executor (bare rclpy.spin(node)), the two threads
    # race to iterate the same executor's internal generator and rclpy
    # raises `ValueError: generator already executing`, killing this whole
    # process over one slow motion call (observed in practice). Giving
    # `node` a private executor removes it from that shared pool entirely,
    # so an abandoned motion call now just means this one roll fails
    # (dice_task_orchestrator already treats 'failed to reach home' etc.
    # as recoverable/retryable) instead of taking the node down.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        if node.get_parameter('run_on_start').value:
            ok, message = manipulator.pick_rotate_place()
            node.get_logger().info(f'run_on_start result: {ok} ({message})')
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        motion.destroy_node()
        client_node.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
