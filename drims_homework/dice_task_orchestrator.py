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
Dice task orchestrator — the "smart" decision layer.

Two ways to drive it -- pick one per use case:

**Interactive / supervised** (recommended while tuning a new cell):
``~/plan_target_face`` then ``~/execute_planned_sequence``. The first
call identifies the die and, if the die's full layout is not already
known (see "Calibration" below), does **exactly one** physical roll to
learn it -- never more than one, and never again afterwards unless
something desyncs the tracked state (see below) -- then reports the
complete, provably minimal roll sequence to ``target_face``. Nothing is
executed until you separately call ``~/execute_planned_sequence``, which
runs that sequence end to end. This is the path that lets you watch each
roll happen and stop/inspect between them.

**Fully automatic**: ``~/reach_target_face`` runs the whole thing
unattended as an explicit state machine (``State`` below):

.. code-block:: text

    IDENTIFY ──► CALIBRATE (if needed) ──► CHECK_TARGET ──(match)──► DONE
                                                 │
                                       (no match, attempts left)
                                                 ▼
                                               PLAN ──(no move known)──► FAILED
                                                 │
                                                 ▼
                                               ROLL ──(hard failure)──► FAILED
                                                 │        │
                                                 │   (recoverable: re-identify,
                                                 │    loop back to IDENTIFY)
                                                 ▼
                                              VERIFY ──► back to CHECK_TARGET

Calibration -- why this node ever needs to roll before it can plan
---------------------------------------------------------------------
Real perception can reliably report *which number* is up, never the
die's precise yaw on the table (a top face is a square, and several pip
patterns are themselves rotationally symmetric -- see
``dice_face_map``'s module docstring, "Why two faces, not one"). So a
single ``/dice_identification`` reading is **not** enough to know where
the other five faces are; the only way to find out is to execute one
*known* roll and observe the resulting face number
(``DiceOrientation.from_two_faces()``). This node tracks the die's full
orientation (``self._orientation``) for as long as it keeps matching
reality: every ``IDENTIFY`` compares the freshly-read face against what
is tracked, and only pays for a calibration roll when they disagree (the
very first call ever, or after anything that could have desynced them --
see ``_calibrate()``). Once calibrated, every subsequent roll is both
executed *and* used to re-derive the orientation fresh via
``from_two_faces()`` (cheap, exact, and self-correcting: it never trusts
a mere prediction over what was actually observed).

* **IDENTIFY** asks the perception layer (``/dice_identification``,
  ``dice_common.py``) which face is currently up.
* **CALIBRATE** (only entered when the tracked orientation is missing or
  stale) performs exactly one roll purely to (re)establish the full
  layout -- see above. If the die is found yawed off world Z at this
  point, it is straightened to yaw 0 first, with one extra roll about a
  world-Z axis ``dice_manipulation_node`` accepts only for this -- see
  ``_calibrate()``. This is the *only* place in this node that ever
  looks at the perception layer's reported orientation (still never for
  reconstructing the six-face layout itself, only to decide whether
  straightening is needed) and the only place a world-Z roll is ever
  commanded.
* **CHECK_TARGET** compares the current up face to ``target_face``; also
  where the ``max_attempts`` budget is enforced.
* **PLAN** asks ``dice_face_map.plan_min_sequence()`` -- pure geometry,
  no ROS -- for the *shortest possible* full sequence to ``target_face``
  from the current orientation, and takes its first move (re-planning
  fresh after every physical roll costs nothing: the search space is at
  most the 24 rotations of a cube).
* **ROLL** configures ``dice_manipulation_node`` for that roll (its
  ``set_parameters`` service) and drives it through one full
  pick-lift-roll-place cycle (its ``~/pick_rotate_place`` service). A
  roll that fails during the turn itself (``'rotation failed'``) is a
  motion-layer statement that this exact (orientation, axis) is
  unreachable, so it gets permanently excluded via the ``_blocked`` set
  passed to every subsequent ``plan_min_sequence()`` call -- see
  ``_try_roll()``.
* **VERIFY** re-identifies the die and re-derives the full orientation
  from ``(face before this roll, this roll, face after)`` via
  ``from_two_faces()`` -- exact by construction, not a prediction that
  could turn out wrong.

Both paths share the same roll-execution helper (``_try_roll()``) and the
same ``_blocked`` set of known-kinematically-infeasible rolls, so
exploration done via one shows up in the other.

This node only ever talks to two contracts:

    * ``/dice_identification`` (perception — the simulator today, a real
      vision node tomorrow, see ``dice_common.py``) -- ``face_number``
      drives everything; the reported orientation is read in exactly one
      place (``_calibrate()``'s de-yaw pre-step, to decide whether the
      die needs straightening) and never trusted for reconstructing the
      six-face layout itself (see "Calibration" above);
    * ``dice_manipulation_node``'s ``set_parameters`` and
      ``~/pick_rotate_place`` services.

It never talks to MoveIt / the gripper / TF directly, and it never
assumes the simulator's internal geometry beyond the die's own numbering
(``dice_face_map.STANDARD_BODY_NORMALS``, documented there) -- how a roll
maps orientations to orientations is plain rotation geometry, not
anything learned from the simulator.

Interfaces
----------
* Parameter ``target_face`` (1-6): the face the die should end up showing.
* Service ``~/plan_target_face`` (``std_srvs/srv/Trigger``): identify,
  calibrate with at most one physical roll if needed, and report the
  complete minimal roll sequence towards ``target_face``.
* Service ``~/execute_planned_sequence`` (``std_srvs/srv/Trigger``): run
  the sequence ``~/plan_target_face`` last reported, roll by roll,
  re-deriving the tracked orientation after each one.
* Service ``~/reach_target_face`` (``std_srvs/srv/Trigger``): (re-)runs
  the fully-automatic state machine for the currently configured
  ``target_face``, with no pause for confirmation.
* Parameter ``run_on_start`` (bool): also run ``reach_target_face`` once
  at start-up.

To change the target at runtime without restarting::

    ros2 param set /dice_task_orchestrator target_face 6
    ros2 service call /dice_task_orchestrator/plan_target_face std_srvs/srv/Trigger "{}"
    # inspect the logged sequence, then:
    ros2 service call /dice_task_orchestrator/execute_planned_sequence std_srvs/srv/Trigger "{}"
"""

import math
from enum import Enum, auto
from typing import Optional, Set, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter

from drims_homework.dice_common import create_dice_identification_client, identify_dice
from drims_homework.dice_face_map import CANDIDATE_ROLLS, DiceOrientation, Move, plan_min_sequence


def _yaw_from_quat(q) -> float:
    """
    World-Z yaw (rad) of a ``geometry_msgs/Quaternion``, assumed close to flat.

    Standard ZYX-Euler yaw extraction. Used only by ``_calibrate()``'s
    pre-straightening step, to answer "is this die close enough to
    axis-aligned to grasp/roll cleanly" -- a much weaker question than
    reconstructing the die's six-face layout from a single reading,
    which stays exclusively face-number-based (``from_two_faces()``, see
    the module docstring's "Calibration") because a real vision node's
    yaw is not trustworthy for that.
    """
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

# pick_rotate_place() failure messages that mean the dice was never
# touched, or the roll layer itself already gave up recovering -- not
# specific to whichever roll was being tried, so retrying a different
# roll cannot help. Anything else ('rotation failed', 'lift failed',
# 'place failed') is treated as recoverable: re-check the real state via
# /dice_identification and keep going.
_UNRECOVERABLE_FAILURES = frozenset({
    'failed to reach home', 'pick failed', 'dice identification failed',
})

# (orientation.signature(), move) pairs already found kinematically
# infeasible -- see plan_min_sequence()'s `blocked` argument.
Blocked = Set[Tuple[tuple, Move]]


class State(Enum):
    IDENTIFY = auto()
    CALIBRATE = auto()
    CHECK_TARGET = auto()
    PLAN = auto()
    ROLL = auto()
    VERIFY = auto()
    DONE = auto()
    FAILED = auto()


class DiceTaskOrchestrator:

    def __init__(self, node: Node, client_node: Node):
        self._node = node
        self._log = node.get_logger()

        # Dedicated node for every outbound blocking call this class
        # makes (/dice_identification, .../set_parameters,
        # ~/pick_rotate_place). Must NOT be `node`: `node` is what
        # rclpy.spin(node) owns once main() starts serving
        # ~/reach_target_face, so spin_until_future_complete(node, ...)
        # from inside that very callback would reenter node's own
        # executor and deadlock. client_node is never spun persistently.
        self._client_node = client_node

        gp = node.get_parameter
        self.manipulation_node_name = gp('manipulation_node_name').value
        self.max_attempts = gp('max_attempts').value
        # See _calibrate()'s de-yaw pre-step: below this the die is
        # considered already at yaw 0 and straightening is skipped.
        self.deyaw_tolerance_deg = gp('deyaw_tolerance_deg').value

        self._dice_identification_client = create_dice_identification_client(
            self._client_node, gp('dice_identification_service').value)
        self._pick_rotate_place_client = self._client_node.create_client(
            Trigger, f'/{self.manipulation_node_name}/pick_rotate_place')
        self._set_parameters_client = self._client_node.create_client(
            SetParameters, f'/{self.manipulation_node_name}/set_parameters')

        # The die's tracked orientation, or None if not (yet) known --
        # see the module docstring's "Calibration". Kept for this node's
        # whole lifetime, like _blocked below: nothing but this node
        # drives the arm, so it stays valid across requests until an
        # IDENTIFY ever disagrees with it.
        self._orientation: Optional[DiceOrientation] = None

        # Rolls already found kinematically infeasible at a given exact
        # orientation -- see dice_face_map.plan_min_sequence()'s
        # `blocked` argument. Kept for this node's whole lifetime: a
        # physical property of this robot cell, not of a single
        # reach_target_face()/plan_target_face() request.
        self._blocked: Blocked = set()

        # Target plan_target_face() last found a sequence for, consumed
        # by execute_planned_sequence(). Not the sequence itself -- that
        # is always re-derived live at execute time (see
        # execute_planned_sequence()'s docstring).
        self._pending_target: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Driving dice_manipulation_node                                      #
    # ------------------------------------------------------------------ #
    def _set_rotation(self, axis: str, angle_deg: float) -> bool:
        if not self._set_parameters_client.wait_for_service(timeout_sec=5.0):
            self._log.error(f"'{self.manipulation_node_name}/set_parameters' not available")
            return False

        request = SetParameters.Request()
        request.parameters = [
            Parameter('roll_axis', Parameter.Type.STRING, str(axis)).to_parameter_msg(),
            Parameter('roll_angle_deg', Parameter.Type.DOUBLE,
                      float(angle_deg)).to_parameter_msg(),
        ]
        future = self._set_parameters_client.call_async(request)
        rclpy.spin_until_future_complete(self._client_node, future)
        result = future.result()
        if result is None:
            self._log.error('set_parameters call failed (no response)')
            return False

        ok = all(r.successful for r in result.results)
        if not ok:
            reasons = [r.reason for r in result.results if not r.successful]
            self._log.error(f'set_parameters rejected: {reasons}')
        return ok

    def _pick_rotate_place(self) -> Tuple[bool, str]:
        if not self._pick_rotate_place_client.wait_for_service(timeout_sec=10.0):
            return False, f"'{self.manipulation_node_name}/pick_rotate_place' not available"
        future = self._pick_rotate_place_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self._client_node, future)
        result = future.result()
        if result is None:
            return False, 'pick_rotate_place call failed (no response)'
        return result.success, result.message

    def _identify_face(self) -> Optional[int]:
        """
        Identify the die; return the face number, or None on failure.

        Only ``face_number`` is used -- never the reported orientation,
        see the module docstring's "Calibration" for why that would not
        be trustworthy against real perception.
        """
        face, pose = identify_dice(self._client_node, self._dice_identification_client)
        if face is None or pose is None:
            return None
        return face

    def _identify(self) -> Tuple[Optional[int], Optional[PoseStamped]]:
        """
        Identify the die; return ``(face_number, pose)``.

        Unlike ``_identify_face()``, also hands back the raw pose --
        used only right before a possible ``CALIBRATE``/``_calibrate()``
        (see ``plan_target_face()`` and ``reach_target_face()``'s
        ``IDENTIFY`` state), and only so ``_calibrate()`` can measure the
        die's current yaw about world Z for its one-time straightening
        pre-step. Every other identification in this module still goes
        through ``_identify_face()`` and never looks at the pose, per
        the module docstring's "Calibration".
        """
        return identify_dice(self._client_node, self._dice_identification_client)

    # ------------------------------------------------------------------ #
    # Shared roll execution (used by both the automatic state machine    #
    # and the interactive plan/execute pair below)                       #
    # ------------------------------------------------------------------ #
    def _try_roll(self, orientation: Optional[DiceOrientation], move: Move) -> Tuple[str, str]:
        """
        Configure and execute one physical roll.

        ``orientation`` is the tracked state *before* this roll, used
        only to key ``self._blocked`` on a kinematic failure -- pass
        ``None`` during calibration, when it is not known yet (nothing
        gets blocked in that case; there is no exact orientation to key
        it on).

        Returns ``(outcome, message)`` where ``outcome`` is one of:

        * ``'ok'`` -- the roll executed; the caller should re-identify
          and rebuild the tracked orientation via ``from_two_faces()``.
        * ``'recoverable'`` -- it failed but ``dice_manipulation_node``
          already put the dice back down best-effort; ``move`` has been
          added to ``self._blocked`` at this exact ``orientation`` if the
          failure was kinematic (``'rotation failed'`` -- the motion
          layer could not actually turn the die about this axis from
          here) and ``orientation`` was known; the caller should
          re-identify to get the real (possibly unchanged) state and
          keep going.
        * ``'hard'`` -- unrecoverable; the caller must give up,
          ``message`` explains why.
        """
        axis, angle = move
        if not self._set_rotation(axis, angle):
            return 'hard', 'failed to configure dice_manipulation_node rotation'

        ok, message = self._pick_rotate_place()
        if ok:
            return 'ok', message

        if message in _UNRECOVERABLE_FAILURES:
            return 'hard', f'pick_rotate_place failed: {message}'

        if message == 'rotation failed':
            # The motion layer could not actually turn the die about this
            # axis from this exact orientation -- never try it again from
            # here (plan_min_sequence() will route around it). Only
            # recordable if we actually knew the orientation this roll
            # started from (not during calibration).
            if orientation is not None:
                self._blocked.add((orientation.signature(), move))
                self._log.warn(
                    f'Roll axis={axis} {angle:+.0f} deg is infeasible at orientation '
                    f'[{orientation.describe()}]; avoiding it from now on.')
            else:
                self._log.warn(
                    f'Calibration roll axis={axis} {angle:+.0f} deg failed '
                    f'(rotation failed); trying the next candidate.')
        else:
            # 'lift failed' / 'place failed': not necessarily this roll's
            # fault -- re-check state, keep going, don't blacklist the move.
            self._log.warn(f'pick_rotate_place failed ({message}); re-checking state.')
        return 'recoverable', message

    def _calibrate(self, face: int, pose: Optional[PoseStamped] = None) -> Tuple[Optional[int], str]:
        """
        (Re-)establish the die's full tracked orientation with one physical roll.

        Tries each of ``CANDIDATE_ROLLS`` in turn (a die's layout does
        not depend on which roll calibrates it) until one executes;
        ``self._orientation`` is set from the observed
        ``(face, move, new_face)`` via ``from_two_faces()`` on success.
        Returns ``(new_face, '')`` on success or ``(None, error)`` if
        every candidate failed.

        If ``pose`` is given and its yaw is more than
        ``deyaw_tolerance_deg`` off world Z, the die is straightened to
        yaw 0 *first* -- one extra physical roll, about the synthetic
        ``'z'`` axis ``dice_manipulation_node.roll_dice()`` accepts only
        for this (see its docstring) -- before any ``CANDIDATE_ROLLS``
        attempt. This runs only here, i.e. only the first time (or the
        first time again after a desync) this node has to work out how
        the die is actually laid out before it can plan: every other
        roll in this module is a normal ``'x'``/``'y'`` face-changing
        roll and never touches yaw. A failed straightening roll is not
        fatal -- calibration still proceeds on whatever face resulted,
        only a ``'hard'`` failure (arm/perception genuinely broken)
        aborts calibration entirely.
        """
        if pose is not None:
            yaw_deg = math.degrees(_yaw_from_quat(pose.pose.orientation))
            if abs(yaw_deg) > self.deyaw_tolerance_deg:
                self._log.info(
                    f'Die yaw={yaw_deg:+.1f} deg off world Z; straightening to 0 '
                    f'before calibrating.')
                outcome, message = self._try_roll(None, ('z', -yaw_deg))
                if outcome == 'hard':
                    return None, f'de-yaw roll failed: {message}'
                new_face = self._identify_face()
                if new_face is None:
                    return None, 'post-deyaw dice identification failed'
                if outcome != 'ok':
                    self._log.warn(
                        f'De-yaw roll did not complete cleanly ({message}); '
                        f'calibrating anyway from face {new_face}.')
                face = new_face

        for move in CANDIDATE_ROLLS:
            outcome, message = self._try_roll(None, move)
            if outcome == 'hard':
                return None, f'calibration roll failed: {message}'
            new_face = self._identify_face()
            if new_face is None:
                return None, 'post-calibration-roll dice identification failed'
            if outcome == 'ok':
                self._orientation = DiceOrientation.from_two_faces(face, move, new_face)
                self._log.info(
                    f'Calibrated: face {face} --{move[0]}{move[1]:+.0f}deg--> face '
                    f'{new_face}; full layout now known: {self._orientation.describe()}')
                return new_face, ''
            face = new_face  # 'recoverable' -- try the next candidate from here
        return None, f'could not calibrate: every candidate roll failed from face {face}'

    def _describe_sequence(self, face: int, target: int, sequence) -> str:
        if not sequence:
            return f'Face {face} is already {target}.'
        steps = ', '.join(f'{axis}{angle:+.0f}deg' for axis, angle in sequence)
        return (f'Minimal sequence from face {face} to face {target}: '
                f'{len(sequence)} roll(s) [{steps}]. Call '
                f'~/execute_planned_sequence to run it.')

    # ------------------------------------------------------------------ #
    # Interactive: plan (at most 1 physical roll, only if not already    #
    # calibrated -- see module docstring), then execute on OK             #
    # ------------------------------------------------------------------ #
    def plan_target_face(self, target_face: int) -> Tuple[bool, str]:
        """
        Identify the die and report the complete minimal roll sequence to ``target_face``.

        Rolls physically at most once, and only if the tracked
        orientation is missing or no longer matches the live face (see
        the module docstring's "Calibration") -- never again afterwards
        while it keeps matching. Call ``execute_planned_sequence()``
        separately to actually run the reported sequence.
        """
        if not 1 <= target_face <= 6:
            return False, f'invalid target_face={target_face} (must be 1-6)'

        face, pose = self._identify()
        if face is None:
            return False, 'dice identification failed'

        if self._orientation is None or self._orientation.up_face != face:
            face, err = self._calibrate(face, pose)
            if face is None:
                self._pending_target = None
                return False, err

        sequence = plan_min_sequence(self._orientation, target_face, blocked=self._blocked)
        if sequence is None:
            self._pending_target = None
            return False, (f'no feasible roll sequence found from face {face} to '
                           f'{target_face} (every candidate blocked at every reachable '
                           f'orientation -- a robot/kinematics limitation, not a planning one)')

        self._pending_target = target_face
        message = self._describe_sequence(face, target_face, sequence)
        self._log.info(message)
        return True, message

    def execute_planned_sequence(self) -> Tuple[bool, str]:
        """
        Run the sequence ``plan_target_face()`` last reported, roll by roll.

        Re-derives the sequence fresh from the die's *live* orientation
        right before executing (cheap -- a BFS over at most 24 states --
        and correct regardless of anything that happened since
        ``plan_target_face()``), and re-derives the tracked orientation
        again after every single roll via ``from_two_faces()`` (exact,
        not a prediction -- see the module docstring). If the tracked
        orientation is stale when this is called (e.g. the die was moved
        since ``plan_target_face()``), this does **not** silently spend a
        calibration roll -- it fails and asks for ``~/plan_target_face``
        again, so a caller reviewing the plan before running it is never
        surprised by an extra, unplanned roll. Stops at the first roll
        that does not succeed and reports how far it got.
        """
        if self._pending_target is None:
            return False, 'no plan pending; call ~/plan_target_face first'
        target = self._pending_target

        face = self._identify_face()
        if face is None:
            return False, 'dice identification failed'
        if self._orientation is None or self._orientation.up_face != face:
            self._pending_target = None
            return False, ('tracked die orientation is stale (observed face does not '
                           'match); call ~/plan_target_face again to recalibrate')
        if face == target:
            self._pending_target = None
            return True, f'Already showing face {target}.'

        sequence = plan_min_sequence(self._orientation, target, blocked=self._blocked)
        if not sequence:
            self._pending_target = None
            return False, (f'no feasible roll sequence to reach face {target} from face '
                           f'{face}; call ~/plan_target_face again')

        total = len(sequence)
        for i, move in enumerate(sequence, start=1):
            axis, angle = move
            self._log.info(f'[{i}/{total}] rolling axis={axis} {angle:+.0f} deg')
            outcome, message = self._try_roll(self._orientation, move)
            if outcome != 'ok':
                self._orientation = None  # stale -- next call must recalibrate
                self._identify_face()  # best-effort, just to leave state current
                self._pending_target = None
                return False, (f'sequence execution stopped after {i - 1} successful '
                               f'roll(s): {message}')

            new_face = self._identify_face()
            if new_face is None:
                self._orientation = None
                self._pending_target = None
                return False, 'post-roll dice identification failed'
            self._orientation = DiceOrientation.from_two_faces(face, move, new_face)
            face = new_face

        self._pending_target = None
        if face == target:
            return True, f'Target face {target} reached in {total} roll(s).'
        return False, (f'sequence completed but face is {face}, expected '
                       f'{target} (die may have settled differently than predicted)')

    # ------------------------------------------------------------------ #
    # State machine                                                       #
    # ------------------------------------------------------------------ #
    def reach_target_face(self, target_face: int) -> Tuple[bool, str]:
        if not 1 <= target_face <= 6:
            return False, f'invalid target_face={target_face} (must be 1-6)'

        state = State.IDENTIFY
        face: Optional[int] = None
        pose: Optional[PoseStamped] = None
        move: Optional[Move] = None
        fail_reason = ''
        attempts = 0

        while True:
            if state == State.IDENTIFY:
                face, pose = self._identify()
                if face is None:
                    state = State.FAILED
                    fail_reason = 'dice identification failed'
                elif self._orientation is None or self._orientation.up_face != face:
                    state = State.CALIBRATE
                else:
                    state = State.CHECK_TARGET

            elif state == State.CALIBRATE:
                new_face, err = self._calibrate(face, pose)
                if new_face is None:
                    state = State.FAILED
                    fail_reason = err
                else:
                    attempts += 1
                    face = new_face
                    state = State.CHECK_TARGET

            elif state == State.CHECK_TARGET:
                if face == target_face:
                    state = State.DONE
                elif attempts >= self.max_attempts:
                    state = State.FAILED
                    fail_reason = f'gave up after {self.max_attempts} rolls, face is now {face}'
                else:
                    state = State.PLAN

            elif state == State.PLAN:
                sequence = plan_min_sequence(self._orientation, target_face, blocked=self._blocked)
                if not sequence:
                    state = State.FAILED
                    fail_reason = f'no feasible roll known from face {face}'
                else:
                    move = sequence[0]
                    attempts += 1
                    axis, angle = move
                    self._log.info(
                        f'[attempt {attempts}/{self.max_attempts}] face {face} -> '
                        f'target {target_face}: rolling axis={axis} {angle:+.0f} deg '
                        f'(minimal sequence has {len(sequence)} roll(s))')
                    state = State.ROLL

            elif state == State.ROLL:
                outcome, message = self._try_roll(self._orientation, move)
                if outcome == 'hard':
                    state = State.FAILED
                    fail_reason = message
                    continue
                if outcome == 'ok':
                    state = State.VERIFY
                    continue

                # 'recoverable' -- tracked orientation may no longer be
                # valid (the die could have ended up anywhere); force a
                # fresh IDENTIFY (and recalibration if needed) rather
                # than assuming it is still correct.
                self._orientation = None
                state = State.IDENTIFY

            elif state == State.VERIFY:
                new_face = self._identify_face()
                if new_face is None:
                    state = State.FAILED
                    fail_reason = 'post-roll dice identification failed'
                    continue
                self._orientation = DiceOrientation.from_two_faces(face, move, new_face)
                self._log.info(f'Now: {self._orientation.describe()}')
                face = new_face
                state = State.CHECK_TARGET

            elif state == State.DONE:
                return True, f'Target face {target_face} reached in {attempts} roll(s).'

            elif state == State.FAILED:
                self._log.error(fail_reason)
                return False, fail_reason


def _declare_parameters(node: Node) -> None:
    node.declare_parameter('target_face', 0)  # 0 = unset / invalid
    # Every roll is a 90 deg turn about a fixed WORLD axis. Once the die's
    # full orientation is known (at most 1 calibration roll, see the
    # module docstring), reaching any target from any start takes at most
    # 3 more rolls (proven by BFS over the reachable orientation graph,
    # see test_dice_face_map.py) -- so 4 total, worst case, per request.
    # Kept generous anyway: a blocked roll (see _blocked) can force a
    # longer detour, and a 'recoverable' failure forces a re-IDENTIFY
    # (and possibly a re-calibration roll) without making progress. This
    # bounds runaway retries on a persistently misbehaving cell, not
    # normal operation.
    node.declare_parameter('max_attempts', 15)
    # See _calibrate()'s de-yaw pre-step: below this (deg) the die is
    # considered already at yaw 0 and straightening is skipped.
    node.declare_parameter('deyaw_tolerance_deg', 1.0)
    node.declare_parameter('manipulation_node_name', 'dice_manipulation_node')
    node.declare_parameter('dice_identification_service', 'dice_identification')
    node.declare_parameter('run_on_start', False)


def main(args=None):
    rclpy.init(args=args)

    # Keep global arguments on: target_face/max_attempts/... must be
    # overridable via the launch --params-file, same reasoning as
    # dice_manipulation_node.
    node = rclpy.create_node('dice_task_orchestrator')
    _declare_parameters(node)

    # Never spun in a persistent loop -- see the comment in
    # DiceTaskOrchestrator.__init__ for why this must be a separate node
    # from `node` (which rclpy.spin(node) below takes ownership of).
    client_node = rclpy.create_node('dice_task_orchestrator_clients')

    orchestrator = DiceTaskOrchestrator(node, client_node)

    def _srv_cb(request, response):
        # Safety net, same reasoning as dice_manipulation_node's own
        # ~/pick_rotate_place callback: an unhandled exception here must
        # never leave this Trigger response unsent.
        target_face = node.get_parameter('target_face').value
        try:
            ok, message = orchestrator.reach_target_face(target_face)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see above
            node.get_logger().error(f'reach_target_face() raised: {exc!r}')
            ok, message = False, f'unhandled exception: {exc!r}'
        response.success = ok
        response.message = message
        return response

    def _plan_srv_cb(request, response):
        target_face = node.get_parameter('target_face').value
        try:
            ok, message = orchestrator.plan_target_face(target_face)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see above
            node.get_logger().error(f'plan_target_face() raised: {exc!r}')
            ok, message = False, f'unhandled exception: {exc!r}'
        response.success = ok
        response.message = message
        return response

    def _execute_srv_cb(request, response):
        try:
            ok, message = orchestrator.execute_planned_sequence()
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see above
            node.get_logger().error(f'execute_planned_sequence() raised: {exc!r}')
            ok, message = False, f'unhandled exception: {exc!r}'
        response.success = ok
        response.message = message
        return response

    node.create_service(Trigger, '~/reach_target_face', _srv_cb)
    node.create_service(Trigger, '~/plan_target_face', _plan_srv_cb)
    node.create_service(Trigger, '~/execute_planned_sequence', _execute_srv_cb)
    node.get_logger().info(
        'dice_task_orchestrator ready. Interactive: ~/plan_target_face then '
        '~/execute_planned_sequence. Automatic: ~/reach_target_face. '
        '(param: target_face)')

    try:
        target_face = node.get_parameter('target_face').value
        if node.get_parameter('run_on_start').value and 1 <= target_face <= 6:
            ok, message = orchestrator.reach_target_face(target_face)
            node.get_logger().info(f'run_on_start result: {ok} ({message})')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        client_node.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
