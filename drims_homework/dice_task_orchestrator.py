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
call identifies the die and, if a full path to ``target_face`` is not
known yet, does **exactly one** physical roll to learn a bit more --
never more than one per call -- then reports either the complete
sequence found or that another call is needed to explore further.
Nothing is executed beyond that single exploratory roll until you
separately call ``~/execute_planned_sequence``, which runs the last
planned sequence end to end. This is the only path that lets you watch
each roll happen and stop/inspect between them. Depending on how much of
the die is already known, reaching a genuinely new target can still take
several ``~/plan_target_face`` calls, each contributing one roll of
information (see ``dice_face_map``'s module docstring for exactly why
this cannot be shortcut to "always one roll" on this robot cell).

**Fully automatic**: ``~/reach_target_face`` runs the whole thing
unattended as an explicit state machine (``State`` below):

.. code-block:: text

    IDENTIFY ──► CHECK_TARGET ──(match)──► DONE
                      │
                (no match, attempts left)
                      ▼
                    PLAN ──(no move known)──► FAILED
                      │
                      ▼
                    ROLL ──(hard failure)──► FAILED
                      │        │
                      │   (recoverable failure: re-identify, loop back to
                      │    CHECK_TARGET without a new roll being recorded)
                      ▼
                   VERIFY ──► back to CHECK_TARGET

* **IDENTIFY** asks the perception layer (``/dice_identification``,
  ``dice_common.py``) which face is currently up.
* **CHECK_TARGET** compares the current face to ``target_face``; also
  where the ``max_attempts`` budget is enforced.
* **PLAN** asks ``dice_face_map.DiceFaceMap`` (a pure-Python model of
  *this* die — see that module) for the next roll to try, purely from
  what has actually been observed so far (see that module's docstring
  for why this is deliberately *not* solved from live orientation).
* **ROLL** configures ``dice_manipulation_node`` for that roll (its
  ``set_parameters`` service) and drives it through one full
  pick-lift-roll-place cycle (its ``~/pick_rotate_place`` service). A
  roll that fails because the tool could not be *aligned* to the axis
  (``'grip alignment failed'``) is treated exactly like a roll that
  fails during the turn itself (``'rotation failed'``): both are a
  motion-layer statement that this exact (face, axis) is unreachable,
  so both get permanently excluded via ``mark_infeasible()`` -- see
  ``_try_roll()``.
* **VERIFY** re-identifies the die and feeds the (roll, resulting face)
  observation back into the map.

Both paths share the same underlying ``DiceFaceMap`` instance and the
same roll-execution helper (``_try_roll()``), so exploration done via
one shows up in the other.

This node only ever talks to two contracts:

    * ``/dice_identification`` (perception — the simulator today, a real
      vision node tomorrow, see ``dice_common.py``);
    * ``dice_manipulation_node``'s ``set_parameters`` and
      ``~/pick_rotate_place`` services.

It never talks to MoveIt / the gripper / TF directly, and it never
assumes the simulator's internal geometry — everything about how a roll
maps faces to faces is learned online by ``dice_face_map`` from what
actually happens.

Interfaces
----------
* Parameter ``target_face`` (1-6): the face the die should end up showing.
* Service ``~/plan_target_face`` (``std_srvs/srv/Trigger``): identify,
  explore at most one roll if needed, report the minimal sequence found
  (or that more exploration is needed) without executing it.
* Service ``~/execute_planned_sequence`` (``std_srvs/srv/Trigger``): run
  the sequence ``~/plan_target_face`` last reported, roll by roll.
* Service ``~/reach_target_face`` (``std_srvs/srv/Trigger``): (re-)runs
  the fully-automatic state machine for the currently configured
  ``target_face``, with no pause for confirmation.
* Parameter ``run_on_start`` (bool): also run ``reach_target_face`` once
  at start-up.

To change the target at runtime without restarting::

    ros2 param set /dice_task_orchestrator target_face 6
    ros2 service call /dice_task_orchestrator/plan_target_face std_srvs/srv/Trigger "{}"
    # inspect the logged sequence (repeat the call above if it reports
    # more exploration is needed), then:
    ros2 service call /dice_task_orchestrator/execute_planned_sequence std_srvs/srv/Trigger "{}"
"""

from enum import Enum, auto
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from std_srvs.srv import Trigger
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter

from drims_homework.dice_common import create_dice_identification_client, identify_dice
from drims_homework.dice_face_map import DiceFaceMap, Move

# pick_rotate_place() failure messages that mean the dice was never
# touched, or the roll layer itself already gave up recovering -- not
# specific to whichever roll was being tried, so retrying a different
# roll cannot help. Anything else ('rotation failed', 'lift failed',
# 'place failed') is treated as recoverable: re-check the real state via
# /dice_identification and keep going.
_UNRECOVERABLE_FAILURES = frozenset({
    'failed to reach home', 'pick failed', 'dice identification failed',
})


class State(Enum):
    IDENTIFY = auto()
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

        self._dice_identification_client = create_dice_identification_client(
            self._client_node, gp('dice_identification_service').value)
        self._pick_rotate_place_client = self._client_node.create_client(
            Trigger, f'/{self.manipulation_node_name}/pick_rotate_place')
        self._set_parameters_client = self._client_node.create_client(
            SetParameters, f'/{self.manipulation_node_name}/set_parameters')

        # The die's roll->face transition graph, learned online. Kept for
        # this node's whole lifetime: a physical property of the die, not
        # of a single reach_target_face()/plan_target_face() request.
        self._map = DiceFaceMap()

        # Last target plan_target_face() found a full path for, consumed
        # by execute_planned_sequence(). Not persisted across a node
        # restart -- deliberately: re-planning is cheap and re-deriving
        # it fresh at execute time from the die's *live* face is safer
        # than trusting a stale plan if something moved the die meanwhile.
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

    def _identify(self) -> Optional[int]:
        """Identify the die; return the face number, or None on failure."""
        face, pose = identify_dice(self._client_node, self._dice_identification_client)
        if face is None or pose is None:
            return None
        return face

    # ------------------------------------------------------------------ #
    # Shared roll execution (used by both the automatic state machine    #
    # and the interactive plan/execute pair below)                       #
    # ------------------------------------------------------------------ #
    def _try_roll(self, face: int, move: Move) -> Tuple[str, str]:
        """
        Configure and execute one physical roll.

        Returns ``(outcome, message)`` where ``outcome`` is one of:

        * ``'ok'`` -- the roll executed; the caller should re-identify
          and ``self._map.record(face, move, new_face)``.
        * ``'recoverable'`` -- it failed but ``dice_manipulation_node``
          already put the dice back down best-effort; ``(face, move)``
          has been marked infeasible if the failure was kinematic
          (``'rotation failed'`` or ``'grip alignment failed'`` -- both
          mean this exact axis is unreachable from this exact face, see
          the module docstring); the caller should re-identify to get
          the real (possibly unchanged) state and keep going.
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

        if message in ('rotation failed', 'grip alignment failed'):
            # Both mean the motion layer could not reach this exact axis
            # from this exact face -- never try it again from here.
            self._map.mark_infeasible(face, move)
            self._log.warn(
                f'Roll axis={axis} {angle:+.0f} deg from face {face} is '
                f'infeasible for this robot configuration ({message}); '
                f'avoiding it from now on.')
        else:
            # 'lift failed' / 'place failed': not necessarily this roll's
            # fault -- re-check state, keep going, don't blacklist the move.
            self._log.warn(f'pick_rotate_place failed ({message}); re-checking state.')
        return 'recoverable', message

    def _describe_sequence(self, face: int, target: int, sequence: List[Move]) -> str:
        if not sequence:
            return f'Face {face} is already {target}.'
        steps = ', '.join(f'{axis}{angle:+.0f}deg' for axis, angle in sequence)
        return (f'Minimal known sequence from face {face} to face {target}: '
                f'{len(sequence)} roll(s) [{steps}]. Call '
                f'~/execute_planned_sequence to run it.')

    def _report_if_known(self, face: int, target_face: int) -> Optional[Tuple[bool, str]]:
        """
        If ``target_face`` is already reachable from ``face`` via the known table, report it.

        Returns None (nothing to report yet) if there is no fully-known
        path -- the caller should keep exploring in that case.
        """
        if face == target_face:
            self._pending_target = None
            return True, f'Already showing face {target_face}; nothing to do.'
        path = self._map.known_path(face, target_face)
        if path is None:
            return None
        self._pending_target = target_face
        message = self._describe_sequence(face, target_face, path)
        self._log.info(message)
        return True, message

    # ------------------------------------------------------------------ #
    # Interactive: plan (>= 0, <= 1 physical roll), then execute on OK    #
    # ------------------------------------------------------------------ #
    def plan_target_face(self, target_face: int) -> Tuple[bool, str]:
        """
        Identify the die, then report the minimal known roll sequence towards ``target_face``.

        Does **at most one** physical roll -- only if a full path to
        ``target_face`` is not known yet -- purely to learn one more
        (face, move) transition, never to actually make progress towards
        the target by trial and error. Never executes the sequence
        itself: call ``execute_planned_sequence()`` separately once
        you've looked at the result.
        """
        if not 1 <= target_face <= 6:
            return False, f'invalid target_face={target_face} (must be 1-6)'

        face = self._identify()
        if face is None:
            return False, 'dice identification failed'

        result = self._report_if_known(face, target_face)
        if result is not None:
            return result

        move = self._map.plan_next(face, target_face)
        if move is None:
            return False, f'no feasible roll known from face {face}'
        axis, angle = move
        self._log.info(
            f'Sequence to face {target_face} not known yet from face {face}; '
            f'exploratory roll: axis={axis} {angle:+.0f} deg.')

        outcome, message = self._try_roll(face, move)
        if outcome == 'hard':
            return False, message
        if outcome == 'ok':
            new_face = self._identify()
            if new_face is None:
                return False, 'post-roll dice identification failed'
            self._map.record(face, move, new_face)
            face = new_face
        else:
            recovered = self._identify()
            if recovered is None:
                return False, (f'pick_rotate_place failed ({message}) and could '
                               f'not re-identify the dice afterwards')
            self._pending_target = None
            return True, (
                f'The exploratory roll failed ({message}), so nothing new was '
                f'learned (still showing face {recovered}); this is a motion/robot '
                f'issue, not a planning one -- call ~/plan_target_face again to '
                f'retry it.')

        result = self._report_if_known(face, target_face)
        if result is not None:
            return result

        self._pending_target = None
        self._log.info(self._map.describe())
        return True, (
            f'Explored one roll (now showing face {face}); sequence to face '
            f'{target_face} not fully known yet -- call ~/plan_target_face '
            f'again to explore further.')

    def execute_planned_sequence(self) -> Tuple[bool, str]:
        """
        Run the sequence ``plan_target_face()`` last reported, roll by roll.

        Re-derives the sequence fresh from the die's *live* face right
        before executing (cheap, and correct regardless -- see
        ``dice_face_map``) rather than trusting a possibly-stale stored
        plan; if nothing changed since ``plan_target_face()`` this is the
        exact same sequence. Stops at the first roll that does not
        succeed and reports how far it got.
        """
        if self._pending_target is None:
            return False, 'no plan pending; call ~/plan_target_face first'
        target = self._pending_target

        face = self._identify()
        if face is None:
            return False, 'dice identification failed'
        if face == target:
            self._pending_target = None
            return True, f'Already showing face {target}.'

        sequence = self._map.known_path(face, target)
        if not sequence:
            self._pending_target = None
            return False, (f'no known sequence to reach face {target} from face '
                           f'{face} anymore; call ~/plan_target_face again')

        for i, move in enumerate(sequence, start=1):
            axis, angle = move
            self._log.info(f'[{i}/{len(sequence)}] rolling axis={axis} {angle:+.0f} deg')
            outcome, message = self._try_roll(face, move)
            if outcome != 'ok':
                self._identify()  # best-effort, just to leave the map up to date
                self._pending_target = None
                return False, (f'sequence execution stopped after {i - 1} successful '
                               f'roll(s): {message}')

            new_face = self._identify()
            if new_face is None:
                self._pending_target = None
                return False, 'post-roll dice identification failed'
            self._map.record(face, move, new_face)
            face = new_face

        self._pending_target = None
        if face == target:
            return True, f'Target face {target} reached in {len(sequence)} roll(s).'
        return False, (f'sequence completed but face is {face}, expected {target} '
                       f'(die may have settled differently than predicted)')

    # ------------------------------------------------------------------ #
    # State machine                                                       #
    # ------------------------------------------------------------------ #
    def reach_target_face(self, target_face: int) -> Tuple[bool, str]:
        if not 1 <= target_face <= 6:
            return False, f'invalid target_face={target_face} (must be 1-6)'

        state = State.IDENTIFY
        face: Optional[int] = None
        move: Optional[Move] = None
        fail_reason = ''
        attempts = 0

        while True:
            if state == State.IDENTIFY:
                face = self._identify()
                state = State.FAILED if face is None else State.CHECK_TARGET
                fail_reason = 'initial dice identification failed'

            elif state == State.CHECK_TARGET:
                if face == target_face:
                    state = State.DONE
                elif attempts >= self.max_attempts:
                    state = State.FAILED
                    fail_reason = f'gave up after {self.max_attempts} rolls, face is now {face}'
                else:
                    state = State.PLAN

            elif state == State.PLAN:
                move = self._map.plan_next(face, target_face)
                if move is None:
                    state = State.FAILED
                    fail_reason = f'no feasible roll known from face {face}'
                else:
                    attempts += 1
                    axis, angle = move
                    self._log.info(
                        f'[attempt {attempts}/{self.max_attempts}] face {face} -> '
                        f'target {target_face}: rolling axis={axis} {angle:+.0f} deg')
                    state = State.ROLL

            elif state == State.ROLL:
                outcome, message = self._try_roll(face, move)
                if outcome == 'hard':
                    state = State.FAILED
                    fail_reason = message
                    continue
                if outcome == 'ok':
                    state = State.VERIFY
                    continue

                # 'recoverable'
                recovered = self._identify()
                if recovered is None:
                    state = State.FAILED
                    fail_reason = (f'pick_rotate_place failed ({message}) and could not '
                                   f're-identify the dice afterwards')
                    continue
                face = recovered
                state = State.CHECK_TARGET  # re-plan from the (possibly unchanged) real face

            elif state == State.VERIFY:
                new_face = self._identify()
                if new_face is None:
                    state = State.FAILED
                    fail_reason = 'post-roll dice identification failed'
                    continue
                self._map.record(face, move, new_face)
                self._log.info(self._map.describe())
                face = new_face
                state = State.CHECK_TARGET

            elif state == State.DONE:
                return True, f'Target face {target_face} reached in {attempts} roll(s).'

            elif state == State.FAILED:
                self._log.error(fail_reason)
                self._log.info(self._map.describe())
                return False, fail_reason


def _declare_parameters(node: Node) -> None:
    node.declare_parameter('target_face', 0)  # 0 = unset / invalid
    # Every roll is a 90 deg turn about a fixed WORLD axis, learned online
    # by dice_face_map (with its opposite-pair shortcut cutting this down
    # a lot in practice). Only 2 candidate rolls are used on this cell
    # (see dice_face_map.CANDIDATE_ROLLS), so exploration is less
    # efficient than with more/body-relative candidates: verified offline
    # with a full physical-cube model, 300 random targets, 0 failures,
    # average ~5 rolls/target once warmed up, worst case (a first-ever
    # request, cold) 31 rolls. 50 leaves real margin above that; lower it
    # once re-verified against the real robot/simulator.
    node.declare_parameter('max_attempts', 50)
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
