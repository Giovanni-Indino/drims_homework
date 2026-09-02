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
The "dice map": what this particular die's six faces do under a roll.

Pure Python, no ROS — this is a model of the die, not of the robot. It
knows nothing about MoveIt, TF, or services; it only answers, given
observations fed to it by whoever is actually driving the arm
(``dice_task_orchestrator``):

    * ``record(face_before, move, face_after)`` / ``mark_infeasible()`` —
      "this roll, tried from this face, produced that face" / "...could
      not even be planned from this face" (fed after every executed roll);
    * ``plan_next(current, target)`` — "which roll should I try next to
      get from ``current`` towards ``target``?"

Why this is purely empirical, not solved algebraically from orientation
------------------------------------------------------------------------
An earlier version of this module tried to shortcut all of this: since a
roll is a known, fixed rotation about a fixed *world* axis, and the die's
live orientation is (in principle) available, composing rotations is
exact quaternion algebra -- no physical trial needed. That reasoning is
correct in the abstract, but does not hold for *this* robot cell, and the
mistake is worth recording so it does not get re-introduced.

``dice_manipulation_node.align_grip_axis()`` yaws the gripper (and,
being rigidly attached, the die with it) by an amount that depends on
*which face is currently grasped* -- it has to, to compensate for that
face's own arbitrary grasp geometry and still leave the jaws exactly
horizontal at release (see that module's docstring). That per-face yaw
is real, physically necessary, and not predictable in closed form
without already knowing the perception layer's internal face-to-frame
convention. The practical consequence: rolling the same ``move`` from
two *different* faces applies two genuinely different net rotations to
the die -- there is no single world-fixed delta that "a roll" means
independent of which face it starts from, so an orientation composed
purely from ``roll_axis``/``roll_angle_deg`` does not predict the real
outcome. Verified by simulation against this project's own
``drims_dice_simulator`` geometry: composing rolls this way and deriving
the rest of the die's layout from just one or two observations produced
the *wrong* face 20-40% of the time -- confidently. That is a worse
failure mode than the slower, honest alternative below, which is never
wrong, only sometimes still exploring.

What *is* still fully valid, and used here
-------------------------------------------
1. **The empirical (face, move) -> face table** (``record()``,
   ``_bfs_next_move()``). A caveat: ``align_grip_axis()`` picks whichever
   of two 180-deg-apart yaws is smaller (see its own docstring), so the
   die's landing yaw at pick time can put a given ``(face, move)`` pair
   on either of (up to) two outcomes -- forcing that choice to be fully
   deterministic was tried and made things *worse*, not better (see
   ``dice_manipulation_node``'s module docstring): two of the six faces
   turned out to become structurally unreachable as a roll *result* no
   matter which single deterministic target was picked, a property of
   this die's own geometry combined with only two candidate world axes
   being kinematically usable at all, not of the alignment strategy.
   Keeping the two-outcome variability is what lets every face that
   *is* reachable at all stay reachable -- so an already-tried
   ``(face, move)`` is retried (bounded, see ``_MAX_RETRIES_PER_MOVE``)
   rather than assumed to always repeat.
2. **The opposite-pair fact**: a standard die has its numbers in three
   *opposite* pairs summing to 7 (1-6, 2-5, 3-4) -- a property of the
   die's *numbering*, unrelated to orientation tracking, so none of the
   above problem touches it. ``assume_standard_die=True`` (the default)
   simply assumes this outright; set it to ``False`` to instead infer it
   empirically via ``_infer_opposite()`` (two applications of the exact
   same roll add up to a single 180 deg turn about one fixed horizontal
   axis, which always swaps top and bottom, independent of the die's yaw
   at the time -- true regardless of numbering, just slower to learn).

A genuine reachability limit, not a bug
-----------------------------------------
With only two candidate world axes available (see ``CANDIDATE_ROLLS``'s
own comment for why), simulation against this project's die geometry
found that two of the six faces are *never* produced as the result of
either candidate roll, from any face, at any landing yaw -- only reached
if the die happens to already be showing one of them (e.g. right after
spawning). ``plan_next()``/``known_path()`` do not special-case this:
they simply never find a path to such a target and correctly report
"nothing known" (``None``) once exploration is exhausted, rather than
hang or guess. If a target genuinely never becomes reachable in
practice, that points at a robot/kinematics limitation (this candidate
set covering only 2 of the 4 geometrically-possible roll directions) --
worth revisiting on the real cell, not something this module can plan
its way around.

A roll can also be *kinematically infeasible* from a given configuration
(a planning failure at the motion layer, distinct from "produced the
wrong face") — ``mark_infeasible()`` records that so ``plan_next()``
never proposes that exact (face, move) again.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

Move = Tuple[str, float]  # (world axis 'x' | 'y', angle in degrees)

# Sentinel recorded for a roll that failed at the motion layer (planning
# failure) rather than producing a face. Distinct from any real face
# number (1-6), so it is never mistaken for one.
INFEASIBLE = 'INFEASIBLE'

# Rolls dice_manipulation_node knows how to execute, each a quarter turn
# about a FIXED WORLD axis (see its module docstring for why world-fixed
# rather than tied to the die's own body: the wrist sweep a given roll
# requires is then always the same physical motion, so an over-rotating
# direction can be excluded here once instead of failing unpredictably
# per face). Only 2 of the 4 geometrically-possible combinations: live
# testing on this cell's arm/gripper found the other two over-rotate the
# wrist towards a near-singular configuration. Tune per cell/robot.
CANDIDATE_ROLLS: List[Move] = [('x', 90.0), ('y', -90.0)]

# How many times an already-tried (face, move) pair may be retried by
# plan_next()'s last-resort tier, hoping a different landing yaw reveals
# the *other* of its (at most two, see the module docstring) possible
# outcomes. Small on purpose: unlike genuine exploration this can only
# ever reveal one more alternative per pair, so a handful of attempts is
# enough to sample it if it exists, and further retries would only ever
# spin without new information -- plan_next()/known_path() must still be
# able to report "not reachable" in finite time (see "A genuine
# reachability limit, not a bug" above).
_MAX_RETRIES_PER_MOVE = 3


class DiceFaceMap:
    """Learns and queries this die's roll -> face transition graph."""

    def __init__(self, candidate_rolls: Optional[List[Move]] = None,
                 assume_standard_die: bool = True):
        self._candidates = list(candidate_rolls or CANDIDATE_ROLLS)
        # learned[face][move] = resulting face, or INFEASIBLE.
        self._learned: Dict[int, Dict[Move, object]] = {f: {} for f in range(1, 7)}
        # See the module docstring's "What is still fully valid" (1):
        # assumed outright by default (a fact about standard numbering,
        # nothing to do with orientation), or left to be inferred
        # empirically -- see _infer_opposite().
        self._opposite: Dict[int, int] = (
            {f: 7 - f for f in range(1, 7)} if assume_standard_die else {}
        )
        self._opposite_move: Dict[int, Move] = {}
        # How many times each (face, move) pair has been retried after
        # already being known -- see plan_next()'s tier 5 and
        # _MAX_RETRIES_PER_MOVE.
        self._retry_count: Dict[Tuple[int, Move], int] = {}

    # ------------------------------------------------------------------ #
    # Feeding observations in                                             #
    # ------------------------------------------------------------------ #
    def record(self, face_before: int, move: Move, face_after: int) -> None:
        """
        Remember that ``move``, tried from ``face_before``, produced ``face_after``.

        Also updates the opposite-pair inference. Not assumed
        deterministic (see the module docstring): recording the same
        ``(face_before, move)`` again with a *different* result is
        expected and simply overwrites the previous one -- counted
        against ``_MAX_RETRIES_PER_MOVE`` either way, so retrying does
        not go on forever.
        """
        if move in self._learned[face_before]:
            key = (face_before, move)
            self._retry_count[key] = self._retry_count.get(key, 0) + 1
        self._learned[face_before][move] = face_after
        self._infer_opposite(face_before, move, face_after)

    def mark_infeasible(self, face: int, move: Move) -> None:
        """
        Remember that ``move`` cannot be planned at all from ``face``.

        A motion-layer planning failure, not a wrong-face result.
        """
        self._learned[face][move] = INFEASIBLE

    # ------------------------------------------------------------------ #
    # Planning                                                            #
    # ------------------------------------------------------------------ #
    def plan_next(self, current: int, target: int) -> Optional[Move]:
        """
        Pick the next roll to try to get from ``current`` towards ``target``.

        Cheapest/most-informed option first:

        1. the first move of a shortest already-known ``current`` ->
           ``target`` path (BFS over learned, non-infeasible transitions);
        2. the move that realises a known opposite pair, if ``target`` is
           already known to be ``current``'s opposite — valid even if
           never tried from this exact face, by the geometry above;
        3. an untried candidate from ``current`` (keep exploring);
        4. the first move of a shortest path towards whichever reachable
           face still has something untried (keep exploration itself
           converging);
        5. last resort, retry a candidate already tried from ``current``
           (excluding known-INFEASIBLE ones, and capped at
           ``_MAX_RETRIES_PER_MOVE`` — see the module docstring for why
           this is necessary at all, not just belt-and-suspenders).

        None only if every strategy above is exhausted.
        """
        move = self._bfs_next_move(current, target)
        if move is not None:
            return move

        if self._opposite.get(current) == target:
            move = self._opposite_move.get(current)
            if move is not None:
                return move

        untried = [m for m in self._candidates if m not in self._learned[current]]
        if untried:
            return untried[0]

        move = self._bfs_to_unexplored(current)
        if move is not None:
            return move

        retry = [
            m for m in self._candidates
            if self._learned[current].get(m) != INFEASIBLE
            and self._retry_count.get((current, m), 0) < _MAX_RETRIES_PER_MOVE
        ]
        if retry:
            return retry[0]

        return None

    def known_path(self, current: int, target: int) -> Optional[List[Move]]:
        """
        Full shortest already-known ``current`` -> ``target`` path, or None.

        Unlike ``plan_next()`` this returns every step, not just the
        first -- for reporting/confirming a complete plan before running
        it (see ``dice_task_orchestrator.plan_target_face()``).
        """
        if current == target:
            return []
        visited = {current}
        queue = deque([(current, [])])
        while queue:
            face, path = queue.popleft()
            for move, result in self._learned_edges(face):
                if result in visited:
                    continue
                new_path = path + [move]
                if result == target:
                    return new_path
                visited.add(result)
                queue.append((result, new_path))
        return None

    def _learned_edges(self, face: int) -> List[Tuple[Move, int]]:
        return [(mv, res) for mv, res in self._learned[face].items() if res != INFEASIBLE]

    def _bfs_next_move(self, current: int, target: int) -> Optional[Move]:
        """
        First move of a shortest ``current`` -> ``target`` path.

        Uses only already-learned transitions. None if not yet reachable.
        """
        path = self.known_path(current, target)
        return path[0] if path else None

    def _bfs_to_unexplored(self, current: int) -> Optional[Move]:
        """
        First move of a shortest path to the nearest unexplored face.

        Uses learned transitions only; "unexplored" means it still has
        an untried candidate roll.
        """
        visited = {current}
        queue = deque([(current, None)])
        while queue:
            face, first_move = queue.popleft()
            for move, result in self._learned_edges(face):
                if result in visited:
                    continue
                step = first_move if first_move is not None else move
                if any(m not in self._learned[result] for m in self._candidates):
                    return step
                visited.add(result)
                queue.append((result, step))
        return None

    # ------------------------------------------------------------------ #
    # Opposite-pair inference                                             #
    # ------------------------------------------------------------------ #
    def _infer_opposite(self, face_before: int, move: Move, face_after: int) -> None:
        second = self._learned.get(face_after, {}).get(move)
        if second is not None and second != INFEASIBLE and second != face_before:
            self._set_opposite(face_before, second, move)

        # Symmetric case: this roll might complete a pair for some face
        # that was rolled *into* face_before earlier by the same move.
        for other_face, moves in self._learned.items():
            if moves.get(move) == face_before and other_face != face_after:
                self._set_opposite(other_face, face_after, move)

    def _set_opposite(self, a: int, b: int, move: Move) -> None:
        already_known = self._opposite.get(a) == b and self._opposite.get(b) == a
        if already_known:
            return
        if a in self._opposite and self._opposite[a] != b:
            return  # inconsistent observation; keep the first pairing
        self._opposite[a] = b
        self._opposite[b] = a
        self._opposite_move.setdefault(a, move)
        self._opposite_move.setdefault(b, move)
        self._complete_by_elimination()

    def _complete_by_elimination(self) -> None:
        """
        Derive the third opposite pair by elimination, if forced.

        If exactly two of the three opposite pairs are known, the third
        is forced: only two faces are left unassigned out of six.
        """
        paired = set(self._opposite)
        if len(paired) == 4:
            remaining = sorted(f for f in range(1, 7) if f not in paired)
            if len(remaining) == 2:
                a, b = remaining
                self._opposite[a] = b
                self._opposite[b] = a

    # ------------------------------------------------------------------ #
    # Reporting                                                           #
    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        """
        Human-readable snapshot of everything ``plan_next()`` reasons over.

        One line per face (``->N`` = leads to face N, ``X`` = known
        infeasible, ``?`` = untried), plus any opposite pairs found so far.
        """
        lines = ['Dice map (what plan_next() currently knows):']
        for face in range(1, 7):
            cells = []
            for axis, angle in self._candidates:
                label = f'{axis}{angle:+.0f}'
                result = self._learned[face].get((axis, angle))
                if result is None:
                    cells.append(f'{label}=?')
                elif result == INFEASIBLE:
                    cells.append(f'{label}=X')
                else:
                    cells.append(f'{label}->{result}')
            lines.append(f'  face {face}: ' + '  '.join(cells))
        lines.append(f'  opposite pairs: {self._format_opposite_pairs() or "none yet"}')
        return '\n'.join(lines)

    def _format_opposite_pairs(self) -> str:
        seen = set()
        pairs = []
        for a, b in sorted(self._opposite.items()):
            if a not in seen:
                pairs.append(f'{a}<->{b}')
                seen.add(a)
                seen.add(b)
        return ', '.join(pairs)
