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
The "dice map": an exact, closed-form model of this die's orientation.

Pure Python, no ROS -- this is a model of the die's geometry, not of the
robot. It knows nothing about MoveIt or services; it only answers, given
what the perception layer can actually, reliably report:

    * ``DiceOrientation.from_two_faces(face_before, move, face_after)`` --
      "this is exactly where every face of the die is right now", built
      from nothing more than two face *numbers* and one *commanded* roll
      -- see "Why two faces, not one" below for why this is the one to
      use against real perception;
    * ``plan_min_sequence(orientation, target_face)`` -- "the *provably
      shortest* sequence of rolls that brings ``target_face`` up".

Why two faces, not one -- what a single reading can and cannot tell you
--------------------------------------------------------------------------
It is tempting to think one ``/dice_identification`` reading is enough:
if it reported the die's true full orientation, composing known rolls on
top of it would be exact quaternion algebra (``from_identification()``
below does exactly this, and is exact -- verified numerically -- *when*
its input is trustworthy). The catch is that assumption: a real
top-down read of "which face is up" cannot, even in principle, also
recover the die's yaw on the table. The top face is a square, which
looks identical every 90 deg; the pip patterns for 1, 4 and 5 are
*themselves* rotationally symmetric (a single centred pip; four corner
pips; four corners plus a centre) so even reading pips more precisely
never breaks that tie for those three faces. No amount of vision quality
fixes this -- it is a symmetry of the object being looked at, not a
sensor limitation. So a real perception layer can, at best, reliably
report *which number* is up (``face_number``) -- never a trustworthy
in-plane rotation. (``drims_dice_simulator`` happens to also hand back
its own internal ground-truth orientation quaternion today, which is why
``from_identification()`` exists and is exact against it -- but relying
on that is relying on the simulator cheating on the perception layer's
behalf, not on anything a real vision node could promise -- see
``dice_common.py``'s "simulator today, real camera tomorrow" contract.)

What *is* always available, exactly, without any vision precision at
all: the face number *before* a roll, the face number *after* it, and
the roll itself -- because we are the ones who commanded it (a known
world axis and angle, ``CANDIDATE_ROLLS`` below), not measuring it. Two
face normals related by an exact right angle (any two distinct,
non-opposite faces of a cube) together with the two world directions
they are known to occupy are enough to fix a rigid body's entire
remaining orientation -- there is no rotational freedom left once two
non-parallel body directions are pinned to two known world directions.
``from_two_faces()`` builds exactly that, in closed integer arithmetic
(cross/dot products only, see the module's "How the reconstruction
works" in ``from_two_faces()``'s own docstring), and verified numerically
against 3000 randomized starting orientations (including ones already
yawed away from axis-aligned): exact, 0 mismatches, regardless of the
die's yaw at the time -- because it never uses that yaw at all.

Why this can be solved algebraically at all (it could not before)
-----------------------------------------------------------------
An earlier version of this module avoided closed-form reasoning
entirely: ``dice_manipulation_node`` used to yaw the gripper -- and,
being rigidly attached, the die with it -- about world Z before every
roll, by an amount that depended on which face was grasped (see git
history for that version's reasoning). That made the die's yaw drift
unpredictably even between two rolls of the *same* commanded axis, so no
fixed model could track it at all, closed-form or otherwise.
``dice_manipulation_node`` no longer does that (see its module
docstring): every roll is a pure quarter turn about a fixed WORLD axis
(``CANDIDATE_ROLLS`` below -- only ``x +90`` or ``y -90``, the two this
cell's arm/gripper can reliably reach, see that module for why the other
two are excluded) with **no rotation about world Z at any point** in the
pick/roll/place sequence -- exactly the invariant ``from_two_faces()``
relies on (the roll is known and it is the *only* rotation applied). Once
built, an orientation composes forward through further known rolls by
plain integer arithmetic (``DiceOrientation.rolled()``), no re-reading of
yaw needed -- though re-deriving it fresh via ``from_two_faces()`` after
every single executed roll (as ``dice_task_orchestrator`` does) costs
nothing extra and stays correct even if something unexpected happened.

``body_normals`` (default ``STANDARD_BODY_NORMALS``) is this die's own
numbering -- which body axis each face number sits on. A standard die has
opposite faces summing to 7 (1-6, 2-5, 3-4); the default further assumes
a specific right-handed chirality matching this project's own simulator.
Pass a different mapping if a real die/vision node ever disagrees --
notably, unlike the die's *yaw*, its fixed numbering/chirality can be
established once in advance (read off the physical die by hand, or
inferred with a handful of ``from_two_faces()`` calls against known
rolls) rather than needing to be re-derived on every use.

How many rolls, really
------------------------
With only the two candidate rolls above, they still generate the *full*
24-element cube rotation group (verified computationally, not assumed --
every one of the 24 reachable orientations is reachable from every
other), so every target face is reachable from every starting
orientation -- unlike the old empirical version, nothing is structurally
unreachable. The worst case, any start to any target, is 3 rolls; most
pairs take 1-2. ``plan_min_sequence()`` finds this exactly via BFS.

A roll can still be *kinematically infeasible* at a given orientation (a
planning failure at the motion layer, distinct from "wrong face") --
``plan_min_sequence()`` accepts a ``blocked`` set (orientation signature,
move) so ``dice_task_orchestrator`` can steer around a roll already found
to fail there, and replans (the group is generated by both rolls, so a
single blocked edge essentially never removes every path -- BFS simply
finds the next-shortest one that avoids it).
"""

import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

Move = Tuple[str, float]  # (world axis 'x' | 'y', angle in degrees)
Vec3 = Tuple[int, int, int]
Quat = Tuple[float, float, float, float]  # (x, y, z, w)

# Rolls dice_manipulation_node knows how to execute, each a quarter turn
# about a FIXED WORLD axis, with the die never yawed about world Z at any
# point in the sequence (see dice_manipulation_node's module docstring).
# Only 2 of the 4 geometrically-possible combinations: live testing on
# this cell's arm/gripper found the other two over-rotate the wrist
# towards a near-singular configuration. Tune per cell/robot -- see this
# module's docstring for why both are still enough to reach every face
# from every orientation.
#
# Only the *sign* of the angle is ever read here (apply_roll()): the
# planning geometry always treats a roll as an exact 90 deg quarter turn.
# The magnitude is what dice_manipulation_node actually commands the wrist
# to sweep -- deliberately 70, not 90: the die is released ~20 deg before
# the full quarter turn and gravity finishes tipping it onto the new
# face, which keeps the wrist away from the configuration where a full
# 90 deg sweep was failing. If the die does not reliably complete the
# tip on your cell, put these back to +-90.
CANDIDATE_ROLLS: List[Move] = [('x', 70.0), ('y', -70.0)]

# This die's numbering: which body axis each face sits on. Opposite faces
# sum to 7 (a standard die), and the specific signs/axes below match
# drims_dice_simulator's own ``DiceSpawner.face_normals`` exactly (see the
# module docstring's "Why two faces, not one" for why this has to be
# known, not derived, and is the one piece of this module that is
# specific to *this* die rather than pure geometry -- unlike the die's
# yaw, it can be fixed once in advance). Pass a different mapping to
# ``DiceOrientation.from_two_faces()``/``from_identification()`` if a
# future die/vision node uses a different numbering.
STANDARD_BODY_NORMALS: Dict[int, Vec3] = {
    1: (0, 0, -1),
    2: (-1, 0, 0),
    3: (0, 1, 0),
    4: (0, -1, 0),
    5: (1, 0, 0),
    6: (0, 0, 1),
}

# Sentinel for a roll recorded as failing at the motion layer (planning
# failure) at a given orientation -- distinct from any real outcome, kept
# only for readability at call sites (see mark_infeasible-style use in
# dice_task_orchestrator, which builds the ``blocked`` set passed to
# plan_min_sequence()).
INFEASIBLE = 'INFEASIBLE'


# ---------------------------------------------------------------------- #
# Small quaternion/vector helpers (dependency-free on purpose -- this     #
# module stays pure Python, no tf_transformations/numpy, so it is        #
# trivially unit-testable without ROS installed).                        #
# ---------------------------------------------------------------------- #
def _quat_mul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_inverse(q: Quat) -> Quat:
    """Inverse of a *unit* quaternion (conjugate)."""
    x, y, z, w = q
    return (-x, -y, -z, w)


def _rotate_vector(v: Vec3, q: Quat) -> Tuple[float, float, float]:
    v_q = (float(v[0]), float(v[1]), float(v[2]), 0.0)
    q_conj = _quat_inverse(q)
    x, y, z, _ = _quat_mul(_quat_mul(q, v_q), q_conj)
    return (x, y, z)


def _quat_from_z_to(target: Vec3) -> Quat:
    """
    Shortest-arc rotation taking +Z to ``target`` (a unit axis direction).

    Same convention documented for every ``faceN_tf`` in
    ``docs/ARCHITECTURE.md`` ("+Z out of the face") -- see this module's
    docstring for how it is used to recover the die's plain rigid-body
    orientation from one ``/dice_identification`` reading. Tie-broken
    exactly like the perception layer's own convention when
    ``target == -Z`` (the cross product vanishes there): a 180 deg
    rotation about X, not an arbitrary axis, so it stays deterministic.
    """
    tx, ty, tz = target
    cross = (-ty, tx, 0.0)  # cross((0,0,1), target)
    dot = float(tz)  # dot((0,0,1), target)
    norm = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
    if norm < 1e-6:
        return (0.0, 0.0, 0.0, 1.0) if dot > 0 else (1.0, 0.0, 0.0, 0.0)
    axis = (cross[0] / norm, cross[1] / norm, cross[2] / norm)
    angle = math.acos(max(-1.0, min(1.0, dot)))
    s = math.sin(angle / 2.0)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0))


def _snap_axis(v: Tuple[float, float, float]) -> Vec3:
    """Round a near-unit vector to the nearest of the 6 signed axis directions."""
    ax, ay, az = abs(v[0]), abs(v[1]), abs(v[2])
    m = max(ax, ay, az)
    if m == ax:
        return (1 if v[0] > 0 else -1, 0, 0)
    if m == ay:
        return (0, 1 if v[1] > 0 else -1, 0)
    return (0, 0, 1 if v[2] > 0 else -1)


def apply_roll(move: Move, v: Vec3) -> Vec3:
    """
    Rotate a unit axis vector ``v`` by one quarter-turn ``move``.

    Exact integer arithmetic (no trig, no rounding): only ``sign(angle_deg)``
    is used -- every roll is treated as an exact 90 deg quarter turn about
    world x or y regardless of the commanded magnitude (see
    ``CANDIDATE_ROLLS`` for why the magnitude is 70, not 90).
    """
    axis, angle_deg = move
    x, y, z = v
    if axis == 'x':
        return (x, -z, y) if angle_deg > 0 else (x, z, -y)
    if axis == 'y':
        return (z, y, -x) if angle_deg > 0 else (-z, y, x)
    raise ValueError(f"move axis must be 'x' or 'y', got {axis!r}")


_AXIS_VECTORS: Tuple[Vec3, ...] = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a: Vec3, b: Vec3) -> int:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _preimage_of_up(move: Move) -> Vec3:
    """Return the pre-roll direction that ``apply_roll(move, ...)`` sends to +Z."""
    for v in _AXIS_VECTORS:
        if apply_roll(move, v) == (0, 0, 1):
            return v
    raise ValueError(f'invalid move {move!r}')  # pragma: no cover -- unreachable for x/y turns


_AXIS_NAMES = {
    (0, 0, 1): '+z(up)', (0, 0, -1): '-z(down)',
    (1, 0, 0): '+x', (-1, 0, 0): '-x',
    (0, 1, 0): '+y', (0, -1, 0): '-y',
}


class DiceOrientation:
    """
    The die's full, exact orientation: which face points which world direction.

    See the module docstring for why this is exact (not an
    approximation or an empirical guess) as long as
    ``dice_manipulation_node`` upholds its own invariant of never yawing
    the die about world Z.
    """

    def __init__(self, world_of_face: Dict[int, Vec3]):
        self._world_of_face = dict(world_of_face)

    @classmethod
    def from_identification(
            cls, face_up: int, quat: Quat,
            body_normals: Optional[Dict[int, Vec3]] = None) -> 'DiceOrientation':
        """
        Build the die's full layout from one measured ``dice_tf`` orientation.

        Requires ``quat`` to be a *trustworthy* full orientation, yaw
        included -- true of ``drims_dice_simulator``'s own ground truth
        today, not assumed true of a real vision node's pip-reading
        (see the module docstring's "Why two faces, not one" for exactly
        why a single top-down reading generally cannot recover yaw).
        Prefer ``from_two_faces()`` wherever the caller cannot vouch for
        ``quat``'s yaw. ``face_up``/``quat`` are exactly what
        ``/dice_identification`` returns
        (``face_number``/``pose.pose.orientation`` -- see
        ``dice_common.py``).
        """
        normals = body_normals or STANDARD_BODY_NORMALS
        q_face = _quat_from_z_to(normals[face_up])
        q_die = _quat_mul(quat, _quat_inverse(q_face))
        return cls({f: _snap_axis(_rotate_vector(n, q_die)) for f, n in normals.items()})

    @classmethod
    def from_two_faces(
            cls, face_before: int, move: Move, face_after: int,
            body_normals: Optional[Dict[int, Vec3]] = None) -> 'DiceOrientation':
        """
        Build the die's full layout from one *executed* roll, face numbers only.

        The realistic way to know the die's layout: real perception can
        reliably report *which number* is up, never the die's precise
        yaw (see the module docstring's "Why two faces, not one").
        ``face_before`` (up before the roll), ``move`` (the roll --
        exactly known because it was commanded, not measured) and
        ``face_after`` (up after the roll, once ``dice_manipulation_node``
        reports success) are the only three exact quantities available in
        reality, and they are enough.

        How the reconstruction works: ``face_before``'s body normal
        occupied world +Z, and ``face_after``'s body normal occupied
        whichever world direction ``move`` sends to +Z (i.e. its
        pre-roll position) -- two known, perpendicular body vectors
        pinned to two known, perpendicular world directions, which fixes
        the remaining rotation completely (the third body/world axis
        pair is forced by the right-hand rule, ``_cross()``). Every
        other face's direction then follows by expressing its own body
        normal in the ``(face_before, face_after, their cross product)``
        basis and reassembling it in the corresponding world directions
        -- plain dot/cross products, exact integers throughout, no trig.
        Returns the orientation *after* the roll (the die's current,
        physical state) by composing the reconstructed pre-roll
        orientation with ``move`` via ``rolled()``. Verified numerically
        against 3000 randomized starting orientations (arbitrary yaw
        included): exact, 0 mismatches -- see ``test_dice_face_map.py``.
        """
        normals = body_normals or STANDARD_BODY_NORMALS
        b1, b2 = normals[face_before], normals[face_after]
        b3 = _cross(b1, b2)
        w1, w2 = (0, 0, 1), _preimage_of_up(move)
        w3 = _cross(w1, w2)

        def _to_world(n: Vec3) -> Vec3:
            a, b, c = _dot(n, b1), _dot(n, b2), _dot(n, b3)
            return (a * w1[0] + b * w2[0] + c * w3[0],
                    a * w1[1] + b * w2[1] + c * w3[1],
                    a * w1[2] + b * w2[2] + c * w3[2])

        pre_roll = cls({f: _to_world(n) for f, n in normals.items()})
        return pre_roll.rolled(move)

    def rolled(self, move: Move) -> 'DiceOrientation':
        """Return the orientation after one more roll, world-frame; does not mutate self."""
        return DiceOrientation({f: apply_roll(move, d) for f, d in self._world_of_face.items()})

    @property
    def up_face(self) -> int:
        for f, d in self._world_of_face.items():
            if d == (0, 0, 1):
                return f
        raise RuntimeError('no face points up -- invalid/incomplete orientation')

    def world_direction(self, face: int) -> Vec3:
        return self._world_of_face[face]

    def signature(self) -> Tuple[Tuple[int, Vec3], ...]:
        """Hashable, comparable snapshot -- used as the BFS visited-set key."""
        return tuple(sorted(self._world_of_face.items()))

    def describe(self) -> str:
        parts = (f'{f}->{_AXIS_NAMES.get(d, d)}'
                 for f, d in sorted(self._world_of_face.items()))
        return ', '.join(parts)


def plan_min_sequence(
        orientation: DiceOrientation, target_face: int,
        candidate_rolls: Optional[List[Move]] = None,
        blocked: Optional[Set[Tuple[Tuple, Move]]] = None,
        max_depth: int = 8) -> Optional[List[Move]]:
    """
    Return the provably shortest sequence of rolls that brings ``target_face`` up.

    Exact breadth-first search over the reachable orientation graph (at
    most the 24 rotations of a cube -- see the module docstring's "How
    many rolls, really", worst case 3 with the default candidates) --
    the true minimum, not an empirically-discovered one. ``[]`` if
    ``target_face`` is already up. ``blocked`` -- optional set of
    ``(orientation.signature(), move)`` pairs already found kinematically
    infeasible at that exact orientation -- is skipped; BFS naturally
    finds the next-shortest path around it. ``None`` only if exhausted
    within ``max_depth`` (should not happen with both default candidates
    available: together they reach every orientation from every other).
    """
    rolls = list(candidate_rolls or CANDIDATE_ROLLS)
    blocked = blocked or set()
    if orientation.up_face == target_face:
        return []

    start_sig = orientation.signature()
    visited = {start_sig}
    queue = deque([(orientation, [])])
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        sig = state.signature()
        for move in rolls:
            if (sig, move) in blocked:
                continue
            nxt = state.rolled(move)
            if nxt.up_face == target_face:
                return path + [move]
            nsig = nxt.signature()
            if nsig in visited:
                continue
            visited.add(nsig)
            queue.append((nxt, path + [move]))
    return None
