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
Unit tests for dice_face_map (the closed-form die orientation model).

Pure Python, no ROS/robot needed.
"""

import math
import random

from drims_homework.dice_face_map import (
    CANDIDATE_ROLLS,
    STANDARD_BODY_NORMALS,
    DiceOrientation,
    apply_roll,
    plan_min_sequence,
)

# Track CANDIDATE_ROLLS rather than hard-coding the magnitude: only the
# sign of the angle carries geometric meaning (see apply_roll()), the
# magnitude is whatever dice_manipulation_node is told to sweep.
ROLL_X = next(m for m in CANDIDATE_ROLLS if m[0] == 'x')
ROLL_Y = next(m for m in CANDIDATE_ROLLS if m[0] == 'y')


def _identity_orientation() -> DiceOrientation:
    return DiceOrientation(dict(STANDARD_BODY_NORMALS))


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_from_axis_angle(axis, angle_rad):
    ax, ay, az = axis
    s = math.sin(angle_rad / 2.0)
    return (ax * s, ay * s, az * s, math.cos(angle_rad / 2.0))


def _rotate_vector(v, q):
    x, y, z, w = q
    vq = (float(v[0]), float(v[1]), float(v[2]), 0.0)
    q_conj = (-x, -y, -z, w)
    rx, ry, rz, _ = _quat_mul(_quat_mul(q, vq), q_conj)
    return (rx, ry, rz)


def _quat_from_z_to(target):
    """Re-implement the module's shortest-arc formula, kept test-local on purpose."""
    tx, ty, tz = target
    cross = (-ty, tx, 0.0)
    dot = float(tz)
    norm = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
    if norm < 1e-6:
        return (0.0, 0.0, 0.0, 1.0) if dot > 0 else (1.0, 0.0, 0.0, 0.0)
    axis = (cross[0] / norm, cross[1] / norm, cross[2] / norm)
    angle = math.acos(max(-1.0, min(1.0, dot)))
    s = math.sin(angle / 2.0)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0))


def _snap(v):
    ax, ay, az = abs(v[0]), abs(v[1]), abs(v[2])
    m = max(ax, ay, az)
    if m == ax:
        return (1 if v[0] > 0 else -1, 0, 0)
    if m == ay:
        return (0, 1 if v[1] > 0 else -1, 0)
    return (0, 0, 1 if v[2] > 0 else -1)


def _simulated_identification(q_die_world, body_normals=STANDARD_BODY_NORMALS):
    """
    Stand in for drims_dice_simulator's own /dice_identification response.

    Given the die's true rigid-body orientation, find the up face and
    return (face_up, dice_tf_quat) the same way the simulator does --
    dice_tf == the up face's own +Z-out-of-the-face frame.
    """
    best_face, best_dot = None, -2.0
    for f, n in body_normals.items():
        wn = _rotate_vector(n, q_die_world)
        if wn[2] > best_dot:
            best_dot = wn[2]
            best_face = f
    q_face = _quat_from_z_to(body_normals[best_face])
    q_ret = _quat_mul(q_die_world, q_face)
    return best_face, q_ret


# ------------------------------------------------------------------ #
# apply_roll / DiceOrientation.rolled                                 #
# ------------------------------------------------------------------ #
def test_apply_roll_x90_moves_up_to_minus_y():
    # Pins apply_roll()'s own geometric law for a literal move, independent
    # of whichever sign combo CANDIDATE_ROLLS currently picks.
    assert apply_roll(('x', 90.0), (0, 0, 1)) == (0, -1, 0)


def test_apply_roll_y_minus90_moves_up_to_minus_x():
    assert apply_roll(('y', -90.0), (0, 0, 1)) == (-1, 0, 0)


def test_apply_roll_y_plus90_moves_up_to_plus_x():
    assert apply_roll(('y', 90.0), (0, 0, 1)) == (1, 0, 0)


def test_apply_roll_is_a_proper_quarter_turn_period_four():
    v = (0, 0, 1)
    for move in CANDIDATE_ROLLS:
        w = v
        for _ in range(4):
            w = apply_roll(move, w)
        assert w == v


def test_rolled_does_not_mutate_original():
    o = _identity_orientation()
    before = o.signature()
    o.rolled(ROLL_X)
    assert o.signature() == before


def test_identity_orientation_up_face_matches_body_normals():
    o = _identity_orientation()
    assert o.up_face == 6  # STANDARD_BODY_NORMALS[6] == (0, 0, 1)


# ------------------------------------------------------------------ #
# from_identification: reconstructing the whole die from one reading  #
# ------------------------------------------------------------------ #
def test_from_identification_matches_ground_truth_at_identity():
    face_up, quat = _simulated_identification((0.0, 0.0, 0.0, 1.0))
    o = DiceOrientation.from_identification(face_up, quat)
    for f, n in STANDARD_BODY_NORMALS.items():
        assert o.world_direction(f) == n


def test_from_identification_recovers_full_layout_randomized():
    random.seed(1234)
    for _ in range(200):
        # Build a random axis-aligned die orientation (as if reached by
        # some sequence of world-frame quarter turns, possibly about z
        # too -- from_identification must reconstruct it regardless of
        # how it got there, it only reads the current state).
        q = (0.0, 0.0, 0.0, 1.0)
        for _ in range(random.randint(0, 6)):
            axis = random.choice([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
            angle = random.choice([math.pi / 2, -math.pi / 2])
            q = _quat_mul(_quat_from_axis_angle(axis, angle), q)

        truth = {f: _snap(_rotate_vector(n, q)) for f, n in STANDARD_BODY_NORMALS.items()}
        face_up, quat = _simulated_identification(q)

        o = DiceOrientation.from_identification(face_up, quat)
        for f in range(1, 7):
            assert o.world_direction(f) == truth[f], f'face {f} mismatch'
        assert o.up_face == face_up


def test_from_identification_is_valid_starting_point_for_rolls():
    face_up, quat = _simulated_identification((0.0, 0.0, 0.0, 1.0))
    o = DiceOrientation.from_identification(face_up, quat)
    rolled = o.rolled(ROLL_X)
    expected = _identity_orientation().rolled(ROLL_X)
    assert rolled.signature() == expected.signature()


# ------------------------------------------------------------------ #
# from_two_faces: reconstructing the whole die from face NUMBERS only #
# (what real perception can actually promise -- no yaw involved at    #
# all, unlike from_identification above).                             #
# ------------------------------------------------------------------ #
def test_from_two_faces_matches_ground_truth_at_identity():
    # face 6 up (identity); rolling ROLL_X brings body +y (face 3) up.
    o = DiceOrientation.from_two_faces(6, ROLL_X, 3)
    expected = _identity_orientation().rolled(ROLL_X)
    assert o.signature() == expected.signature()


def test_from_two_faces_recovers_full_layout_randomized_regardless_of_yaw():
    random.seed(99)
    for _ in range(300):
        # An arbitrary TRUE pre-roll orientation, including yaws about
        # world Z that from_two_faces() never gets to see -- it must
        # reconstruct the post-roll state correctly regardless, using
        # only the two face numbers and the (known) move.
        true_pre = dict(STANDARD_BODY_NORMALS)
        for _ in range(random.randint(0, 6)):
            axis = random.choice(['x', 'y', 'z'])
            angle = random.choice([math.pi / 2, -math.pi / 2])
            if axis == 'z':
                def _rz(v, ang):
                    x, y, z = v
                    return (-y, x, z) if ang > 0 else (y, -x, z)
                true_pre = {f: _rz(d, angle) for f, d in true_pre.items()}
            else:
                true_pre = {f: apply_roll((axis, math.degrees(angle)), d)
                            for f, d in true_pre.items()}

        face_before = next(f for f, d in true_pre.items() if d == (0, 0, 1))
        move = random.choice(CANDIDATE_ROLLS)
        true_post = {f: apply_roll(move, d) for f, d in true_pre.items()}
        face_after = next(f for f, d in true_post.items() if d == (0, 0, 1))

        o = DiceOrientation.from_two_faces(face_before, move, face_after)
        for f in range(1, 7):
            assert o.world_direction(f) == true_post[f], (
                f'face {f} mismatch (face_before={face_before}, move={move}, '
                f'face_after={face_after})')


def test_from_two_faces_needs_no_orientation_reading_at_all():
    # Same (face_before, move, face_after) triple must reconstruct
    # identically no matter what the die's actual yaw was -- the whole
    # point (see the module docstring's "Why two faces, not one").
    o1 = DiceOrientation.from_two_faces(6, ROLL_X, 3)
    o2 = DiceOrientation.from_two_faces(6, ROLL_X, 3)
    assert o1.signature() == o2.signature()


# ------------------------------------------------------------------ #
# plan_min_sequence                                                    #
# ------------------------------------------------------------------ #
def test_plan_min_sequence_empty_when_already_there():
    o = _identity_orientation()
    assert plan_min_sequence(o, o.up_face) == []


def test_plan_min_sequence_single_roll_when_one_suffices():
    o = _identity_orientation()  # face 6 up
    seq = plan_min_sequence(o, 3)  # ROLL_X brings body +y (face 3) up
    assert seq == [ROLL_X]


def test_plan_min_sequence_is_provably_shortest():
    o = _identity_orientation()
    for target in range(1, 7):
        seq = plan_min_sequence(o, target)
        assert seq is not None
        # Executing it must actually reach the target...
        state = o
        for move in seq:
            state = state.rolled(move)
        assert state.up_face == target
        # ...and no shorter sequence can (brute-force check up to len - 1).
        if len(seq) > 0:
            import itertools
            for shorter_len in range(len(seq)):
                for combo in itertools.product(CANDIDATE_ROLLS, repeat=shorter_len):
                    s = o
                    for move in combo:
                        s = s.rolled(move)
                    assert s.up_face != target, (
                        f'found a shorter path to {target}: {combo}')


def test_plan_min_sequence_worst_case_is_three_rolls_from_any_start():
    # Verified analytically (see dice_face_map's module docstring): with
    # both candidate rolls, every face is reachable from every starting
    # orientation in at most 3 rolls.
    reachable = {_identity_orientation().signature(): _identity_orientation()}
    frontier = [_identity_orientation()]
    while frontier:
        nxt = []
        for state in frontier:
            for move in CANDIDATE_ROLLS:
                r = state.rolled(move)
                if r.signature() not in reachable:
                    reachable[r.signature()] = r
                    nxt.append(r)
        frontier = nxt
    assert len(reachable) == 24  # the full cube rotation group

    worst = 0
    for state in reachable.values():
        for target in range(1, 7):
            seq = plan_min_sequence(state, target)
            assert seq is not None
            worst = max(worst, len(seq))
    assert worst == 3


def test_plan_min_sequence_none_when_every_roll_blocked():
    o = _identity_orientation()
    blocked = {(o.signature(), move) for move in CANDIDATE_ROLLS}
    assert plan_min_sequence(o, 3, blocked=blocked) is None


def test_plan_min_sequence_routes_around_a_blocked_roll():
    o = _identity_orientation()
    direct = plan_min_sequence(o, 3)
    assert direct == [ROLL_X]
    blocked = {(o.signature(), ROLL_X)}
    rerouted = plan_min_sequence(o, 3, blocked=blocked)
    assert rerouted is not None
    assert rerouted[0] != ROLL_X
    state = o
    for move in rerouted:
        state = state.rolled(move)
    assert state.up_face == 3


def test_describe_does_not_raise():
    o = _identity_orientation()
    assert 'up' in o.describe()
