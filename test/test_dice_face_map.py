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
Unit tests for dice_face_map.DiceFaceMap.

Pure Python, no ROS/robot needed. Uses a small toy die (an explicit
roll->face table) so the tests do not depend on the real simulator's
geometry.
"""

from drims_homework.dice_face_map import DiceFaceMap, INFEASIBLE

ROLL_A = ('x', 90.0)
ROLL_B = ('y', -90.0)

# A toy die: face 1 up top, opposite pairs 1<->6, 2<->5, 3<->4.
# ROLL_A cycles 1->2->6->5->1 (and 3->4->3... simplified: keep 3/4 fixed
# under ROLL_A), ROLL_B cycles 1->3->6->4->1.
TRANSITIONS = {
    (1, ROLL_A): 2, (2, ROLL_A): 6, (6, ROLL_A): 5, (5, ROLL_A): 1,
    (3, ROLL_A): 3, (4, ROLL_A): 4,
    (1, ROLL_B): 3, (3, ROLL_B): 6, (6, ROLL_B): 4, (4, ROLL_B): 1,
    (2, ROLL_B): 2, (5, ROLL_B): 5,
}


def _drive(dice_map: DiceFaceMap, face: int, move) -> int:
    """Apply `move` to the toy die from `face`, recording the outcome."""
    result = TRANSITIONS[(face, move)]
    dice_map.record(face, move, result)
    return result


def test_plan_next_prefers_untried_when_nothing_learned():
    m = DiceFaceMap(candidate_rolls=[ROLL_A, ROLL_B])
    move = m.plan_next(1, 6)
    assert move in (ROLL_A, ROLL_B)


def test_plan_next_returns_none_when_current_equals_target():
    # Not a normal call path (the orchestrator checks this before
    # planning), but plan_next() itself should not crash or loop.
    m = DiceFaceMap(candidate_rolls=[ROLL_A, ROLL_B])
    assert m._bfs_next_move(1, 1) is None


def test_plan_next_finds_shortest_learned_path():
    m = DiceFaceMap(candidate_rolls=[ROLL_A, ROLL_B])
    face = _drive(m, 1, ROLL_A)   # 1 -> 2
    _drive(m, face, ROLL_A)       # 2 -> 6
    # Now a direct 1->2->6 path is known; plan_next(1, 6) must return the
    # first step of it, not blindly re-explore.
    assert m.plan_next(1, 6) == ROLL_A


def test_known_path_returns_the_full_shortest_sequence():
    m = DiceFaceMap(candidate_rolls=[ROLL_A, ROLL_B])
    face = _drive(m, 1, ROLL_A)   # 1 -> 2
    _drive(m, face, ROLL_A)       # 2 -> 6
    assert m.known_path(1, 6) == [ROLL_A, ROLL_A]


def test_known_path_none_when_not_yet_connected():
    m = DiceFaceMap(candidate_rolls=[ROLL_A, ROLL_B])
    assert m.known_path(1, 6) is None


def test_known_path_empty_list_when_already_there():
    m = DiceFaceMap(candidate_rolls=[ROLL_A, ROLL_B])
    assert m.known_path(1, 1) == []


def test_opposite_pair_assumed_by_default_before_any_roll():
    # assume_standard_die defaults to True: opposite pairs (summing to 7)
    # are known from construction, no exploration needed at all.
    m = DiceFaceMap(candidate_rolls=[ROLL_A, ROLL_B])
    assert m._opposite.get(1) == 6
    assert m._opposite.get(6) == 1


def test_opposite_pair_empirically_inferred_when_assumption_disabled():
    m = DiceFaceMap(candidate_rolls=[ROLL_A, ROLL_B], assume_standard_die=False)
    assert m._opposite == {}
    face = _drive(m, 1, ROLL_A)   # 1 -> 2
    _drive(m, face, ROLL_A)       # 2 -> 6  => 1 and 6 are opposite
    assert m._opposite.get(1) == 6
    assert m._opposite.get(6) == 1


def test_opposite_pair_shortcut_used_even_when_never_tried_from_target_face():
    m = DiceFaceMap(candidate_rolls=[ROLL_A, ROLL_B], assume_standard_die=False)
    face = _drive(m, 1, ROLL_A)   # 1 -> 2
    _drive(m, face, ROLL_A)       # 2 -> 6  => opposite pair (1, 6) known
    # plan_next(1, 6) should now immediately suggest ROLL_A, using the
    # opposite-pair shortcut, not fall through to blind exploration.
    assert m.plan_next(1, 6) == ROLL_A


def test_infeasible_roll_is_never_retried_and_does_not_block_planning():
    m = DiceFaceMap(candidate_rolls=[ROLL_A, ROLL_B])
    m.mark_infeasible(1, ROLL_A)
    assert m._learned[1][ROLL_A] == INFEASIBLE
    # plan_next() must fall back to the other candidate instead of
    # proposing the known-infeasible one.
    move = m.plan_next(1, 6)
    assert move == ROLL_B


def test_plan_next_none_when_every_candidate_infeasible():
    m = DiceFaceMap(candidate_rolls=[ROLL_A, ROLL_B])
    m.mark_infeasible(1, ROLL_A)
    m.mark_infeasible(1, ROLL_B)
    assert m.plan_next(1, 6) is None


def test_full_toy_die_converges_from_any_face_to_any_target():
    m = DiceFaceMap(candidate_rolls=[ROLL_A, ROLL_B])
    for _ in range(50):
        for start in range(1, 7):
            for target in range(1, 7):
                if start == target:
                    continue
                face = start
                for _step in range(20):  # generous bound, must not hang
                    if face == target:
                        break
                    move = m.plan_next(face, target)
                    assert move is not None, (
                        f'no move from {face} towards {target} on a fully '
                        f'connected toy die')
                    face = _drive(m, face, move)
                assert face == target


def test_describe_does_not_raise_before_or_after_learning():
    m = DiceFaceMap(candidate_rolls=[ROLL_A, ROLL_B])
    assert 'face 1' in m.describe()
    _drive(m, 1, ROLL_A)
    assert 'opposite pairs' in m.describe()
