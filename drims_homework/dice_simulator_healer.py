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
Startup healer for ``drims_dice_simulator``'s known first-launch race.

``drims_dice_simulator``'s ``dice_spawner_node`` sometimes loses the
very first ``ApplyPlanningScene`` call it makes at start-up -- the
underlying planning scene monitor is not always fully warmed up yet,
even though the ``/apply_planning_scene`` *service* itself is already
being served, so the very first spawn attempt gets
``ApplyPlanningScene_Response(success=False)`` and the dice is never
actually added to the scene. Nothing in that node retries automatically:
``/dice_identification`` then fails forever, logging "Dice object
'dice' not found in planning scene." every 0.5s. Observed workaround
until now: Ctrl-C and relaunch ``spawn_dice.launch.py`` by hand.

This node automates exactly that manual workaround instead of touching
``drims_dice_simulator`` itself (out of scope for this package -- see
``docs/ARCHITECTURE.md``): it polls ``/dice_identification`` a few times
after start-up and, if it never succeeds, calls ``dice_spawner_node``'s
own ``/reset_dice`` service (which redoes the spawn from scratch, by
then well past the warm-up race) and polls again, up to a bounded number
of reset attempts. Logs clearly and exits (0 on success, 1 on giving up)
either way -- meant to run once per simulator start-up, e.g. from
``dice_simulator_start.launch.py``, not as a persistent node.

Interfaces
----------
* Parameter ``poll_interval_sec`` (float, default 1.0): delay between
  ``/dice_identification`` polls.
* Parameter ``max_polls_per_attempt`` (int, default 8): polls to try
  before giving up on the current attempt (first one, or after a reset).
* Parameter ``max_reset_attempts`` (int, default 3): how many times to
  call ``/reset_dice`` and re-poll before giving up entirely.
"""

import time

import rclpy
from rclpy.node import Node

from std_srvs.srv import Trigger

from drims_homework.dice_common import create_dice_identification_client, identify_dice


def _poll_until_identified(node: Node, client, poll_interval_sec: float, max_polls: int) -> bool:
    for attempt in range(1, max_polls + 1):
        face, pose = identify_dice(node, client, timeout_sec=poll_interval_sec)
        if face is not None and pose is not None:
            node.get_logger().info(f'Dice identification OK on poll {attempt}/{max_polls}.')
            return True
        time.sleep(poll_interval_sec)
    return False


def _reset_dice(node: Node, client) -> bool:
    if not client.wait_for_service(timeout_sec=10.0):
        node.get_logger().error("'/reset_dice' service not available")
        return False
    future = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, future)
    result = future.result()
    if result is None or not result.success:
        node.get_logger().error(f'/reset_dice call failed: {getattr(result, "message", None)}')
        return False
    node.get_logger().info(f'/reset_dice succeeded: {result.message}')
    return True


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('dice_simulator_healer')

    node.declare_parameter('poll_interval_sec', 1.0)
    node.declare_parameter('max_polls_per_attempt', 8)
    node.declare_parameter('max_reset_attempts', 3)

    poll_interval_sec = node.get_parameter('poll_interval_sec').value
    max_polls = node.get_parameter('max_polls_per_attempt').value
    max_reset_attempts = node.get_parameter('max_reset_attempts').value

    dice_identification_client = create_dice_identification_client(node)
    reset_dice_client = node.create_client(Trigger, '/reset_dice')

    healed = False
    try:
        node.get_logger().info(
            'Watching for the dice to appear in the planning scene '
            f'(up to {max_polls} polls every {poll_interval_sec:.1f}s)...')
        if _poll_until_identified(node, dice_identification_client, poll_interval_sec, max_polls):
            healed = True
        else:
            for attempt in range(1, max_reset_attempts + 1):
                node.get_logger().warn(
                    f"Dice still not in the planning scene (known dice_spawner_node "
                    f'start-up race) -- forcing a respawn via /reset_dice '
                    f'(attempt {attempt}/{max_reset_attempts})...')
                if not _reset_dice(node, reset_dice_client):
                    continue
                if _poll_until_identified(
                        node, dice_identification_client, poll_interval_sec, max_polls):
                    healed = True
                    break

        if healed:
            node.get_logger().info('Dice simulator healthy; healer exiting.')
        else:
            node.get_logger().error(
                'Dice still not identifiable after all reset attempts -- giving up. '
                'A full stop/relaunch of the simulator is likely needed.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0 if healed else 1


if __name__ == '__main__':
    raise SystemExit(main())
