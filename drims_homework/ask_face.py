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
``ask_face`` -- one-shot CLI to request another dice face, no remapping.

``dice_task_orchestrator`` already keeps the die's full tracked
orientation in memory for its whole lifetime (``self._orientation``, see
its module docstring's "Calibration"): as long as the node keeps running
and nothing else moved the die, asking for a *different* face never
re-triggers the one-time calibration roll -- ``IDENTIFY`` sees the live
face still matches what is tracked and goes straight to
``CHECK_TARGET -> PLAN -> ROLL``. That already works today via the two
calls documented in ``dice_task_orchestrator``'s module docstring::

    ros2 param set /dice_task_orchestrator target_face 6
    ros2 service call /dice_task_orchestrator/reach_target_face std_srvs/srv/Trigger "{}"

This script is nothing more than those two calls in one command, so
asking for a new face is one line instead of two::

    ros2 run drims_homework ask_face 6
"""

import sys
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from std_srvs.srv import Trigger
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter

DEFAULT_ORCHESTRATOR_NODE = '/dice_task_orchestrator'


def _parse_args(argv: List[str]):
    if len(argv) not in (1, 2):
        return None, None, (
            'usage: ros2 run drims_homework ask_face <face 1-6> '
            f'[orchestrator_node={DEFAULT_ORCHESTRATOR_NODE}]')
    try:
        target_face = int(argv[0])
    except ValueError:
        return None, None, f'invalid face {argv[0]!r}: must be an integer 1-6'
    if not 1 <= target_face <= 6:
        return None, None, f'invalid face {target_face}: must be 1-6'
    orchestrator_node = argv[1] if len(argv) == 2 else DEFAULT_ORCHESTRATOR_NODE
    return target_face, orchestrator_node, None


def ask_face(node: Node, orchestrator_node: str, target_face: int) -> Optional[Tuple[bool, str]]:
    """Set ``target_face`` then call ``~/reach_target_face``; return ``(success, message)`` or ``None``."""
    set_params_client = node.create_client(
        SetParameters, f'{orchestrator_node}/set_parameters')
    reach_client = node.create_client(
        Trigger, f'{orchestrator_node}/reach_target_face')

    if not set_params_client.wait_for_service(timeout_sec=10.0):
        return None

    request = SetParameters.Request()
    request.parameters = [
        Parameter('target_face', Parameter.Type.INTEGER, target_face).to_parameter_msg(),
    ]
    future = set_params_client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    result = future.result()
    if result is None or not all(r.successful for r in result.results):
        reasons = [] if result is None else [r.reason for r in result.results if not r.successful]
        node.get_logger().error(f'failed to set target_face={target_face}: {reasons}')
        return None

    if not reach_client.wait_for_service(timeout_sec=10.0):
        return None

    # No timeout here on purpose -- reach_target_face() runs the whole
    # physical sequence (possibly several rolls) before responding, same
    # as a plain "ros2 service call ... reach_target_face" would block.
    future = reach_client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, future)
    result = future.result()
    if result is None:
        return None
    return result.success, result.message


def main(args=None):
    argv = sys.argv[1:] if args is None else args
    target_face, orchestrator_node, err = _parse_args(argv)
    if err:
        print(err)
        return 1

    rclpy.init()
    node = rclpy.create_node('ask_face')
    try:
        outcome = ask_face(node, orchestrator_node, target_face)
        if outcome is None:
            print(f"could not reach '{orchestrator_node}' "
                  f"(not running, or wrong node name?)")
            return 1
        success, message = outcome
        print(f"{'OK' if success else 'FAILED'}: {message}")
        return 0 if success else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
