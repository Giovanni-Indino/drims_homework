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
Shared helpers for talking to the *perception layer*.

Every node in this package that needs to know where the dice is and which
face is up (``dice_manipulation_node``, ``dice_task_orchestrator``, ...)
goes exclusively through the ``/dice_identification`` service contract
defined here. Today that service is served by the simulator
(``drims_dice_simulator``); tomorrow it will be served by a real
camera-based node. As long as the replacement node:

  * serves ``easy_motion_msgs/srv/DiceIdentification`` on the same service
    name (``dice_identification`` by default), returning
    ``face_number`` + ``pose`` (+ ``success``);
  * keeps broadcasting the ``dice_tf`` / ``dice_com_tf`` TF frames used for
    grasping and for computing the place height (see
    ``dice_manipulation_node``);

nothing in this package needs to change. This module exists precisely to
make that boundary explicit and to avoid duplicating the same service-call
boilerplate in every node.
"""

from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.client import Client

from geometry_msgs.msg import PoseStamped
from easy_motion_msgs.srv import DiceIdentification

DEFAULT_DICE_IDENTIFICATION_SERVICE = 'dice_identification'


def create_dice_identification_client(node: Node,
                                      service_name: str = DEFAULT_DICE_IDENTIFICATION_SERVICE
                                      ) -> Client:
    """Create a client for the perception service on ``node``."""
    return node.create_client(DiceIdentification, service_name)


def identify_dice(node: Node, client: Client,
                  timeout_sec: float = 5.0
                  ) -> Tuple[Optional[int], Optional[PoseStamped]]:
    """
    Call ``/dice_identification`` and return ``(face_number, pose)``.

    Returns ``(None, None)`` if the service is unavailable, times out, or
    reports failure — callers should treat that as "perception layer not
    ready / could not localize the dice", not as a face number.
    """
    if not client.wait_for_service(timeout_sec=timeout_sec):
        node.get_logger().error(
            f"'{client.srv_name}' service not available")
        return None, None

    future = client.call_async(DiceIdentification.Request())
    rclpy.spin_until_future_complete(node, future)
    result = future.result()

    if result is None or not result.success:
        node.get_logger().error('Dice identification failed')
        return None, None

    node.get_logger().info(
        f'Dice identified: face={result.face_number}, '
        f'pos=[{result.pose.pose.position.x:.3f}, '
        f'{result.pose.pose.position.y:.3f}, '
        f'{result.pose.pose.position.z:.3f}] ({result.pose.header.frame_id})')
    return result.face_number, result.pose
