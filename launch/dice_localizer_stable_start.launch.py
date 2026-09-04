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
Launch the real-camera dice localizer as the ``/dice_identification`` server.

Drop-in replacement for the simulator's ``dice_spawner_node`` on the real
robot -- see ``dice_common.py``'s module docstring ("simulator today, a
real camera-based node tomorrow") and ``dice_localizer_stable.py``'s own
in-code docstrings for exactly what it does: identify the face number,
transform the detection into ``world_frame``, and continuously broadcast
``dice_tf_frame`` for ``dice_manipulation_node`` to actually grasp
against.

Assumes, already running, before this:
  * the physical OAK camera driver (publishing ``image_topic``/
    ``camera_info_topic``);
  * the cell's hand-eye calibration static transform -- part of
    ``ur5e_N_start.launch.py`` when launched with ``fake:=false`` (see
    ``camera_calibration_cellN.launch.py`` in ``drims_description``).

Do NOT also launch ``dice_simulator_start.launch.py`` at the same time --
only one node may serve ``/dice_identification``.
"""

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory('drims_homework')
    config_argument = DeclareLaunchArgument(
        'config_path',
        default_value=package_dir + '/config/dice_localizer_stable_config.yaml',
        description='Full path to the dice_localizer_stable configuration.')
    localizer = Node(
        package='drims_homework',
        executable='dice_localizer_stable',
        name='dice_detector_node',
        output='screen',
        parameters=[LaunchConfiguration('config_path')])
    return LaunchDescription([config_argument, localizer])
