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
Convenience launch: dice_manipulation_node + dice_task_orchestrator.

Assumes the robot cell (which already brings up the easy_motion server,
see docs/ARCHITECTURE.md) and the perception layer (the dice simulator
today, /dice_identification service either way) are already running.
"""

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_dir = get_package_share_directory('drims_homework')

    manipulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            pkg_dir + '/launch/dice_manipulation_start.launch.py'))

    orchestrator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            pkg_dir + '/launch/dice_task_orchestrator_start.launch.py'))

    ld = LaunchDescription()
    ld.add_action(manipulation_launch)
    ld.add_action(orchestrator_launch)
    return ld
