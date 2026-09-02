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
Convenience launch: ``drims_dice_simulator``'s own spawner + a healer.

Wraps ``drims_dice_simulator``'s ``spawn_dice.launch.py`` unmodified
(all its launch arguments -- ``face_up``, ``dice_size``, ``position``,
``random_position``, ... -- are simply forwarded) and adds
``dice_simulator_healer`` alongside it, which automates the manual
Ctrl-C-and-relaunch workaround for a known first-launch race in that
package -- see ``dice_simulator_healer.py``'s module docstring. Use this
instead of calling ``spawn_dice.launch.py`` directly and you should never
need to manually restart the simulator again because of that race.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


_FORWARDED_ARGS = [
    'face_up', 'dice_size', 'position', 'random_position',
    'x_min', 'x_max', 'y_min', 'y_max',
    'surface_height', 'pips_distance', 'pip_diameter',
]


def generate_launch_description():
    declare_args = [
        DeclareLaunchArgument(name, default_value='__default__')
        for name in _FORWARDED_ARGS
    ]

    spawn_dice_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('drims_dice_simulator'), '/launch/spawn_dice.launch.py']),
        launch_arguments=[(name, LaunchConfiguration(name)) for name in _FORWARDED_ARGS])

    healer_node = Node(
        package='drims_homework',
        executable='dice_simulator_healer',
        name='dice_simulator_healer',
        output='screen')

    ld = LaunchDescription()
    for action in declare_args:
        ld.add_action(action)
    ld.add_action(spawn_dice_launch)
    ld.add_action(healer_node)
    return ld
