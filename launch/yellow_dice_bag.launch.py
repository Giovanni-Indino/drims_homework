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

"""Run yellow-die localization on a ROS 2 bag, without a live camera."""

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Create a launch description for the localizer and a bag player."""
    package_dir = get_package_share_directory('drims_homework')
    config_argument = DeclareLaunchArgument(
        'config_path',
        default_value=package_dir + '/config/yellow_dice_localizer_config.yaml',
        description='Full path to the localizer configuration.')
    bag_argument = DeclareLaunchArgument(
        'bag_path',
        description='Directory that contains metadata.yaml for a ROS 2 bag.')
    rate_argument = DeclareLaunchArgument(
        'rate', default_value='1.0',
        description='Playback rate; use 0.2 when inspecting the debug image.')

    localizer = Node(
        package='drims_homework',
        executable='yellow_dice_localizer',
        name='yellow_dice_localizer',
        output='screen',
        parameters=[LaunchConfiguration('config_path')])
    player = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_path'), '--rate',
             LaunchConfiguration('rate')],
        output='screen')
    return LaunchDescription([config_argument, bag_argument, rate_argument,
                              localizer, player])
