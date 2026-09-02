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


from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('drims_homework')

    config_path_cmd = DeclareLaunchArgument(
        'dice_manipulation_config_path',
        default_value=pkg_dir + '/config/dice_manipulation_config.yaml',
        description='Full path to the dice_manipulation_node config file')

    dice_manipulation_node = Node(
        package='drims_homework',
        executable='dice_manipulation_node',
        name='dice_manipulation_node',
        output='screen',
        parameters=[
            LaunchConfiguration('dice_manipulation_config_path'),
        ])

    ld = LaunchDescription()
    ld.add_action(config_path_cmd)
    ld.add_action(dice_manipulation_node)
    return ld
