from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    config_node = os.path.join(
        get_package_share_directory('points_shall_not_pass'),
        'config',
        'config.yaml'
        )

    node = Node(
            package='points_shall_not_pass',
            name='gandalf_node',
            executable='points_shall_not_pass_node',
            output='screen',
            parameters=[config_node]
        )
    return LaunchDescription([node])
