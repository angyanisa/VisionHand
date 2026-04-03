from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    csv_file_arg = DeclareLaunchArgument(
        'csv_file',
        default_value='',
        description='Path to CSV file for playback'
    )

    rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='30.0',
        description='Publishing rate in Hz'
    )

    loop_arg = DeclareLaunchArgument(
        'loop_playback',
        default_value='False',
        description='Whether to loop the playback'
    )

    # Rokoko listener node in playback mode
    rokoko_node = Node(
        package='hands',
        executable='rokoko_listener',
        name='rokoko_listener',
        output='screen',
        parameters=[{
            'playback_mode': True,
            'csv_file': LaunchConfiguration('csv_file'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'loop_playback': LaunchConfiguration('loop_playback')
        }]
    )

    return LaunchDescription([
        csv_file_arg,
        rate_arg,
        loop_arg,
        rokoko_node
    ])
