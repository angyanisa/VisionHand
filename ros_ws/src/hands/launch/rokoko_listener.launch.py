from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    ip_arg = DeclareLaunchArgument(
        'ip_address',
        default_value='0.0.0.0',
        description='IP address to listen on (0.0.0.0 = all interfaces)'
    )
    
    port_arg = DeclareLaunchArgument(
        'port',
        default_value='14043',
        description='UDP port for Rokoko Studio stream'
    )
    
    rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='30.0',
        description='Publishing rate in Hz'
    )

    # Rokoko listener node
    rokoko_node = Node(
        package='hands',
        executable='rokoko_listener',
        name='rokoko_listener',
        output='screen',
        parameters=[{
            'ip_address': LaunchConfiguration('ip_address'),
            'port': LaunchConfiguration('port'),
            'publish_rate': LaunchConfiguration('publish_rate')
        }]
    )

    return LaunchDescription([
        ip_arg,
        port_arg,
        rate_arg,
        rokoko_node
    ])