import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
import xacro

def generate_launch_description():
    # Declare arguments
    csv_file_arg = DeclareLaunchArgument(
        'csv_file',
        default_value='',
        description='Path to CSV file with motion capture data'
    )

    use_ik_arg = DeclareLaunchArgument(
        'use_ik',
        default_value='true',
        description='Use IK solver (true) or fallback animation (false)'
    )

    loop_arg = DeclareLaunchArgument(
        'loop',
        default_value='true',
        description='Loop the motion data'
    )

    frame_rate_arg = DeclareLaunchArgument(
        'frame_rate',
        default_value='30',
        description='Playback frame rate (Hz)'
    )

    # Get configuration values
    csv_file = LaunchConfiguration('csv_file')
    use_ik = LaunchConfiguration('use_ik')
    loop = LaunchConfiguration('loop')
    frame_rate = LaunchConfiguration('frame_rate')

    # Get the path to the URDF file
    urdf_path = PathJoinSubstitution([
        get_package_share_directory('hands'),
        'urdf',
        'orca',
        'orca_hand_right.urdf'
    ])

    robot_desc = Command(['xacro ', urdf_path])

    # Publish robot's state to /tf and /robot_description
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    # IK-based hand controller node
    node_hand_controller_ik = Node(
        package='hands',
        executable='hand_controller_ik',
        name='hand_controller_ik',
        output='screen',
        parameters=[{
            'hand_name': 'orca',
            'csv_file': csv_file,
            'use_ik': use_ik,
            'loop': loop,
            'frame_rate': frame_rate
        }]
    )

    # RViz2 for visualization
    node_rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(get_package_share_directory('hands'), 'rviz', 'view_hand.rviz')]
    )

    # Create and return the launch description
    return LaunchDescription([
        csv_file_arg,
        use_ik_arg,
        loop_arg,
        frame_rate_arg,
        node_robot_state_publisher,
        node_hand_controller_ik,
        node_rviz
    ])
