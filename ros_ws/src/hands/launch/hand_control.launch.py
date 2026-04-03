import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch.launch_description_sources import PythonLaunchDescriptionSource
import xacro

def generate_launch_description():
    hand_name_arg = DeclareLaunchArgument(
        'hand_name',
        default_value='inspire',
        description='Name of the hand to launch (inspire, leap, or orca)'
    )
    hand_name = LaunchConfiguration('hand_name')

    # Get the path to the URDF file
    urdf_path = PathJoinSubstitution([
        get_package_share_directory('hands'),
        'urdf',
        hand_name,
        [hand_name, '_hand_right.urdf']
    ])

    robot_desc = Command(['xacro ', urdf_path])

    # Publish robot's state to /tf and /robot_description
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    # GUI to control the robot's joints
    # node_joint_state_publisher_gui = Node(
    #     package='joint_state_publisher_gui',
    #     executable='joint_state_publisher_gui',
    #     name='joint_state_publisher_gui'
    # )

    # Package name for controller node
    node_hand_controller = Node(
        package='hands',
        executable='hand_controller',
        name='hand_controller',
        output='screen',
        parameters=[{'hand_name': hand_name}]
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
        hand_name_arg,
        node_robot_state_publisher,
        # node_joint_state_publisher_gui,
        node_hand_controller,
        node_rviz
    ])