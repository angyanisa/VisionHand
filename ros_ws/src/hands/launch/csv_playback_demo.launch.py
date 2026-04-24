"""
Launch file for CSV playback with full hand retargeting and visualization pipeline.

This demonstrates how to use pre-recorded CSV data instead of live Rokoko Studio data.

Usage:
    ros2 launch hands csv_playback_demo.launch.py csv_file:=/path/to/your/file.csv hand_type:=orca loop:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    """Dynamically launch robot_state_publisher and retargeting node based on hand_type"""
    hand_type = LaunchConfiguration('hand_type').perform(context)

    # Map hand type to executable name
    retargeting_map = {
        'orca': 'orca_retargeting',
        'inspire': 'inspire_retargeting',
        'leap': 'leap_retargeting',
        'nano': 'nano_retargeting',
        'nano_physics': 'nano_retargeting_physics',
    }

    executable = retargeting_map.get(hand_type)
    if not executable:
        raise ValueError(f"Invalid hand_type '{hand_type}'. Must be one of: orca, inspire, leap, nano")

    # Get the path to the URDF file
    # nano_physics shares the nano URDF; inspire uses the left variant
    urdf_folder = {'inspire': 'inspire', 'nano_physics': 'nano'}.get(hand_type, hand_type)
    urdf_base   = {'inspire': 'inspire_hand_left', 'nano_physics': 'nano_hand_right'}.get(
                    hand_type, f'{hand_type}_hand_right')
    urdf_path = os.path.join(
        get_package_share_directory('hands'),
        'urdf', urdf_folder,
        f'{urdf_base}.urdf'
    )

    # Read URDF file directly (not using xacro since these are plain .urdf files)
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    # Robot state publisher - publishes URDF and TF transforms
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    # Retargeting node with CSV playback parameter
    retargeting_node = Node(
        package='hands',
        executable=executable,
        name=f'{hand_type}_retargeting',
        output='screen',
        parameters=[{
            'data_in_degrees': True  # CSV playback data is in degrees
        }]
    )

    # Select RViz config based on hand type:
    # leap has no 'base' link so needs its own fixed frame (palm_lower)
    if hand_type == 'leap':
        rviz_config_file = 'view_hand_leap.rviz'
    else:
        rviz_config_file = 'view_hand.rviz'

    rviz_config = os.path.join(
        get_package_share_directory('hands'), 'rviz', rviz_config_file
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    nodes = [robot_state_publisher, retargeting_node, rviz_node]

    if hand_type in ('nano', 'nano_physics'):
        nodes.append(Node(
            package='hands',
            executable='nano_hardware_control',
            name='nano_hardware_control',
            output='screen',
            parameters=[{
                'hardware_mode': LaunchConfiguration('hardware_mode'),
                'move_time': LaunchConfiguration('move_time'),
            }]
        ))

    return nodes


def generate_launch_description():
    # Declare launch arguments
    csv_file_arg = DeclareLaunchArgument(
        'csv_file',
        default_value='',
        description='Path to CSV file for playback (required)'
    )

    hand_type_arg = DeclareLaunchArgument(
        'hand_type',
        default_value='orca',
        description='Type of hand to control: orca, inspire, or leap'
    )

    rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='30.0',
        description='Publishing rate in Hz'
    )

    loop_arg = DeclareLaunchArgument(
        'loop',
        default_value='false',
        description='Whether to loop the playback'
    )

    control_method_arg = DeclareLaunchArgument(
        'control_method',
        default_value='direct',
        description='Control method: direct, fingertip_ik, or jparse_ik'
    )

    hardware_mode_arg = DeclareLaunchArgument(
        'hardware_mode',
        default_value='false',
        description='Enable hardware control (true/false)'
    )

    move_time_arg = DeclareLaunchArgument(
        'move_time',
        default_value='50',
        description='Servo move time in ms for nano hardware control'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to ORCA hand model (required if hardware_mode=true)'
    )

    calibrate_arg = DeclareLaunchArgument(
        'calibrate',
        default_value='false',
        description='Calibrate ORCA hand on startup (true/false)'
    )

    # Rokoko listener node in playback mode
    rokoko_playback_node = Node(
        package='hands',
        executable='rokoko_listener',
        name='rokoko_listener',
        output='screen',
        parameters=[{
            'playback_mode': True,
            'csv_file': LaunchConfiguration('csv_file'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'loop_playback': LaunchConfiguration('loop')
        }]
    )

    # Visualization node
    visualization_node = Node(
        package='hands',
        executable='visualization_node',
        name='visualization_node',
        output='screen',
        parameters=[{
            'hand_type': LaunchConfiguration('hand_type')
        }]
    )

    # Control type publisher node (publishes to /control_type topic)
    control_type_node = Node(
        package='hands',
        executable='control_type_publisher',
        name='control_type_publisher',
        output='screen',
        parameters=[{
            'control_method': LaunchConfiguration('control_method')
        }]
    )

    # Hardware control node (only active when hardware_mode=true)
    orca_hardware_node = Node(
        package='hands',
        executable='orca_hardware_control',
        name='orca_hardware_control',
        output='screen',
        parameters=[{
            'hardware_mode': LaunchConfiguration('hardware_mode'),
            'model_path': LaunchConfiguration('model_path'),
            'calibrate': LaunchConfiguration('calibrate')
        }]
    )

    # Fingertip error plotter (plots x, y, z error vs time for each finger)
    # Automatically saves plots when Ctrl+C is pressed or playback ends
    # Delayed by 2 seconds to let RViz and other nodes start first
    error_plotter_node = TimerAction(
        period=2.0,  # 2 second delay
        actions=[
            Node(
                package='hands',
                executable='fingertip_error_plotter',
                name='fingertip_error_plotter',
                output='screen'
            )
        ]
    )

    # Conditionally launch the robot_state_publisher and retargeting node based on hand_type
    dynamic_launch = OpaqueFunction(function=launch_setup)

    return LaunchDescription([
        csv_file_arg,
        hand_type_arg,
        rate_arg,
        loop_arg,
        control_method_arg,
        hardware_mode_arg,
        move_time_arg,
        model_path_arg,
        calibrate_arg,
        dynamic_launch,
        rokoko_playback_node,
        control_type_node,
        visualization_node,
        orca_hardware_node,
        error_plotter_node
    ])
