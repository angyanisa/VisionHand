import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command, PythonExpression
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch_ros.parameter_descriptions import ParameterValue


def launch_setup(context, *args, **kwargs):
    """Dynamically launch nodes based on hand_type"""
    hand_type = LaunchConfiguration('hand_type').perform(context)

    # Map hand type to executable name
    retargeting_map = {
        'orca': 'orca_retargeting',
        'inspire': 'inspire_retargeting',
        'leap': 'leap_retargeting',
        'nano': 'nano_retargeting'
    }

    executable = retargeting_map.get(hand_type)

    if not executable:
        raise ValueError(f"Invalid hand_type '{hand_type}'. Must be one of: orca, inspire, leap")

    # Get the path to the URDF file (must be done inside OpaqueFunction)
    if hand_type == 'inspire':
        urdf_file = f'{hand_type}_hand_left.urdf'
    else:
        urdf_file = f'{hand_type}_hand_right.urdf'
    urdf_path = PathJoinSubstitution([
        get_package_share_directory('hands'),
        'urdf',
        hand_type,
        urdf_file
    ])

    # robot_desc = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)
    urdf_path_str = urdf_path.perform(context)

    robot_desc = ParameterValue(
        Command(['xacro ', urdf_path_str]),
        value_type=str
    ) 

    # Robot state publisher - publishes URDF and TF transforms
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    # Retargeting node
    retargeting_node = Node(
        package='hands',
        executable=executable,
        name=f'{hand_type}_retargeting',
        output='screen'
    )

    return [robot_state_publisher, retargeting_node]


def generate_launch_description():
    # Declare launch arguments
    hand_type_arg = DeclareLaunchArgument(
        'hand_type',
        default_value='orca',
        description='Hand type to use (orca, inspire, or leap)'
    )

    control_type_arg = DeclareLaunchArgument(
        'control_type',
        default_value='direct',
        description='Control method (direct, fingertip_ik, or jparse_ik)'
    )

    ip_arg = DeclareLaunchArgument(
        'ip_address',
        default_value='0.0.0.0',
        description='IP address for Rokoko listener (0.0.0.0 = all interfaces)'
    )

    port_arg = DeclareLaunchArgument(
        'port',
        default_value='14043',
        description='UDP port for Rokoko Studio stream'
    )

    vive_ip_arg = DeclareLaunchArgument(
        'vive_ip',
        default_value='0.0.0.0',
        description='IP address for VIVE listener (0.0.0.0 = all interfaces)'
    )

    vive_port_arg = DeclareLaunchArgument(
        'vive_port',
        default_value='9001',
        description='UDP port for VIVE tracker stream'
    )

    hardware_mode_arg = DeclareLaunchArgument(
        'hardware_mode',
        default_value='false',
        description='Enable hardware control for hand (true/false)'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to hand model (required if hardware_mode=true)'
    )

    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='',
        description='Serial port for hardware control (required if hardware_mode=true)'
    )

    calibrate_arg = DeclareLaunchArgument(
        'calibrate',
        default_value='false',
        description='Calibrate hand on startup (true/false)'
    )


    # RViz2 for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(get_package_share_directory('hands'), 'rviz', 'view_hand.rviz')]
    )

    # Visualization node - publishes hand_type and control_type
    visualization_node = Node(
        package='hands',
        executable='visualization_node',
        name='visualization_node',
        output='screen',
        parameters=[{
            'hand_type': LaunchConfiguration('hand_type'),
            'control_type': LaunchConfiguration('control_type')
        }]
    )

    # Rokoko listener node - receives UDP data and publishes to topics
    rokoko_listener_node = Node(
        package='hands',
        executable='rokoko_listener',
        name='rokoko_listener',
        output='screen',
        parameters=[{
            'ip_address': LaunchConfiguration('ip_address'),
            'port': LaunchConfiguration('port'),
            'publish_rate': 30.0
        }]
    )

    vive_tracker_node = Node(
        package='hands',
        executable='vive_listener',
        name='vive_listener',
        output='screen',
        parameters=[{
            'vive_ip': LaunchConfiguration('vive_ip'),
            'vive_port': LaunchConfiguration('vive_port'),
            'target_serial': 'ANY', # specify tracker serial number here if needed
            'publish_rate': 30.0
        }]
    )

    # Control type publisher node
    control_type_publisher = Node(
        package='hands',
        executable='control_type_publisher',
        name='control_type_publisher',
        output='screen',
        parameters=[{
            'control_method': LaunchConfiguration('control_type')
        }]
    )

    # Hardware control node for ORCA hand (optional)
    orca_hardware_node = Node(
        package='hands',
        executable='orca_hardware_control',
        name='orca_hardware_control',
        output='screen',
        condition=IfCondition(
            PythonExpression([
                "'", LaunchConfiguration('hardware_mode'), "' == 'true' and ",
                "'", LaunchConfiguration('hand_type'), "' == 'orca'"
            ])
        ),
        parameters=[{
            'hardware_mode': LaunchConfiguration('hardware_mode'),
            'model_path': LaunchConfiguration('model_path'),
            'calibrate': LaunchConfiguration('calibrate')
        }]
    )

    # Hardware control node for leap hand (optional)
    leap_hardware_node = Node(
        package='hands',
        executable='leap_hardware_control',
        name='leap_hardware_control',
        output='screen',
        condition=IfCondition(
            PythonExpression([
                "'", LaunchConfiguration('hardware_mode'), "' == 'true' and ",
                "'", LaunchConfiguration('hand_type'), "' == 'leap'"
            ])
        ),
        parameters=[{
            'hardware_mode': LaunchConfiguration('hardware_mode'),
            'serial_port': LaunchConfiguration('serial_port'),
            'calibrate': LaunchConfiguration('calibrate')
        }]
    )

    # Hardware control node for inspire hand (optional)
    inspire_hardware_node = Node(
        package='hands',
        executable='inspire_hardware_control',
        name='inspire_hardware_control',
        output='screen',
        condition=IfCondition(
            PythonExpression([
                "'", LaunchConfiguration('hardware_mode'), "' == 'true' and ",
                "'", LaunchConfiguration('hand_type'), "' == 'inspire'"
            ])
        ),
        parameters=[{
            'hardware_mode': LaunchConfiguration('hardware_mode'),
            'serial_port': LaunchConfiguration('serial_port'),
            'calibrate': LaunchConfiguration('calibrate')
        }]
    )

    # Conditionally launch the appropriate retargeting node and robot_state_publisher
    dynamic_launch = OpaqueFunction(function=launch_setup)

    return LaunchDescription([
        hand_type_arg,
        control_type_arg,
        ip_arg,
        port_arg,
        vive_ip_arg,
        vive_port_arg,
        hardware_mode_arg,
        model_path_arg,
        serial_port_arg,
        calibrate_arg,
        rviz_node,
        visualization_node,
        rokoko_listener_node,
        vive_tracker_node,
        control_type_publisher,
        orca_hardware_node,
        leap_hardware_node,
        inspire_hardware_node,
        dynamic_launch
    ])