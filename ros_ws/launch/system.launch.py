from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
import os
from pathlib import Path

def generate_launch_description():
    launch_file_dir = Path(__file__).parent
    workspace_root = launch_file_dir.parent  # Goes up to ros_ws/

    aria_path = Path.home() / 'RealtimeProjectAria'
    host_utils_path = aria_path / 'host_pc_utils'

    aria_path = str(aria_path.resolve())
    host_utils_path = str(host_utils_path.resolve())
    
    return LaunchDescription([
        # Start the shell scripts
        ExecuteProcess(
            cmd=[f'{host_utils_path}/start_streaming_usb.sh'],
            cwd=aria_path,
            output='log',
            name='streaming_usb',
            shell=True
        ),

        TimerAction(
            period=15.0,
            actions=[
                ExecuteProcess(
                    cmd=[f'{host_utils_path}/start_ros_publishing.sh'],
                    cwd=aria_path,
                    output='log',
                    name='ros_publishing',
                    shell=True
                )
            ]
        ),

        TimerAction(
            period=17.0,
            actions=[
                ExecuteProcess(
                    cmd=[f'{host_utils_path}/start_eye_track.sh'],
                    cwd=aria_path,
                    output='screen',
                    name='eye_track',
                    shell=True
                )
            ]
        ),
                    
        # Include the SAM server launch file
        TimerAction(
            period=19.0,
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'launch', 'ros2_sam', 'server.launch.py'],
                    output='screen',
                    name='sam_server'
                )
            ]
        ),
        
        # Launch ROS2 nodes
        TimerAction(
            period=21.0,
            actions=[
                Node(
                    package='ros2_sam',
                    executable='aria_to_sam_node',
                    output='screen',
                    name='aria_to_sam'
                ),
                Node(
                    package='ros2_gemini',
                    executable='gemini_image_label',
                    output='screen',
                    name='gemini_label'
                ),
                Node(
                    package='emg_interface',
                    executable='emg_live_classifier',
                    output='screen',
                    name='emg_classifier'
                ),
                Node(
                    package='inspire_hand',
                    executable='inspire_hand_listener',
                    output='screen',
                    name='hand_listener'
                ),
                Node(
                    package='inspire_hand',
                    executable='EMG_to_inspire',
                    output='screen',
                    name='emg_to_inspire'
                ),
            ]
        ),
    ])