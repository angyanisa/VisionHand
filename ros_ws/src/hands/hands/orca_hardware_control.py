#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

# Hardware connection
try:
    from orca_core import OrcaHand
    ORCA_AVAILABLE = True
except ImportError:
    ORCA_AVAILABLE = False
    print("Warning: orca_core not available. Hardware control will be disabled.")


class OrcaHardwareController(Node):
    """
    Hardware control node for ORCA hand.
    Subscribes to /joint_states and sends commands to real ORCA hardware.
    """

    def __init__(self):
        super().__init__('orca_hardware_controller')

        # Parameters
        self.declare_parameter('hardware_mode', False)
        self.declare_parameter('model_path', '')
        self.declare_parameter('calibrate', False)

        self.hardware_mode = self.get_parameter('hardware_mode').get_parameter_value().bool_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.calibrate_on_start = self.get_parameter('calibrate').get_parameter_value().bool_value

        # Hardware connection
        self.hand = None
        self.hardware_connected = False

        if self.hardware_mode:
            if not ORCA_AVAILABLE:
                self.get_logger().error('Hardware mode enabled but orca_core not available!')
                self.get_logger().error('Install orca_core or disable hardware_mode')
                return

            if not self.model_path:
                self.get_logger().error('Hardware mode enabled but model_path not provided!')
                self.get_logger().error('Set model_path parameter (e.g., /path/to/orcahand_v1_right)')
                return

            self.get_logger().info(f'Connecting to ORCA hand at: {self.model_path}')
            self.connect_to_hardware()
        else:
            self.get_logger().info('Hardware mode disabled - running in simulation only')

        # Joint mapping from ROS joint names (URDF) to ORCA hardware joint names
        # The URDF publishes joints like "right_thumb_mcp" to /joint_states
        # The ORCA hardware expects joints like "thumb_mcp" (without "right_" prefix)
        self.joint_mapping = {
            # Thumb
            'right_thumb_mcp': 'thumb_mcp',
            'right_thumb_abd': 'thumb_abd',
            'right_thumb_pip': 'thumb_pip',
            'right_thumb_dip': 'thumb_dip',

            # Index finger
            'right_index_mcp': 'index_mcp',
            'right_index_abd': 'index_abd',
            'right_index_pip': 'index_pip',

            # Middle finger
            'right_middle_mcp': 'middle_mcp',
            'right_middle_abd': 'middle_abd',
            'right_middle_pip': 'middle_pip',

            # Ring finger
            'right_ring_mcp': 'ring_mcp',
            'right_ring_abd': 'ring_abd',
            'right_ring_pip': 'ring_pip',

            # Pinky
            'right_pinky_mcp': 'pinky_mcp',
            'right_pinky_abd': 'pinky_abd',
            'right_pinky_pip': 'pinky_pip',

            # Wrist (if needed)
            'right_wrist': 'wrist',
        }

        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Debug counter for periodic logging
        self.callback_count = 0
        self.log_every_n = 30  # Log every 30 callbacks (about 1 second at 30Hz)

        self.get_logger().info('ORCA Hardware Controller initialized')
        if self.hardware_mode and self.hardware_connected:
            self.get_logger().info('Hardware control ENABLED and CONNECTED')
        elif self.hardware_mode:
            self.get_logger().warn('Hardware control ENABLED but NOT CONNECTED')
        else:
            self.get_logger().info('Hardware control DISABLED (simulation mode)')

    def connect_to_hardware(self):
        """Connect to the ORCA hand hardware"""
        try:
            self.hand = OrcaHand(self.model_path)
            status = self.hand.connect()

            if not status[0]:
                self.get_logger().error('Failed to connect to ORCA hand!')
                self.hardware_connected = False
                return

            self.get_logger().info('Successfully connected to ORCA hand!')
            self.hardware_connected = True

            # Calibrate if requested
            if self.calibrate_on_start:
                self.get_logger().info('Starting hand calibration...')
                self.hand.calibrate()
                self.get_logger().info('Calibration completed!')

        except Exception as e:
            self.get_logger().error(f'Error connecting to hardware: {e}')
            self.hardware_connected = False

    def joint_state_callback(self, msg: JointState):
        """
        Callback for /joint_states topic.
        Extracts joint positions and sends to hardware.
        """
        self.callback_count += 1

        if not self.hardware_mode or not self.hardware_connected:
            return

        # Extract joint positions
        joint_positions = {}

        for i, joint_name in enumerate(msg.name):
            if joint_name in self.joint_mapping:
                orca_joint_name = self.joint_mapping[joint_name]
                # Convert from radians to degrees (ORCA expects degrees)
                angle_degrees = math.degrees(msg.position[i])
                joint_positions[orca_joint_name] = angle_degrees

        # Debug logging every N callbacks
        if self.callback_count % self.log_every_n == 0:
            self.get_logger().info(f'=== Hardware Control Debug (callback #{self.callback_count}) ===')
            self.get_logger().info(f'Received {len(msg.name)} joints from /joint_states')
            self.get_logger().info(f'Mapped {len(joint_positions)} joints to hardware')
            if joint_positions:
                # Show a few sample joints
                sample_joints = list(joint_positions.items())[:3]
                for joint_name, angle in sample_joints:
                    self.get_logger().info(f'  {joint_name}: {angle:.2f}°')

        # Send to hardware
        if joint_positions:
            try:
                self.hand.set_joint_pos(joint_positions)
            except Exception as e:
                self.get_logger().error(f'Error sending to hardware: {e}')

    def reset_hand(self):
        """Reset hand to neutral position"""
        if self.hand and self.hardware_connected:
            try:
                neutral_positions = {joint: 0 for joint in self.hand.joint_ids}
                self.hand.set_joint_pos(neutral_positions)
                self.get_logger().info('Hand reset to neutral position')
            except Exception as e:
                self.get_logger().error(f'Error resetting hand: {e}')

    def disconnect(self):
        """Disconnect from hardware"""
        if self.hardware_connected:
            self.reset_hand()
            self.get_logger().info('Disconnected from ORCA hand')
            self.hardware_connected = False


def main(args=None):
    rclpy.init(args=args)
    node = OrcaHardwareController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.disconnect()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
