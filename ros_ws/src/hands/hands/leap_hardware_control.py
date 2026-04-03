#!/usr/bin/env python3
"""
LEAP Hand Hardware Controller for ROS2
Interfaces with Dynamixel motors using the LEAP SDK utility files.
"""

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState

# Ensure these files are in your PYTHONPATH or your package folder
from leap_hand_utils.dynamixel_client import DynamixelClient
import leap_hand_utils.leap_hand_utils as lhu

class LeapHardwareController(Node):
    def __init__(self):
        super().__init__('leap_hardware_controller')

        # 1. Parameters & Configuration
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 4000000) # LEAP standard is 4M
        self.declare_parameter('curr_lim', 550)      # 350 for Lite, 550 for Full
        
        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value
        self.curr_lim = self.get_parameter('curr_lim').get_parameter_value().integer_value

        self.motor_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.kP = 600
        self.kI = 0
        self.kD = 200

        # 2. Initialize Dynamixel Client
        try:
            self.dxl_client = DynamixelClient(self.motor_ids, serial_port, baud_rate)
            self.dxl_client.connect()
            self.setup_motors()
            self.get_logger().info(f'LEAP Hand connected on {serial_port} at {baud_rate} baud.')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to LEAP Hand: {e}')
            self.dxl_client = None

        # 3. Joint Mapping (LEAP order: Index 0-3, Middle 4-7, Ring 8-11, Thumb 12-15)
        self.joint_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        # self.joint_names = [
        #     '1','0','2','3',      # index (abduction, MCP, PIP, DIP)
        #     '5','4','6','7',      # middle (abduction, MCP, PIP, DIP)
        #     '9','8','10','11',    # ring (abduction, MCP, PIP, DIP)
        #     '12','13','14','15'   # thumb (abduction, MCP, PIP, DIP)
        # ]

        self.joint_labels = [
            "Index Abduction", "Index MCP", "Index PIP", "Index DIP",
            "Middle Abduction", "Middle MCP", "Middle PIP", "Middle DIP",
            "Ring Abduction", "Ring MCP", "Ring PIP", "Ring DIP",
            "Thumb Abduction", "Thumb MCP", "Thumb PIP", "Thumb DIP"
        ]

        # self.test_index = 0
        # self.test_direction = 1
        # self.test_timer = self.create_timer(1.5, self.run_joint_test)

        # 4. ROS Subscription
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

    def setup_motors(self):
        """Configure motor gains and limits according to LEAP example code"""
        # Set Operating Mode (Position-Current Control)
        self.dxl_client.sync_write(self.motor_ids, np.ones(16)*5, 11, 1)
        
        # Set Gains
        self.dxl_client.sync_write(self.motor_ids, np.ones(16) * self.kP, 84, 2)
        self.dxl_client.sync_write(self.motor_ids, np.ones(16) * self.kI, 82, 2)
        self.dxl_client.sync_write(self.motor_ids, np.ones(16) * self.kD, 80, 2)
        
        # Reduced gains for side-to-side (abduction) motors for stability
        abduction_motors = [0, 4, 8]
        self.dxl_client.sync_write(abduction_motors, np.ones(3) * (self.kP * 0.75), 84, 2)
        self.dxl_client.sync_write(abduction_motors, np.ones(3) * (self.kD * 0.75), 80, 2)

        # Set Current Limit
        self.dxl_client.sync_write(self.motor_ids, np.ones(16) * self.curr_lim, 102, 2)
        
        # Enable Torque
        self.dxl_client.set_torque_enabled(self.motor_ids, True)

    # def run_joint_test(self):
    #     if not self.dxl_client:
    #         return

    #     # 1. Start with the hardware 'Home' (pi radians)
    #     leap_pose = np.ones(16) * np.pi  

    #     # 2. Apply the Calibration Offset
    #     # If your hand points 90 degrees forward at pi, we subtract 1.57 rad
    #     # to bring it back to an upright/flat position.
    #     # MCP indices for Index, Middle, Ring are 1, 5, and 9.
    #     mcp_offset = -np.pi/2  # -90 degrees in radians
    #     mcp_indices = [1, 5, 9]
    #     for idx in mcp_indices:
    #         leap_pose[idx] += mcp_offset

    #     # 3. Perform the Joint-by-Joint Test
    #     # We move the current 'test_index' slightly to see it move
    #     test_magnitude = 0.4 * self.test_direction
    #     leap_pose[self.test_index] += test_magnitude

    #     # 4. Safety Clip before sending to motors
    #     safe_pose = lhu.angle_safety_clip(leap_pose, type="modified")

    #     try:
    #         self.dxl_client.write_desired_pos(self.motor_ids, safe_pose)
    #     except Exception as e:
    #         self.get_logger().error(f'Write error: {e}')
    #         return

    #     # Print info so you can verify which motor is which
    #     self.get_logger().info(
    #         f"""
    #         ======== JOINT TEST ========
    #         Array Index : {self.test_index}
    #         Motor ID    : {self.motor_ids[self.test_index]}
    #         Label       : {self.joint_labels[self.test_index]}
    #         Value       : {safe_pose[self.test_index]:.3f} rad
    #         ============================
    #         """
    #     )

    #     # Advance to the next joint
    #     self.test_index += 1
    #     if self.test_index >= 16:
    #         self.test_index = 0
    #         self.test_direction *= -1  # Reverse wiggle direction for the next loop
    #         self.get_logger().info("---- Starting next loop (Reversed) ----")

    def joint_state_callback(self, msg):
        if not self.dxl_client:
            return

        joint_map = {name: pos for name, pos in zip(msg.name, msg.position)}
        # Build 16-joint array in Allegro order
        allegro_pose = np.zeros(16)
        for i, name in enumerate(self.joint_names):
            if name in joint_map:
                allegro_pose[i] = joint_map[name]

        # Convert to LEAP hardware convention (Home = 3.14159)
        # Using lhu.allegro_to_LEAPhand handles the offset automatically
        leap_pose = lhu.allegro_to_LEAPhand(allegro_pose, zeros=False)
        mcp_indices = [1, 5, 9]
        for idx in mcp_indices:
            leap_pose[idx] -= np.pi/2 # 90 degree offset for MCP joints on hardware
        # Safety Clip (from leap_hand_utils.py)
        safe_pose = lhu.angle_safety_clip(leap_pose, type="modified")

        # Write to hardware
        try:
            self.dxl_client.write_desired_pos(self.motor_ids, safe_pose)
        except Exception as e:
            self.get_logger().error(f'Error writing to motors: {e}')

    def destroy_node(self):
        if self.dxl_client:
            self.dxl_client.set_torque_enabled(self.motor_ids, False)
            self.dxl_client.disconnect()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LeapHardwareController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()