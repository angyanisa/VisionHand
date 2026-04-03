#!/usr/bin/env python3
"""
Inspire Hand Hardware Controller

This node subscribes to joint_states from the retargeting node and sends
commands to the real Inspire hand hardware via serial communication.
"""

import rclpy
from rclpy.node import Node
import serial
from sensor_msgs.msg import JointState


class InspireHardwareController(Node):
    def __init__(self):
        super().__init__('inspire_hardware_controller')

        # Declare parameters
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('hand_id', 1)

        self.serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        self.baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value
        self.hand_id = self.get_parameter('hand_id').get_parameter_value().integer_value

        # Serial connection
        self.serial_conn = None
        self.connect_serial()

        # Joint limits (radians) for normalizing to [0, 1]
        # These match the limits from inspire_retargeting.py

        # 19 degrees to 176.7 degrees (index to pinky)
        # -13 degrees to 53.6 degrees (thumb bending)
        # 90 degrees to 165 degrees (thumb rotation)
        # self.joint_limits = {
        #     'thumb_proximal_yaw_joint': (1.57, 2.88),          # 90 to 165 degrees
        #     'thumb_proximal_pitch_joint': (-0.227, 0.935),     # -13 to 53.6 degrees
        #     'thumb_intermediate_joint': (0.0, 0.935),          # 0 to 53.6 degrees
        #     'thumb_distal_joint': (0.0, 0.698),                # 0 to 40 degrees
        #     'index_proximal_joint': (0.332, 3.09),          # 19 to 176.7 degrees
        #     'index_intermediate_joint': (0.0, 3.09),
        #     'middle_proximal_joint': (0.332, 3.09),
        #     'middle_intermediate_joint': (0.0, 3.09),
        #     'ring_proximal_joint': (0.332, 3.09),
        #     'ring_intermediate_joint': (0.0, 3.09),
        #     'pinky_proximal_joint': (0.332, 3.09),
        #     'pinky_intermediate_joint': (0.0, 3.09),
        # }
        self.joint_limits = {
            # THUMB
            'thumb_proximal_yaw_joint': (0.0, 1.308),      # rotation
            'thumb_proximal_pitch_joint': (0.0, 0.6),
            'thumb_intermediate_joint': (0.0, 0.8),
            'thumb_distal_joint': (0.0, 0.4),

            # FINGERS (proximal + intermediate only; distal is mechanically coupled)
            'index_proximal_joint': (0.0, 1.47),
            'index_intermediate_joint': (-0.04545, 1.56),

            'middle_proximal_joint': (0.0, 1.47),
            'middle_intermediate_joint': (-0.04545, 1.56),

            'ring_proximal_joint': (0.0, 1.47),
            'ring_intermediate_joint': (-0.04545, 1.56),

            'pinky_proximal_joint': (0.0, 1.47),
            'pinky_intermediate_joint': (-0.04545, 1.56),
        }


        # Mapping from URDF joints (12) to hardware joints (6)
        self.hardware_joint_mapping = {
            0: ['pinky_proximal_joint', 'pinky_intermediate_joint'],
            1: ['ring_proximal_joint', 'ring_intermediate_joint'],
            2: ['middle_proximal_joint', 'middle_intermediate_joint'],
            3: ['index_proximal_joint', 'index_intermediate_joint'],

            # Thumb motors (confirmed order for your hardware)
            
            4: ['thumb_proximal_yaw_joint'],  # ROTATION motor
            5: [  # BENDING tendon motor
                'thumb_proximal_pitch_joint',
                'thumb_intermediate_joint',
                'thumb_distal_joint'
            ],
        }

        self.joint_gains = {
            'thumb_proximal_yaw_joint': 3.0,   # BOOST rotation
            'thumb_proximal_pitch_joint': 1.5,
            'thumb_intermediate_joint': 1.5,
            'thumb_distal_joint': 1.5,
        }

        # Subscribe to joint states from retargeting
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.get_logger().info(f'Inspire Hardware Controller started on {self.serial_port}')

    def connect_serial(self):
        """Establish serial connection to Inspire hand"""
        try:
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.05
            )
            self.get_logger().info(f'Connected to Inspire hand on {self.serial_port}')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to Inspire hand: {e}')
            self.serial_conn = None

    def normalize_joint(self, joint_name, value):
        """Convert joint angle (radians) to normalized [0, 1] range"""
        if joint_name not in self.joint_limits:
            return 0.5

        lower, upper = self.joint_limits[joint_name]
        value = max(lower, min(upper, value))
        if upper - lower > 0:
            return (value - lower) / (upper - lower)
        return 0.0
    
    def combine_and_normalize(self, joint_list, joint_positions):
        total = 0.0
        total_min = 0.0
        total_max = 0.0

        for j in joint_list:
            if j in joint_positions and j in self.joint_limits:
                lower, upper = self.joint_limits[j]
                val = joint_positions[j]
                gain = self.joint_gains.get(j, 1.0)
                val = val * gain
                val = max(lower, min(upper, val))
                total += val
                total_min += lower
                total_max += upper

        if total_max > total_min:
            norm = (total - total_min) / (total_max - total_min)
            return max(0.0, min(1.0, norm))

        return 0.0

    def data2bytes(self, data):
        """Convert integer to 2 bytes (little-endian)"""
        if data == -1:
            return [0xFF, 0xFF]
        return [int(data) & 0xFF, (int(data) >> 8) & 0xFF]

    def num2str(self, num):
        """Convert number to hex byte"""
        return bytes.fromhex(f'{num:02x}')

    def checknum(self, data, leng):
        """Calculate checksum"""
        return sum(data[2:leng]) & 0xFF

    def send_joint_positions(self, positions):
        """
        Send joint positions to Inspire hand.

        Args:
            positions: list of 6 normalized positions [0.0, 1.0]
        """
        if not self.serial_conn:
            self.get_logger().warn('Serial connection not available', throttle_duration_sec=5.0)
            return

        try:
            datanum = 0x0F
            b = [0] * (datanum + 5)

            b[0] = 0xEB
            b[1] = 0x90
            b[2] = self.hand_id
            b[3] = datanum
            b[4] = 0x12
            # b[5] = 0xC2
            b[5] = 0xCE # use ANGLE_SET
            b[6] = 0x05

            # def map_position(p, idx):
            #     p = max(0.0, min(1.0, p))
            #     if idx in [5]:  # Thumb joints
            #         return int(p * 1000)
            #     else:
            #         return int(1000 - p * 1000)
            def map_position(p, idx):
                p = max(0.0, min(1.0, p))
                return int(1000 - p * 1000)  # 0=open, 1000=closed (Inspire convention)

            scaled_positions = [map_position(p, i) for i, p in enumerate(positions)]

            for i in range(6):
                dbytes = self.data2bytes(scaled_positions[i]) # little-endian
                b[7 + i*2] = dbytes[0]
                b[8 + i*2] = dbytes[1]

            b[19] = self.checknum(b, datanum + 4)

            putdata = b''
            for byte in b:
                putdata += self.num2str(byte)

            self.serial_conn.write(putdata)
            self.get_logger().debug(f'Sent joint command: {scaled_positions}')

        except Exception as e:
            self.get_logger().error(f'Failed to send joint positions: {e}')

    def joint_state_callback(self, msg):
        joint_positions = {name: pos for name, pos in zip(msg.name, msg.position)}
        hardware_positions = [0.0] * 6

        for hw_idx, urdf_joints in self.hardware_joint_mapping.items():
            hardware_positions[hw_idx] = self.combine_and_normalize(urdf_joints, joint_positions)

        # Debug print (remove later if spammy)
        # self.get_logger().info(f'HW normalized: {[round(p, 2) for p in hardware_positions]}')

        self.send_joint_positions(hardware_positions)

    # def joint_state_callback(self, msg):
    #     """Process joint states and send to hardware"""
    #     joint_positions = {}
    #     for name, position in zip(msg.name, msg.position):
    #         joint_positions[name] = position

    #     hardware_positions = [0.0] * 6

    #     for hw_idx, urdf_joints in self.hardware_joint_mapping.items():
    #         normalized_values = []
    #         for joint_name in urdf_joints:
    #             if joint_name in joint_positions:
    #                 norm_val = self.normalize_joint(joint_name, joint_positions[joint_name])
    #                 normalized_values.append(norm_val)
    #                 # normalized_values.append(joint_positions[joint_name])
    #                 self.get_logger().info(f'Joint name: {joint_name}, position: {joint_positions[joint_name]:.3f}, normalized: {norm_val:.3f}')
    #         if normalized_values:
    #             hardware_positions[hw_idx] = sum(normalized_values) / len(normalized_values)
    #     # self.send_joint_positions(hardware_positions)
    #     self.send_joint_positions([0,0,0,0,0,0])  # TESTING ONLY


    def destroy_node(self):
        """Clean up serial connection on shutdown"""
        if self.serial_conn:
            self.serial_conn.close()
            self.get_logger().info('Serial connection closed')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    controller = InspireHardwareController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Inspire Hardware Controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()