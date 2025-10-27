#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import serial
import time
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, ReliabilityPolicy

class InspireHandDriver(Node):
    def __init__(self):
        super().__init__('inspire_hand_driver')

        self.serial_port = "/dev/ttyUSB0"
        self.baud_rate = 115200
        self.timeout = 0.05

        try:
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            self.get_logger().info(f"Connected to Inspire hand on {self.serial_port}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to Inspire hand: {e}")
            self.serial_conn = None

        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)

        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.RELIABLE

        self.trajectory_sub = self.create_subscription(
            JointTrajectory,
            '/inspire_hand/joint_trajectory',
            self.trajectory_callback,
            qos_profile
        )

        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.current_positions = [0.0] * len(self.joint_names)

        self.timer = self.create_timer(0.1, self.publish_joint_states)

        self.get_logger().info("Inspire Hand Driver initialized")
        self.test_connection()

    def test_connection(self):
        if not self.serial_conn:
            return
        try:
            self.serial_conn.write(b'\x00')
            self.get_logger().info("Sent basic test byte to Inspire hand.")
        except Exception as e:
            self.get_logger().error(f"Connection test failed: {e}")

    def data2bytes(self, data):
        if data == -1:
            return [0xFF, 0xFF]
        return [int(data) & 0xFF, (int(data) >> 8) & 0xFF]

    def num2str(self, num):
        return bytes.fromhex(f'{num:02x}')

    def checknum(self, data, leng):
        return sum(data[2:leng]) & 0xFF

    def send_joint_positions(self, positions):
        if not self.serial_conn:
            return
        try:
            datanum = 0x0F
            b = [0] * (datanum + 5)

            b[0] = 0xEB
            b[1] = 0x90
            b[2] = 1  # hand_id
            b[3] = datanum
            b[4] = 0x12
            b[5] = 0xC2
            b[6] = 0x05

            # Map from [0.0, 1.0] → [200, 1500]
            def map_position(p):
                p = max(0.0, min(1.0, p))  # clamp to [0, 1]
                return int(1500 - p * (1500 - 200))

            scaled_positions = [map_position(p) for p in positions]

            for i in range(6):
                dbytes = self.data2bytes(scaled_positions[i])
                b[7 + i*2] = dbytes[0]
                b[8 + i*2] = dbytes[1]

            b[19] = self.checknum(b, datanum + 4)

            putdata = b''
            for byte in b:
                putdata += self.num2str(byte)

            self.serial_conn.write(putdata)
            self.get_logger().info(f"Sent mapped joint command: {scaled_positions}")

        except Exception as e:
            self.get_logger().error(f"Failed to send mapped joint positions: {e}")

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Received trajectory with {len(msg.points)} points")
        for point in msg.points:
            self.send_joint_positions(point.positions)
            self.current_positions = point.positions
            time.sleep(0.1)

    def publish_joint_states(self):
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = self.joint_names
        joint_state.position = self.current_positions
        self.joint_state_pub.publish(joint_state)

    def destroy_node(self):
        if self.serial_conn:
            self.serial_conn.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    driver = InspireHandDriver()
    try:
        rclpy.spin(driver)
    except KeyboardInterrupt:
        driver.get_logger().info("Shutting down Inspire Hand Driver")
    finally:
        driver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
