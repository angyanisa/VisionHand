#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import serial
import time
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, ReliabilityPolicy

from hamsa import firmware
from hamsa import hand

class NanoHandDriver(Node):
    def __init__(self):
        super().__init__('nano_hand_listener')

        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)

        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.RELIABLE

        self.trajectory_sub = self.create_subscription(
            JointTrajectory,
            '/nano_hand/joint_trajectory',
            self.trajectory_callback,
            qos_profile
        )

        self.joint_names = ['servo1', 'servo2', 'servo3', 'servo4', 'servo5', 'servo6', 'servo7', 'servo8', 'servo9', 'servo10']
        self.current_positions = [0.0] * len(self.joint_names)

        self.timer = self.create_timer(0.1, self.publish_joint_states)

        self.get_logger().info("Nano Hand Driver initialized")
        # self.test_connection()

    # def test_connection(self):
    #     if not self.serial_conn:
    #         return
    #     try:
    #         self.serial_conn.write(b'\x00')
    #         self.get_logger().info("Sent basic test byte to Inspire hand.")
    #     except Exception as e:
    #         self.get_logger().error(f"Connection test failed: {e}")

    def send_joint_positions(self, positions):
        try:
            hand.wiggle_pinky(positions[0], 100)
            hand.curl_pinky(positions[1], 100)
            hand.wiggle_ring(positions[2], 100)
            hand.curl_ring(positions[3], 100)
            hand.wiggle_middle(positions[4], 100)
            hand.curl_middle(positions[5], 100)
            hand.wiggle_index(positions[6], 100)
            hand.curl_index(positions[7], 100)
            hand.wiggle_thumb(positions[8], 100)
            hand.curl_thumb(positions[9], 100)

            self.get_logger().info(f"Sent mapped joint command: {positions}")

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
    driver = NanoHandDriver()
    try:
        rclpy.spin(driver)
    except KeyboardInterrupt:
        driver.get_logger().info("Shutting down Nano Hand Driver")
    finally:
        driver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
