#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, ReliabilityPolicy

from hamsa import firmware
from hamsa import hand


class NanoHandDriver(Node):
    def __init__(self):
        super().__init__('nano_hand_listener')

        self.declare_parameter('move_time', 100)
        self.move_time = self.get_parameter('move_time').get_parameter_value().integer_value

        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)

        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.RELIABLE

        self.trajectory_sub = self.create_subscription(
            JointTrajectory,
            '/nano_hand/joint_trajectory',
            self.trajectory_callback,
            qos_profile
        )

        self.joint_names = ['servo1', 'servo2', 'servo3', 'servo4', 'servo5',
                            'servo6', 'servo7', 'servo8', 'servo9', 'servo10']
        self.current_positions = [0.0] * len(self.joint_names)

        self.timer = self.create_timer(0.1, self.publish_joint_states)

        self.get_logger().info(f"Nano Hand Driver initialized (move_time={self.move_time}ms)")

    def send_joint_positions(self, positions):
        try:
            hand.wiggle_pinky(positions[0], self.move_time)
            hand.curl_pinky(positions[1], self.move_time)
            hand.wiggle_ring(positions[2], self.move_time)
            hand.curl_ring(positions[3], self.move_time)
            hand.wiggle_middle(positions[4], self.move_time)
            hand.curl_middle(positions[5], self.move_time)
            hand.wiggle_index(positions[6], self.move_time)
            hand.curl_index(positions[7], self.move_time)
            hand.wiggle_thumb(positions[8], self.move_time)
            hand.curl_thumb(positions[9], self.move_time)
            self.get_logger().info(f"Sent: {[f'{v:.2f}' for v in positions]}", throttle_duration_sec=0.5)
        except Exception as e:
            self.get_logger().error(f"Failed to send joint positions: {e}")

    def trajectory_callback(self, msg):
        if not msg.points:
            return
        point = msg.points[-1]  # use latest point; EMG_to_nano sends single-point msgs
        self.send_joint_positions(point.positions)
        self.current_positions = list(point.positions)

    def publish_joint_states(self):
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = self.joint_names
        joint_state.position = self.current_positions
        self.joint_state_pub.publish(joint_state)

    def destroy_node(self):
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
