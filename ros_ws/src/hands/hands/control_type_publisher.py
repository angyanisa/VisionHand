#!/usr/bin/env python3
"""
Simple node to publish the control method type to /control_type topic.

This allows retargeting nodes to know which control method to use:
- direct: Direct joint angle mapping
- fingertip_ik: Fingertip IK-based control
- jparse_ik: IK with JPARSE
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ControlTypePublisher(Node):
    def __init__(self):
        super().__init__('control_type_publisher')

        # Declare parameter
        self.declare_parameter('control_method', 'direct')

        # Get parameter
        self.control_method = self.get_parameter('control_method').get_parameter_value().string_value

        # Validate control method
        valid_methods = ['direct', 'fingertip_ik', 'jparse_ik']
        if self.control_method not in valid_methods:
            self.get_logger().error(
                f'Invalid control method: {self.control_method}. '
                f'Valid options: {valid_methods}'
            )
            self.control_method = 'direct'

        # Create publisher
        self.publisher_ = self.create_publisher(String, 'control_type', 10)

        # Publish immediately at startup and then periodically
        self.timer_callback()  # Publish right away
        self.timer = self.create_timer(0.1, self.timer_callback)  # Then every 100ms

        self.get_logger().info('='*60)
        self.get_logger().info(f'Control Type Publisher started')
        self.get_logger().info(f'Control Method: {self.control_method}')
        self.get_logger().info('='*60)

    def timer_callback(self):
        """Periodically publish the control type"""
        msg = String()
        msg.data = self.control_method
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ControlTypePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
