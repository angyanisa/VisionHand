#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')

        # Declare parameters
        self.declare_parameter('hand_type', 'orca')

        # Get parameter values
        self.hand_type = self.get_parameter('hand_type').get_parameter_value().string_value

        # Validate hand_type
        valid_hands = ['orca', 'inspire', 'leap', 'nano', 'nano_physics']
        if self.hand_type not in valid_hands:
            self.get_logger().error(f"Invalid hand_type '{self.hand_type}'. Must be one of: {valid_hands}")
            raise ValueError(f"Invalid hand_type: {self.hand_type}")

        # Create publishers (only hand_type, control_type is handled by control_type_publisher)
        self.hand_type_pub = self.create_publisher(String, 'hand_type', 10)

        # Publish at 10 Hz to ensure subscribers receive the data
        self.timer = self.create_timer(0.1, self.publish_config)

        self.get_logger().info(f'Visualization Node started')
        self.get_logger().info(f'Hand Type: {self.hand_type}')
        self.get_logger().info(f'Publishing to /hand_type topic')

    def publish_config(self):
        """Continuously publish hand type configuration"""
        hand_msg = String()
        hand_msg.data = self.hand_type
        self.hand_type_pub.publish(hand_msg)


def main(args=None):
    rclpy.init(args=args)

    try:
        visualization_node = VisualizationNode()
        rclpy.spin(visualization_node)
    except (KeyboardInterrupt, ValueError):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
