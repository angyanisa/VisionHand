import rclpy
from ros2_sam.aria_to_sam import AriaToSAM

def main(args=None):
    rclpy.init(args=args)
    node = AriaToSAM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down AriaToSAM node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()