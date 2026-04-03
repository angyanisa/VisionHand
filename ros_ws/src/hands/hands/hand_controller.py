import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math
import time

class HandController(Node):
    def __init__(self):
        super().__init__('hand_controller')
        
        self.declare_parameter('hand_name', 'inspire')
        hand_name = self.get_parameter('hand_name').get_parameter_value().string_value
        self.get_logger().info(f"Controlling joints for: {hand_name}")

        joint_map = {
            'inspire': [
                'thumb_proximal_yaw_joint', 'thumb_proximal_pitch_joint',
                'thumb_intermediate_joint', 'thumb_distal_joint',
                'index_proximal_joint', 'index_intermediate_joint',
                'middle_proximal_joint', 'middle_intermediate_joint',
                'ring_proximal_joint', 'ring_intermediate_joint',
                'pinky_proximal_joint', 'pinky_intermediate_joint'
            ],
            'leap': [
                '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'
            ],
            'orca': [
                'right_wrist',
                'right_thumb_mcp', 'right_thumb_abd', 'right_thumb_pip', 'right_thumb_dip',
                'right_index_abd', 'right_index_mcp', 'right_index_pip',
                'right_middle_abd', 'right_middle_mcp', 'right_middle_pip',
                'right_ring_abd', 'right_ring_mcp', 'right_ring_pip',
                'right_pinky_abd', 'right_pinky_mcp', 'right_pinky_pip'
            ]
        }

        self.joint_names = joint_map.get(hand_name)
        if self.joint_names is None:
            self.get_logger().error(f"Hand name '{hand_name}' not found in joint map. Shutting down.")
            self.destroy_node()
            rclpy.shutdown()
            return

        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.1, self.publish_joint_states)
        self.get_logger().info('Hand Controller node has been started.')

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        
        current_time = time.time()
        angle = 0.3 * (1 + math.sin(current_time))
        
        msg.position = [angle] * len(self.joint_names)
        
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    hand_controller = HandController()
    if rclpy.ok():
        rclpy.spin(hand_controller)
    
    if rclpy.ok():
      hand_controller.destroy_node()
      rclpy.shutdown()

if __name__ == '__main__':
    main()
