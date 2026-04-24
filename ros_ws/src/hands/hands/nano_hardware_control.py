#!/usr/bin/env python3
"""
Nano Hand Hardware Controller

Subscribes to /joint_states from nano_retargeting and sends commands
to the real Nano hand via the hamsa library.

Joint names from nano_retargeting (10 independent joints):
  pinky_wiggle, pinky_curl
  ring_wiggle,  ring_curl
  middle_wiggle, middle_curl
  index_wiggle, index_curl
  thumb_wiggle, thumb_curl

Hamsa API: hand.wiggle_X(proportion, time) / hand.curl_X(proportion, time)
  proportion is 0.0–1.0:
    curl:   0.0 = open (out position), 1.0 = closed (in position)
    wiggle: 0.0 = left,               1.0 = right
  The firmware maps proportion → raw servo units using hamsa.config internally.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from hamsa import hand


# URDF joint limits (radians) — must match nano_retargeting.py
# curl:   lower = open, upper = closed
# wiggle: lower = left, upper = right
JOINT_LIMITS = {
    'pinky_wiggle':  (-0.17,  0.26 ),
    'pinky_curl':    ( 0.0,   0.97 ),
    'ring_wiggle':   (-0.2,   0.07 ),
    'ring_curl':     ( 0.0,   0.83 ),
    'middle_wiggle': (-0.13,  0.24 ),
    'middle_curl':   ( 0.0,   1.09 ),
    'index_wiggle':  (-0.17,  0.17 ),
    'index_curl':    ( 0.0,   1.06 ),
    'thumb_wiggle':  (-0.78,  1.48 ),
    'thumb_curl':    ( 0.0,   0.68 ),
}

# Maps joint name → hamsa send function
SEND_FUNCTIONS = {
    'pinky_wiggle':  hand.wiggle_pinky,
    'pinky_curl':    hand.curl_pinky,
    'ring_wiggle':   hand.wiggle_ring,
    'ring_curl':     hand.curl_ring,
    'middle_wiggle': hand.wiggle_middle,
    'middle_curl':   hand.curl_middle,
    'index_wiggle':  hand.wiggle_index,
    'index_curl':    hand.curl_index,
    'thumb_wiggle':  hand.wiggle_thumb,
    'thumb_curl':    hand.curl_thumb,
}


def angle_to_proportion(joint_name, angle_rad):
    """Convert URDF joint angle (radians) to hamsa proportion (0.0–1.0).

    Hamsa convention (confirmed from EMG_to_nano.py open_hand sending curl=1.0):
      curl:   0.0 = closed (in position),  1.0 = open (out position)
      wiggle: 0.0 = left,                  1.0 = right

    URDF curl angles go 0.0 (open) → hi (closed), so curl proportions are inverted.
    URDF wiggle angles go lo (left) → hi (right), so wiggle proportions are direct.
    """
    lo, hi = JOINT_LIMITS[joint_name]
    angle_rad = max(lo, min(hi, angle_rad))
    if hi == lo:
        return 0.0
    t = (angle_rad - lo) / (hi - lo)
    return (1.0 - t) if 'curl' in joint_name else t


class NanoHardwareControl(Node):
    def __init__(self):
        super().__init__('nano_hardware_control')

        self.declare_parameter('hardware_mode', False)
        self.declare_parameter('move_time', 50)

        self.hardware_mode = self.get_parameter('hardware_mode').get_parameter_value().bool_value
        self.move_time = self.get_parameter('move_time').get_parameter_value().integer_value

        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        if not self.hardware_mode:
            self.get_logger().info('DRY-RUN mode — logging proportions without sending to hand')
        else:
            self.get_logger().info(f'Nano Hardware Control started (move_time={self.move_time}ms)')

    def joint_state_callback(self, msg):
        joint_positions = dict(zip(msg.name, msg.position))

        proportions = {}
        for joint_name in SEND_FUNCTIONS:
            if joint_name in joint_positions:
                proportions[joint_name] = angle_to_proportion(joint_name, joint_positions[joint_name])

        self.get_logger().info(
            ('SEND' if self.hardware_mode else 'DRY-RUN') + ' proportions: ' +
            ', '.join(f'{k}={v:.2f}' for k, v in proportions.items()),
            throttle_duration_sec=0.5
        )

        if not self.hardware_mode:
            return

        for joint_name, send_fn in SEND_FUNCTIONS.items():
            if joint_name not in proportions:
                continue
            try:
                send_fn(proportions[joint_name], self.move_time)
            except Exception as e:
                self.get_logger().error(f'Failed to send {joint_name}: {e}', throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = NanoHardwareControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
