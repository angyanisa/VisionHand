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

Hamsa API: hand.wiggle_X(position, time) / hand.curl_X(position, time)
  position is in raw servo units — taken from hamsa.config:
    curl:   out=open position, in=closed position
    wiggle: left / right positions
  We interpolate linearly from URDF joint angle range → servo unit range.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from hamsa import hand


# URDF joint limits (radians) — must match nano_retargeting.py
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

# Servo unit ranges from hamsa.config.
# curl:   (out, in)   → angle lower maps to out (open), upper maps to in (closed)
# wiggle: (left, right) → angle lower maps to left, upper maps to right
SERVO_RANGES = {
    'pinky_curl':    (0,    800 ),
    'ring_curl':     (0,    1023),
    'middle_curl':   (1023, 200 ),
    'index_curl':    (1023, 0   ),
    'thumb_curl':    (0,    800 ),
    'pinky_wiggle':  (550,  450 ),
    'ring_wiggle':   (575,  475 ),
    'middle_wiggle': (550,  450 ),
    'index_wiggle':  (550,  450 ),
    'thumb_wiggle':  (800,  435 ),
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


def angle_to_servo(joint_name, angle_rad):
    """Convert URDF joint angle (radians) to servo position units."""
    lo, hi = JOINT_LIMITS[joint_name]
    servo_lo, servo_hi = SERVO_RANGES[joint_name]

    # Clamp to joint limits
    angle_rad = max(lo, min(hi, angle_rad))

    # Linear interpolation: angle [lo, hi] → servo [servo_lo, servo_hi]
    if hi - lo == 0:
        t = 0.0
    else:
        t = (angle_rad - lo) / (hi - lo)

    return int(round(servo_lo + t * (servo_hi - servo_lo)))


class NanoHardwareControl(Node):
    def __init__(self):
        super().__init__('nano_hardware_control')

        self.declare_parameter('hardware_mode', False)
        self.declare_parameter('move_time', 50)

        self.hardware_mode = self.get_parameter('hardware_mode').get_parameter_value().bool_value
        self.move_time = self.get_parameter('move_time').get_parameter_value().integer_value

        if not self.hardware_mode:
            self.get_logger().info('Hardware mode disabled — not sending commands to hand')
            return

        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.get_logger().info(f'Nano Hardware Control started (move_time={self.move_time}ms)')

    def joint_state_callback(self, msg):
        joint_positions = dict(zip(msg.name, msg.position))

        for joint_name, send_fn in SEND_FUNCTIONS.items():
            if joint_name not in joint_positions:
                continue
            angle = joint_positions[joint_name]
            servo_pos = angle_to_servo(joint_name, angle)
            try:
                send_fn(servo_pos, self.move_time)
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
