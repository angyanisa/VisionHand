#!/usr/bin/env python3
import json
import socket
import threading
import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

class ViveTrackerNode(Node):
    def __init__(self):
        super().__init__("vive_tracker_node")
        
        # Parameters
        self.declare_parameter("ip_address", "0.0.0.0")
        self.declare_parameter("vive_port", 9001)
        self.declare_parameter("target_serial", "ANY") # serial number of vive tracker, specify if needed
        self.declare_parameter('publish_rate', 30.0)
        
        # Calibration offsets (in radians) to align tracker with hand base
        self.roll_offset = 1.46
        self.pitch_offset = -0.35
        self.yaw_offset = 0.19

        self.ip = self.get_parameter("ip_address").value
        self.vive_port = self.get_parameter("vive_port").value
        self.target_serial = self.get_parameter("target_serial").value
        self.publish_rate = self.get_parameter('publish_rate').value

        # Publisher for the specific wrist tracker
        self.pose_pub = self.create_publisher(PoseStamped, "vive_wrist_pose", 10)

        # UDP listener in background thread
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.vive_port))
        self.get_logger().info(f"Listening for VIVE tracker data on {self.ip}:{self.vive_port}")

        self.thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.thread.start()

    def apply_calibration(self, q_raw):
        """
        q_raw is [x, y, z, w] from the listener's publish_pose logic.
        Returns calibrated [x, y, z, w] for ROS2.
        """
        # Get offsets from parameters
        roll = self.roll_offset
        pitch = self.pitch_offset
        yaw = self.yaw_offset
        
        # 1. Create the raw rotation object from tracker data
        # Note: SciPy uses [x, y, z, w] format by default
        raw_rotation = R.from_quat(q_raw)
        
        # 2. Create the calibration rotation from Euler angles
        calib_rotation = R.from_euler('xyz', [roll, pitch, yaw])
        
        # 3. Combine rotations (Calibration * Raw)
        final_rotation = calib_rotation * raw_rotation
        
        return final_rotation.as_quat() # Returns [x, y, z, w]

    def recv_loop(self):
        while rclpy.ok():
            try:
                data, _ = self.sock.recvfrom(65535)
                msg_data = json.loads(data.decode())
                
                for tracker in msg_data.get("trackers", []):
                    # Only process the specific tracker assigned to the wrist
                    if tracker["serial"] == self.target_serial or self.target_serial == "ANY":
                        self.publish_pose(tracker)
            except Exception as e:
                self.get_logger().warn(f"Parse error: {e}")

    def publish_pose(self, tracker):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world" # Matches the frame used in retargeting

        pos = tracker["pos"]
        # Vive sender usually sends [w, x, y, z], but ROS uses [x, y, z, w]
        raw_quat = [tracker["quat"][1], tracker["quat"][2], tracker["quat"][3], tracker["quat"][0]]

        # Apply calibration if needed
        calibrated_quat = self.apply_calibration(raw_quat)

        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        
        msg.pose.orientation.x = float(calibrated_quat[0])
        msg.pose.orientation.y = float(calibrated_quat[1])
        msg.pose.orientation.z = float(calibrated_quat[2])
        msg.pose.orientation.w = float(calibrated_quat[3])

        self.pose_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ViveTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()