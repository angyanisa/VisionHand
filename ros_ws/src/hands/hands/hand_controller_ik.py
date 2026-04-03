import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import math
import time
import numpy as np
import pandas as pd
import pybullet as p  # Only for IK calculations, not visualization


class HandControllerIK(Node):
    def __init__(self):
        super().__init__('hand_controller_ik')

        # Declare parameters
        self.declare_parameter('hand_name', 'orca')
        self.declare_parameter('csv_file', '')
        self.declare_parameter('use_ik', True)
        self.declare_parameter('loop', True)
        self.declare_parameter('frame_rate', 30)

        hand_name = self.get_parameter('hand_name').get_parameter_value().string_value
        csv_file = self.get_parameter('csv_file').get_parameter_value().string_value
        self.use_ik = self.get_parameter('use_ik').get_parameter_value().bool_value
        self.loop = self.get_parameter('loop').get_parameter_value().bool_value
        frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value

        self.get_logger().info(f"Controlling joints for: {hand_name}")
        self.get_logger().info(f"CSV file: {csv_file}")
        self.get_logger().info(f"Use IK: {self.use_ik}")

        # Joint names for ORCA hand
        self.joint_names = [
            'right_wrist',
            'right_thumb_mcp', 'right_thumb_abd', 'right_thumb_pip', 'right_thumb_dip',
            'right_index_abd', 'right_index_mcp', 'right_index_pip',
            'right_middle_abd', 'right_middle_mcp', 'right_middle_pip',
            'right_ring_abd', 'right_ring_mcp', 'right_ring_pip',
            'right_pinky_abd', 'right_pinky_mcp', 'right_pinky_pip'
        ]

        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'fingertip_targets', 10)

        # Load CSV data
        if csv_file:
            try:
                self.data = pd.read_csv(csv_file)
                self.get_logger().info(f"Loaded {len(self.data)} frames from CSV")
                self.current_frame = 0
            except Exception as e:
                self.get_logger().error(f"Failed to load CSV: {e}")
                self.data = None
                self.current_frame = 0
        else:
            self.data = None
            self.current_frame = 0

        # Initialize PyBullet for IK (headless mode)
        if self.use_ik and self.data is not None:
            self.init_pybullet_ik()
        else:
            self.robot_id = None

        # Setup coordinate transformations
        self.tip_position_mapping = {
            'thumb': ['RightDigit1DistalPhalanx_position_x', 'RightDigit1DistalPhalanx_position_y', 'RightDigit1DistalPhalanx_position_z'],
            'index': ['RightDigit2DistalPhalanx_position_x', 'RightDigit2DistalPhalanx_position_y', 'RightDigit2DistalPhalanx_position_z'],
            'middle': ['RightDigit3DistalPhalanx_position_x', 'RightDigit3DistalPhalanx_position_y', 'RightDigit3DistalPhalanx_position_z'],
            'ring': ['RightDigit4DistalPhalanx_position_x', 'RightDigit4DistalPhalanx_position_y', 'RightDigit4DistalPhalanx_position_z'],
            'pinky': ['RightDigit5DistalPhalanx_position_x', 'RightDigit5DistalPhalanx_position_y', 'RightDigit5DistalPhalanx_position_z']
        }

        # Joint mapping from CSV columns to ROS joint names (for direct joint angle control)
        self.csv_to_ros_joint_mapping = {
            'RightWrist_flexion': 'right_wrist',
            'RightDigit1Carpometacarpal_flexion': 'right_thumb_mcp',
            'RightDigit1Carpometacarpal_ulnarDeviation': 'right_thumb_abd',
            'RightDigit1Metacarpophalangeal_flexion': 'right_thumb_pip',
            'RightDigit1Interphalangeal_flexion': 'right_thumb_dip',
            'RightDigit2Metacarpophalangeal_ulnarDeviation': 'right_index_abd',
            'RightDigit2Metacarpophalangeal_flexion': 'right_index_mcp',
            'RightDigit2ProximalInterphalangeal_flexion': 'right_index_pip',
            'RightDigit3Metacarpophalangeal_ulnarDeviation': 'right_middle_abd',
            'RightDigit3Metacarpophalangeal_flexion': 'right_middle_mcp',
            'RightDigit3ProximalInterphalangeal_flexion': 'right_middle_pip',
            'RightDigit4Metacarpophalangeal_ulnarDeviation': 'right_ring_abd',
            'RightDigit4Metacarpophalangeal_flexion': 'right_ring_mcp',
            'RightDigit4ProximalInterphalangeal_flexion': 'right_ring_pip',
            'RightDigit5Metacarpophalangeal_ulnarDeviation': 'right_pinky_abd',
            'RightDigit5Metacarpophalangeal_flexion': 'right_pinky_mcp',
            'RightDigit5ProximalInterphalangeal_flexion': 'right_pinky_pip'
        }

        # Timer for publishing
        timer_period = 1.0 / frame_rate
        self.timer = self.create_timer(timer_period, self.publish_callback)
        self.get_logger().info(f'Hand Controller IK node started at {frame_rate} Hz')

    def init_pybullet_ik(self):
        """Initialize PyBullet in DIRECT mode (no GUI) for IK calculations only"""
        self.physics_client = p.connect(p.DIRECT)

        # Load ORCA hand URDF
        urdf_path = "/home/shalika/Desktop/orca_core/orcahand_description/models/urdf/orcahand_right.urdf"
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)

        # Find palm link
        self.palm_link_id = None
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode("utf-8")
            if link_name == "right_palm":
                self.palm_link_id = i
                break

        if self.palm_link_id is None:
            self.get_logger().error("Palm link not found!")
            return

        # Fingertip link indices
        self.link_indices = [12, 19, 26, 33, 40]  # thumb, index, middle, ring, pinky

        # IK parameters (from your working code)
        self.lower_limits = [
            -0.67360,  # wrist
            -0.87266, -1.08211, -0.79440, -0.85384,  # thumb
            -1.04577, -0.34907, -0.34907,  # index
            -0.64577, -0.34907, -0.34907,  # middle
            -0.47577, -0.34907, -0.34907,  # ring
            -0.12244, -0.34907, -0.34907   # pinky
        ]
        self.upper_limits = [
            0.89720,  # wrist
            0.87266, 0.00000, 1.23000, 1.45000,  # thumb
            0.24577, 1.65806, 1.88496,  # index
            0.64577, 1.58825, 1.86750,  # middle
            0.80577, 1.58825, 1.86750,  # ring
            1.16910, 1.71042, 1.88496   # pinky
        ]

        # Rest poses from config.yaml
        self.rest_poses = [
            math.radians(0),    # wrist
            math.radians(-13), math.radians(43), math.radians(33), math.radians(19),  # thumb
            math.radians(25), math.radians(0), math.radians(0),    # index
            math.radians(-2), math.radians(0), math.radians(0),    # middle
            math.radians(-20), math.radians(-1), math.radians(0),  # ring
            math.radians(-55), math.radians(1), math.radians(0),   # pinky
        ]

        self.joint_ranges = [u - l for u, l in zip(self.upper_limits, self.lower_limits)]

        # Finger to joint index mapping
        self.finger_joint_indices = {
            0: [4, 6, 8, 10],    # thumb
            1: [13, 15, 17],     # index
            2: [20, 22, 24],     # middle
            3: [27, 29, 31],     # ring
            4: [34, 36, 38]      # pinky
        }

        self.get_logger().info("PyBullet IK initialized (headless mode)")

    def rotation_rokoko_to_orca(self, vec):
        """Transform Rokoko coordinates to ORCA coordinates"""
        rx, ry, rz = vec
        return np.array([rx, rz, ry], dtype=float)

    def get_joint_angles_from_csv(self, row):
        """Extract joint angles directly from CSV (no IK)"""
        joint_angles_dict = {}

        # Extract joint angles from CSV columns
        for csv_col, ros_joint in self.csv_to_ros_joint_mapping.items():
            if csv_col in row.index and pd.notna(row[csv_col]):
                angle_deg = float(row[csv_col])
                angle_rad = math.radians(angle_deg)
                joint_angles_dict[ros_joint] = angle_rad

        # Convert to ordered list matching self.joint_names
        joint_angles = []
        for joint_name in self.joint_names:
            if joint_name in joint_angles_dict:
                joint_angles.append(joint_angles_dict[joint_name])
            else:
                joint_angles.append(0.0)  # Default to 0 if not found

        # Also extract fingertip targets for visualization
        targets = {}
        for finger_name, pos_columns in self.tip_position_mapping.items():
            if all(col in row.index and pd.notna(row[col]) for col in pos_columns):
                rokoko_tip = np.array([row[col] for col in pos_columns], dtype=float)
                orca_tip = self.rotation_rokoko_to_orca(rokoko_tip)
                targets[finger_name] = orca_tip

        return joint_angles, targets

    def compute_ik_from_frame(self, row):
        """Compute joint angles using IK from CSV frame data"""
        if self.robot_id is None:
            return None

        # Get palm position
        palm_cols = ("RightHand_position_x", "RightHand_position_y", "RightHand_position_z")
        if not all(c in row.index for c in palm_cols):
            return None

        # Step PyBullet simulation to update link states
        p.stepSimulation()

        # Get ORCA palm world position from PyBullet
        palm_link_info = p.getLinkState(self.robot_id, self.palm_link_id, computeForwardKinematics=True)
        orca_palm_in_world = np.array(palm_link_info[0], dtype=float)

        # Get Rokoko palm position
        rokoko_palm = np.array([row[c] for c in palm_cols], dtype=float)
        orca_palm = self.rotation_rokoko_to_orca(rokoko_palm)

        # Compute fingertip targets in world frame
        orca_tips_in_world = {}
        for finger_name, pos_columns in self.tip_position_mapping.items():
            if all(col in row.index for col in pos_columns):
                rokoko_tip = np.array([row[col] for col in pos_columns], dtype=float)
                orca_tip = self.rotation_rokoko_to_orca(rokoko_tip)
                orca_tip_in_palm = orca_tip - orca_palm
                orca_tips_in_world[finger_name] = orca_tip_in_palm + orca_palm_in_world

        # Run IK for each finger
        finger_order = ["thumb", "index", "middle", "ring", "pinky"]

        # Map PyBullet joint indices to ROS joint name indices
        # ROS order: [wrist, thumb_mcp, thumb_abd, thumb_pip, thumb_dip,
        #             index_abd, index_mcp, index_pip, middle_abd, middle_mcp, middle_pip,
        #             ring_abd, ring_mcp, ring_pip, pinky_abd, pinky_mcp, pinky_pip]
        pybullet_to_ros_index = {
            2: 0,   # wrist
            4: 1, 6: 2, 8: 3, 10: 4,     # thumb
            13: 5, 15: 6, 17: 7,          # index
            20: 8, 22: 9, 24: 10,         # middle
            27: 11, 29: 12, 31: 13,       # ring
            34: 14, 36: 15, 38: 16        # pinky
        }

        joint_angles = [0.0] * 17  # Initialize all joints

        for i, finger_name in enumerate(finger_order):
            if finger_name not in orca_tips_in_world:
                continue

            target_pos = orca_tips_in_world[finger_name]

            try:
                ik_result = p.calculateInverseKinematics(
                    self.robot_id,
                    self.link_indices[i],
                    target_pos,
                    lowerLimits=self.lower_limits,
                    upperLimits=self.upper_limits,
                    jointRanges=self.joint_ranges,
                    restPoses=self.rest_poses,
                    solver=p.IK_SDLS,
                    maxNumIterations=200,
                    residualThreshold=1e-4
                )

                # Extract joint angles for this finger and map to ROS indices
                pybullet_joints = self.finger_joint_indices[i]
                for pb_joint_idx in pybullet_joints:
                    if pb_joint_idx < len(ik_result) and pb_joint_idx in pybullet_to_ros_index:
                        ros_idx = pybullet_to_ros_index[pb_joint_idx]
                        joint_angles[ros_idx] = ik_result[pb_joint_idx]
                        # Update PyBullet state so next IK call uses updated pose
                        p.resetJointState(self.robot_id, pb_joint_idx, ik_result[pb_joint_idx])

            except Exception as e:
                self.get_logger().warn(f"IK failed for {finger_name}: {e}")
                continue

        return joint_angles, orca_tips_in_world

    def publish_markers(self, targets):
        """Publish RViz markers for fingertip targets"""
        marker_array = MarkerArray()

        colors = {
            "thumb": (1.0, 0.0, 0.0),    # Red
            "index": (0.0, 1.0, 0.0),    # Green
            "middle": (0.0, 0.0, 1.0),   # Blue
            "ring": (1.0, 1.0, 0.0),     # Yellow
            "pinky": (1.0, 0.0, 1.0)     # Magenta
        }

        for i, (finger_name, position) in enumerate(targets.items()):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "fingertip_targets"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = float(position[2])
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.01  # 1cm diameter
            marker.scale.y = 0.01
            marker.scale.z = 0.01

            color = colors.get(finger_name, (1.0, 1.0, 1.0))
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def publish_callback(self):
        """Main callback to publish joint states"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        if self.data is not None:
            # Check if we've reached the end
            if self.current_frame >= len(self.data):
                if self.loop:
                    self.current_frame = 0
                    self.get_logger().info("Looping motion data...")
                else:
                    self.get_logger().info("Playback complete")
                    return

            row = self.data.iloc[self.current_frame]

            if self.use_ik:
                # IK mode: compute joint angles from fingertip positions
                result = self.compute_ik_from_frame(row)

                if result is not None:
                    joint_angles, targets = result
                    msg.position = joint_angles

                    # Publish markers
                    self.publish_markers(targets)
                else:
                    msg.position = [0.0] * len(self.joint_names)

            else:
                # Direct joint angle mode: read joint angles from CSV
                joint_angles, targets = self.get_joint_angles_from_csv(row)
                msg.position = joint_angles

                # Publish markers (targets in absolute world coordinates)
                if targets:
                    self.publish_markers(targets)

            self.current_frame += 1

        else:
            # Fallback: simple sine wave animation (no CSV loaded)
            current_time = time.time()
            angle = 0.3 * (1 + math.sin(current_time))
            msg.position = [angle] * len(self.joint_names)

        self.joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    hand_controller = HandControllerIK()

    try:
        rclpy.spin(hand_controller)
    except KeyboardInterrupt:
        pass
    finally:
        if hand_controller.robot_id is not None:
            p.disconnect()
        hand_controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
