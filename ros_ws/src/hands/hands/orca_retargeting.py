#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import pybullet as p
import numpy as np
import math
import os

# Import jparse for advanced IK solving
try:
    import jparse_robotics as jparse
    JPARSE_AVAILABLE = True
except ImportError:
    JPARSE_AVAILABLE = False
    print("Warning: jparse not available. Install with: pip install git+https://github.com/armlabstanford/jparse.git")


class OrcaRetargeting(Node):
    def __init__(self):
        super().__init__('orca_retargeting')

        # Declare parameter for data format (degrees vs radians)
        self.declare_parameter('data_in_degrees', False)
        self.data_in_degrees = self.get_parameter('data_in_degrees').get_parameter_value().bool_value

        # Declare parameter for using joint limits
        self.declare_parameter('use_joint_limits', True)
        self.use_joint_limits = self.get_parameter('use_joint_limits').get_parameter_value().bool_value

        if self.data_in_degrees:
            self.get_logger().info('Data format: DEGREES (will convert to radians)')
        else:
            self.get_logger().info('Data format: RADIANS (no conversion needed)')

        if self.use_joint_limits:
            self.get_logger().info('Joint limits: ENABLED (will clamp to safe ranges)')
        else:
            self.get_logger().info('Joint limits: DISABLED')

        # # ORCA hand joint names
        # self.joint_names = [
        #     'right_wrist',
        #     'right_thumb_mcp', 'right_thumb_abd', 'right_thumb_pip', 'right_thumb_dip',
        #     'right_index_abd', 'right_index_mcp', 'right_index_pip',
        #     'right_middle_abd', 'right_middle_mcp', 'right_middle_pip',
        #     'right_ring_abd', 'right_ring_mcp', 'right_ring_pip',
        #     'right_pinky_abd', 'right_pinky_mcp', 'right_pinky_pip'
        # ]

        # Joint mapping from CSV columns to ORCA Hand joint names
        self.joint_mapping = {
            # Thumb (Digit1)
            'right_thumb_dip': 'RightDigit1Interphalangeal_flexion',
            'right_thumb_pip': 'RightDigit1Metacarpophalangeal_flexion',
            'right_thumb_mcp': 'RightDigit1Carpometacarpal_flexion',
            'right_thumb_abd': 'RightDigit1Carpometacarpal_ulnarDeviation',

            #If ProximalInterphalangeal_flexion doesn't work well, try DistalInterphalangeal_flexion instead? -- kind of the problem ig
            # Index finger (Digit2)

            'right_index_pip': 'RightDigit2ProximalInterphalangeal_flexion',
            'right_index_mcp': 'RightDigit2Metacarpophalangeal_flexion',
            'right_index_abd': 'RightDigit2Metacarpophalangeal_ulnarDeviation',

            # Middle finger (Digit3)
            'right_middle_pip': 'RightDigit3ProximalInterphalangeal_flexion',
            'right_middle_mcp': 'RightDigit3Metacarpophalangeal_flexion',
            'right_middle_abd': 'RightDigit3Metacarpophalangeal_ulnarDeviation',

            # Ring finger (Digit4)
            'right_ring_pip': 'RightDigit4ProximalInterphalangeal_flexion',
            'right_ring_mcp': 'RightDigit4Metacarpophalangeal_flexion',
            'right_ring_abd': 'RightDigit4Metacarpophalangeal_ulnarDeviation',

            # Pinky (Digit5)
            'right_pinky_pip': 'RightDigit5ProximalInterphalangeal_flexion',
            'right_pinky_mcp': 'RightDigit5Metacarpophalangeal_flexion',
            'right_pinky_abd': 'RightDigit5Metacarpophalangeal_ulnarDeviation'
        }

        # Joint limits (ROM - Range of Motion) from ORCA hand config.yaml
        # These are the physical limits of the hardware to prevent damage
        # Values in radians [min, max]
        self.joint_limits_deg = {
            'right_thumb_mcp': [-50, 50],
            'right_thumb_abd': [-20, 42],
            'right_thumb_pip': [-12, 108],
            'right_thumb_dip': [-20, 112],
            'right_index_abd': [-37, 37],
            'right_index_mcp': [-20, 95],
            'right_index_pip': [-20, 108],
            'right_middle_abd': [-37, 37],
            'right_middle_mcp': [-20, 91],
            'right_middle_pip': [-20, 107],
            'right_ring_abd': [-37, 37],
            'right_ring_mcp': [-20, 91],
            'right_ring_pip': [-20, 107],
            'right_pinky_abd': [-37, 37],
            'right_pinky_mcp': [-20, 98],
            'right_pinky_pip': [-20, 108],
            'right_wrist': [-50, 30]
        }

        # Convert joint limits to radians for internal use
        self.joint_limits_rad = {}
        for joint, (min_deg, max_deg) in self.joint_limits_deg.items():
            self.joint_limits_rad[joint] = [math.radians(min_deg), math.radians(max_deg)]

        # Nullspace control: home configuration is midpoint of joint limits
        # Nullspace motion pushes joints toward this "comfortable" mid-range pose
        self.home_config = {}
        for joint, (lo, hi) in self.joint_limits_rad.items():
            self.home_config[joint] = (lo + hi) / 2.0

        # Nullspace gain: how strongly joints are pushed toward home
        self.nullspace_gain = 0.8

        # Position gain (Kp) for jparse IK: scales the position error
        # Higher = more aggressive tracking, but can overshoot/oscillate
        self.jparse_position_gain = 1.0

        # Neutral position offsets from ORCA config.yaml
        # These represent the hand's resting position (when glove is at zero)
        self.declare_parameter('use_neutral_offsets', True)
        self.use_neutral_offsets = self.get_parameter('use_neutral_offsets').get_parameter_value().bool_value

        self.neutral_offsets_deg = {
            # 'right_thumb_mcp': -13,
            # 'right_thumb_abd': 43,   # Large offset - thumb naturally sits away from palm
            # 'right_thumb_pip': 33,
            # 'right_thumb_dip': 19,
            # 'right_index_abd': 25,
            # 'right_index_mcp': 0,
            # 'right_index_pip': 0,
            # 'right_middle_abd': -2,
            # 'right_middle_mcp': 0,
            # 'right_middle_pip': 0,
            # 'right_ring_abd': -20,
            # 'right_ring_mcp': -1,
            # 'right_ring_pip': 0,
            # 'right_pinky_abd': -55,
            # 'right_pinky_mcp': 1,
            # 'right_pinky_pip': 0,
            # 'right_wrist': 0
            'right_thumb_mcp': 0,
            'right_thumb_abd': 0,   # Large offset - thumb naturally sits away from palm
            'right_thumb_pip': 0,
            'right_thumb_dip': 0,
            'right_index_abd': 0,
            'right_index_mcp': 0,
            'right_index_pip': 0,
            'right_middle_abd': 0,
            'right_middle_mcp': 0,
            'right_middle_pip': 0,
            'right_ring_abd': 0,
            'right_ring_mcp': 0,
            'right_ring_pip': 0,
            'right_pinky_abd': 0,
            'right_pinky_mcp': 0,
            'right_pinky_pip': 0,
            'right_wrist': 0
        }

        # Convert neutral offsets to radians
        self.neutral_offsets_rad = {}
        for joint, offset_deg in self.neutral_offsets_deg.items():
            self.neutral_offsets_rad[joint] = math.radians(offset_deg)

        if self.use_neutral_offsets:
            self.get_logger().info('Neutral offsets: ENABLED (using hardware neutral position)')
        else:
            self.get_logger().info('Neutral offsets: DISABLED')

        # Position columns for fingertips (for inverse kinematics)
        # Match the actual CSV column names
        self.tip_position_mapping = {
            'thumb': ['RightDigit1DistalPhalanx_position_x', 'RightDigit1DistalPhalanx_position_y', 'RightDigit1DistalPhalanx_position_z'],
            'index': ['RightDigit2DistalPhalanx_position_x', 'RightDigit2DistalPhalanx_position_y', 'RightDigit2DistalPhalanx_position_z'],
            'middle': ['RightDigit3DistalPhalanx_position_x', 'RightDigit3DistalPhalanx_position_y', 'RightDigit3DistalPhalanx_position_z'],
            'ring': ['RightDigit4DistalPhalanx_position_x', 'RightDigit4DistalPhalanx_position_y', 'RightDigit4DistalPhalanx_position_z'],
            'pinky': ['RightDigit5DistalPhalanx_position_x', 'RightDigit5DistalPhalanx_position_y', 'RightDigit5DistalPhalanx_position_z']
        }

        self.pybullet_to_orca = {
            2: "right_wrist",
            4: "right_thumb_mcp",
            6: "right_thumb_abd",
            8: "right_thumb_pip",
            10: "right_thumb_dip",
            13: "right_index_abd",
            15: "right_index_mcp",
            17: "right_index_pip",
            20: "right_middle_abd",
            22: "right_middle_mcp",
            24: "right_middle_pip",
            27: "right_ring_abd",
            29: "right_ring_mcp",
            31: "right_ring_pip",
            34: "right_pinky_abd",
            36: "right_pinky_mcp",
            38: "right_pinky_pip"
        }

        # Fingertip link indices in ORCA URDF (for IK)
        # self.fingertip_link_indices = {
        #     "thumb": 12,   # right_thumb fingertip fixed offset
        #     "index": 19,   # right_index fingertip fixed offset
        #     "middle": 26,  # right_middle fingertip fixed offset
        #     "ring": 33,    # right_ring fingertip fixed offset
        #     "pinky": 40    # right_pinky fingertip fixed offset
        # }

        self.fingertip_link_indices = {
            "thumb": 11,   # right_thumb fingertip fixed offset
            "index": 18,   # right_index fingertip fixed offset
            "middle": 25,  # right_middle fingertip fixed offset
            "ring": 32,    # right_ring fingertip fixed offset
            "pinky": 39    # right_pinky fingertip fixed offset
        }

        # Initialize PyBullet for IK solving (DIRECT mode - no GUI)
        self.physics_client = p.connect(p.DIRECT)
        self.get_logger().info('PyBullet physics client connected in DIRECT mode')

        # Load ORCA hand URDF
        # self.urdf_path = "/home/shalika/Desktop/hand_control_ws/src/orcahand_description/models/urdf/orcahand_right.urdf"
        # try:
        #     self.robot_id = p.loadURDF(self.urdf_path, useFixedBase=True)
        #     self.get_logger().info(f'Loaded URDF from: {self.urdf_path}')
        # except Exception as e:
        #     self.get_logger().error(f'Failed to load URDF: {e}')
        #     raise

        # Load Inspire hand URDF
        share_directory = get_package_share_directory('hands')
        urdf_path = os.path.join(share_directory, 'urdf', 'orca', 'orcahand_right.urdf')
        try:
            self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
            self.get_logger().info(f'Loaded URDF from: {urdf_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            raise

        # Find the palm link index
        self.palm_link_id = None
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode("utf-8")
            if link_name == "right_wrist_jointbody":
                self.palm_link_id = i
                self.get_logger().info(f'Found palm link index: {self.palm_link_id}')
                break

        if self.palm_link_id is None:
            self.get_logger().warn("Palm link 'right_palm' not found in URDF!")

        # Define joint limits for IK (extracted from URDF)
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

        # Rest poses (neutral position)
        self.rest_poses = [
            -0.0515, 0.7540, -0.2736, 0.1516, 0.0965,
            0.0087, 0.0657, 0.1435, -0.0714, -0.3261,
            0.1222, -0.1143, -0.7266, 0.2105, -0.0214,
            -0.8825, 0.1695
        ]

        # Joint ranges (for damping)
        self.joint_ranges = [upper - lower for upper, lower in zip(self.upper_limits, self.lower_limits)]

        # Subscribers
        self.control_type_sub = self.create_subscription(
            String, 'control_type', self.control_type_callback, 10)
        self.rokoko_data_sub = self.create_subscription(
            String, 'rokoko_ref_data', self.rokoko_data_callback, 10)

        # Publisher for retargeted joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Publisher for fingertip target markers (visualization in RViz)
        self.marker_pub = self.create_publisher(MarkerArray, 'fingertip_targets', 10)

        # Publisher for fingertip position errors (for plotting)
        # Format: [time, thumb_x, thumb_y, thumb_z, index_x, index_y, index_z, ...]
        self.error_pub = self.create_publisher(Float64MultiArray, 'fingertip_errors', 10)
        self.start_time = self.get_clock().now()

        self.control_type = 'direct'  # Default
        self.latest_rokoko_data = None

        self.get_logger().info('ORCA Retargeting Node started')

    def publish_rokoko_fingertip_markers(self, parsed_data):
        """
        Compute Rokoko fingertip positions in the URDF world frame and publish
        them as sphere markers on /fingertip_targets so RViz can show where the
        glove says each fingertip should be.
        """
        # Get palm position from PyBullet (URDF world frame)
        palm_link_info = p.getLinkState(self.robot_id, self.palm_link_id, computeForwardKinematics=True)
        pb_palm = np.array(palm_link_info[0], dtype=float)

        # Get Rokoko palm position (if available)
        palm_cols = ['RightHand_position_x', 'RightHand_position_y', 'RightHand_position_z']
        rokoko_palm = None
        if all(col in parsed_data for col in palm_cols):
            rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)

        tips_in_world = {}
        for finger_name, pos_columns in self.tip_position_mapping.items():
            # Try primary CSV naming
            rokoko_tip = None
            if all(col in parsed_data for col in pos_columns):
                rokoko_tip = np.array([parsed_data[col] for col in pos_columns], dtype=float)
            else:
                # Try live teleop naming
                digit_num = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5}[finger_name]
                alt_cols = [f'RightDigit{digit_num}Tip_pos_x', f'RightDigit{digit_num}Tip_pos_y', f'RightDigit{digit_num}Tip_pos_z']
                if all(col in parsed_data for col in alt_cols):
                    rokoko_tip = np.array([parsed_data[col] for col in alt_cols], dtype=float)

            if rokoko_tip is None:
                continue

            orca_tip = np.array(self.rotation_rokoko_to_orca(rokoko_tip), dtype=float)

            if rokoko_palm is not None:
                orca_palm = np.array(self.rotation_rokoko_to_orca(rokoko_palm), dtype=float)
                tips_in_world[finger_name] = (orca_tip - orca_palm) + pb_palm
            else:
                tips_in_world[finger_name] = orca_tip

        if not tips_in_world:
            return

        # Build and publish MarkerArray
        colors = {
            'thumb':  (1.0, 0.0, 0.0),   # Red
            'index':  (0.0, 1.0, 0.0),   # Green
            'middle': (0.0, 0.0, 1.0),   # Blue
            'ring':   (1.0, 1.0, 0.0),   # Yellow
            'pinky':  (1.0, 0.0, 1.0),   # Magenta
        }
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        marker_id = 0

        # --- Rokoko target spheres (translucent) ---
        for finger_name, pos in tips_in_world.items():
            m = Marker()
            m.header.frame_id = 'base'
            m.header.stamp = stamp
            m.ns = 'fingertip_targets'
            m.id = marker_id; marker_id += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            m.pose.orientation.w = 1.0
            m.scale.x = 0.01
            m.scale.y = 0.01
            m.scale.z = 0.01
            r, g, b = colors.get(finger_name, (1.0, 1.0, 1.0))
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.5
            marker_array.markers.append(m)

        # --- PyBullet palm axes (reference frame) ---
        axis_length = 0.03  # Larger for palm
        axis_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]  # RGB = XYZ

        palm_orn = palm_link_info[1]  # Quaternion
        palm_rot_matrix = np.array(p.getMatrixFromQuaternion(palm_orn)).reshape(3, 3)

        for axis_idx in range(3):
            axis_vec = palm_rot_matrix[:, axis_idx]
            end_pos = pb_palm + axis_length * axis_vec

            m = Marker()
            m.header.frame_id = 'base'
            m.header.stamp = stamp
            m.ns = 'palm_axes'
            m.id = marker_id; marker_id += 1
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.003  # Thicker lines for palm

            start_pt = Point()
            start_pt.x, start_pt.y, start_pt.z = float(pb_palm[0]), float(pb_palm[1]), float(pb_palm[2])
            end_pt = Point()
            end_pt.x, end_pt.y, end_pt.z = float(end_pos[0]), float(end_pos[1]), float(end_pos[2])
            m.points = [start_pt, end_pt]

            r, g, b = axis_colors[axis_idx]
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
            marker_array.markers.append(m)

        # Palm label
        m = Marker()
        m.header.frame_id = 'base'
        m.header.stamp = stamp
        m.ns = 'palm_label'
        m.id = marker_id; marker_id += 1
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(pb_palm[0])
        m.pose.position.y = float(pb_palm[1])
        m.pose.position.z = float(pb_palm[2]) + 0.04
        m.scale.z = 0.012
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, 1.0
        m.text = 'PALM (PyBullet)'
        marker_array.markers.append(m)

        # --- Rokoko palm axes (transformed to ORCA frame) ---
        # Show the Rokoko coordinate frame axes at the PyBullet palm position
        # This visualizes how Rokoko axes map to ORCA axes
        # Always show these axes to visualize the frame transformation
        rokoko_axis_length = 0.04  # Slightly longer than PyBullet axes for visibility

        # Rokoko axes in Rokoko frame (unit vectors)
        rokoko_axes = {
            'X': np.array([1.0, 0.0, 0.0]),  # Rokoko +X
            'Y': np.array([0.0, 1.0, 0.0]),  # Rokoko +Y (up in Rokoko)
            'Z': np.array([0.0, 0.0, 1.0]),  # Rokoko +Z (forward in Rokoko)
        }

        # Transform each axis to ORCA frame using rotation_rokoko_to_orca
        # Mapping: ORCA.x = Rokoko.x, ORCA.y = Rokoko.z, ORCA.z = Rokoko.y
        orca_axes = {}
        for axis_name, axis_vec in rokoko_axes.items():
            orca_axes[axis_name] = np.array(self.rotation_rokoko_to_orca(axis_vec))

        # Colors: R=Rokoko X, G=Rokoko Y, B=Rokoko Z (lighter to distinguish from PyBullet)
        rokoko_axis_colors = {
            'X': (1.0, 0.5, 0.5),  # Light red - Rokoko X -> ORCA X
            'Y': (0.5, 1.0, 0.5),  # Light green - Rokoko Y (up) -> ORCA Z
            'Z': (0.5, 0.5, 1.0),  # Light blue - Rokoko Z (forward) -> ORCA Y
        }

        for axis_name, orca_axis_vec in orca_axes.items():
            end_pos = pb_palm + rokoko_axis_length * orca_axis_vec

            m = Marker()
            m.header.frame_id = 'base'
            m.header.stamp = stamp
            m.ns = 'rokoko_palm_axes'
            m.id = marker_id; marker_id += 1
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.004  # Slightly thicker than PyBullet axes

            start_pt = Point()
            start_pt.x, start_pt.y, start_pt.z = float(pb_palm[0]), float(pb_palm[1]), float(pb_palm[2])
            end_pt = Point()
            end_pt.x, end_pt.y, end_pt.z = float(end_pos[0]), float(end_pos[1]), float(end_pos[2])
            m.points = [start_pt, end_pt]

            r, g, b = rokoko_axis_colors[axis_name]
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
            marker_array.markers.append(m)

        # Rokoko palm label
        m = Marker()
        m.header.frame_id = 'base'
        m.header.stamp = stamp
        m.ns = 'rokoko_palm_label'
        m.id = marker_id; marker_id += 1
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(pb_palm[0])
        m.pose.position.y = float(pb_palm[1])
        m.pose.position.z = float(pb_palm[2]) + 0.06  # Above PyBullet label
        m.scale.z = 0.010
        m.color.r, m.color.g, m.color.b, m.color.a = 0.8, 0.8, 0.8, 1.0
        m.text = 'Rokoko: X=red Y=grn->Z Z=blu->Y'
        marker_array.markers.append(m)

        # --- Actual fingertip axes (from PyBullet) ---
        axis_length = 0.015  # Smaller for fingertips
        axis_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]  # RGB = XYZ

        for finger_name, link_idx in self.fingertip_link_indices.items():
            link_state = p.getLinkState(self.robot_id, link_idx, computeForwardKinematics=True)
            link_pos = np.array(link_state[0], dtype=float)
            link_orn = link_state[1]

            rot_matrix = np.array(p.getMatrixFromQuaternion(link_orn)).reshape(3, 3)

            for axis_idx in range(3):
                axis_vec = rot_matrix[:, axis_idx]
                end_pos = link_pos + axis_length * axis_vec

                m = Marker()
                m.header.frame_id = 'base'
                m.header.stamp = stamp
                m.ns = 'fingertip_axes'
                m.id = marker_id; marker_id += 1
                m.type = Marker.LINE_LIST
                m.action = Marker.ADD
                m.scale.x = 0.002

                start_pt = Point()
                start_pt.x, start_pt.y, start_pt.z = float(link_pos[0]), float(link_pos[1]), float(link_pos[2])
                end_pt = Point()
                end_pt.x, end_pt.y, end_pt.z = float(end_pos[0]), float(end_pos[1]), float(end_pos[2])
                m.points = [start_pt, end_pt]

                r, g, b = axis_colors[axis_idx]
                m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
                marker_array.markers.append(m)

            # Text label
            m = Marker()
            m.header.frame_id = 'base'
            m.header.stamp = stamp
            m.ns = 'fingertip_labels'
            m.id = marker_id; marker_id += 1
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = float(link_pos[0])
            m.pose.position.y = float(link_pos[1])
            m.pose.position.z = float(link_pos[2]) + 0.02
            m.scale.z = 0.008
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, 1.0
            m.text = finger_name
            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)

        # --- Compute and publish fingertip position errors ---
        # Error = target (Rokoko) - actual (PyBullet FK)
        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']
        error_msg = Float64MultiArray()

        # Time since start (in seconds)
        current_time = self.get_clock().now()
        elapsed = (current_time - self.start_time).nanoseconds / 1e9
        error_data = [elapsed]

        for finger_name in finger_order:
            if finger_name in tips_in_world:
                target_pos = np.array(tips_in_world[finger_name])
                # Get actual fingertip position from PyBullet
                link_idx = self.fingertip_link_indices[finger_name]
                link_state = p.getLinkState(self.robot_id, link_idx, computeForwardKinematics=True)
                actual_pos = np.array(link_state[0], dtype=float)
                # Error = target - actual
                error = target_pos - actual_pos
                error_data.extend([error[0], error[1], error[2]])
            else:
                # No data for this finger, use NaN
                error_data.extend([float('nan'), float('nan'), float('nan')])

        error_msg.data = error_data
        self.error_pub.publish(error_msg)

    def rotation_rokoko_to_orca(self, vec):
        """
        Map a 3-vector in Rokoko character axes into ORCA/PyBullet world axes.

        Coordinate mapping:
          - ORCA.x == Rokoko.x
          - ORCA.y == Rokoko.z  (ORCA y faces outward; Rokoko z is forward/out)
          - ORCA.z == Rokoko.y  (ORCA z is up; Rokoko y is up)
        """
        import numpy as np
        rx, ry, rz = vec
        return [rx, rz, ry]

    def control_type_callback(self, msg):
        """Update the current control method"""
        self.control_type = msg.data
        self.get_logger().info(f'Control type updated to: {self.control_type}')

    def rokoko_data_callback(self, msg):
        """Receive and process Rokoko reference data"""
        self.latest_rokoko_data = msg.data

        # Parse CSV data
        parsed_data = self.parse_csv_data(msg.data)

        if not parsed_data:
            return

        # Publish fingertip target markers for RViz visualization
        self.publish_rokoko_fingertip_markers(parsed_data)

        # Execute the appropriate control method
        try:
            if self.control_type == 'direct':
                joint_angles = self.direct_joint_angle_control(parsed_data)
            elif self.control_type == 'fingertip_ik':
                self.get_logger().info('CALLING fingertip_ik_control', once=True)
                joint_angles = self.fingertip_ik_control(parsed_data)
            elif self.control_type == 'jparse_ik':
                joint_angles = self.jparse_ik_control(parsed_data)
            else:
                self.get_logger().warn(f'Unknown control type: {self.control_type}')
                return
        except Exception as e:
            self.get_logger().error(f'Error in control method: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return

        # Publish joint states
        if joint_angles:
            self.publish_joint_state(joint_angles)

    def parse_csv_data(self, csv_string):
        """
        Parse CSV-format Rokoko data
        Format: timestamp,joint_name,value
        Returns dict with joint names as keys and values as floats
        """
        parsed = {}
        lines = csv_string.strip().split('\n')

        for line in lines:
            parts = line.split(',')
            if len(parts) < 3:
                continue

            timestamp, joint_name, value = parts[0], parts[1], parts[2]

            try:
                parsed[joint_name] = float(value)
            except ValueError:
                self.get_logger().warn(f'Could not parse value for {joint_name}: {value}')
                continue

        return parsed

    def direct_joint_angle_control(self, parsed_data):
        """
        Control method 1: Direct joint angle mapping
        Maps Rokoko finger angles directly to ORCA hand joints
        """
        self.get_logger().debug('Using direct joint angle control')

        # Create joint_names list from pybullet_to_orca mapping (sorted by index)
        joint_names = [self.pybullet_to_orca[idx] for idx in sorted(self.pybullet_to_orca.keys())]

        # Initialize with zero positions for all joints
        joint_angles = {}

        # DEBUG: Log what Rokoko data we received (first time only)
        if not hasattr(self, '_logged_rokoko_keys'):
            self._logged_rokoko_keys = True
            self.get_logger().info('='*80)
            self.get_logger().info('ROKOKO DATA KEYS RECEIVED:')
            for key in sorted(parsed_data.keys()):
                self.get_logger().info(f'  {key}: {parsed_data[key]:.3f}')
            self.get_logger().info('='*80)

        # Map parsed_data to ORCA joint angles using joint_mapping
        # joint_mapping: 'right_thumb_dip' -> 'RightDigit1Interphalangeal_flexion'
        # We reverse it to get values from parsed_data
        for orca_joint_name, rokoko_joint_name in self.joint_mapping.items():
            # Get value from parsed Rokoko data
            value = parsed_data.get(rokoko_joint_name, 0.0)

            # Convert to radians if data is in degrees (CSV playback mode)
            if self.data_in_degrees:
                value_radians = math.radians(value)
            else:
                # Live Rokoko data is already in radians
                value_radians = value

            
            # CONTINUOUS DEBUG: Log thumb values BEFORE offset
            if 'thumb' in orca_joint_name:
                raw_value = value_radians
                offset = self.neutral_offsets_rad.get(orca_joint_name, 0.0) if self.use_neutral_offsets else 0.0
                self.get_logger().info(
                    f'THUMB: {orca_joint_name} <- {rokoko_joint_name} | '
                    f'Raw: {math.degrees(raw_value):6.1f}° | '
                    f'Offset: {math.degrees(offset):6.1f}° | '
                    f'Final: {math.degrees(raw_value + offset):6.1f}°'
                )

            # Apply neutral position offset if enabled
            if self.use_neutral_offsets and orca_joint_name in self.neutral_offsets_rad:
                value_radians += self.neutral_offsets_rad[orca_joint_name]

            joint_angles[orca_joint_name] = value_radians

        # Periodic debug logging for ALL joints to check for issues
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1

        if self._debug_counter % 60 == 0:  # Log every 60 frames (~2 seconds at 30Hz)
            self.get_logger().info('='*80)
            self.get_logger().info(f'ALL JOINT VALUES (frame {self._debug_counter}):')
            for orca_joint_name, value_rad in joint_angles.items():
                self.get_logger().info(f'  {orca_joint_name}: {math.degrees(value_rad):6.1f}°')
            self.get_logger().info('='*80)

        # DEBUG: Log the mapping (first time only)
        if not hasattr(self, '_logged_mapping'):
            self._logged_mapping = True
            self.get_logger().info('='*80)
            self.get_logger().info('JOINT MAPPING:')
            for orca_name, rokoko_name in self.joint_mapping.items():
                value = parsed_data.get(rokoko_name, 0.0)
                self.get_logger().info(f'  {orca_name} <- {rokoko_name} = {value:.3f}')
            self.get_logger().info('='*80)

        # Convert to ordered list matching joint_names order
        joint_angles_list = []
        for joint_name in joint_names:
            joint_angles_list.append(joint_angles.get(joint_name, 0.0)) 

        # DEBUG: Log final joint angles (first time only)
        if not hasattr(self, '_logged_final'):
            self._logged_final = True
            self.get_logger().info('='*80)
            self.get_logger().info('FINAL JOINT ANGLES (in publish order):')
            for i, (name, angle) in enumerate(zip(joint_names, joint_angles_list)):
                self.get_logger().info(f'  [{i}] {name}: {angle:.3f}')
            self.get_logger().info('='*80)

        # Store joint_names for publish function
        self.joint_names = joint_names

        return joint_angles_list

    def fingertip_ik_control(self, parsed_data):
        """
        Control method 2: Fingertip IK-based control
        Uses PyBullet's IK solver to compute joint angles from fingertip positions

        Based on the PyBullet implementation from hand_control_rec_vis.py
        """
        self.get_logger().info('ENTERED fingertip_ik_control function', once=True)
        self.get_logger().debug('Using fingertip IK control with PyBullet')

        # Create joint_names list from pybullet_to_orca mapping (sorted by index)
        joint_names = [self.pybullet_to_orca[idx] for idx in sorted(self.pybullet_to_orca.keys())]

        self.get_logger().info(f'Getting palm link state (robot_id={self.robot_id}, palm_link_id={self.palm_link_id})', once=True)
        # Get palm position from PyBullet
        palm_link_info = p.getLinkState(self.robot_id, self.palm_link_id, computeForwardKinematics=True)
        orca_palm_in_world = np.array(palm_link_info[0], dtype=float)
        # Convert to list for addition (matching working implementation)
        orca_palm_in_world_arr = orca_palm_in_world.tolist()
        self.get_logger().info(f'Palm position: {orca_palm_in_world}', once=True)

        # Extract fingertip positions and convert to ORCA world frame
        orca_tips_in_world = {}
        for finger_name, pos_columns in self.tip_position_mapping.items():
            # Check if all position components are available (CSV format with full names)
            rokoko_tip = None
            if all(col in parsed_data for col in pos_columns):
                # Get Rokoko fingertip position (in world frame) - CSV naming
                rokoko_tip = np.array([parsed_data[col] for col in pos_columns], dtype=float)
            else:
                # Try alternative naming convention for live teleop (Tip_pos format)
                digit_num = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5}[finger_name]
                alt_columns = [f'RightDigit{digit_num}Tip_pos_x', f'RightDigit{digit_num}Tip_pos_y', f'RightDigit{digit_num}Tip_pos_z']

                if all(col in parsed_data for col in alt_columns):
                    # Get Rokoko fingertip position using alternative naming - live teleop
                    rokoko_tip = np.array([parsed_data[col] for col in alt_columns], dtype=float)

            # Skip this finger if no fingertip data is available
            if rokoko_tip is None:
                continue

            # Convert to ORCA orientation
            orca_tip = np.array(self.rotation_rokoko_to_orca(rokoko_tip), dtype=float)

            # Also get Rokoko palm position if available
            palm_cols = ['RightHand_position_x', 'RightHand_position_y', 'RightHand_position_z']
            if all(col in parsed_data for col in palm_cols):
                rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)
                orca_palm = np.array(self.rotation_rokoko_to_orca(rokoko_palm), dtype=float)

                # Fingertip relative to palm
                orca_tip_in_palm = orca_tip - orca_palm

                # Transform to ORCA world frame (using list for addition)
                orca_tips_in_world[finger_name] = orca_tip_in_palm + orca_palm_in_world_arr
            else:
                # If no palm data, use fingertip position directly
                orca_tips_in_world[finger_name] = orca_tip

            # DEBUG: Log first frame
            if not hasattr(self, '_logged_ik_fingertips'):
                self.get_logger().info(f'{finger_name} tip (world): {orca_tips_in_world[finger_name]}')

        if not hasattr(self, '_logged_ik_fingertips'):
            self._logged_ik_fingertips = True
            self.get_logger().info('='*80)
            self.get_logger().info(f'Using PyBullet IK for {len(orca_tips_in_world)} fingertips')
            self.get_logger().info('='*80)

        # Solve IK for each fingertip using PyBullet
        finger_order = ["thumb", "index", "middle", "ring", "pinky"]

        # Initialize joint angles array (will be updated by IK)
        # Start with current joint states or rest poses
        current_joint_angles = self.rest_poses.copy()

        # Define finger-specific joint ranges in the joint_angles array
        finger_joint_ranges = {
            0: (1, 5),   # thumb: joint_angles[1:5] → PyBullet joints [4, 6, 8, 10]
            1: (5, 8),   # index: joint_angles[5:8] → PyBullet joints [13, 15, 17]
            2: (8, 11),  # middle: joint_angles[8:11] → PyBullet joints [20, 22, 24]
            3: (11, 14), # ring: joint_angles[11:14] → PyBullet joints [27, 29, 31]
            4: (14, 17)  # pinky: joint_angles[14:17] → PyBullet joints [34, 36, 38]
        }

        # Map finger index to actual PyBullet joint indices
        finger_pybullet_joints = {
            0: [4, 6, 8, 10],    # thumb
            1: [13, 15, 17],     # index
            2: [20, 22, 24],     # middle
            3: [27, 29, 31],     # ring
            4: [34, 36, 38]      # pinky
        }

        # Solve IK for each finger
        for i, finger_name in enumerate(finger_order):
            if finger_name not in orca_tips_in_world:
                continue

            target_pos = orca_tips_in_world[finger_name]
            link_index = self.fingertip_link_indices[finger_name]

            try:
                # Call PyBullet IK (matching working implementation)
                joint_angles = p.calculateInverseKinematics(
                    self.robot_id,
                    link_index,
                    target_pos,
                    # lowerLimits=self.lower_limits,
                    # upperLimits=self.upper_limits,
                    # jointRanges=self.joint_ranges,
                    # restPoses=self.rest_poses,
                    maxNumIterations=1000,
                    residualThreshold=1e-4
                )

                # Convert to list and set wrist to 0
                joint_angles = list(joint_angles)
                joint_angles[0] = 0.0

                # Update only this finger's joints in current_joint_angles
                # Extract the correct joints from the IK solution based on finger_joint_ranges
                if i in finger_joint_ranges and i in finger_pybullet_joints:
                    start_idx, end_idx = finger_joint_ranges[i]
                    pybullet_joints = finger_pybullet_joints[i]

                    # Update each joint for this finger
                    for j, pb_joint_idx in enumerate(pybullet_joints):
                        if start_idx + j < len(joint_angles):
                            # Extract from IK solution
                            target_angle = joint_angles[start_idx + j]

                            # CRITICAL: Update PyBullet's internal state so next finger's IK builds on this solution
                            # This matches the working implementation in hand_control_rec_vis.py
                            p.resetJointState(self.robot_id, pb_joint_idx, target_angle)

                            # Map PyBullet joint index to array index for later publishing
                            # PyBullet indices: [2, 4, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38]
                            # Array indices:    [0, 1, 2, 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
                            pb_to_array_idx = {
                                2: 0, 4: 1, 6: 2, 8: 3, 10: 4,
                                13: 5, 15: 6, 17: 7,
                                20: 8, 22: 9, 24: 10,
                                27: 11, 29: 12, 31: 13,
                                34: 14, 36: 15, 38: 16
                            }

                            if pb_joint_idx in pb_to_array_idx:
                                array_idx = pb_to_array_idx[pb_joint_idx]
                                current_joint_angles[array_idx] = target_angle

            except Exception as e:
                self.get_logger().warn(f'IK failed for {finger_name}: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
                continue

        # Convert from array to ORCA joint order for publishing
        joint_angles_list = []
        for pb_idx in sorted(self.pybullet_to_orca.keys()):
            # Map PyBullet index to position in current_joint_angles array
            pb_to_array_idx = {
                2: 0, 4: 1, 6: 2, 8: 3, 10: 4,
                13: 5, 15: 6, 17: 7,
                20: 8, 22: 9, 24: 10,
                27: 11, 29: 12, 31: 13,
                34: 14, 36: 15, 38: 16
            }

            if pb_idx in pb_to_array_idx:
                array_idx = pb_to_array_idx[pb_idx]
                if array_idx < len(current_joint_angles):
                    joint_angles_list.append(current_joint_angles[array_idx])
                else:
                    joint_angles_list.append(0.0)
            else:
                joint_angles_list.append(0.0)

        # Store joint_names for publish function
        self.joint_names = joint_names

        return joint_angles_list

    def jparse_ik_control(self, parsed_data):
        """
        Control method 3: IK with JPARSE (Joint Position And Rotation Solver Engine)
        Uses PyBullet's Jacobian calculation with jparse for advanced IK solving

        Solves IK for each finger independently from base to fingertip
        """
        self.get_logger().debug('Using JPARSE IK control')

        if not JPARSE_AVAILABLE:
            self.get_logger().error('jparse not available! Falling back to zero positions')
            return [0.0] * len(self.joint_names)

        # Create joint_names list from pybullet_to_orca mapping (sorted by index)
        joint_names = [self.pybullet_to_orca[idx] for idx in sorted(self.pybullet_to_orca.keys())]

        # Initialize joint angles dictionary with CURRENT PyBullet state
        # This ensures fingers without IK updates maintain their current positions
        joint_angles = {}
        for pb_idx, joint_name in self.pybullet_to_orca.items():
            joint_state = p.getJointState(self.robot_id, pb_idx)
            joint_angles[joint_name] = joint_state[0]  # Current joint position

        # Initialize debug counters (do this once at the start)
        if not hasattr(self, '_jparse_log_count'):
            self._jparse_log_count = 0
        if not hasattr(self, '_pos_debug_count'):
            self._pos_debug_count = 0
        if not hasattr(self, '_jac_debug_count'):
            self._jac_debug_count = 0

        # Define finger chain information: (finger_name, base_link_index, joint_indices)
        # Uses self.fingertip_link_indices for end_link values
        finger_chains = {
            'thumb': {
                'base_link': 3,  # right_palm
                'ee_link_idx': self.fingertip_link_indices['thumb'],
                'joint_indices': [4, 6, 8, 10],  # mcp, abd, pip, dip
                'joint_names': ['right_thumb_mcp', 'right_thumb_abd', 'right_thumb_pip', 'right_thumb_dip']
            },
            'index': {
                'base_link': 3,  # right_palm
                'ee_link_idx': self.fingertip_link_indices['index'],
                'joint_indices': [13, 15, 17],  # abd, mcp, pip
                'joint_names': ['right_index_abd', 'right_index_mcp', 'right_index_pip']
            },
            'middle': {
                'base_link': 3,  # right_palm
                'ee_link_idx': self.fingertip_link_indices['middle'],
                'joint_indices': [20, 22, 24],  # abd, mcp, pip
                'joint_names': ['right_middle_abd', 'right_middle_mcp', 'right_middle_pip']
            },
            'ring': {
                'base_link': 3,  # right_palm
                'ee_link_idx': self.fingertip_link_indices['ring'],
                'joint_indices': [27, 29, 31],  # abd, mcp, pip
                'joint_names': ['right_ring_abd', 'right_ring_mcp', 'right_ring_pip']
            },
            'pinky': {
                'base_link': 3,  # right_palm
                'ee_link_idx': self.fingertip_link_indices['pinky'],
                'joint_indices': [34, 36, 38],  # abd, mcp, pip
                'joint_names': ['right_pinky_abd', 'right_pinky_mcp', 'right_pinky_pip']
            }
        }

        # Debug: Log how many fingers we're processing
        # Get palm position from PyBullet (ORCA world frame)
        palm_link_info = p.getLinkState(self.robot_id, self.palm_link_id, computeForwardKinematics=True)
        orca_palm_in_world = np.array(palm_link_info[0], dtype=float)
        orca_palm_in_world_arr = orca_palm_in_world.tolist()

        # Get Rokoko palm position if available
        palm_cols = ['RightHand_position_x', 'RightHand_position_y', 'RightHand_position_z']
        rokoko_palm_available = all(col in parsed_data for col in palm_cols)
        if rokoko_palm_available:
            rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)
            orca_palm_from_rokoko = np.array(self.rotation_rokoko_to_orca(rokoko_palm), dtype=float)

        fingers_processed = 0

        # Solve IK for each finger independently
        for finger_name, chain_info in finger_chains.items():
            # Get target fingertip position from parsed data
            pos_columns = self.tip_position_mapping[finger_name]

            # Try primary naming convention
            rokoko_tip = None
            if all(col in parsed_data for col in pos_columns):
                rokoko_tip = np.array([parsed_data[col] for col in pos_columns], dtype=float)
            else:
                # Try alternative naming (live teleop)
                digit_num = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5}[finger_name]
                alt_columns = [f'RightDigit{digit_num}Tip_pos_x',
                              f'RightDigit{digit_num}Tip_pos_y',
                              f'RightDigit{digit_num}Tip_pos_z']
                if all(col in parsed_data for col in alt_columns):
                    rokoko_tip = np.array([parsed_data[col] for col in alt_columns], dtype=float)

            # Skip this finger if no fingertip data available
            if rokoko_tip is None:
                if self._jparse_log_count < 3:
                    self.get_logger().warn(f'No position data for {finger_name}, skipping')
                continue

            fingers_processed += 1

            # Transform Rokoko fingertip to ORCA coordinate frame
            orca_tip = np.array(self.rotation_rokoko_to_orca(rokoko_tip), dtype=float)

            # Compute target position in PyBullet world frame
            # Make fingertip position relative to palm, then add PyBullet palm position
            if rokoko_palm_available:
                orca_tip_in_palm = orca_tip - orca_palm_from_rokoko
                target_pos_world = orca_tip_in_palm + orca_palm_in_world_arr
            else:
                # No palm data - use absolute position (less accurate)
                target_pos_world = orca_tip

            target_pos_world = np.array(target_pos_world, dtype=float)

            # Get current joint positions for this finger
            current_joint_positions = []
            for joint_idx in chain_info['joint_indices']:
                joint_state = p.getJointState(self.robot_id, joint_idx)
                current_joint_positions.append(joint_state[0])

            # Get current end effector position
            link_state = p.getLinkState(self.robot_id, chain_info['ee_link_idx'])
            current_ee_pos = np.array(link_state[0])

            # Calculate position error
            pos_error = target_pos_world - current_ee_pos

            # Debug: Log positions for first finger (first few times)
            if self._pos_debug_count < 3 and finger_name == 'thumb':
                self.get_logger().info(f'=== {finger_name} Position Debug ===')
                self.get_logger().info(f'Rokoko tip (raw): {rokoko_tip}')
                self.get_logger().info(f'ORCA tip: {orca_tip}')
                if rokoko_palm_available:
                    self.get_logger().info(f'Rokoko palm: {rokoko_palm}')
                    self.get_logger().info(f'ORCA palm (Rokoko): {orca_palm_from_rokoko}')
                    self.get_logger().info(f'ORCA palm (PyBullet): {orca_palm_in_world}')
                self.get_logger().info(f'Target pos (world): {target_pos_world}')
                self.get_logger().info(f'Current pos: {current_ee_pos}')
                self.get_logger().info(f'Error: {pos_error}')
                self.get_logger().info(f'Error magnitude: {np.linalg.norm(pos_error):.6f}m')
                self._pos_debug_count += 1

            # Calculate Jacobian using PyBullet
            # Get all joint states (positions, velocities, accelerations)
            # and create mapping from joint index to Jacobian column index
            num_joints = p.getNumJoints(self.robot_id)
            joint_positions = []
            joint_velocities = []
            joint_accelerations = []
            joint_index_to_col = {}  # Maps joint index to Jacobian column index

            col_idx = 0
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                joint_type = joint_info[2]  # Joint type

                # Only include movable joints (revolute or prismatic)
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    joint_state = p.getJointState(self.robot_id, i)
                    joint_positions.append(joint_state[0])  # Position
                    joint_velocities.append(joint_state[1])  # Velocity
                    joint_accelerations.append(0.0)  # Acceleration (assumed zero)
                    joint_index_to_col[i] = col_idx
                    col_idx += 1

            # Calculate Jacobian for the fingertip link
            jacobian_linear, jacobian_angular = p.calculateJacobian(
                bodyUniqueId=self.robot_id,
                linkIndex=chain_info['ee_link_idx'],
                localPosition=[0, 0, 0],
                objPositions=joint_positions,
                objVelocities=joint_velocities,
                objAccelerations=joint_accelerations
            )

            # Convert to numpy arrays
            J_linear = np.array(jacobian_linear)
            J_angular = np.array(jacobian_angular)

            # Extract only the columns corresponding to this finger's joints
            # Map joint indices to Jacobian column indices
            finger_col_indices = []
            for joint_idx in chain_info['joint_indices']:
                if joint_idx in joint_index_to_col:
                    finger_col_indices.append(joint_index_to_col[joint_idx])

            J_finger_linear = J_linear[:, finger_col_indices]

            # Use jparse to solve the IK problem (velocity-based)
            # Create a jparse solver for this finger
            try:
                # Create JParseCore solver with singularity threshold
                # gamma: directions with σᵢ/σₘₐₓ < gamma are treated as singular
                jp_solver = jparse.JParseCore(gamma=0.1)

                # Compute the jparse pseudo-inverse of the Jacobian
                # This uses the jparse algorithm (not standard damped least squares)
                J_pinv, nullspace = jp_solver.compute(
                    jacobian=J_finger_linear,
                    singular_direction_gain_position=1.0,
                    position_dimensions=3,  # We're only controlling position (x, y, z)
                    return_nullspace=True #idk if changing this did anything
                )

                # J_pinv = jp_solver.pinv(J_finger_linear)

                # Clamp position error magnitude to prevent huge delta_q
                # This limits how far we try to move per iteration
                max_step = 0.005  # 5mm max step per iteration
                error_norm = np.linalg.norm(pos_error)
                if error_norm > max_step:
                    pos_error_clamped = pos_error * (max_step / error_norm)
                else:
                    pos_error_clamped = pos_error  # Small error, use as-is for convergence

                # Direct position solve with clamped error
                # Apply position gain (Kp) to scale the velocity command
                v_desired = self.jparse_position_gain * pos_error_clamped
                dq_task = J_pinv @ v_desired

                # Nullspace control: push joints toward home (midpoint of limits)
                # This uses the redundant DOFs to keep joints in a comfortable range
                # without affecting the fingertip position
                dq_nullspace = np.zeros_like(dq_task)
                if nullspace is not None:
                    # Build home config vector for this finger's joints
                    q_home = np.array([self.home_config.get(jn, 0.0)
                                       for jn in chain_info['joint_names']])
                    q_current = np.array(current_joint_positions)

                    # Gradient of ||q - q_home||^2 drives joints toward home
                    q_error = q_current - q_home
                    nullspace_velocity = -self.nullspace_gain * q_error

                    # Project through nullspace so it doesn't fight the task
                    dq_nullspace = nullspace @ nullspace_velocity
                    # dq_nullspace = np.asarray(dq_nullspace).flatten()

                delta_q = dq_task + dq_nullspace
                # delta_q = dq_nullspace

                # DIAGNOSTIC: Check delta_q values
                delta_q_max_deg = np.degrees(np.max(np.abs(delta_q)))
                if delta_q_max_deg > 30:  # Still too large even after clamping
                    self.get_logger().warn(
                        f'{finger_name}: LARGE delta_q! max={delta_q_max_deg:.1f}° per iteration. '
                        f'Error={error_norm*1000:.1f}mm (clamped to {np.linalg.norm(pos_error_clamped)*1000:.1f}mm)'
                    )

                # Debug: Check Jacobian and delta_q
                if self._jac_debug_count < 2 and finger_name == 'thumb':
                    self.get_logger().info(f'J_finger_linear shape: {J_finger_linear.shape}')
                    self.get_logger().info(f'J_pinv shape: {J_pinv.shape}')
                    self.get_logger().info(f'pos_error: {pos_error} (norm={error_norm*1000:.1f}mm)')
                    self.get_logger().info(f'pos_error_clamped: {pos_error_clamped} (norm={np.linalg.norm(pos_error_clamped)*1000:.1f}mm)')
                    self.get_logger().info(f'dq_task: {dq_task}')
                    self.get_logger().info(f'dq_nullspace: {dq_nullspace} (norm={np.linalg.norm(dq_nullspace):.4f})')
                    self.get_logger().info(f'delta_q (task+null): {delta_q}')
                    q_home = np.array([self.home_config.get(jn, 0.0) for jn in chain_info['joint_names']])
                    self.get_logger().info(f'q_home (deg): {[f"{np.degrees(q):.1f}" for q in q_home]}')
                    self.get_logger().info(f'q_current (deg): {[f"{np.degrees(q):.1f}" for q in current_joint_positions]}')

                    self._jac_debug_count += 1

                # Update joint angles for this finger with limit clamping
                for i, joint_name in enumerate(chain_info['joint_names']):
                    new_angle = current_joint_positions[i] + delta_q[i]

                    # Clamp to joint limits
                    if joint_name in self.joint_limits_rad:
                        min_limit, max_limit = self.joint_limits_rad[joint_name]
                        new_angle = max(min_limit, min(max_limit, new_angle))

                    joint_angles[joint_name] = new_angle

                # Debug logging (first few iterations)
                self._jparse_log_count += 1

                # Log first 20 iterations, then every 60th frame (~2 sec at 30Hz)
                if self._jparse_log_count < 20 or self._jparse_log_count % 60 == 0:
                    delta_q_max = np.max(np.abs(delta_q))
                    self.get_logger().info(
                        f'{finger_name}: err={error_norm*1000:.1f}mm->clamped={np.linalg.norm(pos_error_clamped)*1000:.1f}mm, '
                        f'delta_q_max={np.degrees(delta_q_max):.1f}°, '
                        f'null_norm={np.linalg.norm(dq_nullspace):.3f}, '
                        f'delta_q(deg)={[f"{np.degrees(dq):.1f}" for dq in delta_q]}'
                    )

            except Exception as e:
                self.get_logger().error(f'jparse IK failed for {finger_name}: {e}')
                # Keep current positions on failure
                for i, joint_name in enumerate(chain_info['joint_names']):
                    joint_angles[joint_name] = current_joint_positions[i]

        # Debug: Log processing summary
        if self._jparse_log_count < 3:
            self.get_logger().info(f'Processed {fingers_processed}/5 fingers')
            self.get_logger().info(f'Joint angles computed: {len(joint_angles)} joints')
            if fingers_processed == 0:
                self.get_logger().error('NO FINGERS PROCESSED - No position data available!')

        # Convert to ordered list matching joint_names order
        joint_angles_list = []
        for joint_name in joint_names:
            angle = joint_angles.get(joint_name, 0.0)
            joint_angles_list.append(angle)

        # Debug: Show some actual joint angles
        if self._jparse_log_count < 3:
            self.get_logger().info(f'Sample joint angles (degrees):')
            for i, (name, angle) in enumerate(zip(joint_names[:5], joint_angles_list[:5])):
                self.get_logger().info(f'  {name}: {math.degrees(angle):.2f}°')

        # CRITICAL: Update PyBullet simulation with new joint angles
        # This ensures the next iteration uses updated positions for Jacobian calculation
        updated_count = 0
        for pb_idx, joint_name in self.pybullet_to_orca.items():
            if joint_name in joint_angles:
                p.resetJointState(self.robot_id, pb_idx, joint_angles[joint_name])
                updated_count += 1

        if self._jparse_log_count < 3:
            self.get_logger().info(f'Updated {updated_count} PyBullet joints')

        # Store joint_names for publish function
        self.joint_names = joint_names

        return joint_angles_list

    def clamp_joint_angles(self, joint_angles):
        """
        Clamp joint angles to stay within physical limits.
        Prevents fingers from crossing each other or moving beyond safe ranges.

        Args:
            joint_angles: List of joint angles in radians

        Returns:
            List of clamped joint angles
        """
        if not self.use_joint_limits:
            return joint_angles

        clamped_angles = []
        clamped_count = 0

        for i, joint_name in enumerate(self.joint_names):
            angle = joint_angles[i]

            if joint_name in self.joint_limits_rad:
                min_limit, max_limit = self.joint_limits_rad[joint_name]

                if angle < min_limit or angle > max_limit:
                    clamped_angle = max(min_limit, min(max_limit, angle))

                    # Log when clamping occurs (but not too frequently)
                    if not hasattr(self, '_clamp_log_count'):
                        self._clamp_log_count = {}
                    if joint_name not in self._clamp_log_count:
                        self._clamp_log_count[joint_name] = 0

                    self._clamp_log_count[joint_name] += 1

                    # Log every 100th clamp for each joint
                    if self._clamp_log_count[joint_name] % 100 == 1:
                        self.get_logger().warn(
                            f'Clamping {joint_name}: {math.degrees(angle):.1f}° → {math.degrees(clamped_angle):.1f}° '
                            f'(limits: {math.degrees(min_limit):.1f}° to {math.degrees(max_limit):.1f}°)'
                        )

                    clamped_angles.append(clamped_angle)
                    clamped_count += 1
                else:
                    clamped_angles.append(angle)
            else:
                # No limit defined for this joint, pass through
                clamped_angles.append(angle)

        return clamped_angles

    def publish_joint_state(self, joint_angles):
        """Publish the retargeted joint state with joint limit clamping"""
        # Apply joint limits to prevent impossible movements
        clamped_angles = self.clamp_joint_angles(joint_angles)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = clamped_angles

        # Debug logging for publishing
        if not hasattr(self, '_publish_log_count'):
            self._publish_log_count = 0

        self._publish_log_count += 1
        if self._publish_log_count % 30 == 0:  # Log every 30 publishes (about 1 second)
            self.get_logger().info(f'=== Publishing Joint States (#{self._publish_log_count}) ===')
            self.get_logger().info(f'Publishing {len(msg.name)} joints to /joint_states')
            if len(msg.name) > 0 and len(clamped_angles) > 0:
                # Show first 5 joints as samples
                num_samples = min(5, len(msg.name))
                for i in range(num_samples):
                    self.get_logger().info(f'  {msg.name[i]}: {clamped_angles[i]:.4f} rad ({math.degrees(clamped_angles[i]):.2f}°)')
            else:
                self.get_logger().warn('WARNING: Empty joint names or angles!')

        self.joint_pub.publish(msg)

    def destroy_node(self):
        """Clean up PyBullet connection"""
        if hasattr(self, 'physics_client'):
            p.disconnect(self.physics_client)
            self.get_logger().info('PyBullet physics client disconnected')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    orca_retargeting = OrcaRetargeting()

    try:
        rclpy.spin(orca_retargeting)
    except KeyboardInterrupt:
        pass
    finally:
        orca_retargeting.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
