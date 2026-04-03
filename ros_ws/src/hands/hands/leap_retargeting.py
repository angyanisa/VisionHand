#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory
import pybullet as p
import numpy as np
import math
import os
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# Import jparse for advanced IK solving
try:
    import jparse_robotics as jparse
    JPARSE_AVAILABLE = True
except ImportError:
    JPARSE_AVAILABLE = False
    print("Warning: jparse not available. Install with: pip install git+https://github.com/armlabstanford/jparse.git")


class LeapRetargeting(Node):
    def __init__(self):
        super().__init__('leap_retargeting')
        self.marker_pub = self.create_publisher(MarkerArray, 'fingertip_targets', 10)
        self.debug_marker_pub = self.create_publisher(MarkerArray, 'leap_link_frames', 10)

        self.declare_parameter('data_in_degrees', False)
        self.data_in_degrees = self.get_parameter('data_in_degrees').get_parameter_value().bool_value

        if self.data_in_degrees:
            self.get_logger().info('Data format: DEGREES (will convert to radians)')
        else:
            self.get_logger().info('Data format: RADIANS (no conversion needed)')

        # leap_hand_right.urdf uses a clean 0-15 joint sequence
        self.joint_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        # self.joint_names = ['1', '0', '2', '3', '5', '4', '6', '7', '9', '8', '10', '11', '12', '13', '14', '15']
        
        self.joint_mapping = {
            # Thumb
            '12': 'RightDigit1Carpometacarpal_ulnarDeviation',
            '13': 'RightDigit1Carpometacarpal_flexion',
            '14': 'RightDigit1Metacarpophalangeal_flexion',
            '15': 'RightDigit1Interphalangeal_flexion',
            # Index
            '0': 'RightDigit2Metacarpophalangeal_ulnarDeviation',
            '1': 'RightDigit2Metacarpophalangeal_flexion',
            '2': 'RightDigit2ProximalInterphalangeal_flexion',
            '3': 'RightDigit2DistalInterphalangeal_flexion',
            # Middle
            '4': 'RightDigit3Metacarpophalangeal_ulnarDeviation',
            '5': 'RightDigit3Metacarpophalangeal_flexion',
            '6': 'RightDigit3ProximalInterphalangeal_flexion',
            '7': 'RightDigit3DistalInterphalangeal_flexion',
            # Ring
            '8': 'RightDigit4Metacarpophalangeal_ulnarDeviation',
            '9': 'RightDigit4Metacarpophalangeal_flexion',
            '10': 'RightDigit4ProximalInterphalangeal_flexion',
            '11': 'RightDigit4DistalInterphalangeal_flexion',
        }

        self.tip_position_mapping = {
            'thumb': ['RightDigit1DistalPhalanx_position_x', 'RightDigit1DistalPhalanx_position_y', 'RightDigit1DistalPhalanx_position_z'],
            'index': ['RightDigit2DistalPhalanx_position_x', 'RightDigit2DistalPhalanx_position_y', 'RightDigit2DistalPhalanx_position_z'],
            'middle': ['RightDigit3DistalPhalanx_position_x', 'RightDigit3DistalPhalanx_position_y', 'RightDigit3DistalPhalanx_position_z'],
            'ring': ['RightDigit4DistalPhalanx_position_x', 'RightDigit4DistalPhalanx_position_y', 'RightDigit4DistalPhalanx_position_z']
        }

        self.pybullet_to_leap = {
            0: '1', 1: '0', 2: '2', 3: '3',        # Index
            5: '5', 6: '4', 7: '6', 8: '7',        # Middle
            10: '9', 11: '8', 12: '10', 13: '11',  # Ring
            15: '12', 16: '13', 17: '14', 18: '15' # Thumb
        }
        
        self.physics_client_id = p.connect(p.DIRECT)
        self.get_logger().info(f'PyBullet physics client connected in DIRECT mode')

        
        share_directory = get_package_share_directory('hands')
        urdf_path = os.path.join(share_directory, 'urdf', 'leap', 'leap_hand_right.urdf')
        
        try:
            self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
            self.get_logger().info(f'Loaded leap_hand_right.urdf from: {urdf_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            raise

        self.find_link_indices()
        
        # Initializing thumb neutral positions
        self.thumb_neutral = {}
        for i in range(12, 16):
            self.thumb_neutral[str(i)] = p.getJointState(self.robot_id, i)[0]
        
        self.palm_link_id = -1 

        self.movable_joints = []
        self.joint_name_to_index = {}
        self.names = []
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            joint_type = info[2]
            joint_name = info[1].decode("utf-8")
            if joint_type != p.JOINT_FIXED:
                self.joint_name_to_index[joint_name] = i
                self.movable_joints.append(i)
                self.names.append(joint_name)

        # Extract limits from leap_hand_right.urdf
        self.lower_limits, self.upper_limits, self.joint_ranges = self.get_joint_limits(self.robot_id)
        self.joint_limits_rad = {}
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode("utf-8")
            joint_type = joint_info[2]

            if joint_type == p.JOINT_REVOLUTE:
                lower = joint_info[8]
                upper = joint_info[9]
                self.joint_limits_rad[joint_name] = [lower, upper]
        self.joint_limits_deg = {joint: [math.degrees(lim) for lim in limits] for joint, limits in self.joint_limits_rad.items()}

        self.rest_poses = [
            np.pi/6, -np.pi/6, np.pi/3, np.pi/3, # Index
            np.pi/6, 0.0,      np.pi/3, np.pi/3, # Middle
            np.pi/6, np.pi/6,  np.pi/3, np.pi/3, # Ring
            np.pi/6, np.pi/6,  np.pi/3, np.pi/3  # Thumb
        ]

        self.jparse_position_gain = 1.0
        # home config is halfway point of joint limits
        self.home_config = {}
        for joint, (lower, upper) in self.joint_limits_rad.items():
            self.home_config[joint] = (lower + upper) / 2.0
        self.nullspace_gain = 0.8

        self.get_logger().info(f"Movable joint order (PyBullet IK order): {self.movable_joints}")
        self.get_logger().info(f"Joint names: {self.names}")
        self.get_logger().info(f"Joint name to index mapping: {self.joint_name_to_index}")

        # Subscribers
        self.control_type_sub = self.create_subscription(
            String, 'control_type', self.control_type_callback, 10)
        self.rokoko_data_sub = self.create_subscription(
            String, 'rokoko_ref_data', self.rokoko_data_callback, 10)

        # Publisher for retargeted joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        self.control_type = 'direct'  # Default
        self.latest_rokoko_data = None

        # Dump full joint/link table for debugging
        self.dump_joint_link_table()

        # Timer to publish link frame visualization at 10Hz
        self.create_timer(0.1, self.publish_link_frames)

        self.get_logger().info('Leap Retargeting Node started')


    def dump_joint_link_table(self):
        """Print full joint/link table at startup for debugging the kinematic chain."""
        self.get_logger().info('=' * 90)
        self.get_logger().info('FULL JOINT/LINK TABLE (PyBullet)')
        self.get_logger().info(f'{"Idx":<5} {"Joint Name":<12} {"Type":<10} {"Parent Link":<20} {"Child Link":<20} {"Limits"}')
        self.get_logger().info('-' * 90)

        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = {0: 'REVOLUTE', 1: 'PRISMATIC', 2: 'SPHERICAL', 3: 'PLANAR', 4: 'FIXED'}[info[2]]
            parent_idx = info[16]
            parent_name = 'palm_lower (base)' if parent_idx == -1 else p.getJointInfo(self.robot_id, parent_idx)[12].decode('utf-8')
            child_name = info[12].decode('utf-8')
            lower, upper = info[8], info[9]

            self.get_logger().info(
                f'{i:<5} {joint_name:<12} {joint_type:<10} {parent_name:<20} {child_name:<20} [{lower:.3f}, {upper:.3f}]'
            )

        self.get_logger().info('=' * 90)

        # Also dump fingertip link indices found
        self.get_logger().info(f'Fingertip link indices: {self.fingertip_link_indices}')
        self.get_logger().info(f'Palm link id: {self.palm_link_id}')

    def publish_link_frames(self):
        """
        Publish coordinate frame axes (RGB=XYZ) and text labels for every link.
        Subscribe to /leap_link_frames MarkerArray in RViz to see them.
        """
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        marker_id = 0

        num_joints = p.getNumJoints(self.robot_id)
        axis_length = 0.015  # 15mm axes
        axis_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]  # RGB = XYZ

        # Base link (palm_lower)
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        base_pos = np.array(base_pos)
        base_rot = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)

        for axis_idx in range(3):
            axis_vec = base_rot[:, axis_idx]
            end_pos = base_pos + axis_length * axis_vec

            m = Marker()
            m.header.frame_id = 'palm_lower'
            m.header.stamp = stamp
            m.ns = 'link_axes'
            m.id = marker_id; marker_id += 1
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.002

            start_pt = Point()
            start_pt.x, start_pt.y, start_pt.z = float(base_pos[0]), float(base_pos[1]), float(base_pos[2])
            end_pt = Point()
            end_pt.x, end_pt.y, end_pt.z = float(end_pos[0]), float(end_pos[1]), float(end_pos[2])
            m.points = [start_pt, end_pt]

            r, g, b = axis_colors[axis_idx]
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
            marker_array.markers.append(m)

        # Base label
        m = Marker()
        m.header.frame_id = 'palm_lower'
        m.header.stamp = stamp
        m.ns = 'link_labels'
        m.id = marker_id; marker_id += 1
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(base_pos[0])
        m.pose.position.y = float(base_pos[1])
        m.pose.position.z = float(base_pos[2]) + 0.02
        m.scale.z = 0.008
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, 1.0
        m.text = 'base: palm_lower'
        marker_array.markers.append(m)

        # Palm link — larger, thicker coordinate frame for visibility
        palm_axis_length = 0.04  # 40mm axes (bigger than other links)
        if self.palm_link_id == -1:
            palm_pos, palm_quat = p.getBasePositionAndOrientation(self.robot_id)
        else:
            palm_state = p.getLinkState(self.robot_id, self.palm_link_id, computeForwardKinematics=True)
            palm_pos, palm_quat = palm_state[0], palm_state[1]

        palm_pos = np.array(palm_pos)
        palm_rot = np.array(p.getMatrixFromQuaternion(palm_quat)).reshape(3, 3)
        axis_labels = ['X', 'Y', 'Z']

        for axis_idx in range(3):
            axis_vec = palm_rot[:, axis_idx]
            end_pos = palm_pos + palm_axis_length * axis_vec

            m = Marker()
            m.header.frame_id = 'palm_lower'
            m.header.stamp = stamp
            m.ns = 'palm_axes'
            m.id = marker_id; marker_id += 1
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.004  # Thicker lines

            start_pt = Point()
            start_pt.x, start_pt.y, start_pt.z = float(palm_pos[0]), float(palm_pos[1]), float(palm_pos[2])
            end_pt = Point()
            end_pt.x, end_pt.y, end_pt.z = float(end_pos[0]), float(end_pos[1]), float(end_pos[2])
            m.points = [start_pt, end_pt]

            r, g, b = axis_colors[axis_idx]
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
            marker_array.markers.append(m)

            # Axis label at the tip
            m = Marker()
            m.header.frame_id = 'palm_lower'
            m.header.stamp = stamp
            m.ns = 'palm_axis_labels'
            m.id = marker_id; marker_id += 1
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = float(end_pos[0])
            m.pose.position.y = float(end_pos[1])
            m.pose.position.z = float(end_pos[2])
            m.scale.z = 0.01
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
            m.text = f'PALM {axis_labels[axis_idx]}'
            marker_array.markers.append(m)

        # Palm label
        m = Marker()
        m.header.frame_id = 'palm_lower'
        m.header.stamp = stamp
        m.ns = 'palm_label'
        m.id = marker_id; marker_id += 1
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(palm_pos[0])
        m.pose.position.y = float(palm_pos[1])
        m.pose.position.z = float(palm_pos[2]) + 0.05
        m.scale.z = 0.012
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, 1.0
        m.text = f'PALM (id={self.palm_link_id})'
        marker_array.markers.append(m)

        # All child links
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            child_link_name = info[12].decode('utf-8')

            link_state = p.getLinkState(self.robot_id, i, computeForwardKinematics=True)
            link_pos = np.array(link_state[0], dtype=float)
            link_quat = link_state[1]
            rot_matrix = np.array(p.getMatrixFromQuaternion(link_quat)).reshape(3, 3)

            # Draw XYZ axes
            for axis_idx in range(3):
                axis_vec = rot_matrix[:, axis_idx]
                end_pos = link_pos + axis_length * axis_vec

                m = Marker()
                m.header.frame_id = 'palm_lower'
                m.header.stamp = stamp
                m.ns = 'link_axes'
                m.id = marker_id; marker_id += 1
                m.type = Marker.LINE_LIST
                m.action = Marker.ADD
                m.scale.x = 0.002 if joint_type != p.JOINT_FIXED else 0.001

                start_pt = Point()
                start_pt.x, start_pt.y, start_pt.z = float(link_pos[0]), float(link_pos[1]), float(link_pos[2])
                end_pt = Point()
                end_pt.x, end_pt.y, end_pt.z = float(end_pos[0]), float(end_pos[1]), float(end_pos[2])
                m.points = [start_pt, end_pt]

                r, g, b = axis_colors[axis_idx]
                m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
                marker_array.markers.append(m)

            # Text label: "idx: link_name (joint_name, TYPE)"
            type_str = 'REV' if joint_type == p.JOINT_REVOLUTE else 'FIX' if joint_type == p.JOINT_FIXED else '?'
            m = Marker()
            m.header.frame_id = 'palm_lower'
            m.header.stamp = stamp
            m.ns = 'link_labels'
            m.id = marker_id; marker_id += 1
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = float(link_pos[0])
            m.pose.position.y = float(link_pos[1])
            m.pose.position.z = float(link_pos[2]) + 0.012
            m.scale.z = 0.006
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, 1.0
            if joint_type == p.JOINT_FIXED:
                m.color.r, m.color.g, m.color.b = 0.5, 0.5, 0.5  # Gray for fixed
            m.text = f'[{i}] {child_link_name} (j:{joint_name} {type_str})'
            marker_array.markers.append(m)

        self.debug_marker_pub.publish(marker_array)

    def publish_target_markers(self, targets_dict):
        marker_array = MarkerArray()
        
        for i, (finger_name, position) in enumerate(targets_dict.items()):
            marker = Marker()
            marker.header.frame_id = "palm_lower"  # Or "world" depending on your TF tree
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "fingertip_targets"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Position from your IK logic
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = float(position[2])
            
            # Scale and Color
            marker.scale.x = 0.01  # 1cm sphere
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.a = 1.0 
            # diff finger colors
            if finger_name == 'thumb': # orange
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.0
            elif finger_name == 'index': # green
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            elif finger_name == 'middle': # blue
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            elif finger_name == 'ring': # purple
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            
            marker_array.markers.append(marker)
            
        self.marker_pub.publish(marker_array)

    def find_link_indices(self):
        """Find link indices for fingertips and palm from robot.urdf"""
        self.fingertip_link_indices = {}
        self.palm_link_id = -1  # Default to root link for Leap Hand

        num_joints = p.getNumJoints(self.robot_id)
        
        # Iterate through all joints to find child links
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode("utf-8")
            
            # Match the specific link names in robot.urdf [cite: 1, 4, 7, 10]
            if link_name == "fingertip":
                self.fingertip_link_indices['index'] = i
            elif link_name == "fingertip_2":
                self.fingertip_link_indices['middle'] = i
            elif link_name == "fingertip_3":
                self.fingertip_link_indices['ring'] = i
            elif link_name == "thumb_fingertip":
                self.fingertip_link_indices['thumb'] = i
            
            # Check if this link is the palm
            if link_name == "palm_lower":
                self.palm_link_id = i

        # If palm_lower wasn't found in the joint list, it is the root base (index -1)
        if self.palm_link_id is None:
            # Check the base name directly
            base_name = p.getVisualShapeData(self.robot_id)[0][4].decode("utf-8")
            if "palm_lower" in base_name:
                self.palm_link_id = -1 # PyBullet base index
                self.get_logger().info('Found palm_lower as the root base link')

        if self.palm_link_id is None:
            self.get_logger().warn("Could not find 'palm_lower' link in robot.urdf!")
        
    def rotation_rokoko_to_leap(self, vec, finger_name=None):
        rx, ry, rz = vec
        # return [-rx, -rz, ry]
        return [ry, -rx, -rz]

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

        # Execute the appropriate control method
        if self.control_type == 'direct':
            joint_angles = self.direct_joint_angle_control(parsed_data)
        elif self.control_type == 'fingertip_ik':
            joint_angles = self.fingertip_ik_control(parsed_data)
        elif self.control_type == 'jparse_ik':
            joint_angles = self.jparse_ik_control(parsed_data)
        else:
            self.get_logger().warn(f'Unknown control type: {self.control_type}')
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
        Direct joint angle control with corrected Cartesian marker visualization.
        Matches the working IK logic to prevent 'far away' marker drift.
        """
        self.get_logger().debug('Using direct joint angle control')
        joint_angles = {}
        
        # Standard gains for the thumb
        thumb_gains = {"12": 1.8, "13": 1.5, "14": 1.2, "15": 1.0}
        
        # 1. Map joints directly (with selective inversion if needed)
        for leap_joint, rokoko_joint in self.joint_mapping.items():
            val = parsed_data.get(rokoko_joint, 0.0)
            rad = math.radians(val) if self.data_in_degrees else val

            if leap_joint in thumb_gains:
                neutral = self.thumb_neutral[leap_joint]
                angle = neutral + thumb_gains[leap_joint] * (rad - neutral)
            else:
                angle = rad

            # Clamp to limits
            if leap_joint in self.joint_limits_rad:
                min_limit, max_limit = self.joint_limits_rad[leap_joint]
                angle = max(min_limit, min(max_limit, angle))

            joint_angles[leap_joint] = angle
        
        # 2. Extract Cartesian positions for visualization
        leap_tips_in_world = {}
        
        # Get current palm state
        if self.palm_link_id is None or self.palm_link_id == -1:
            palm_pos, palm_quat = p.getBasePositionAndOrientation(self.robot_id)
        else:
            palm_state = p.getLinkState(self.robot_id, self.palm_link_id)
            palm_pos, palm_quat = palm_state[0], palm_state[1]

        palm_pos = np.array(palm_pos)
        rot_matrix = np.array(p.getMatrixFromQuaternion(palm_quat)).reshape(3, 3)

        # Reference offsets used in working IK logic
        # Using a slightly smaller offset for visualization to sit on the mesh
        fingertip_offset = np.array([0.08, 0.0, -0.02]) 
        thumb_offset = np.array([0.08, 0.0, -0.01])
        
        rotated_finger_offset = rot_matrix.dot(fingertip_offset)
        rotated_thumb_offset = rot_matrix.dot(thumb_offset)

        for finger_name, pos_columns in self.tip_position_mapping.items():
            rokoko_tip = None
            if all(col in parsed_data for col in pos_columns):
                rokoko_tip = np.array([parsed_data[col] for col in pos_columns], dtype=float)
            else:
                digit_num = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4}[finger_name]
                alt = [f'RightDigit{digit_num}Tip_pos_{c}' for c in ['x', 'y', 'z']]
                if all(col in parsed_data for col in alt):
                    rokoko_tip = np.array([parsed_data[col] for col in alt], dtype=float)

            if rokoko_tip is not None:
                palm_cols = ['RightHand_position_x', 'RightHand_position_y', 'RightHand_position_z']
                if all(col in parsed_data for col in palm_cols):
                    rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)
                    rokoko_rel_vec = rokoko_tip - rokoko_palm
                    
                    leap_rel_vec = np.array([rokoko_rel_vec[1], -rokoko_rel_vec[0], -rokoko_rel_vec[2]])
                    
                    # Apply rotated offsets
                    offset = rotated_thumb_offset if finger_name == 'thumb' else rotated_finger_offset
                    scale_factor = 1.7
                    
                    # Combine: World Palm + Rotated Relative Data + Rotated Hardware Offset
                    # This should yield a magnitude between 0.12m and 0.18m
                    leap_tips_in_world[finger_name] = palm_pos + (leap_rel_vec * scale_factor) - offset

        # 3. Publish the markers to RViz
        if leap_tips_in_world:
            self.publish_target_markers(leap_tips_in_world)

        joint_angles_list = [joint_angles.get(name, 0.0) for name in self.joint_names]
        return joint_angles_list

    def rotation_rokoko_to_leap(self, vec, finger_name=None):
        """
        Updated to match the reference switch_vector_from_rokoko logic.
        """
        rx, ry, rz = vec
        return [ry, -rx, -rz]

    def get_joint_limits(self, robot):
        joint_lower_limits = []
        joint_upper_limits = []
        joint_ranges = []
        for i in range(p.getNumJoints(robot)):
            joint_info = p.getJointInfo(robot, i)
            if joint_info[2] == p.JOINT_FIXED:
                continue
            joint_lower_limits.append(joint_info[8])
            joint_upper_limits.append(joint_info[9])
            joint_ranges.append(joint_info[9] - joint_info[8])
        return joint_lower_limits, joint_upper_limits, joint_ranges
    
    def fingertip_ik_control(self, parsed_data):
        """
        Revised IK control based on LeapPybulletIK implementation.
        Uses Null Space optimization and RealTip targeting.
        """
        # Get Current Palm State
        if self.palm_link_id is None or self.palm_link_id == -1:
            palm_pos, palm_quat = p.getBasePositionAndOrientation(self.robot_id)
        else:
            palm_state = p.getLinkState(self.robot_id, self.palm_link_id)
            palm_pos, palm_quat = palm_state[0], palm_state[1]

        palm_pos = np.array(palm_pos)

        leap_tips_in_world = {}
        for finger_name, pos_columns in self.tip_position_mapping.items():
            rokoko_tip = None
            if all(col in parsed_data for col in pos_columns):
                rokoko_tip = np.array([parsed_data[col] for col in pos_columns], dtype=float)
            else:
                digit_num = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4}[finger_name]
                alt = [f'RightDigit{digit_num}Tip_pos_{c}' for c in ['x', 'y', 'z']]
                if all(col in parsed_data for col in alt):
                    rokoko_tip = np.array([parsed_data[col] for col in alt], dtype=float)

            if rokoko_tip is None: continue

            palm_cols = ['RightHand_position_x', 'RightHand_position_y', 'RightHand_position_z']
            if all(col in parsed_data for col in palm_cols):
                rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)
                rokoko_rel_vec = rokoko_tip - rokoko_palm
                leap_rel_vec = self.rotation_rokoko_to_leap(rokoko_rel_vec)
                leap_tips_in_world[finger_name] = palm_pos + leap_rel_vec

        # Solve IK for the 'realtip' indices (4, 9, 14, 19)
        finger_order = ["index", "middle", "ring", "thumb"]
        # target_indices = {"index": 4, "middle": 9, "ring": 14, "thumb": 19}
        target_indices = {"index": 3, "middle": 8, "ring": 13, "thumb": 18} # Updated to match the correct fingertip link indices
        
        final_joint_angles = [0.0] * 16

        for i, finger_name in enumerate(finger_order):
            if finger_name not in leap_tips_in_world: continue

            target_pos = leap_tips_in_world[finger_name]
            target_link_idx = target_indices[finger_name]

            # Calculate IK using Null Space (restPoses)
            # This returns results for ALL joints (including fixed ones)
            ik_result = p.calculateInverseKinematics(
                self.robot_id,
                target_link_idx,
                target_pos,
                lowerLimits=self.lower_limits,
                upperLimits=self.upper_limits,
                jointRanges=self.joint_ranges,
                restPoses=self.rest_poses,
                maxNumIterations=1000,
                residualThreshold=0.001
            )

            # Slice for the relevant finger joints only
            start_idx = i * 4
            for j in range(4):
                angle = ik_result[start_idx + j]
                final_joint_angles[start_idx + j] = angle
                p.resetJointState(self.robot_id, start_idx + j, angle)
            self.publish_target_markers(leap_tips_in_world)

        # 6. Post-process: Swap first two joints of each finger (MCP/PIP swap)
        for start in [0, 4, 8]:
            final_joint_angles[start], final_joint_angles[start+1] = \
                final_joint_angles[start+1], final_joint_angles[start]

        return final_joint_angles

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

        # Create joint_names list from pybullet_to_leap mapping (sorted by index)
        joint_names = [self.pybullet_to_leap[idx] for idx in sorted(self.pybullet_to_leap.keys())]

        # Initialize joint angles dictionary with CURRENT PyBullet state
        # This ensures fingers without IK updates maintain their current positions
        joint_angles = {}
        for pb_idx, joint_name in self.pybullet_to_leap.items():
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
            'index': {
                'base_link': self.palm_link_id,
                'ee_link_idx': self.fingertip_link_indices['index'],
                'joint_indices': [0, 1, 2, 3],  # MCP_Flex -> MCP_Yaw -> PIP -> DIP
                'joint_names': ['1', '0', '2', '3']
            },
            'middle': {
                'base_link': self.palm_link_id,
                'ee_link_idx': self.fingertip_link_indices['middle'],
                'joint_indices': [5, 6, 7, 8],
                'joint_names': ['5', '4', '6', '7']
            },
            'ring': {
                'base_link': self.palm_link_id,
                'ee_link_idx': self.fingertip_link_indices['ring'],
                'joint_indices': [10, 11, 12, 13],
                'joint_names': ['9', '8', '10', '11']
            },
            'thumb': {
                'base_link': self.palm_link_id,
                'ee_link_idx': self.fingertip_link_indices['thumb'],
                'joint_indices': [15, 16, 17, 18],
                'joint_names': ['12', '13', '14', '15']
            }
        }

        # Debug: Log how many fingers we're processing
        # Get palm position from PyBullet (LEAP world frame)
        if self.palm_link_id == -1:
            # Use getBasePositionAndOrientation for the root link
            palm_pos, palm_quat = p.getBasePositionAndOrientation(self.robot_id)
        else:
            # Use getLinkState for any other link
            palm_link_info = p.getLinkState(self.robot_id, self.palm_link_id, computeForwardKinematics=True)
            palm_pos, palm_quat = palm_link_info[0], palm_link_info[1]
        # palm_link_info = p.getLinkState(self.robot_id, self.palm_link_id, computeForwardKinematics=True)
        leap_palm_in_world = np.array(palm_pos, dtype=float)
        leap_palm_in_world_arr = leap_palm_in_world.tolist()

        # debug
        # for fname, lidx in self.fingertip_link_indices.items():
        #     tip_state = p.getLinkState(self.robot_id, lidx)
        #     tip_pos = np.array(tip_state[0])
        #     rel = tip_pos - leap_palm_in_world
        #     self.get_logger().info(f'LEAP rest tip {fname}: rel={np.round(rel,4)}')

        # Get Rokoko palm position if available
        palm_cols = ['RightHand_position_x', 'RightHand_position_y', 'RightHand_position_z']
        rokoko_palm_available = all(col in parsed_data for col in palm_cols)
        if rokoko_palm_available:
            rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)
            leap_palm_from_rokoko = np.array(self.rotation_rokoko_to_leap(rokoko_palm), dtype=float)

        fingers_processed = 0
        targets_for_viz = {}

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
                digit_num = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4}[finger_name]
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

            # Transform Rokoko fingertip to LEAP coordinate frame
            leap_tip = np.array(self.rotation_rokoko_to_leap(rokoko_tip), dtype=float)

            # Compute target position in PyBullet world frame
            # Make fingertip position relative to palm, then add PyBullet palm position
            scale = 1.0
            if rokoko_palm_available:
                leap_tip_in_palm = (leap_tip - leap_palm_from_rokoko) * scale
                # y_scale = 2.0
                # leap_tip_in_palm[1] *= y_scale
                target_pos_world = leap_tip_in_palm + leap_palm_in_world_arr
            else:
                # No palm data - use absolute position (less accurate)
                target_pos_world = leap_tip
            
            # self.get_logger().info(
            #     f'{finger_name}: raw_rel={rokoko_tip - rokoko_palm}, '
            #     f'after_swizzle={leap_tip - leap_palm_from_rokoko}'
            # )

            target_pos_world = np.array(target_pos_world, dtype=float)
            targets_for_viz[finger_name] = target_pos_world.tolist()

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
                self.get_logger().info(f'LEAP tip: {leap_tip}')
                if rokoko_palm_available:
                    self.get_logger().info(f'Rokoko palm: {rokoko_palm}')
                    self.get_logger().info(f'LEAP palm (Rokoko): {leap_palm_from_rokoko}')
                    self.get_logger().info(f'LEAP palm (PyBullet): {leap_palm_in_world}')
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
                    return_nullspace=True
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
                delta_q = dq_task + dq_nullspace

                # Direct position solve with clamped error
                # delta_q = J_pinv @ pos_error_clamped

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
                    self.get_logger().info(f'delta_q: {delta_q}')
                    
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
                        f'delta_q(deg)={[f"{np.degrees(dq):.1f}" for dq in delta_q]}'
                    )

            except Exception as e:
                self.get_logger().error(f'jparse IK failed for {finger_name}: {e}')
                # Keep current positions on failure
                for i, joint_name in enumerate(chain_info['joint_names']):
                    joint_angles[joint_name] = current_joint_positions[i]

        # Debug: Log processing summary
        if self._jparse_log_count < 3:
            self.get_logger().info(f'Processed {fingers_processed}/4 fingers')
            self.get_logger().info(f'Joint angles computed: {len(joint_angles)} joints')
            if fingers_processed == 0:
                self.get_logger().error('NO FINGERS PROCESSED - No position data available!')
        
        if len(targets_for_viz) > 0:
            self.publish_target_markers(targets_for_viz)
        
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
        for pb_idx, joint_name in self.pybullet_to_leap.items():
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

    def publish_joint_state(self, angles):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = angles
        self.joint_pub.publish(msg)

    def destroy_node(self):
        """Clean up PyBullet connection"""
        if hasattr(self, 'physics_client_id'):
            p.disconnect(self.physics_client_id)
            self.get_logger().info('PyBullet physics client disconnected')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LeapRetargeting()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()