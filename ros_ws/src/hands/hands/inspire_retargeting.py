#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import pybullet as p
import os
from ament_index_python.packages import get_package_share_directory
import numpy as np
import math
from visualization_msgs.msg import Marker, MarkerArray


# Import jparse for advanced IK solving
try:
    import jparse_robotics as jparse
    JPARSE_AVAILABLE = True
except ImportError:
    JPARSE_AVAILABLE = False
    print("Warning: jparse not available. Install with: pip install git+https://github.com/armlabstanford/jparse.git")



class InspireRetargeting(Node):
    def __init__(self):
        super().__init__('inspire_retargeting')
        self.marker_pub = self.create_publisher(MarkerArray, 'fingertip_targets', 10)

        # Declare parameter for data format (degrees vs radians)
        self.declare_parameter('data_in_degrees', False)
        self.data_in_degrees = self.get_parameter('data_in_degrees').get_parameter_value().bool_value

        csv_path = os.path.join(get_package_share_directory('hands'), 'data', 'move_fingers_2.csv')
        self.declare_parameter('csv_file', csv_path)
        self.csv_file = self.get_parameter('csv_file').get_parameter_value().string_value

        if self.data_in_degrees:
            self.get_logger().info('Data format: DEGREES (will convert to radians)')
        else:
            self.get_logger().info('Data format: RADIANS (no conversion needed)')

        self.physics_client_id = p.connect(p.DIRECT)
        self.get_logger().info(f'PyBullet physics client connected in DIRECT mode')

        # Load Inspire hand URDF
        share_directory = get_package_share_directory('hands')
        urdf_path = os.path.join(share_directory, 'urdf', 'inspire', 'inspire_hand_left.urdf')
        try:
            self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
            self.get_logger().info(f'Loaded URDF from: {urdf_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            raise

        # Unified joint definition: pb_idx and limits per joint
        self.joints = {
            'thumb_proximal_yaw_joint':  {'pb_idx': 1,  'limits': [0.0, 1.308]},
            'thumb_proximal_pitch_joint':{'pb_idx': 2,  'limits': [0.0, 0.6]},
            'thumb_intermediate_joint':  {'pb_idx': 3,  'limits': [0.0, 0.8]},
            'thumb_distal_joint':        {'pb_idx': 4,  'limits': [0.0, 0.4]},

            'index_proximal_joint':      {'pb_idx': 6,  'limits': [0.0, 1.47]},
            'index_intermediate_joint':  {'pb_idx': 7,  'limits': [-0.04545, 1.56]},

            'middle_proximal_joint':     {'pb_idx': 9,  'limits': [0.0, 1.47]},
            'middle_intermediate_joint': {'pb_idx': 10, 'limits': [-0.04545, 1.56]},

            'ring_proximal_joint':       {'pb_idx': 12, 'limits': [0.0, 1.47]},
            'ring_intermediate_joint':   {'pb_idx': 13, 'limits': [-0.04545, 1.56]},

            'pinky_proximal_joint':      {'pb_idx': 15, 'limits': [0.0, 1.47]},
            'pinky_intermediate_joint':  {'pb_idx': 16, 'limits': [-0.04545, 1.56]},
        }

        # Derive convenience structures from unified joint definition
        self.pybullet_to_inspire = {v['pb_idx']: k for k, v in self.joints.items()}
        self.joint_names = [self.pybullet_to_inspire[idx] for idx in sorted(self.pybullet_to_inspire.keys())]

        # Joint mapping from CSV columns to INSPIRE Hand joint names
        self.joint_mapping = {
            'thumb_proximal_yaw_joint':   'LeftDigit1Carpometacarpal_ulnarDeviation',
            'thumb_proximal_pitch_joint': 'LeftDigit1Carpometacarpal_flexion',
            'thumb_intermediate_joint':   'LeftDigit1Metacarpophalangeal_flexion',
            'thumb_distal_joint':         'LeftDigit1Interphalangeal_flexion',
            'index_proximal_joint':       'LeftDigit2Metacarpophalangeal_flexion',
            'index_intermediate_joint':   'LeftDigit2ProximalInterphalangeal_flexion',
            'middle_proximal_joint':      'LeftDigit3Metacarpophalangeal_flexion',
            'middle_intermediate_joint':  'LeftDigit3ProximalInterphalangeal_flexion',
            'ring_proximal_joint':        'LeftDigit4Metacarpophalangeal_flexion',
            'ring_intermediate_joint':    'LeftDigit4ProximalInterphalangeal_flexion',
            'pinky_proximal_joint':       'LeftDigit5Metacarpophalangeal_flexion',
            'pinky_intermediate_joint':   'LeftDigit5ProximalInterphalangeal_flexion',
        }

        # Position columns for fingertips (for inverse kinematics)
        self.tip_position_mapping = {
            'thumb':  ['LeftDigit1DistalPhalanx_position_x', 'LeftDigit1DistalPhalanx_position_y', 'LeftDigit1DistalPhalanx_position_z'],
            'index':  ['LeftDigit2DistalPhalanx_position_x', 'LeftDigit2DistalPhalanx_position_y', 'LeftDigit2DistalPhalanx_position_z'],
            'middle': ['LeftDigit3DistalPhalanx_position_x', 'LeftDigit3DistalPhalanx_position_y', 'LeftDigit3DistalPhalanx_position_z'],
            'ring':   ['LeftDigit4DistalPhalanx_position_x', 'LeftDigit4DistalPhalanx_position_y', 'LeftDigit4DistalPhalanx_position_z'],
            'pinky':  ['LeftDigit5DistalPhalanx_position_x', 'LeftDigit5DistalPhalanx_position_y', 'LeftDigit5DistalPhalanx_position_z'],
        }

        # Identify movable joints in PyBullet
        self.movable_joints = []
        self.num_joints = p.getNumJoints(self.robot_id)
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] != p.JOINT_FIXED:
                self.movable_joints.append(i)

        self.find_link_indices()

        # Derive joint limits and ranges from unified joint definition
        sorted_joints = sorted(self.joints.items(), key=lambda x: x[1]['pb_idx'])
        self.lower_limits = [j[1]['limits'][0] for j in sorted_joints]
        self.upper_limits = [j[1]['limits'][1] for j in sorted_joints]
        self.joint_ranges = [upper - lower for lower, upper in zip(self.lower_limits, self.upper_limits)]
        self.joint_limits_rad = {name: data['limits'] for name, data in self.joints.items()}
        self.joint_limits_deg = {j: [math.degrees(l[0]), math.degrees(l[1])] for j, l in self.joint_limits_rad.items()}
        self.use_joint_limits = True

        # Finger chains for jparse IK — only independent (non-mimic) joints
        self.finger_chains = {
            'thumb': {
                'base_link':    self.palm_link_id,
                'ee_link_idx':  self.fingertip_link_indices['thumb'],
                'joint_indices': [1, 2],
                'joint_names':  ['thumb_proximal_yaw_joint', 'thumb_proximal_pitch_joint'],
            },
            'index': {
                'base_link':    self.palm_link_id,
                'ee_link_idx':  self.fingertip_link_indices['index'],
                'joint_indices': [6],
                'joint_names':  ['index_proximal_joint'],
            },
            'middle': {
                'base_link':    self.palm_link_id,
                'ee_link_idx':  self.fingertip_link_indices['middle'],
                'joint_indices': [9],
                'joint_names':  ['middle_proximal_joint'],
            },
            'ring': {
                'base_link':    self.palm_link_id,
                'ee_link_idx':  self.fingertip_link_indices['ring'],
                'joint_indices': [12],
                'joint_names':  ['ring_proximal_joint'],
            },
            'pinky': {
                'base_link':    self.palm_link_id,
                'ee_link_idx':  self.fingertip_link_indices['pinky'],
                'joint_indices': [15],
                'joint_names':  ['pinky_proximal_joint'],
            },
        }

        # Mimic joint relationships from URDF
        # mimic_joint_idx: (parent_joint_name, parent_pb_idx, multiplier, offset)
        self.mimic_joints = {
            3:  ('thumb_proximal_pitch_joint', 2,  1.334,   0.0),
            4:  ('thumb_proximal_pitch_joint', 2,  0.667,   0.0),
            7:  ('index_proximal_joint',       6,  1.06399, -0.04545),
            10: ('middle_proximal_joint',      9,  1.06399, -0.04545),
            13: ('ring_proximal_joint',        12, 1.06399, -0.04545),
            16: ('pinky_proximal_joint',       15, 1.06399, -0.04545),
        }

        # Rest poses
        self.rest_poses = [p.getJointState(self.robot_id, pb_idx)[0] for pb_idx in self.movable_joints]

        for i in range(self.num_joints):
            self.get_logger().info(f'Joint info: {p.getJointInfo(self.robot_id, i)[1].decode("utf-8")}')

        # Subscribers
        self.control_type_sub = self.create_subscription(
            String, 'control_type', self.control_type_callback, 10)
        self.rokoko_data_sub = self.create_subscription(
            String, 'rokoko_ref_data', self.rokoko_data_callback, 10)

        # Publisher for retargeted joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        self.control_type = 'direct'
        self.latest_rokoko_data = None

        self.get_logger().info('Inspire Retargeting Node started')

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def find_link_indices(self):
        """Find link indices for fingertips and palm from URDF."""
        self.fingertip_link_indices = {}
        self.palm_link_id = None

        self.get_logger().info(f'URDF has {self.num_joints} joints')

        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode("utf-8")

            if link_name == "thumb_tip":
                self.fingertip_link_indices['thumb'] = i
                self.get_logger().info(f'Found thumb tip at link index {i}')
            elif link_name == "index_tip":
                self.fingertip_link_indices['index'] = i
                self.get_logger().info(f'Found index tip at link index {i}')
            elif link_name == "middle_tip":
                self.fingertip_link_indices['middle'] = i
                self.get_logger().info(f'Found middle tip at link index {i}')
            elif link_name == "ring_tip":
                self.fingertip_link_indices['ring'] = i
                self.get_logger().info(f'Found ring tip at link index {i}')
            elif link_name == "pinky_tip":
                self.fingertip_link_indices['pinky'] = i
                self.get_logger().info(f'Found pinky tip at link index {i}')
            elif link_name == "hand_base_link":
                self.palm_link_id = i
                self.get_logger().info(f'Found palm link at index {i}')

        if self.palm_link_id is None:
            self.get_logger().warn("Palm link not found in URDF!")

        self.get_logger().info(f'Fingertip link indices: {self.fingertip_link_indices}')

    def rotation_rokoko_to_inspire(self, vec, finger_name=None):
        """Map a 3-vector in Rokoko character axes into Inspire/PyBullet world axes."""
        rx, ry, rz = vec
        return [rz, -rx, ry]

    def get_rokoko_tip(self, parsed_data, finger_name):
        """Return fingertip position array for a finger, trying both naming conventions."""
        pos_columns = self.tip_position_mapping[finger_name]
        if all(col in parsed_data for col in pos_columns):
            return np.array([parsed_data[col] for col in pos_columns], dtype=float)

        digit_num = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5}[finger_name]
        alt = [f'LeftDigit{digit_num}Tip_pos_{c}' for c in ['x', 'y', 'z']]
        if all(col in parsed_data for col in alt):
            return np.array([parsed_data[col] for col in alt], dtype=float)

        return None

    def publish_target_markers(self, targets_dict):
        marker_array = MarkerArray()
        colors = {
            'thumb':  (1.0, 0.5, 0.0),
            'index':  (0.0, 1.0, 0.0),
            'middle': (0.0, 0.0, 1.0),
            'ring':   (1.0, 0.0, 1.0),
            'pinky':  (0.0, 1.0, 1.0),
        }
        for i, (finger_name, position) in enumerate(targets_dict.items()):
            marker = Marker()
            marker.header.frame_id = "base"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "fingertip_targets"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = float(position[2])
            marker.scale.x = marker.scale.y = marker.scale.z = 0.01
            marker.color.a = 1.0
            r, g, b = colors.get(finger_name, (1.0, 1.0, 1.0))
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def publish_joint_state(self, joint_angles):
        """Publish the retargeted joint state."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = joint_angles
        self.joint_pub.publish(msg)

    def clamp_joint_angles(self, joint_angles):
        """Clamp joint angles to stay within physical limits."""
        if not self.use_joint_limits:
            return joint_angles

        clamped_angles = []
        for i, joint_name in enumerate(self.joint_names):
            angle = joint_angles[i]
            if joint_name in self.joint_limits_rad:
                min_limit, max_limit = self.joint_limits_rad[joint_name]
                if angle < min_limit or angle > max_limit:
                    clamped_angle = max(min_limit, min(max_limit, angle))
                    if not hasattr(self, '_clamp_log_count'):
                        self._clamp_log_count = {}
                    self._clamp_log_count.setdefault(joint_name, 0)
                    self._clamp_log_count[joint_name] += 1
                    if self._clamp_log_count[joint_name] % 100 == 1:
                        self.get_logger().warn(
                            f'Clamping {joint_name}: {math.degrees(angle):.1f}° → '
                            f'{math.degrees(clamped_angle):.1f}° '
                            f'(limits: {math.degrees(min_limit):.1f}° to {math.degrees(max_limit):.1f}°)'
                        )
                    clamped_angles.append(clamped_angle)
                else:
                    clamped_angles.append(angle)
            else:
                clamped_angles.append(angle)
        return clamped_angles

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def control_type_callback(self, msg):
        self.control_type = msg.data

    def rokoko_data_callback(self, msg):
        self.latest_rokoko_data = msg.data
        parsed_data = self.parse_csv_data(msg.data)
        if not parsed_data:
            return

        if self.control_type == 'direct':
            joint_angles = self.direct_joint_angle_control(parsed_data)
        elif self.control_type == 'fingertip_ik':
            joint_angles = self.fingertip_ik_control(parsed_data)
        elif self.control_type == 'jparse_ik':
            joint_angles = self.jparse_ik_control(parsed_data)
        else:
            self.get_logger().warn(f'Unknown control type: {self.control_type}')
            return

        if joint_angles:
            self.publish_joint_state(joint_angles)

    def parse_csv_data(self, csv_string):
        """Parse CSV-format Rokoko data. Returns dict of {joint_name: float}."""
        parsed = {}
        for line in csv_string.strip().split('\n'):
            parts = line.split(',')
            if len(parts) < 3:
                continue
            _, joint_name, value = parts[0], parts[1], parts[2]
            try:
                parsed[joint_name] = float(value)
            except ValueError:
                self.get_logger().warn(f'Could not parse value for {joint_name}: {value}')
        return parsed

    # -------------------------------------------------------------------------
    # Control methods
    # -------------------------------------------------------------------------

    def direct_joint_angle_control(self, parsed_data):
        """
        Control method 1: Direct joint angle mapping with Cartesian marker visualization.
        """
        joint_angles = {}

        # Map joints directly
        for inspire_joint, rokoko_joint in self.joint_mapping.items():
            val = parsed_data.get(rokoko_joint, 0.0)
            joint_angles[inspire_joint] = math.radians(val) if self.data_in_degrees else val

        # Compute fingertip markers via palm-relative transform
        inspire_tips_in_world = {}

        palm_state = p.getLinkState(self.robot_id, self.palm_link_id) if self.palm_link_id is not None \
            else None
        if palm_state is not None:
            palm_pos = np.array(palm_state[0])
            palm_cols = ['LeftHand_position_x', 'LeftHand_position_y', 'LeftHand_position_z']

            if all(col in parsed_data for col in palm_cols):
                rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)

                for finger_name in self.tip_position_mapping:
                    rokoko_tip = self.get_rokoko_tip(parsed_data, finger_name)
                    if rokoko_tip is not None:
                        rokoko_rel = rokoko_tip - rokoko_palm
                        inspire_rel = np.array(self.rotation_rokoko_to_inspire(rokoko_rel, finger_name))
                        inspire_tips_in_world[finger_name] = palm_pos + inspire_rel

        if inspire_tips_in_world:
            self.publish_target_markers(inspire_tips_in_world)

        return [joint_angles.get(name, 0.0) for name in self.joint_names]

    def fingertip_ik_control(self, parsed_data):
        """
        Control method 2: Fingertip IK-based control using PyBullet.
        Uses fingertip positions to solve inverse kinematics.
        """
        self.get_logger().info('ENTERED fingertip_ik_control function', once=True)

        # Get palm position from PyBullet
        palm_link_info = p.getLinkState(self.robot_id, self.palm_link_id, computeForwardKinematics=True)
        inspire_palm_in_world = np.array(palm_link_info[0], dtype=float)

        # Build fingertip world-frame targets
        inspire_tips_in_world = {}
        palm_cols = ['LeftHand_position_x', 'LeftHand_position_y', 'LeftHand_position_z']

        if all(col in parsed_data for col in palm_cols):
            rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)
            inspire_palm_rokoko = np.array(self.rotation_rokoko_to_inspire(rokoko_palm), dtype=float)

            for finger_name in self.tip_position_mapping:
                rokoko_tip = self.get_rokoko_tip(parsed_data, finger_name)
                if rokoko_tip is None:
                    continue
                inspire_tip = np.array(self.rotation_rokoko_to_inspire(rokoko_tip), dtype=float)
                inspire_tip_in_palm = inspire_tip - inspire_palm_rokoko
                inspire_tips_in_world[finger_name] = inspire_tip_in_palm + inspire_palm_in_world

                if not hasattr(self, '_logged_ik_fingertips'):
                    self.get_logger().info(f'{finger_name} tip (world): {inspire_tips_in_world[finger_name]}')

        if not hasattr(self, '_logged_ik_fingertips'):
            self._logged_ik_fingertips = True
            self.get_logger().info('=' * 80)
            self.get_logger().info(f'Using PyBullet IK for {len(inspire_tips_in_world)} fingertips')
            self.get_logger().info('=' * 80)

        # Build per-finger PyBullet joint index lists
        finger_order = ["thumb", "index", "middle", "ring", "pinky"]
        finger_joint_map = {
            'thumb':  ['thumb_proximal_yaw_joint', 'thumb_proximal_pitch_joint',
                       'thumb_intermediate_joint', 'thumb_distal_joint'],
            'index':  ['index_proximal_joint', 'index_intermediate_joint'],
            'middle': ['middle_proximal_joint', 'middle_intermediate_joint'],
            'ring':   ['ring_proximal_joint', 'ring_intermediate_joint'],
            'pinky':  ['pinky_proximal_joint', 'pinky_intermediate_joint'],
        }
        finger_to_pb_indices = {
            finger: [idx for idx, name in self.pybullet_to_inspire.items() if name in joint_names]
            for finger, joint_names in finger_joint_map.items()
        }

        pb_to_array_idx = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5, 9: 6, 10: 7, 12: 8, 13: 9, 15: 10, 16: 11}
        current_joint_angles = self.rest_poses.copy()

        for finger_name in finger_order:
            if finger_name not in inspire_tips_in_world:
                continue

            target_pos = inspire_tips_in_world[finger_name]
            link_index = self.fingertip_link_indices[finger_name]

            try:
                ik_result = p.calculateInverseKinematics(
                    self.robot_id, link_index, target_pos,
                    lowerLimits=self.lower_limits,
                    upperLimits=self.upper_limits,
                    jointRanges=self.joint_ranges,
                    restPoses=current_joint_angles,
                    maxNumIterations=200,
                    residualThreshold=1e-4,
                )
                if finger_name == 'thumb':
                    self.get_logger().info(f"Thumb IK Output: {ik_result[0:4]}")

                for pb_idx in finger_to_pb_indices[finger_name]:
                    try:
                        ik_idx = self.movable_joints.index(pb_idx)
                        angle = ik_result[ik_idx]
                        current_joint_angles[pb_to_array_idx[pb_idx]] = angle
                        p.resetJointState(self.robot_id, pb_idx, angle)
                    except ValueError:
                        self.get_logger().warn(f'IK mapping error: {pb_idx} not in movable joints')

            except Exception as e:
                self.get_logger().warn(f'IK failed for {finger_name}: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())

        self.publish_target_markers(inspire_tips_in_world)
        return [current_joint_angles[idx] for idx in range(12)]

    def jparse_ik_control(self, parsed_data):
        """
        Control method 3: IK with JPARSE (Joint Position And Rotation Solver Engine).
        Uses PyBullet's Jacobian calculation with jparse for advanced IK solving.
        Solves IK for each finger independently from base to fingertip.
        """
        if not JPARSE_AVAILABLE:
            self.get_logger().error('jparse not available! Falling back to zero positions')
            return [0.0] * len(self.joint_names)

        # Initialize debug counters once
        if not hasattr(self, '_jparse_log_count'):
            self._jparse_log_count = 0
        if not hasattr(self, '_pos_debug_count'):
            self._pos_debug_count = 0
        if not hasattr(self, '_jac_debug_count'):
            self._jac_debug_count = 0

        # Initialize joint angles from current PyBullet state
        joint_angles = {
            joint_name: p.getJointState(self.robot_id, pb_idx)[0]
            for pb_idx, joint_name in self.pybullet_to_inspire.items()
        }

        # Get palm position from PyBullet
        palm_link_info = p.getLinkState(self.robot_id, self.palm_link_id, computeForwardKinematics=True)
        inspire_palm_in_world = np.array(palm_link_info[0], dtype=float)

        # Get Rokoko palm if available
        palm_cols = ['LeftHand_position_x', 'LeftHand_position_y', 'LeftHand_position_z']
        rokoko_palm_available = all(col in parsed_data for col in palm_cols)
        rokoko_palm = None
        inspire_palm_from_rokoko = None
        if rokoko_palm_available:
            rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)
            inspire_palm_from_rokoko = np.array(self.rotation_rokoko_to_inspire(rokoko_palm), dtype=float)

        fingers_processed = 0
        targets_for_viz = {}

        for finger_name, chain_info in self.finger_chains.items():
            rokoko_tip = self.get_rokoko_tip(parsed_data, finger_name)
            if rokoko_tip is None:
                if self._jparse_log_count < 3:
                    self.get_logger().warn(f'No position data for {finger_name}, skipping')
                continue

            fingers_processed += 1

            # Transform to world frame
            inspire_tip = np.array(self.rotation_rokoko_to_inspire(rokoko_tip), dtype=float)
            if rokoko_palm_available:
                target_pos_world = (inspire_tip - inspire_palm_from_rokoko) + inspire_palm_in_world
            else:
                target_pos_world = inspire_tip
            targets_for_viz[finger_name] = target_pos_world.tolist()

            # Current finger joint positions and EE position
            current_joint_positions = [
                p.getJointState(self.robot_id, idx)[0] for idx in chain_info['joint_indices']
            ]
            current_ee_pos = np.array(p.getLinkState(self.robot_id, chain_info['ee_link_idx'])[0])
            pos_error = target_pos_world - current_ee_pos

            # Debug: Log positions for first finger (first few times)
            if self._pos_debug_count < 3 and finger_name == 'thumb':
                self.get_logger().info(f'=== {finger_name} Position Debug ===')
                self.get_logger().info(f'Rokoko tip (raw): {rokoko_tip}')
                self.get_logger().info(f'INSPIRE tip: {inspire_tip}')
                if rokoko_palm_available:
                    self.get_logger().info(f'Rokoko palm: {rokoko_palm}')
                    self.get_logger().info(f'INSPIRE palm (Rokoko): {inspire_palm_from_rokoko}')
                    self.get_logger().info(f'INSPIRE palm (PyBullet): {inspire_palm_in_world}')
                self.get_logger().info(f'Target pos (world): {target_pos_world}')
                self.get_logger().info(f'Current pos: {current_ee_pos}')
                self.get_logger().info(f'Error magnitude: {np.linalg.norm(pos_error):.6f}m')
                self._pos_debug_count += 1

            # Build Jacobian
            joint_positions, joint_velocities, joint_accelerations = [], [], []
            joint_index_to_col = {}
            col_idx = 0
            for i in range(self.num_joints):
                joint_type = p.getJointInfo(self.robot_id, i)[2]
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    js = p.getJointState(self.robot_id, i)
                    joint_positions.append(js[0])
                    joint_velocities.append(js[1])
                    joint_accelerations.append(0.0)
                    joint_index_to_col[i] = col_idx
                    col_idx += 1

            jacobian_linear, _ = p.calculateJacobian(
                bodyUniqueId=self.robot_id,
                linkIndex=chain_info['ee_link_idx'],
                localPosition=[0, 0, 0],
                objPositions=joint_positions,
                objVelocities=joint_velocities,
                objAccelerations=joint_accelerations,
            )
            J_linear = np.array(jacobian_linear)
            finger_col_indices = [
                joint_index_to_col[idx] for idx in chain_info['joint_indices']
                if idx in joint_index_to_col
            ]
            J_finger_linear = J_linear[:, finger_col_indices]

            try:
                jp_solver = jparse.JParseCore(gamma=0.1)
                J_pinv = jp_solver.compute(
                    jacobian=J_finger_linear,
                    singular_direction_gain_position=1.0,
                    position_dimensions=3,
                    return_nullspace=False,
                )

                # Clamp step size
                max_step = 0.005
                error_norm = np.linalg.norm(pos_error)
                pos_error_clamped = pos_error * (max_step / error_norm) if error_norm > max_step else pos_error
                delta_q = J_pinv @ pos_error_clamped

                delta_q_max_deg = np.degrees(np.max(np.abs(delta_q)))
                if delta_q_max_deg > 30:
                    self.get_logger().warn(
                        f'{finger_name}: LARGE delta_q! max={delta_q_max_deg:.1f}° '
                        f'Error={error_norm * 1000:.1f}mm'
                    )

                if self._jac_debug_count < 2 and finger_name == 'thumb':
                    self.get_logger().info(f'J_finger_linear shape: {J_finger_linear.shape}')
                    self.get_logger().info(f'pos_error norm: {error_norm * 1000:.1f}mm')
                    self.get_logger().info(f'delta_q: {delta_q}')
                    self._jac_debug_count += 1

                for i, jname in enumerate(chain_info['joint_names']):
                    new_angle = current_joint_positions[i] + delta_q[i]
                    if jname in self.joint_limits_rad:
                        lo, hi = self.joint_limits_rad[jname]
                        new_angle = max(lo, min(hi, new_angle))
                    joint_angles[jname] = new_angle

                self._jparse_log_count += 1
                if self._jparse_log_count < 20 or self._jparse_log_count % 60 == 0:
                    self.get_logger().info(
                        f'{finger_name}: err={error_norm * 1000:.1f}mm, '
                        f'delta_q_max={np.degrees(np.max(np.abs(delta_q))):.1f}°'
                    )

            except Exception as e:
                self.get_logger().error(f'jparse IK failed for {finger_name}: {e}')
                for i, jname in enumerate(chain_info['joint_names']):
                    joint_angles[jname] = current_joint_positions[i]

        if self._jparse_log_count < 3:
            self.get_logger().info(f'Processed {fingers_processed}/5 fingers')
            if fingers_processed == 0:
                self.get_logger().error('NO FINGERS PROCESSED - No position data available!')

        if targets_for_viz:
            self.publish_target_markers(targets_for_viz)

        joint_angles_list = [joint_angles.get(name, 0.0) for name in self.joint_names]

        # Update PyBullet with new independent joint angles
        for pb_idx, joint_name in self.pybullet_to_inspire.items():
            if joint_name in joint_angles:
                p.resetJointState(self.robot_id, pb_idx, joint_angles[joint_name])

        # Update mimic joints
        for mimic_pb_idx, (parent_name, _, multiplier, offset) in self.mimic_joints.items():
            if parent_name in joint_angles:
                mimic_angle = joint_angles[parent_name] * multiplier + offset
                p.resetJointState(self.robot_id, mimic_pb_idx, mimic_angle)
                mimic_name = self.pybullet_to_inspire.get(mimic_pb_idx)
                if mimic_name:
                    joint_angles[mimic_name] = mimic_angle

        return joint_angles_list


def main(args=None):
    rclpy.init(args=args)
    inspire_retargeting = InspireRetargeting()
    try:
        rclpy.spin(inspire_retargeting)
    except KeyboardInterrupt:
        pass
    finally:
        inspire_retargeting.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
    

# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String
# from sensor_msgs.msg import JointState
# import pybullet as p
# import os
# from ament_index_python.packages import get_package_share_directory
# import numpy as np
# import math
# from visualization_msgs.msg import Marker, MarkerArray
# from geometry_msgs.msg import PoseStamped


# # Import jparse for advanced IK solving
# try:
#     import jparse_robotics as jparse
#     JPARSE_AVAILABLE = True
# except ImportError:
#     JPARSE_AVAILABLE = False
#     print("Warning: jparse not available. Install with: pip install git+https://github.com/armlabstanford/jparse.git")



# class InspireRetargeting(Node):
#     def __init__(self):
#         super().__init__('inspire_retargeting')
#         self.marker_pub = self.create_publisher(MarkerArray, 'fingertip_targets', 10)

#         # Declare parameter for data format (degrees vs radians)
#         self.declare_parameter('data_in_degrees', False)
#         self.data_in_degrees = self.get_parameter('data_in_degrees').get_parameter_value().bool_value
        
#         csv_path = os.path.join(get_package_share_directory('hands'), 'data','move_fingers_2.csv')
#         self.declare_parameter('csv_file', csv_path)
#         self.csv_file = self.get_parameter('csv_file').get_parameter_value().string_value

#         if self.data_in_degrees:
#             self.get_logger().info('Data format: DEGREES (will convert to radians)')
#         else:
#             self.get_logger().info('Data format: RADIANS (no conversion needed)')

#         self.physics_client_id = p.connect(p.DIRECT)
#         self.get_logger().info(f'PyBullet physics client connected in DIRECT mode')

#         # Load Inspire hand URDF
#         share_directory = get_package_share_directory('hands')
#         urdf_path = os.path.join(share_directory, 'urdf', 'inspire', 'inspire_hand_left.urdf')
#         try:
#             self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
#             self.get_logger().info(f'Loaded URDF from: {urdf_path}')
#         except Exception as e:
#             self.get_logger().error(f'Failed to load URDF: {e}')
#             raise

#         self.calibrated = False
#         self.T_vive_to_rokoko = np.eye(4)

#         self.joints = {
#             'thumb_proximal_yaw_joint':  {'pb_idx': 1,  'limits': [0.0, 1.308]},
#             'thumb_proximal_pitch_joint':{'pb_idx': 2,  'limits': [0.0, 0.6]},
#             'thumb_intermediate_joint':  {'pb_idx': 3,  'limits': [0.0, 0.8]},
#             'thumb_distal_joint':        {'pb_idx': 4,  'limits': [0.0, 0.4]},

#             'index_proximal_joint':      {'pb_idx': 6,  'limits': [0.0, 1.47]},
#             'index_intermediate_joint':  {'pb_idx': 7,  'limits': [-0.04545, 1.56]},

#             'middle_proximal_joint':     {'pb_idx': 9,  'limits': [0.0, 1.47]},
#             'middle_intermediate_joint': {'pb_idx': 10, 'limits': [-0.04545, 1.56]},

#             'ring_proximal_joint':       {'pb_idx': 12, 'limits': [0.0, 1.47]},
#             'ring_intermediate_joint':   {'pb_idx': 13, 'limits': [-0.04545, 1.56]},

#             'pinky_proximal_joint':      {'pb_idx': 15, 'limits': [0.0, 1.47]},
#             'pinky_intermediate_joint':  {'pb_idx': 16, 'limits': [-0.04545, 1.56]},
#         }

#         # Inspire hand joint names
#         self.pybullet_to_inspire = {v['pb_idx']: k for k, v in self.joints.items()}
#         self.joint_names = [self.pybullet_to_inspire[idx] for idx in sorted(self.pybullet_to_inspire.keys())]

#         # Joint mapping from CSV columns to INSPIRE Hand joint names
#         self.joint_mapping = {
#             # Thumb (Digit 1)
#             'thumb_proximal_yaw_joint': 'LeftDigit1Carpometacarpal_ulnarDeviation',
#             'thumb_proximal_pitch_joint': 'LeftDigit1Carpometacarpal_flexion',
#             'thumb_intermediate_joint': 'LeftDigit1Metacarpophalangeal_flexion',
#             'thumb_distal_joint': 'LeftDigit1Interphalangeal_flexion',
#             # Index (Digit 2)
#             'index_proximal_joint': 'LeftDigit2Metacarpophalangeal_flexion',
#             'index_intermediate_joint': 'LeftDigit2ProximalInterphalangeal_flexion',
#             # Middle (Digit 3)
#             'middle_proximal_joint': 'LeftDigit3Metacarpophalangeal_flexion',
#             'middle_intermediate_joint': 'LeftDigit3ProximalInterphalangeal_flexion',
#             # Ring (Digit 4)
#             'ring_proximal_joint': 'LeftDigit4Metacarpophalangeal_flexion',
#             'ring_intermediate_joint': 'LeftDigit4ProximalInterphalangeal_flexion',
#             # Pinky (Digit 5)
#             'pinky_proximal_joint': 'LeftDigit5Metacarpophalangeal_flexion',
#             'pinky_intermediate_joint': 'LeftDigit5ProximalInterphalangeal_flexion',
#         }
        
#         # Position columns for fingertips (for inverse kinematics)
#         self.tip_position_mapping = {
#             'thumb': ['LeftDigit1DistalPhalanx_position_x', 'LeftDigit1DistalPhalanx_position_y', 'LeftDigit1DistalPhalanx_position_z'],
#             'index': ['LeftDigit2DistalPhalanx_position_x', 'LeftDigit2DistalPhalanx_position_y', 'LeftDigit2DistalPhalanx_position_z'],
#             'middle': ['LeftDigit3DistalPhalanx_position_x', 'LeftDigit3DistalPhalanx_position_y', 'LeftDigit3DistalPhalanx_position_z'],
#             'ring': ['LeftDigit4DistalPhalanx_position_x', 'LeftDigit4DistalPhalanx_position_y', 'LeftDigit4DistalPhalanx_position_z'],
#             'pinky': ['LeftDigit5DistalPhalanx_position_x', 'LeftDigit5DistalPhalanx_position_y', 'LeftDigit5DistalPhalanx_position_z']
#         }

#         # Identify movable joints in PyBullet
#         self.movable_joints = []
#         self.num_joints = p.getNumJoints(self.robot_id)
#         for i in range(self.num_joints):
#             info = p.getJointInfo(self.robot_id, i)
#             joint_type = info[2]
#             if joint_type != p.JOINT_FIXED:
#                 self.movable_joints.append(i)

#         self.find_link_indices()

#         # Find joint limits and ranges
#         sorted_joints = sorted(self.joints.items(), key=lambda x: x[1]['pb_idx'])
#         self.lower_limits = [j[1]['limits'][0] for j in sorted_joints]
#         self.upper_limits = [j[1]['limits'][1] for j in sorted_joints]
#         self.joint_ranges = [upper - lower for lower, upper in zip(self.lower_limits, self.upper_limits)]
#         self.joint_limits_rad = {name: data['limits'] for name, data in self.joints.items()}
#         self.joint_limits_deg = {joint: [math.degrees(lim[0]), math.degrees(lim[1])] for joint, lim in self.joint_limits_rad.items()}
#         self.use_joint_limits = True # change to False to disable joint limits in jparse
 
#         # Define finger chain information — ONLY independent (non-mimic) joints
#         # Mimic joints are set manually after IK to keep PyBullet in sync
#         # Old chains (included mimic joints):
#         #   thumb: [1, 2, 3, 4] = Yaw, Pitch, Inter(mimic), Distal(mimic)
#         #   index: [6, 7] = Proximal, Intermediate(mimic)
#         #   middle: [9, 10], ring: [12, 13], pinky: [15, 16]
#         self.finger_chains = {
#             'thumb': {
#                 'base_link': self.palm_link_id,
#                 'ee_link_idx': self.fingertip_link_indices['thumb'],
#                 'joint_indices': [1, 2],  # Only Yaw and Pitch (independent)
#                 'joint_names': ['thumb_proximal_yaw_joint', 'thumb_proximal_pitch_joint']
#             },
#             'index': {
#                 'base_link': self.palm_link_id,
#                 'ee_link_idx': self.fingertip_link_indices['index'],
#                 'joint_indices': [6],  # Only Proximal (independent)
#                 'joint_names': ['index_proximal_joint']
#             },
#             'middle': {
#                 'base_link': self.palm_link_id,
#                 'ee_link_idx': self.fingertip_link_indices['middle'],
#                 'joint_indices': [9],
#                 'joint_names': ['middle_proximal_joint']
#             },
#             'ring': {
#                 'base_link': self.palm_link_id,
#                 'ee_link_idx': self.fingertip_link_indices['ring'],
#                 'joint_indices': [12],
#                 'joint_names': ['ring_proximal_joint']
#             },
#             'pinky': {
#                 'base_link': self.palm_link_id,
#                 'ee_link_idx': self.fingertip_link_indices['pinky'],
#                 'joint_indices': [15],
#                 'joint_names': ['pinky_proximal_joint']
#             }
#         }

#         # Mimic joint relationships from URDF:
#         # mimic_joint_idx: (parent_joint_name, parent_pb_idx, multiplier, offset)
#         self.mimic_joints = {
#             3: ('thumb_proximal_pitch_joint', 2, 1.334, 0.0),    # thumb_intermediate
#             4: ('thumb_proximal_pitch_joint', 2, 0.667, 0.0),    # thumb_distal
#             7: ('index_proximal_joint', 6, 1.06399, -0.04545),   # index_intermediate
#             10: ('middle_proximal_joint', 9, 1.06399, -0.04545), # middle_intermediate
#             13: ('ring_proximal_joint', 12, 1.06399, -0.04545),  # ring_intermediate
#             16: ('pinky_proximal_joint', 15, 1.06399, -0.04545)  # pinky_intermediate
#         }

#         # Rest poses
#         self.rest_poses = []
#         for pb_idx in self.movable_joints:
#             joint_state = p.getJointState(self.robot_id, pb_idx)
#             self.rest_poses.append(joint_state[0])

#         for i in range(self.num_joints):
#             self.get_logger().info(f'Joint info: {p.getJointInfo(self.robot_id, i)[1].decode("utf-8")}')

#         # Subscribers
#         self.control_type_sub = self.create_subscription(
#             String, 'control_type', self.control_type_callback, 10)
#         self.rokoko_data_sub = self.create_subscription(
#             String, 'rokoko_ref_data', self.rokoko_data_callback, 10)
#         self.vive_data_sub = self.create_subscription(
#             PoseStamped, 'vive_wrist_pose', self.vive_pose_callback, 10)
#         self.calibrate_sub = self.create_subscription(
#             String, 'calibrate', self.calibrate_callback, 10)

#         # Publisher for retargeted joint states
#         self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

#         self.control_type = 'direct'  # Default
#         self.latest_rokoko_data = None
#         self.latest_vive_wrist_pose = None
#         self.get_logger().info('Inspire Retargeting Node started')

#     def publish_target_markers(self, targets_dict):
#         marker_array = MarkerArray()
        
#         for i, (finger_name, position) in enumerate(targets_dict.items()):
#             marker = Marker()
#             marker.header.frame_id = "base"  # Or "world" depending on your TF tree
#             marker.header.stamp = self.get_clock().now().to_msg()
#             marker.ns = "fingertip_targets"
#             marker.id = i
#             marker.type = Marker.SPHERE
#             marker.action = Marker.ADD
            
#             # Position from your IK logic
#             marker.pose.position.x = float(position[0])
#             marker.pose.position.y = float(position[1])
#             marker.pose.position.z = float(position[2])
            
#             # Scale and Color
#             marker.scale.x = 0.01  # 1cm sphere
#             marker.scale.y = 0.01
#             marker.scale.z = 0.01
#             marker.color.a = 1.0 
#             # diff finger colors
#             if finger_name == 'thumb': # orange
#                 marker.color.r = 1.0
#                 marker.color.g = 0.5
#                 marker.color.b = 0.0
#             elif finger_name == 'index': # green
#                 marker.color.r = 0.0
#                 marker.color.g = 1.0
#                 marker.color.b = 0.0
#             elif finger_name == 'middle': # blue
#                 marker.color.r = 0.0
#                 marker.color.g = 0.0
#                 marker.color.b = 1.0
#             elif finger_name == 'ring': # purple
#                 marker.color.r = 1.0
#                 marker.color.g = 0.0
#                 marker.color.b = 1.0
#             elif finger_name == 'pinky': # cyan
#                 marker.color.r = 0.0
#                 marker.color.g = 1.0
#                 marker.color.b = 1.0
            
#             marker_array.markers.append(marker)
            
#         self.marker_pub.publish(marker_array)

#     def find_link_indices(self):
#         """Find link indices for fingertips and palm from URDF"""
#         self.fingertip_link_indices = {}
#         self.palm_link_id = None
        
#         self.get_logger().info(f'URDF has {self.num_joints} joints')
        
#         for i in range(self.num_joints):
#             joint_info = p.getJointInfo(self.robot_id, i)
#             link_name = joint_info[12].decode("utf-8")
#             joint_name = joint_info[1].decode("utf-8")
            
#             # Find fingertip links
#             if link_name == "thumb_tip":
#                 self.fingertip_link_indices['thumb'] = i
#                 self.get_logger().info(f'Found thumb tip at link index {i}')
#             elif link_name == "index_tip":
#                 self.fingertip_link_indices['index'] = i
#                 self.get_logger().info(f'Found index tip at link index {i}')
#             elif link_name == "middle_tip":
#                 self.fingertip_link_indices['middle'] = i
#                 self.get_logger().info(f'Found middle tip at link index {i}')
#             elif link_name == "ring_tip":
#                 self.fingertip_link_indices['ring'] = i
#                 self.get_logger().info(f'Found ring tip at link index {i}')
#             elif link_name == "pinky_tip":
#                 self.fingertip_link_indices['pinky'] = i
#                 self.get_logger().info(f'Found pinky tip at link index {i}')
            
#             # Find palm link
#             if link_name == "hand_base_link":
#                 self.palm_link_id = i
#                 self.get_logger().info(f'Found palm link at index {i}')
        
#         if self.palm_link_id is None:
#             self.get_logger().warn("Palm link not found in URDF!")
        
#         self.get_logger().info(f'Fingertip link indices: {self.fingertip_link_indices}')

#     def control_type_callback(self, msg):
#         """Update the current control method"""
#         self.control_type = msg.data
#         # self.get_logger().info(f'Control type updated to: {self.control_type}')

#     def rokoko_data_callback(self, msg):
#         """Receive and process Rokoko reference data"""

#         # self.get_logger().info(f'control_type={self.control_type}, calibrated={self.calibrated}, vive={self.latest_vive_wrist_pose}')

#         self.latest_rokoko_data = msg.data

#         # Parse CSV data
#         parsed_data = self.parse_csv_data(msg.data)

#         if not parsed_data:
#             return

#         # Execute the appropriate control method
#         if self.control_type == 'direct':
#             joint_angles = self.direct_joint_angle_control(parsed_data)
#         elif self.control_type == 'fingertip_ik':
#             joint_angles = self.fingertip_ik_control(parsed_data)
#         elif self.control_type == 'jparse_ik':
#             joint_angles = self.jparse_ik_control(parsed_data)
#         else:
#             self.get_logger().warn(f'Unknown control type: {self.control_type}')
#             return

#         # Publish joint states
#         if joint_angles:
#             self.publish_joint_state(joint_angles)

#     def vive_pose_callback(self, msg):
#         """
#         Updates the internal record of the wrist's position and orientation.
#         msg is a geometry_msgs/PoseStamped object.
#         """
#         self.latest_vive_wrist_pose = msg.pose
#         # Optional: Log receipt once to verify connection
#         if not hasattr(self, '_vive_connected'):
#             self.get_logger().info('Successfully receiving Vive PoseStamped data')
#             self._vive_connected = True

#     def calibrate_callback(self, msg):
#         """
#         Triggered manually via ROS topic.
#         Captures current pose as calibration reference.
#         """
#         if self.latest_vive_wrist_pose is None:
#             self.get_logger().warn("Cannot calibrate: no Vive data yet")
#             return

#         if self.latest_rokoko_data is None:
#             self.get_logger().warn("Cannot calibrate: no Rokoko data yet")
#             return

#         parsed_data = self.parse_csv_data(self.latest_rokoko_data)
#         if not parsed_data:
#             self.get_logger().warn("Failed to parse Rokoko data")
#             return

#         self.calibrate_wrist_transform(parsed_data)

#     def parse_csv_data(self, csv_string):
#         """
#         Parse CSV-format Rokoko data
#         Format: timestamp,joint_name,value
#         Returns dict with joint names as keys and values as floats
#         """
#         parsed = {}
#         lines = csv_string.strip().split('\n')

#         for line in lines:
#             parts = line.split(',')
#             if len(parts) < 3:
#                 continue

#             timestamp, joint_name, value = parts[0], parts[1], parts[2]

#             try:
#                 parsed[joint_name] = float(value)
#             except ValueError:
#                 self.get_logger().warn(f'Could not parse value for {joint_name}: {value}')
#                 continue

#         return parsed

#     def rotation_rokoko_to_inspire(self, vec, finger_name=None):
#         """
#         Map a 3-vector in Rokoko character axes into Inspire/PyBullet world axes.
#         """
#         rx, ry, rz = vec
#         if finger_name == 'thumb':
#             return [rz, -rx, ry]  # Swap Y and Z for thumb
#         else:
#             return [rz, -rx, ry]
        
#     def make_transform(self, pos, quat=None):
#         """Create 4x4 transform matrix"""
#         T = np.eye(4)

#         if quat is not None:
#             R = np.array(p.getMatrixFromQuaternion([
#                 quat.x, quat.y, quat.z, quat.w
#             ])).reshape(3, 3)
#         else:
#             R = np.eye(3)

#         T[:3, :3] = R
#         T[:3, 3] = pos
#         return T

#     def invert_transform(self, T):
#         """Efficient inverse of SE3 transform"""
#         R = T[:3, :3]
#         t = T[:3, 3]

#         T_inv = np.eye(4)
#         T_inv[:3, :3] = R.T
#         T_inv[:3, 3] = -R.T @ t
#         return T_inv

#     def calibrate_wrist_transform(self, parsed_data):
#         """
#         Compute transform between Vive wrist and Rokoko palm.
#         Run once when hand is in neutral pose.
#         """

#         if self.latest_vive_wrist_pose is None:
#             self.get_logger().warn("Cannot calibrate: no Vive data")
#             return

#         palm_cols = ['LeftHand_position_x', 'LeftHand_position_y', 'LeftHand_position_z']
#         if not all(col in parsed_data for col in palm_cols):
#             self.get_logger().warn("Cannot calibrate: no Rokoko palm data")
#             return

#         # Vive transform
#         vive_pos = self.latest_vive_wrist_pose.position
#         vive_quat = self.latest_vive_wrist_pose.orientation
#         T_vive = self.make_transform(
#             np.array([vive_pos.x, vive_pos.y, vive_pos.z]),
#             vive_quat
#         )

#         # Rokoko palm
#         rokoko_palm = np.array([parsed_data[col] for col in palm_cols])
#         T_rokoko = self.make_transform(rokoko_palm)

#         # Compute transform
#         self.T_vive_to_rokoko = self.invert_transform(T_vive) @ T_rokoko

#         self.calibrated = True
#         self.get_logger().info("Wrist calibration complete") 

#     def compute_world_target(self, vive_pos, vive_quat, rokoko_tip, rokoko_palm):
#         """Compute world target position for fingertip based on Vive wrist and Rokoko data, used in all control methods"""
#         T_vive = self.make_transform(vive_pos, vive_quat)
#         T_tip = self.make_transform(rokoko_tip)
#         T_palm = self.make_transform(rokoko_palm)

#         T_rel = self.invert_transform(T_palm) @ T_tip
#         T_world = T_vive @ self.T_vive_to_rokoko @ T_rel

#         return T_world[:3, 3]

#     def direct_joint_angle_control(self, parsed_data):
#         """
#         Direct joint angle control with Cartesian marker visualization. 
#         """
#         self.get_logger().debug('Using direct joint angle control')
#         joint_angles = {}

#         if not self.calibrated:
#             if self.latest_vive_wrist_pose is not None:
#                 self.get_logger().warn("Waiting for calibration", throttle_duration_sec=2.0)
#                 return [0.0] * len(self.joint_names)
  
#         # 1. Map joints directly (with selective inversion if needed)
#         for inspire_joint, rokoko_joint in self.joint_mapping.items():
#             val = parsed_data.get(rokoko_joint, 0.0)
#             rad = math.radians(val) if self.data_in_degrees else val
#             joint_angles[inspire_joint] = rad 
        
#         # 2. Extract Cartesian positions for visualization
#         inspire_tips_in_world = {}
        
#         if self.latest_vive_wrist_pose is not None:
#             vive_pos = self.latest_vive_wrist_pose.position
#             vive_quat = self.latest_vive_wrist_pose.orientation
#             vive_wrist_pos = np.array([vive_pos.x, vive_pos.y, vive_pos.z])

#             # Get Rokoko Palm Reference for relative finger offsets
#             palm_cols = ['LeftHand_position_x', 'LeftHand_position_y', 'LeftHand_position_z']
            
#             if all(col in parsed_data for col in palm_cols):
#                 rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)

#                 for finger_name, pos_columns in self.tip_position_mapping.items():
#                     # Get the tip position from Rokoko
#                     rokoko_tip = None
#                     if all(col in parsed_data for col in pos_columns):
#                         rokoko_tip = np.array([parsed_data[col] for col in pos_columns], dtype=float)
#                     else:
#                         digit_num = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5}[finger_name]
#                         alt = [f'LeftDigit{digit_num}Tip_pos_{c}' for c in ['x', 'y', 'z']]
#                         if all(col in parsed_data for col in alt):
#                             rokoko_tip = np.array([parsed_data[col] for col in alt], dtype=float)

                    
#                     if rokoko_tip is not None and all(col in parsed_data for col in palm_cols):
#                         world_target = self.compute_world_target(vive_wrist_pos, vive_quat, rokoko_tip, rokoko_palm) # Compute world target using Vive wrist as reference
#                         inspire_tips_in_world[finger_name] = world_target

#         # 3. Publish the markers to RViz
#         if inspire_tips_in_world:
#             self.publish_target_markers(inspire_tips_in_world)

#         joint_angles_list = [joint_angles.get(name, 0.0) for name in self.joint_names]
#         return joint_angles_list

#     def fingertip_ik_control(self, parsed_data):
#         """
#         Control method 2: Fingertip IK-based control
#         Uses fingertip positions to solve inverse kinematics
#         """
#         self.get_logger().info('ENTERED fingertip_ik_control function', once=True)
#         self.get_logger().debug('Using fingertip IK control with PyBullet')

#         joint_names = self.joint_names

#         if not self.calibrated:
#             self.get_logger().warn("Waiting for calibration", throttle_duration_sec=2.0)
#             return [0.0] * len(self.joint_names)

#         if self.latest_vive_wrist_pose is None:
#             self.get_logger().warn("Waiting for Vive wrist data to solve IK...", once=True)
#             return [0.0] * len(self.joint_names) # Return neutral pose if no Vive data
        
#         vive_pose = self.latest_vive_wrist_pose.position
#         vive_quaternion = self.latest_vive_wrist_pose.orientation
#         vive_wrist_pos = np.array([vive_pose.x, vive_pose.y, vive_pose.z])

#         inspire_tips_in_world = {}
#         palm_cols = ['LeftHand_position_x', 'LeftHand_position_y', 'LeftHand_position_z']

#         if all(col in parsed_data for col in palm_cols):
#             rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)
#             for finger_name, pos_columns in self.tip_position_mapping.items():
#                 rokoko_tip = None
#                 if all(col in parsed_data for col in pos_columns):
#                     rokoko_tip = np.array([parsed_data[col] for col in pos_columns], dtype=float)
#                 else:
#                     digit_num = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5}[finger_name]
#                     alt = [f'LeftDigit{digit_num}Tip_pos_{c}' for c in ['x', 'y', 'z']] 
#                     if all(col in parsed_data for col in alt):
#                         rokoko_tip = np.array([parsed_data[col] for col in alt], dtype=float)
                
#                 if rokoko_tip is not None:
#                     world_target = self.compute_world_target(vive_wrist_pos, vive_quaternion, rokoko_tip, rokoko_palm) # Compute world target using Vive wrist as reference
#                     inspire_tips_in_world[finger_name] = world_target

#                     # DEBUG: Log first frame
#                     if not hasattr(self, '_logged_ik_fingertips'):
#                             self.get_logger().info(f'{finger_name} tip (world): {inspire_tips_in_world[finger_name]}')
                
#                 if not hasattr(self, '_logged_ik_fingertips'):
#                     self._logged_ik_fingertips = True
#                     self.get_logger().info('='*80)
#                     self.get_logger().info(f'Using PyBullet IK for {len(inspire_tips_in_world)} fingertips')
#                     self.get_logger().info('='*80)

#         current_joint_angles = self.rest_poses.copy()
#         finger_order = ["thumb", "index", "middle", "ring", "pinky"]
#         pb_to_array_idx = {1:0, 2:1, 3:2, 4:3, 6:4, 7:5, 9:6, 10:7, 12:8, 13:9, 15:10, 16:11}
#         finger_to_pb_indices = {}
#         for finger in finger_order:
#             if finger == "thumb":
#                 joint_names = ['thumb_proximal_yaw_joint', 'thumb_proximal_pitch_joint', 
#                                 'thumb_intermediate_joint', 'thumb_distal_joint']
#             elif finger == "index":
#                 joint_names = ['index_proximal_joint', 'index_intermediate_joint']
#             elif finger == "middle":
#                 joint_names = ['middle_proximal_joint', 'middle_intermediate_joint']
#             elif finger == "ring":
#                 joint_names = ['ring_proximal_joint', 'ring_intermediate_joint']
#             elif finger == "pinky":
#                 joint_names = ['pinky_proximal_joint', 'pinky_intermediate_joint']
#             finger_to_pb_indices[finger] = [idx for idx, name in self.pybullet_to_inspire.items() if name in joint_names]

#         for finger_name in finger_order:
#             if finger_name not in inspire_tips_in_world:
#                 continue

#             target_pos = inspire_tips_in_world[finger_name]
#             link_index = self.fingertip_link_indices[finger_name]

#             try:
#                 ik_result = p.calculateInverseKinematics(
#                     self.robot_id,
#                     link_index,
#                     target_pos,
#                     lowerLimits=self.lower_limits,
#                     upperLimits=self.upper_limits,
#                     jointRanges=self.joint_ranges,
#                     restPoses=current_joint_angles,
#                     maxNumIterations=200,
#                     residualThreshold=1e-4
#                 )

#                 if finger_name == 'thumb':
#                     self.get_logger().info(f"Thumb IK Output: {ik_result[0:4]}")

#                 # Map PyBullet results back to the joint array
#                 pb_indices = finger_to_pb_indices[finger_name]
#                 for pb_idx in pb_indices:
#                     try:
#                         # Find index of pb_idx in movable_joints
#                         ik_idx = self.movable_joints.index(pb_idx)
#                         angle = ik_result[ik_idx]
#                         array_idx = pb_to_array_idx[pb_idx]
#                         current_joint_angles[array_idx] = angle
#                         p.resetJointState(self.robot_id, pb_idx, angle)
#                     except ValueError:
#                         self.get_logger().warn(f'IK mapping error: {pb_idx} not in movable joints')

#             except Exception as e:
#                 self.get_logger().warn(f'IK failed for {finger_name}: {e}')
#                 import traceback
#                 self.get_logger().error(traceback.format_exc())
#                 continue

#         # 5. Finalize and Publish
#         self.publish_target_markers(inspire_tips_in_world)
#         joint_angles_list = [current_joint_angles[idx] for idx in range(12)]
#         return joint_angles_list

#     def jparse_ik_control(self, parsed_data):
#         """
#         Control method 3: IK with JPARSE (Joint Position And Rotation Solver Engine)
#         Uses PyBullet's Jacobian calculation with jparse for advanced IK solving

#         Solves IK for each finger independently from base to fingertip
#         """
#         self.get_logger().debug('Using JPARSE IK control')

#         if not JPARSE_AVAILABLE:
#             self.get_logger().error('jparse not available! Falling back to zero positions')
#             return [0.0] * len(self.joint_names)
        
#         if self.latest_vive_wrist_pose is None:
#             self.get_logger().warn("Waiting for Vive wrist data to solve JParse IK...", once=True)
#             return [0.0] * len(self.joint_names) # Return neutral pose if no Vive data
        
#         vive_pos = self.latest_vive_wrist_pose.position
#         vive_quat = self.latest_vive_wrist_pose.orientation
#         vive_wrist_pos = np.array([vive_pos.x, vive_pos.y, vive_pos.z])

#         # Create joint_names list from pybullet_to_inspire mapping (sorted by index)
#         joint_names = self.joint_names

#         if not self.calibrated:
#             self.get_logger().warn("Waiting for calibration", throttle_duration_sec=2.0)
#             return [0.0] * len(self.joint_names)

#         # Initialize joint angles dictionary with CURRENT PyBullet state
#         # This ensures fingers without IK updates maintain their current positions
#         joint_angles = {}
#         for pb_idx, joint_name in self.pybullet_to_inspire.items():
#             joint_state = p.getJointState(self.robot_id, pb_idx)
#             joint_angles[joint_name] = joint_state[0]  # Current joint position

#         # Initialize debug counters (do this once at the start)
#         if not hasattr(self, '_jparse_log_count'):
#             self._jparse_log_count = 0
#         if not hasattr(self, '_pos_debug_count'):
#             self._pos_debug_count = 0
#         if not hasattr(self, '_jac_debug_count'):
#             self._jac_debug_count = 0

#         # Debug: Log how many fingers we're processing
#         # Get palm position from PyBullet (INSPIRE world frame)
#         palm_link_info = p.getLinkState(self.robot_id, self.palm_link_id, computeForwardKinematics=True)
#         inspire_palm_in_world = np.array(palm_link_info[0], dtype=float)
#         inspire_palm_in_world_arr = inspire_palm_in_world.tolist()

#         # Get Rokoko palm position if available
#         palm_cols = ['LeftHand_position_x', 'LeftHand_position_y', 'LeftHand_position_z']
#         rokoko_palm_available = all(col in parsed_data for col in palm_cols)
#         if rokoko_palm_available:
#             rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)
#             inspire_palm_from_rokoko = np.array(self.rotation_rokoko_to_inspire(rokoko_palm), dtype=float)

#         fingers_processed = 0
#         targets_for_viz = {}

#         # Solve IK for each finger independently
#         for finger_name, chain_info in self.finger_chains.items():
#             # Get target fingertip position from parsed data
#             pos_columns = self.tip_position_mapping[finger_name]

#             # Try primary naming convention
#             rokoko_tip = None
#             if all(col in parsed_data for col in pos_columns):
#                 rokoko_tip = np.array([parsed_data[col] for col in pos_columns], dtype=float)
#             else:
#                 # Try alternative naming (live teleop)
#                 digit_num = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5}[finger_name]
#                 alt_columns = [f'LeftDigit{digit_num}Tip_pos_x',
#                               f'LeftDigit{digit_num}Tip_pos_y',
#                               f'LeftDigit{digit_num}Tip_pos_z']
#                 if all(col in parsed_data for col in alt_columns):
#                     rokoko_tip = np.array([parsed_data[col] for col in alt_columns], dtype=float)

#             # Skip this finger if no fingertip data available
#             if rokoko_tip is None:
#                 if self._jparse_log_count < 3:
#                     self.get_logger().warn(f'No position data for {finger_name}, skipping')
#                 continue

#             fingers_processed += 1

#             target_pos_world = self.compute_world_target(vive_wrist_pos, vive_quat, rokoko_tip, rokoko_palm) # Compute world target using Vive wrist as reference
#             targets_for_viz[finger_name] = target_pos_world.tolist()

#             # Get current joint positions for this finger
#             current_joint_positions = []
#             for joint_idx in chain_info['joint_indices']:
#                 joint_state = p.getJointState(self.robot_id, joint_idx)
#                 current_joint_positions.append(joint_state[0])

#             # Get current end effector position
#             link_state = p.getLinkState(self.robot_id, chain_info['ee_link_idx'])
#             current_ee_pos = np.array(link_state[0])

#             # Calculate position error
#             pos_error = target_pos_world - current_ee_pos

#             # Debug: Log positions for first finger (first few times)
#             if self._pos_debug_count < 3 and finger_name == 'thumb':
#                 self.get_logger().info(f'=== {finger_name} Position Debug ===')
#                 self.get_logger().info(f'Rokoko tip (raw): {rokoko_tip}')
#                 # self.get_logger().info(f'INSPIRE tip: {inspire_tip}')
#                 if rokoko_palm_available:
#                     self.get_logger().info(f'Rokoko palm: {rokoko_palm}')
#                     self.get_logger().info(f'INSPIRE palm (Rokoko): {inspire_palm_from_rokoko}')
#                     self.get_logger().info(f'INSPIRE palm (PyBullet): {inspire_palm_in_world}')
#                 self.get_logger().info(f'Target pos (world): {target_pos_world}')
#                 self.get_logger().info(f'Current pos: {current_ee_pos}')
#                 self.get_logger().info(f'Error: {pos_error}')
#                 self.get_logger().info(f'Error magnitude: {np.linalg.norm(pos_error):.6f}m')
#                 self._pos_debug_count += 1

#             # Calculate Jacobian using PyBullet
#             # Get all joint states (positions, velocities, accelerations)
#             # and create mapping from joint index to Jacobian column index
#             joint_positions = []
#             joint_velocities = []
#             joint_accelerations = []
#             joint_index_to_col = {}  # Maps joint index to Jacobian column index

#             col_idx = 0
#             for i in range(self.num_joints):
#                 joint_info = p.getJointInfo(self.robot_id, i)
#                 joint_type = joint_info[2]  # Joint type

#                 # Only include movable joints (revolute or prismatic)
#                 if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
#                     joint_state = p.getJointState(self.robot_id, i)
#                     joint_positions.append(joint_state[0])  # Position
#                     joint_velocities.append(joint_state[1])  # Velocity
#                     joint_accelerations.append(0.0)  # Acceleration (assumed zero)
#                     joint_index_to_col[i] = col_idx
#                     col_idx += 1

#             # Calculate Jacobian for the fingertip link
#             jacobian_linear, jacobian_angular = p.calculateJacobian(
#                 bodyUniqueId=self.robot_id,
#                 linkIndex=chain_info['ee_link_idx'],
#                 localPosition=[0, 0, 0],
#                 objPositions=joint_positions,
#                 objVelocities=joint_velocities,
#                 objAccelerations=joint_accelerations
#             )

#             # Convert to numpy arrays
#             J_linear = np.array(jacobian_linear)
#             J_angular = np.array(jacobian_angular)

#             # Extract only the columns corresponding to this finger's joints
#             # Map joint indices to Jacobian column indices
#             finger_col_indices = []
#             for joint_idx in chain_info['joint_indices']:
#                 if joint_idx in joint_index_to_col:
#                     finger_col_indices.append(joint_index_to_col[joint_idx])

#             J_finger_linear = J_linear[:, finger_col_indices]

#             # Use jparse to solve the IK problem (velocity-based)
#             # Create a jparse solver for this finger
#             try:
#                 # Create JParseCore solver with singularity threshold
#                 # gamma: directions with σᵢ/σₘₐₓ < gamma are treated as singular
#                 jp_solver = jparse.JParseCore(gamma=0.1)

#                 # Compute the jparse pseudo-inverse of the Jacobian
#                 # This uses the jparse algorithm (not standard damped least squares)
#                 J_pinv = jp_solver.compute(
#                     jacobian=J_finger_linear,
#                     singular_direction_gain_position=1.0,
#                     position_dimensions=3,  # We're only controlling position (x, y, z)
#                     return_nullspace=False
#                 )

#                 # J_pinv = jp_solver.pinv(J_finger_linear)

#                 # Clamp position error magnitude to prevent huge delta_q
#                 # This limits how far we try to move per iteration
#                 max_step = 0.005  # 5mm max step per iteration
#                 error_norm = np.linalg.norm(pos_error)
#                 if error_norm > max_step:
#                     pos_error_clamped = pos_error * (max_step / error_norm)
#                 else:
#                     pos_error_clamped = pos_error  # Small error, use as-is for convergence

#                 # Direct position solve with clamped error
#                 delta_q = J_pinv @ pos_error_clamped

#                 # DIAGNOSTIC: Check delta_q values
#                 delta_q_max_deg = np.degrees(np.max(np.abs(delta_q)))
#                 if delta_q_max_deg > 30:  # Still too large even after clamping
#                     self.get_logger().warn(
#                         f'{finger_name}: LARGE delta_q! max={delta_q_max_deg:.1f}° per iteration. '
#                         f'Error={error_norm*1000:.1f}mm (clamped to {np.linalg.norm(pos_error_clamped)*1000:.1f}mm)'
#                     )

#                 # Debug: Check Jacobian and delta_q
#                 if self._jac_debug_count < 2 and finger_name == 'thumb':
#                     self.get_logger().info(f'J_finger_linear shape: {J_finger_linear.shape}')
#                     self.get_logger().info(f'J_pinv shape: {J_pinv.shape}')
#                     self.get_logger().info(f'pos_error: {pos_error} (norm={error_norm*1000:.1f}mm)')
#                     self.get_logger().info(f'pos_error_clamped: {pos_error_clamped} (norm={np.linalg.norm(pos_error_clamped)*1000:.1f}mm)')
#                     self.get_logger().info(f'delta_q: {delta_q}')
                    
#                     self._jac_debug_count += 1

#                 # Update joint angles for this finger with limit clamping
#                 for i, joint_name in enumerate(chain_info['joint_names']):
#                     new_angle = current_joint_positions[i] + delta_q[i]

#                     # Clamp to joint limits
#                     if joint_name in self.joint_limits_rad:
#                         min_limit, max_limit = self.joint_limits_rad[joint_name]
#                         new_angle = max(min_limit, min(max_limit, new_angle))

#                     joint_angles[joint_name] = new_angle

#                 # Debug logging (first few iterations)
#                 self._jparse_log_count += 1

#                 # Log first 20 iterations, then every 60th frame (~2 sec at 30Hz)
#                 if self._jparse_log_count < 20 or self._jparse_log_count % 60 == 0:
#                     delta_q_max = np.max(np.abs(delta_q))
#                     self.get_logger().info(
#                         f'{finger_name}: err={error_norm*1000:.1f}mm->clamped={np.linalg.norm(pos_error_clamped)*1000:.1f}mm, '
#                         f'delta_q_max={np.degrees(delta_q_max):.1f}°, '
#                         f'delta_q(deg)={[f"{np.degrees(dq):.1f}" for dq in delta_q]}'
#                     )

#             except Exception as e:
#                 self.get_logger().error(f'jparse IK failed for {finger_name}: {e}')
#                 # Keep current positions on failure
#                 for i, joint_name in enumerate(chain_info['joint_names']):
#                     joint_angles[joint_name] = current_joint_positions[i]

#         # Debug: Log processing summary
#         if self._jparse_log_count < 3:
#             self.get_logger().info(f'Processed {fingers_processed}/5 fingers')
#             self.get_logger().info(f'Joint angles computed: {len(joint_angles)} joints')
#             if fingers_processed == 0:
#                 self.get_logger().error('NO FINGERS PROCESSED - No position data available!')

#         if len(targets_for_viz) > 0:
#             self.publish_target_markers(targets_for_viz)
        
#         # Convert to ordered list matching joint_names order
#         joint_angles_list = [joint_angles.get(name, 0.0) for name in joint_names]

#         # Debug: Show some actual joint angles
#         if self._jparse_log_count < 3:
#             self.get_logger().info(f'Sample joint angles (degrees):')
#             for i, (name, angle) in enumerate(zip(joint_names[:5], joint_angles_list[:5])):
#                 self.get_logger().info(f'  {name}: {math.degrees(angle):.2f}°')

#         # CRITICAL: Update PyBullet simulation with new joint angles
#         # This ensures the next iteration uses updated positions for Jacobian calculation
#         updated_count = 0
#         for pb_idx, joint_name in self.pybullet_to_inspire.items():
#             if joint_name in joint_angles:
#                 p.resetJointState(self.robot_id, pb_idx, joint_angles[joint_name])
#                 updated_count += 1

#         # Set mimic joints in PyBullet to match their parent joints
#         # This keeps PyBullet's kinematic model consistent with the real robot
#         for mimic_pb_idx, (parent_name, parent_pb_idx, multiplier, offset) in self.mimic_joints.items():
#             if parent_name in joint_angles:
#                 mimic_angle = joint_angles[parent_name] * multiplier + offset
#                 p.resetJointState(self.robot_id, mimic_pb_idx, mimic_angle)
#                 # Also store in joint_angles so it gets published
#                 mimic_name = self.pybullet_to_inspire.get(mimic_pb_idx)
#                 if mimic_name:
#                     joint_angles[mimic_name] = mimic_angle

#         if self._jparse_log_count < 3:
#             self.get_logger().info(f'Updated {updated_count} independent + {len(self.mimic_joints)} mimic PyBullet joints')

#         # # Store joint_names for publish function
#         # self.joint_names = joint_names

#         return joint_angles_list

#     def clamp_joint_angles(self, joint_angles):
#         """
#         Clamp joint angles to stay within physical limits.
#         Prevents fingers from crossing each other or moving beyond safe ranges.

#         Args:
#             joint_angles: List of joint angles in radians

#         Returns:
#             List of clamped joint angles
#         """
#         if not self.use_joint_limits:
#             return joint_angles

#         clamped_angles = []
#         clamped_count = 0

#         for i, joint_name in enumerate(self.joint_names):
#             angle = joint_angles[i]

#             if joint_name in self.joint_limits_rad:
#                 min_limit, max_limit = self.joint_limits_rad[joint_name]

#                 if angle < min_limit or angle > max_limit:
#                     clamped_angle = max(min_limit, min(max_limit, angle))

#                     # Log when clamping occurs (but not too frequently)
#                     if not hasattr(self, '_clamp_log_count'):
#                         self._clamp_log_count = {}
#                     if joint_name not in self._clamp_log_count:
#                         self._clamp_log_count[joint_name] = 0

#                     self._clamp_log_count[joint_name] += 1

#                     # Log every 100th clamp for each joint
#                     if self._clamp_log_count[joint_name] % 100 == 1:
#                         self.get_logger().warn(
#                             f'Clamping {joint_name}: {math.degrees(angle):.1f}° → {math.degrees(clamped_angle):.1f}° '
#                             f'(limits: {math.degrees(min_limit):.1f}° to {math.degrees(max_limit):.1f}°)'
#                         )

#                     clamped_angles.append(clamped_angle)
#                     clamped_count += 1
#                 else:
#                     clamped_angles.append(angle)
#             else:
#                 # No limit defined for this joint, pass through
#                 clamped_angles.append(angle)

#         return clamped_angles

#     def publish_joint_state(self, joint_angles):
#         """Publish the retargeted joint state"""
#         msg = JointState()
#         msg.header.stamp = self.get_clock().now().to_msg()
#         msg.name = self.joint_names
#         msg.position = joint_angles

#         self.joint_pub.publish(msg)


# def main(args=None):
#     rclpy.init(args=args)
#     inspire_retargeting = InspireRetargeting()

#     try:
#         rclpy.spin(inspire_retargeting)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         inspire_retargeting.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()


# if __name__ == '__main__':
#     main()
