#!/usr/bin/env python3
"""
Nano Hand Retargeting Node  (RobotNanoHand — robotnanohand.com)

Joint layout from nanohand_test.urdf (PyBullet index order):
  0:  pinky_wiggle       (independent)
  1:  pinky_curl         (independent — MCP, servo-driven)
  2:  pinky_curl_pip     (mimic of pinky_curl,  multiplier=2.92)
  3:  pinky_curl_dip     (mimic of pinky_curl,  multiplier=3.94)
  4:  ring_wiggle        (independent)
  5:  ring_curl          (independent — MCP)
  6:  ring_curl_pip      (mimic of ring_curl,   multiplier=2.31)
  7:  ring_curl_dip      (mimic of ring_curl,   multiplier=2.85)
  8:  middle_wiggle      (independent)
  9:  middle_curl        (independent — MCP)
  10: middle_curl_pip    (mimic of middle_curl, multiplier=2.2)
  11: middle_curl_dip    (mimic of middle_curl, multiplier=2.85)
  12: index_wiggle       (independent)
  13: index_curl         (independent — MCP)
  14: index_curl_pip     (mimic of index_curl,  multiplier=0.7)
  15: index_curl_dip     (mimic of index_curl,  multiplier=0.5)
  16: thumb_wiggle       (independent)
  17: thumb_curl         (independent — MCP)
  18: thumb_curl_pip     (mimic of thumb_curl,  multiplier=0.71)
  19: thumb_curl_dip     (mimic of thumb_curl,  multiplier=0.54)

UPDATE mimic multipliers after physical measurement:
  PIP multiplier (0.7) and DIP multiplier (0.5) are URDF placeholders.
  Measure on the real hand: set MCP to a known angle, read PIP and DIP
  relative to their own parent links, compute ratio = measured / MCP.

  Both PIP and DIP couple to the MCP curl (not to each other) because
  the same tendon runs through all three phalanges.
"""

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


try:
    import jparse_robotics as jparse
    JPARSE_AVAILABLE = True
except ImportError:
    JPARSE_AVAILABLE = False
    print("Warning: jparse not available. Install with: pip install git+https://github.com/armlabstanford/jparse.git")


class NanoRetargeting(Node):
    def __init__(self):
        super().__init__('nano_retargeting')
        self.marker_pub = self.create_publisher(MarkerArray, 'fingertip_targets', 10)

        # Parameters
        self.declare_parameter('data_in_degrees', True)
        self.data_in_degrees = self.get_parameter('data_in_degrees').get_parameter_value().bool_value

        if self.data_in_degrees:
            self.get_logger().info('Data format: DEGREES (will convert to radians)')
        else:
            self.get_logger().info('Data format: RADIANS (no conversion needed)')

        # Connect PyBullet
        self.physics_client_id = p.connect(p.DIRECT)
        self.get_logger().info('PyBullet physics client connected in DIRECT mode')

        # Load URDF
        share_directory = get_package_share_directory('hands')
        urdf_path = os.path.join(share_directory, 'urdf', 'nano', 'nano_hand_right.urdf')
        try:
            self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
            self.get_logger().info(f'Loaded URDF from: {urdf_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            raise

        # ── Independent (servo-driven) joints only ────────────────────────────
        # pb_idx: PyBullet joint index (from URDF parse order above)
        # limits: [lower_rad, upper_rad] taken directly from URDF <limit> tags
        self.joints = {
            # Pinky
            'pinky_wiggle': {'pb_idx': 0,  'limits': [-0.17,  0.26 ]},
            'pinky_curl':   {'pb_idx': 1,  'limits': [0.0,    0.97 ]},
            # Ring
            'ring_wiggle':  {'pb_idx': 4,  'limits': [-0.2,   0.07 ]},
            'ring_curl':    {'pb_idx': 5,  'limits': [0.0,    0.83 ]},
            # Middle
            'middle_wiggle':{'pb_idx': 8,  'limits': [-0.13,  0.24 ]},
            'middle_curl':  {'pb_idx': 9,  'limits': [0.0,    1.09 ]},
            # Index
            'index_wiggle': {'pb_idx': 12, 'limits': [-0.17,  0.17 ]},
            'index_curl':   {'pb_idx': 13, 'limits': [0.0,    1.06 ]},
            # Thumb
            'thumb_wiggle': {'pb_idx': 16, 'limits': [-0.78,  1.48 ]},
            'thumb_curl':   {'pb_idx': 17, 'limits': [0.0,    0.68 ]},
        }

        # ── Mimic joints (PIP and DIP, tendon-coupled to MCP curl) ────────────
        # Format: pb_idx: (parent_joint_name, parent_pb_idx, multiplier, offset)
        # Both PIP and DIP reference the MCP curl joint as their driver —
        # multiplier values match URDF <mimic> tags — UPDATE after measurement.
        self.mimic_joints = {
            # Pinky
            2:  ('pinky_curl',  1,  2.0,  0.0),  # pinky_curl_pip  (PIP)
            3:  ('pinky_curl',  1,  2.23, 0.0),  # pinky_curl_dip  (DIP)
            # Ring
            6:  ('ring_curl',   5,  1.5,  0.0),  # ring_curl_pip
            7:  ('ring_curl',   5,  1.96, 0.0),  # ring_curl_dip
            # Middle
            10: ('middle_curl', 9,  2.0,  0.0),  # middle_curl_pip
            11: ('middle_curl', 9,  2.18, 0.0),  # middle_curl_dip
            # Index
            14: ('index_curl',  13, 2.0,  0.0),  # index_curl_pip
            15: ('index_curl',  13, 2.23, 0.0),  # index_curl_dip
            # Thumb
            18: ('thumb_curl',  17, 0.83, 0.0),  # thumb_curl_pip
            19: ('thumb_curl',  17, 2.0,  0.0),  # thumb_curl_dip
        }

        # ── Joint mapping: nano joint name → Rokoko CSV column ────────────────
        # curl   → Metacarpophalangeal_flexion (MCP flexion)
        # wiggle → Metacarpophalangeal_ulnarDeviation (abduction/adduction)
        # Update Left/Right prefix to match your recording side.
        self.joint_mapping = {
            'thumb_curl':    'RightDigit1Metacarpophalangeal_flexion',
            'thumb_wiggle':  'RightDigit1Metacarpophalangeal_ulnarDeviation',
            'index_curl':    'RightDigit2Metacarpophalangeal_flexion',
            'index_wiggle':  'RightDigit2Metacarpophalangeal_ulnarDeviation',
            'middle_curl':   'RightDigit3Metacarpophalangeal_flexion',
            'middle_wiggle': 'RightDigit3Metacarpophalangeal_ulnarDeviation',
            'ring_curl':     'RightDigit4Metacarpophalangeal_flexion',
            'ring_wiggle':   'RightDigit4Metacarpophalangeal_ulnarDeviation',
            'pinky_curl':    'RightDigit5Metacarpophalangeal_flexion',
            'pinky_wiggle':  'RightDigit5Metacarpophalangeal_ulnarDeviation',
        }

        # Fingertip positions for IK (from Rokoko CSV columns)
        self.tip_position_mapping = {
            'thumb':  ['RightDigit1DistalPhalanx_position_x', 'RightDigit1DistalPhalanx_position_y', 'RightDigit1DistalPhalanx_position_z'],
            'index':  ['RightDigit2DistalPhalanx_position_x', 'RightDigit2DistalPhalanx_position_y', 'RightDigit2DistalPhalanx_position_z'],
            'middle': ['RightDigit3DistalPhalanx_position_x', 'RightDigit3DistalPhalanx_position_y', 'RightDigit3DistalPhalanx_position_z'],
            'ring':   ['RightDigit4DistalPhalanx_position_x', 'RightDigit4DistalPhalanx_position_y', 'RightDigit4DistalPhalanx_position_z'],
            'pinky':  ['RightDigit5DistalPhalanx_position_x', 'RightDigit5DistalPhalanx_position_y', 'RightDigit5DistalPhalanx_position_z'],
        }

        # Build lookup: pb_idx → joint_name (independent joints only)
        self.pybullet_to_nano = {v['pb_idx']: k for k, v in self.joints.items()}
        self.joint_names = [self.pybullet_to_nano[idx] for idx in sorted(self.pybullet_to_nano.keys())]

        # All movable joints (including mimics, needed for PyBullet IK arrays)
        self.movable_joints = []
        self.num_joints = p.getNumJoints(self.robot_id)
        for i in range(self.num_joints):
            if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED:
                self.movable_joints.append(i)

        self.find_link_indices()

        # Joint limit arrays — must cover ALL movable joints in pb_idx order
        # (PyBullet IK requires limit arrays of length == number of movable joints)
        mimic_limits = {
            2:  (0.0, 1.41),  # pinky_curl_pip
            3:  (0.0, 1.64),  # pinky_curl_dip
            6:  (0.0, 1.29),  # ring_curl_pip
            7:  (0.0, 1.55),  # ring_curl_dip
            10: (0.0, 1.65),  # middle_curl_pip
            11: (0.0, 1.6 ),  # middle_curl_dip
            14: (0.0, 1.65),  # index_curl_pip
            15: (0.0, 1.64),  # index_curl_dip
            18: (0.0, 0.66),  # thumb_curl_pip
            19: (0.0, 1.65),  # thumb_curl_dip
        }
        indep_limits = {v['pb_idx']: v['limits'] for v in self.joints.values()}
        self.lower_limits = []
        self.upper_limits = []
        for pb_idx in self.movable_joints:
            if pb_idx in indep_limits:
                lo, hi = indep_limits[pb_idx]
            elif pb_idx in mimic_limits:
                lo, hi = mimic_limits[pb_idx]
            else:
                lo, hi = -3.14, 3.14
            self.lower_limits.append(lo)
            self.upper_limits.append(hi)
        self.joint_ranges  = [u - l for l, u in zip(self.lower_limits, self.upper_limits)]
        self.joint_limits_rad = {name: data['limits'] for name, data in self.joints.items()}
        self.use_joint_limits = True

        # Finger chains for IK — independent joints only (wiggle + curl, no mimics)
        self.finger_chains = {
            'thumb': {
                'base_link':     self.palm_link_id,
                'ee_link_idx':   self.fingertip_link_indices['thumb'],
                'joint_indices': [16, 17],
                'joint_names':   ['thumb_wiggle', 'thumb_curl'],
            },
            'index': {
                'base_link':     self.palm_link_id,
                'ee_link_idx':   self.fingertip_link_indices['index'],
                'joint_indices': [12, 13],
                'joint_names':   ['index_wiggle', 'index_curl'],
            },
            'middle': {
                'base_link':     self.palm_link_id,
                'ee_link_idx':   self.fingertip_link_indices['middle'],
                'joint_indices': [8, 9],
                'joint_names':   ['middle_wiggle', 'middle_curl'],
            },
            'ring': {
                'base_link':     self.palm_link_id,
                'ee_link_idx':   self.fingertip_link_indices['ring'],
                'joint_indices': [4, 5],
                'joint_names':   ['ring_wiggle', 'ring_curl'],
            },
            'pinky': {
                'base_link':     self.palm_link_id,
                'ee_link_idx':   self.fingertip_link_indices['pinky'],
                'joint_indices': [0, 1],
                'joint_names':   ['pinky_wiggle', 'pinky_curl'],
            },
        }

        # Build a table of safe MCP upper limits: the mimic multipliers can
        # drive PIP/DIP past 1.57 rad even when MCP itself is within [0,1.57].
        # Pre-compute the tightest ceiling per curl joint so we can clamp input
        # before it ever reaches _apply_mimic_joints.
        # e.g. pinky DIP multiplier=3.94 → safe MCP ceil = 1.57/3.94 ≈ 0.40 rad
        self._mcp_safe_upper: dict[str, float] = {}
        MIMIC_UPPER = 1.57
        for _, (parent_name, _, multiplier, _) in self.mimic_joints.items():
            if multiplier > 0:
                safe = MIMIC_UPPER / multiplier
                existing = self._mcp_safe_upper.get(parent_name, float('inf'))
                self._mcp_safe_upper[parent_name] = min(existing, safe)
        # Never tighten beyond the URDF curl upper limit itself
        for jname, data in self.joints.items():
            urdf_upper = data['limits'][1]
            if jname in self._mcp_safe_upper:
                self._mcp_safe_upper[jname] = min(self._mcp_safe_upper[jname], urdf_upper)
            else:
                self._mcp_safe_upper[jname] = urdf_upper

        self.get_logger().info(
            'Safe MCP upper limits (rad): ' +
            ', '.join(f'{k}={v:.3f}' for k, v in self._mcp_safe_upper.items())
        )

        # Rest poses (all movable joints including mimics)
        self.rest_poses = [p.getJointState(self.robot_id, i)[0] for i in self.movable_joints]

        # Log all joints at startup for verification
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            self.get_logger().info(f'Joint {i}: {info[1].decode("utf-8")} (type={info[2]})')

        # Subscribers
        self.control_type_sub = self.create_subscription(
            String, 'control_type', self.control_type_callback, 10)
        self.rokoko_data_sub = self.create_subscription(
            String, 'rokoko_ref_data', self.rokoko_data_callback, 10)

        # Publisher
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        self.control_type = 'direct'
        self.latest_rokoko_data = None
        self._ik_prev_joint_angles = self.rest_poses.copy()
        self.get_logger().info('Nano Retargeting Node started')

    # ── Link discovery ────────────────────────────────────────────────────────

    def find_link_indices(self):
        """Find fingertip and palm link indices from URDF child link names."""
        self.fingertip_link_indices = {}
        self.palm_link_id = None

        # Child link name → finger key (from nanohand_test.urdf)
        tip_link_names = {
            'pinky_tip':  'pinky',
            'ring_tip':   'ring',
            'middle_tip': 'middle',
            'index_tip':  'index',
            'thumb_tip':  'thumb',
        }

        # The palm is the URDF root link (index -1 in PyBullet).
        # We use the first direct child of palm as a proxy to get palm position
        # via getLinkState. pinky_base is joint 0's child link.
        PALM_PROXY_CHILD = 'pinky_base'

        self.get_logger().info(f'URDF has {self.num_joints} joints')

        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            link_name = info[12].decode('utf-8')  # child link of this joint

            if link_name in tip_link_names:
                finger = tip_link_names[link_name]
                self.fingertip_link_indices[finger] = i
                self.get_logger().info(f'Found {finger} tip at link index {i}')

            if link_name == PALM_PROXY_CHILD:
                self.palm_link_id = i
                self.get_logger().info(f'Found palm proxy link at index {i} ({link_name})')

        if self.palm_link_id is None:
            self.get_logger().warn('Palm proxy link not found — marker visualisation may be offset')

        self.get_logger().info(f'Fingertip link indices: {self.fingertip_link_indices}')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_rokoko_tip(self, parsed_data, finger_name):
        """Return fingertip position array, trying both CSV naming conventions."""
        pos_columns = self.tip_position_mapping[finger_name]
        if all(col in parsed_data for col in pos_columns):
            return np.array([parsed_data[col] for col in pos_columns], dtype=float)
        digit_num = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5}[finger_name]
        alt = [f'RightDigit{digit_num}Tip_pos_{c}' for c in ['x', 'y', 'z']]
        if all(col in parsed_data for col in alt):
            return np.array([parsed_data[col] for col in alt], dtype=float)
        return None

    def rotation_rokoko_to_nano(self, vec):
        """Remap Rokoko character-space axes to PyBullet world axes."""
        rx, ry, rz = vec
        return [rz, -rx, ry]

    def _apply_mimic_joints(self, joint_angles):
        """
        Compute PIP and DIP angles from their MCP curl parent and update
        PyBullet state. Both PIP and DIP couple to the MCP (not each other)
        because the same tendon drives all three phalanges.

        Mimic angles are clamped to their own URDF limits [0, 1.57] so that
        large multipliers (e.g. 3.94 for pinky DIP) cannot drive a link past
        its physical stop and into the palm or an adjacent finger.
        joint_angles dict is NOT mutated.
        """
        MIMIC_LOWER = 0.0
        MIMIC_UPPER = 1.57   # matches <limit> upper in URDF for all PIP/DIP joints
        for mimic_pb_idx, (parent_name, _, multiplier, offset) in self.mimic_joints.items():
            if parent_name in joint_angles:
                mimic_angle = joint_angles[parent_name] * multiplier + offset
                mimic_angle = max(MIMIC_LOWER, min(MIMIC_UPPER, mimic_angle))
                p.resetJointState(self.robot_id, mimic_pb_idx, mimic_angle)

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
            marker.header.frame_id = 'base'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'fingertip_targets'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = float(position[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.01
            marker.color.a = 1.0
            r, g, b = colors.get(finger_name, (1.0, 1.0, 1.0))
            marker.color.r, marker.color.g, marker.color.b = r, g, b
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def publish_joint_state(self, joint_angles):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(a) for a in joint_angles]
        self.joint_pub.publish(msg)

    # ── Callbacks ─────────────────────────────────────────────────────────────

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
        """Parse timestamp,joint_name,value lines into {joint_name: float}."""
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

    # ── Control methods ───────────────────────────────────────────────────────

    def direct_joint_angle_control(self, parsed_data):
        """
        Direct joint angle control.
        Maps Rokoko CSV columns → nano MCP curl and wiggle joints.
        PIP and DIP mimic joints are computed automatically from the curl angle
        via _apply_mimic_joints — no separate CSV columns needed for them.
        No Vive tracker or calibration required.
        """
        joint_angles = {}

        for nano_joint, rokoko_col in self.joint_mapping.items():
            val = parsed_data.get(rokoko_col, 0.0)
            rad = math.radians(val) if self.data_in_degrees else val
            lo, hi = self.joint_limits_rad.get(nano_joint, (-math.pi, math.pi))
            # Tighten the upper limit for curl joints so their mimic PIP/DIP
            # joints stay within [0, 1.57] and don't drive fingers into the palm.
            safe_hi = self._mcp_safe_upper.get(nano_joint, hi)
            rad = max(lo, min(safe_hi, rad))
            joint_angles[nano_joint] = rad

        # Update independent joints in PyBullet
        for pb_idx, joint_name in self.pybullet_to_nano.items():
            if joint_name in joint_angles:
                p.resetJointState(self.robot_id, pb_idx, joint_angles[joint_name])

        # Propagate tendon coupling to PIP and DIP
        self._apply_mimic_joints(joint_angles)

        # # Visualise fingertip markers using palm-relative Rokoko positions
        # inspire_tips_in_world = {}
        # palm_state = p.getLinkState(self.robot_id, self.palm_link_id) if self.palm_link_id is not None else None
        # if palm_state is not None:
        #     palm_pos = np.array(palm_state[0])
        #     palm_cols = ['RightHand_position_x', 'RightHand_position_y', 'RightHand_position_z']
        #     if all(col in parsed_data for col in palm_cols):
        #         rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)
        #         for finger_name in self.tip_position_mapping:
        #             rokoko_tip = self.get_rokoko_tip(parsed_data, finger_name)
        #             if rokoko_tip is not None:
        #                 rel = rokoko_tip - rokoko_palm
        #                 inspire_tips_in_world[finger_name] = palm_pos + np.array(
        #                     self.rotation_rokoko_to_nano(rel))

        # if inspire_tips_in_world:
        #     self.publish_target_markers(inspire_tips_in_world)

        # Visualise fingertip markers using FK positions from PyBullet.
        # After resetJointState above, getLinkState gives the true world-frame
        # fingertip position.  We express each tip relative to the PyBullet
        # world origin — which maps directly onto the 'base' RViz frame because
        # the URDF root (palm) is loaded at the world origin with useFixedBase.
        # The 'base' link has a 180° X flip relative to 'palm', but PyBullet
        # works in the 'palm' world frame.  To get coordinates in the 'base'
        # frame we apply the same 180° X rotation: [x, y, z] → [x, -y, -z].
        targets = {}
        for finger_name, tip_idx in self.fingertip_link_indices.items():
            link_state = p.getLinkState(self.robot_id, tip_idx, computeForwardKinematics=True)
            pos_palm_world = np.array(link_state[0])   # world frame == palm frame
            # Rotate into 'base' frame (180° around X)
            pos_base = np.array([pos_palm_world[0], -pos_palm_world[1], -pos_palm_world[2]])
            targets[finger_name] = pos_base

        if targets:
            self.publish_target_markers(targets)

        return [joint_angles.get(name, 0.0) for name in self.joint_names]

    def fingertip_ik_control(self, parsed_data):
        """
        Fingertip IK using PyBullet calculateInverseKinematics.
        Computes world-frame fingertip targets from palm-relative Rokoko data,
        solves IK per finger sequentially, then propagates mimic joints.
        """
        palm_link_info = p.getLinkState(self.robot_id, self.palm_link_id, computeForwardKinematics=True)
        inspire_palm_in_world = np.array(palm_link_info[0], dtype=float)

        palm_cols = ['RightHand_position_x', 'RightHand_position_y', 'RightHand_position_z']
        if not all(col in parsed_data for col in palm_cols):
            self.get_logger().warn('No palm position data for IK', once=True)
            return [0.0] * len(self.joint_names)

        rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)
        inspire_palm_rokoko = np.array(self.rotation_rokoko_to_nano(rokoko_palm), dtype=float)

        inspire_tips_in_world = {}
        for finger_name in self.tip_position_mapping:
            rokoko_tip = self.get_rokoko_tip(parsed_data, finger_name)
            if rokoko_tip is None:
                continue
            inspire_tip = np.array(self.rotation_rokoko_to_nano(rokoko_tip), dtype=float)
            inspire_tips_in_world[finger_name] = (inspire_tip - inspire_palm_rokoko) + inspire_palm_in_world

        if not hasattr(self, '_logged_ik_fingertips'):
            self._logged_ik_fingertips = True
            self.get_logger().info(f'Fingertip IK: targeting {len(inspire_tips_in_world)} fingers')

        pb_to_movable_idx = {pb: i for i, pb in enumerate(self.movable_joints)}
        current_joint_angles = self._ik_prev_joint_angles.copy()

        # Max change per frame per joint — thumb gets a tighter clamp to prevent
        # branch-flipping jitter from the underdetermined 2-DOF IK problem.
        MAX_DELTA = {
            'thumb':  0.05,  # ~3 deg/frame at 30Hz — tight to prevent branch jumps
            'index':  0.15,
            'middle': 0.15,
            'ring':   0.15,
            'pinky':  0.15,
        }

        for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            if finger_name not in inspire_tips_in_world:
                continue
            link_index = self.fingertip_link_indices.get(finger_name)
            if link_index is None:
                continue
            try:
                ik_result = p.calculateInverseKinematics(
                    self.robot_id, link_index,
                    inspire_tips_in_world[finger_name],
                    lowerLimits=self.lower_limits,
                    upperLimits=self.upper_limits,
                    jointRanges=self.joint_ranges,
                    restPoses=current_joint_angles,
                    maxNumIterations=200,
                    residualThreshold=1e-4,
                )
                max_delta = MAX_DELTA[finger_name]
                for pb_idx in self.finger_chains[finger_name]['joint_indices']:
                    if pb_idx in pb_to_movable_idx:
                        movable_idx = pb_to_movable_idx[pb_idx]
                        prev = current_joint_angles[movable_idx]
                        clamped = prev + max(min(ik_result[movable_idx] - prev, max_delta), -max_delta)
                        current_joint_angles[movable_idx] = clamped
                        p.resetJointState(self.robot_id, pb_idx, clamped)
            except Exception as e:
                self.get_logger().warn(f'IK failed for {finger_name}: {e}')

        # Persist solution so next frame seeds IK from here (prevents inter-frame flipping)
        self._ik_prev_joint_angles = current_joint_angles.copy()

        # Build joint_angles dict from IK result, then propagate mimics
        joint_angles = {
            self.pybullet_to_nano[pb]: current_joint_angles[pb_to_movable_idx[pb]]
            for pb in self.pybullet_to_nano if pb in pb_to_movable_idx
        }
        self._apply_mimic_joints(joint_angles)

        # Visualise FK fingertip positions (same as direct mode — world frame → base frame)
        targets = {}
        for finger_name, tip_idx in self.fingertip_link_indices.items():
            link_state = p.getLinkState(self.robot_id, tip_idx, computeForwardKinematics=True)
            pos = np.array(link_state[0])
            targets[finger_name] = np.array([pos[0], -pos[1], -pos[2]])
        self.publish_target_markers(targets)
        return [joint_angles.get(name, 0.0) for name in self.joint_names]

    def jparse_ik_control(self, parsed_data):
        """
        Jacobian-based IK using jparse, solved per finger independently.
        PIP and DIP mimic joints are propagated after each finger solve.
        """
        if not JPARSE_AVAILABLE:
            self.get_logger().error('jparse not available!')
            return [0.0] * len(self.joint_names)

        if not hasattr(self, '_jparse_log_count'):
            self._jparse_log_count = 0
        if not hasattr(self, '_pos_debug_count'):
            self._pos_debug_count = 0

        # Initialise from current PyBullet state
        joint_angles = {
            jname: p.getJointState(self.robot_id, pb_idx)[0]
            for pb_idx, jname in self.pybullet_to_nano.items()
        }

        palm_link_info = p.getLinkState(self.robot_id, self.palm_link_id, computeForwardKinematics=True)
        inspire_palm_in_world = np.array(palm_link_info[0], dtype=float)

        palm_cols = ['RightHand_position_x', 'RightHand_position_y', 'RightHand_position_z']
        rokoko_palm_available = all(col in parsed_data for col in palm_cols)
        inspire_palm_from_rokoko = None
        rokoko_palm = None
        if rokoko_palm_available:
            rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)
            inspire_palm_from_rokoko = np.array(self.rotation_rokoko_to_nano(rokoko_palm), dtype=float)

        fingers_processed = 0
        targets_for_viz = {}

        for finger_name, chain_info in self.finger_chains.items():
            rokoko_tip = self.get_rokoko_tip(parsed_data, finger_name)
            if rokoko_tip is None:
                if self._jparse_log_count < 3:
                    self.get_logger().warn(f'No position data for {finger_name}, skipping')
                continue

            fingers_processed += 1
            inspire_tip = np.array(self.rotation_rokoko_to_nano(rokoko_tip), dtype=float)
            if rokoko_palm_available:
                target_pos_world = (inspire_tip - inspire_palm_from_rokoko) + inspire_palm_in_world
            else:
                target_pos_world = inspire_tip
            targets_for_viz[finger_name] = target_pos_world.tolist()

            current_joint_positions = [
                p.getJointState(self.robot_id, idx)[0] for idx in chain_info['joint_indices']
            ]
            current_ee_pos = np.array(p.getLinkState(self.robot_id, chain_info['ee_link_idx'])[0])
            pos_error = target_pos_world - current_ee_pos

            if self._pos_debug_count < 3 and finger_name == 'index':
                self.get_logger().info(f'=== {finger_name} jparse debug ===')
                self.get_logger().info(f'Target: {target_pos_world}  Current: {current_ee_pos}')
                self.get_logger().info(f'Error magnitude: {np.linalg.norm(pos_error)*1000:.1f}mm')
                self._pos_debug_count += 1

            # Build full Jacobian then extract finger columns
            joint_positions, joint_velocities, joint_accelerations = [], [], []
            joint_index_to_col = {}
            col_idx = 0
            for i in range(self.num_joints):
                if p.getJointInfo(self.robot_id, i)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
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
            J_finger = J_linear[:, finger_col_indices]

            try:
                jp_solver = jparse.JParseCore(gamma=0.1)
                J_pinv = jp_solver.compute(
                    jacobian=J_finger,
                    singular_direction_gain_position=1.0,
                    position_dimensions=3,
                    return_nullspace=False,
                )
                max_step = 0.005
                error_norm = np.linalg.norm(pos_error)
                clamped = pos_error * (max_step / error_norm) if error_norm > max_step else pos_error
                delta_q = J_pinv @ clamped

                for i, jname in enumerate(chain_info['joint_names']):
                    new_angle = current_joint_positions[i] + delta_q[i]
                    if jname in self.joint_limits_rad:
                        lo, hi = self.joint_limits_rad[jname]
                        new_angle = max(lo, min(hi, new_angle))
                    joint_angles[jname] = new_angle

                self._jparse_log_count += 1
                if self._jparse_log_count < 20 or self._jparse_log_count % 60 == 0:
                    self.get_logger().info(
                        f'{finger_name}: err={error_norm*1000:.1f}mm  '
                        f'delta_q_max={np.degrees(np.max(np.abs(delta_q))):.1f}°'
                    )

            except Exception as e:
                self.get_logger().error(f'jparse IK failed for {finger_name}: {e}')
                for i, jname in enumerate(chain_info['joint_names']):
                    joint_angles[jname] = current_joint_positions[i]

        if self._jparse_log_count < 3:
            self.get_logger().info(f'Processed {fingers_processed}/5 fingers')

        if targets_for_viz:
            # Flip palm-world → base frame (180° X rotation in URDF base joint)
            tips_in_base = {
                k: np.array([v[0], -v[1], -v[2]]) for k, v in targets_for_viz.items()
            }
            self.publish_target_markers(tips_in_base)

        # Update independent joints in PyBullet, then propagate mimics
        for pb_idx, jname in self.pybullet_to_nano.items():
            if jname in joint_angles:
                p.resetJointState(self.robot_id, pb_idx, joint_angles[jname])
        self._apply_mimic_joints(joint_angles)

        return [joint_angles.get(name, 0.0) for name in self.joint_names]


def main(args=None):
    rclpy.init(args=args)
    node = NanoRetargeting()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()