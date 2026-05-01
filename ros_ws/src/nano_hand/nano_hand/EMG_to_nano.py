#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Bool, Float32MultiArray, Int32, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import pandas as pd
import json
import os
import time
import threading
import numpy as np
import math

N_SERVOS = 10
DOF_ANGLE_SOURCES = {
    1: ['RightDigit5Metacarpophalangeal_ulnarDeviation'],
    2: ['RightDigit5Metacarpophalangeal_flexion', 'RightDigit5ProximalInterphalangeal_flexion'],
    3: ['RightDigit4Metacarpophalangeal_ulnarDeviation'],
    4: ['RightDigit4Metacarpophalangeal_flexion', 'RightDigit4ProximalInterphalangeal_flexion'],
    5: ['RightDigit3Metacarpophalangeal_ulnarDeviation'],
    6: ['RightDigit3Metacarpophalangeal_flexion', 'RightDigit3ProximalInterphalangeal_flexion'],
    7: ['RightDigit2Metacarpophalangeal_ulnarDeviation'],
    8: ['RightDigit2Metacarpophalangeal_flexion', 'RightDigit2ProximalInterphalangeal_flexion'],
    9: ['RightDigit1Carpometacarpal_ulnarDeviation'],
    10: ['RightDigit1Carpometacarpal_flexion']
}

# FSR index → positions-array indices (0-based) for all DOFs of that finger
FSR_TO_DOF_INDICES = {
    0: [8, 9],   # thumb  → DOF 9 (abduction), DOF 10 (flex)
    1: [6, 7],   # index  → DOF 7, 8
    2: [4, 5],   # middle → DOF 5, 6
    3: [2, 3],   # ring   → DOF 3, 4
    4: [0, 1],   # pinky  → DOF 1, 2
}
# Flex-only DOF index per finger, used during extrapolation
FSR_FLEX_DOF_INDEX = {0: 9, 1: 7, 2: 5, 3: 3, 4: 1}

CALIBRATION_FILE = "nano_calibration.json"
FSR_STALE_S = 0.5       # treat FSR as disconnected if no update within this window
EXTRAP_STEP = 0.002     # position units added per frame during extrapolation
EXTRAP_TIMEOUT_S = 5.0  # maximum extrapolation duration
CURL_GAIN = 2.0         # amplifies curl: half the URDF range drives full servo travel

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

try:
    import jparse_robotics as jparse
    JPARSE_AVAILABLE = True
except ImportError:
    JPARSE_AVAILABLE = False


class EMGToNanoMultiCSV(Node):
    def __init__(self):
        super().__init__('emg_to_nano')
        self.declare_parameter('use_fsr', False)
        self.declare_parameter('use_jparse_ik', False)
        self.use_fsr = self.get_parameter('use_fsr').value
        self.use_jparse_ik = self.get_parameter('use_jparse_ik').value

        self.create_subscription(Int32, '/emg_gesture', self.emg_callback, 10)
        self.create_subscription(String, '/gemini/detected_object', self.gemini_callback, 10)
        self.pub = self.create_publisher(JointTrajectory, '/nano_hand/joint_trajectory', 10)

        if self.use_fsr:
            self.create_subscription(Float32MultiArray, '/fsr/data', self._fsr_data_cb, 10)
            self.create_subscription(Bool, '/fsr/connected', self._fsr_connected_cb, 10)

        self.csv_map = {
            "Bottle body": 'bottle_body_fsr.csv',
            "Bottle cap": 'bottle_lid_fsr.csv',
            "Mug body": 'mug_body_fsr.csv',
            "Mug handle": 'mug_handle_fsr.csv',
            "test": 'test.csv'
        }

        self.current_df = None
        self.current_thresholds = {}   # loaded from <recording>_fsr.json
        self.is_open_pose = False

        self.frame_rate = 100
        self.frame_delay = 1.0 / self.frame_rate
        self.active_thread = None
        self.stop_flag = False

        # Stability filter
        self.last_gesture = 0
        self.candidate_gesture = None
        self.candidate_count = 0
        self.required_stability = 3

        # Calibration
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE) as f:
                self.calibration = json.load(f)
        else:
            self.calibration = None

        self.max_open = [1.0] * N_SERVOS
        self.min_closure = [0.0] * N_SERVOS

        # FSR state — written by ROS callbacks, read by stream thread (GIL-safe)
        self.latest_fsr = [0.0] * 5
        self.fsr_connected = False
        self.fsr_last_time = 0.0

        if self.use_jparse_ik:
            if not PYBULLET_AVAILABLE or not JPARSE_AVAILABLE:
                self.get_logger().error('pybullet or jparse not available — falling back to direct control')
                self.use_jparse_ik = False
            else:
                self._setup_ik()

        self.get_logger().info(
            f"EMG to Nano Hand started (use_fsr={self.use_fsr}, use_jparse_ik={self.use_jparse_ik})."
        )

    # ------------------------------------------------------------------
    # FSR callbacks
    # ------------------------------------------------------------------

    def _fsr_data_cb(self, msg):
        self.latest_fsr = list(msg.data)
        self.fsr_last_time = time.time()

    def _fsr_connected_cb(self, msg):
        self.fsr_connected = msg.data

    def _fsr_fresh(self):
        return self.fsr_connected and (time.time() - self.fsr_last_time < FSR_STALE_S)

    # ------------------------------------------------------------------
    # Angle → position helpers (direct calibration path)
    # ------------------------------------------------------------------

    def normalize_with_calibration(self, angle, dof):
        if not self.calibration:
            return 0.0
        cal = self.calibration[str(dof)]
        min_v, max_v = cal["csv_open"], cal["csv_closed"]
        norm = 0.0 if (max_v - min_v) == 0 else (angle - min_v) / (max_v - min_v)
        norm = max(0.0, min(1.0, norm))
        return 1.0 - norm if cal.get("invert", False) else norm

    def extract_positions(self, row):
        dof_positions = []
        for dof, joints in DOF_ANGLE_SOURCES.items():
            values = [row[j] for j in joints if j in row and pd.notnull(row[j])]
            avg = sum(values) / len(values) if values else 0.0
            norm = self.normalize_with_calibration(avg, dof)
            min_val, max_val = self.min_closure[dof-1], self.max_open[dof-1]
            dof_positions.append(min_val + (max_val - min_val) * norm)
        return dof_positions

    def _publish_positions(self, positions):
        msg = JointTrajectory()
        msg.joint_names = [f'servo{i}' for i in range(1, N_SERVOS + 1)]
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start.nanosec = int(self.frame_delay * 1e9)
        msg.points.append(point)
        self.pub.publish(msg)

    # ------------------------------------------------------------------
    # FSR threshold loading
    # ------------------------------------------------------------------

    def _load_fsr_thresholds(self, csv_filename):
        stem = os.path.splitext(csv_filename)[0]
        json_path = os.path.join(
            get_package_share_directory('nano_hand'), 'fsr_csv', f'{stem}.json'
        )
        try:
            with open(json_path) as f:
                data = json.load(f)
            self.get_logger().info(f"Loaded FSR thresholds: {stem}.json")
            return data.get('sensors', {})
        except FileNotFoundError:
            self.get_logger().warn(f"No FSR thresholds found for {csv_filename} — force gating disabled.")
            return {}

    # ------------------------------------------------------------------
    # jparse IK setup (called once when use_jparse_ik=True)
    # ------------------------------------------------------------------

    def _setup_ik(self):
        self._ik_physics_client = p.connect(p.DIRECT)

        share_dir = get_package_share_directory('hands')
        urdf_path = os.path.join(share_dir, 'urdf', 'nano', 'nano_hand_right.urdf')
        self._ik_robot_id = p.loadURDF(urdf_path, useFixedBase=True)
        self.get_logger().info(f'IK: loaded URDF from {urdf_path}')

        # Independent (servo-driven) joints: name → {pb_idx, limits [lo, hi] rad}
        self._ik_joints = {
            'pinky_wiggle': {'pb_idx': 1,  'limits': [-0.17,  0.17 ]},
            'pinky_curl':   {'pb_idx': 2,  'limits': [0.0,    0.97 ]},
            'ring_wiggle':  {'pb_idx': 6,  'limits': [-0.2,   0.13 ]},
            'ring_curl':    {'pb_idx': 7,  'limits': [0.0,    0.83 ]},
            'middle_wiggle':{'pb_idx': 11, 'limits': [-0.13,  0.24 ]},
            'middle_curl':  {'pb_idx': 12, 'limits': [0.0,    1.09 ]},
            'index_wiggle': {'pb_idx': 16, 'limits': [-0.17,  0.17 ]},
            'index_curl':   {'pb_idx': 17, 'limits': [0.0,    1.06 ]},
            'thumb_wiggle': {'pb_idx': 21, 'limits': [-0.78,  1.48 ]},
            'thumb_curl':   {'pb_idx': 22, 'limits': [0.0,    0.68 ]},
        }

        # Mimic joints (PIP and DIP): pb_idx → (parent_name, parent_pb_idx, multiplier, offset)
        self._ik_mimic_joints = {
            3:  ('pinky_curl',  2,  2.0,  0.0),
            4:  ('pinky_curl',  2,  2.23, 0.0),
            8:  ('ring_curl',   7,  1.5,  0.0),
            9:  ('ring_curl',   7,  1.96, 0.0),
            13: ('middle_curl', 12, 2.0,  0.0),
            14: ('middle_curl', 12, 2.18, 0.0),
            18: ('index_curl',  17, 2.0,  0.0),
            19: ('index_curl',  17, 2.23, 0.0),
            23: ('thumb_curl',  22, 0.83, 0.0),
            24: ('thumb_curl',  22, 1.0,  0.0),
        }

        # Distal phalanx world positions — IK targets from CSV
        self._ik_tip_position_mapping = {
            'thumb':  ['RightDigit1DistalPhalanx_position_x', 'RightDigit1DistalPhalanx_position_y', 'RightDigit1DistalPhalanx_position_z'],
            'index':  ['RightDigit2DistalPhalanx_position_x', 'RightDigit2DistalPhalanx_position_y', 'RightDigit2DistalPhalanx_position_z'],
            'middle': ['RightDigit3DistalPhalanx_position_x', 'RightDigit3DistalPhalanx_position_y', 'RightDigit3DistalPhalanx_position_z'],
            'ring':   ['RightDigit4DistalPhalanx_position_x', 'RightDigit4DistalPhalanx_position_y', 'RightDigit4DistalPhalanx_position_z'],
            'pinky':  ['RightDigit5DistalPhalanx_position_x', 'RightDigit5DistalPhalanx_position_y', 'RightDigit5DistalPhalanx_position_z'],
        }

        # Build joint name list (sorted by pb_idx) and limit lookup
        self._ik_pybullet_to_nano = {v['pb_idx']: k for k, v in self._ik_joints.items()}
        self._ik_joint_names = [self._ik_pybullet_to_nano[i] for i in sorted(self._ik_pybullet_to_nano)]
        self._ik_joint_limits_rad = {name: data['limits'] for name, data in self._ik_joints.items()}

        # All movable joints (independent + mimics, needed for Jacobian)
        self._ik_num_joints = p.getNumJoints(self._ik_robot_id)
        self._ik_movable_joints = [
            i for i in range(self._ik_num_joints)
            if p.getJointInfo(self._ik_robot_id, i)[2] != p.JOINT_FIXED
        ]

        self._ik_find_link_indices()

        # Rest fingertip distances from palm origin (used to scale IK targets)
        self._ik_tip_distances = {}
        for fname, link_idx in self._ik_fingertip_link_indices.items():
            ls = p.getLinkState(self._ik_robot_id, link_idx, computeForwardKinematics=True)
            self._ik_tip_distances[fname] = float(np.linalg.norm(ls[0]))

        # Joint limit arrays for all movable joints (needed by PyBullet IK)
        mimic_limits = {
            3:  (0.0, 1.41), 4:  (0.0, 1.64),
            8:  (0.0, 1.29), 9:  (0.0, 1.55),
            13: (0.0, 1.65), 14: (0.0, 1.6 ),
            18: (0.0, 1.65), 19: (0.0, 1.64),
            23: (0.0, 0.66), 24: (0.0, 1.65),
        }
        indep_limits = {v['pb_idx']: v['limits'] for v in self._ik_joints.values()}
        self._ik_lower_limits, self._ik_upper_limits = [], []
        for pb_idx in self._ik_movable_joints:
            lo, hi = indep_limits.get(pb_idx, mimic_limits.get(pb_idx, (-3.14, 3.14)))
            self._ik_lower_limits.append(lo)
            self._ik_upper_limits.append(hi)
        self._ik_joint_ranges = [u - l for l, u in zip(self._ik_lower_limits, self._ik_upper_limits)]

        # Per-finger chains: wiggle + curl + pip + dip (4 DOF, matches nano_retargeting.py)
        self._ik_finger_chains = {
            'thumb':  {'ee_link_idx': self._ik_fingertip_link_indices['thumb'],  'joint_indices': [21, 22, 23, 24]},
            'index':  {'ee_link_idx': self._ik_fingertip_link_indices['index'],  'joint_indices': [16, 17, 18, 19]},
            'middle': {'ee_link_idx': self._ik_fingertip_link_indices['middle'], 'joint_indices': [11, 12, 13, 14]},
            'ring':   {'ee_link_idx': self._ik_fingertip_link_indices['ring'],   'joint_indices': [6,  7,  8,  9 ]},
            'pinky':  {'ee_link_idx': self._ik_fingertip_link_indices['pinky'],  'joint_indices': [1,  2,  3,  4 ]},
        }

        # Pre-compute safe MCP upper limits so mimic PIP/DIP stay within [0, 1.57]
        MIMIC_UPPER = 1.57
        self._ik_mcp_safe_upper = {}
        for _, (parent_name, _, multiplier, _) in self._ik_mimic_joints.items():
            if multiplier > 0:
                safe = MIMIC_UPPER / multiplier
                self._ik_mcp_safe_upper[parent_name] = min(
                    self._ik_mcp_safe_upper.get(parent_name, float('inf')), safe
                )
        for jname, data in self._ik_joints.items():
            urdf_hi = data['limits'][1]
            self._ik_mcp_safe_upper[jname] = min(self._ik_mcp_safe_upper.get(jname, urdf_hi), urdf_hi)

        self._ik_jparse_log_count = 0
        self.get_logger().info('jparse IK setup complete.')

    def _ik_find_link_indices(self):
        self._ik_fingertip_link_indices = {}
        self._ik_palm_link_id = None
        tip_link_names = {
            'pinky_fingertip': 'pinky', 'ring_fingertip': 'ring',
            'middle_fingertip': 'middle', 'index_fingertip': 'index', 'thumb_fingertip': 'thumb',
        }
        for i in range(self._ik_num_joints):
            info = p.getJointInfo(self._ik_robot_id, i)
            link_name = info[12].decode('utf-8')
            if link_name in tip_link_names:
                self._ik_fingertip_link_indices[tip_link_names[link_name]] = i
            if link_name == 'pinky_base':
                self._ik_palm_link_id = i

    def _ik_apply_mimic_joints(self, joint_angles):
        MIMIC_LOWER, MIMIC_UPPER = 0.0, 1.57
        for mimic_pb_idx, (parent_name, _, multiplier, offset) in self._ik_mimic_joints.items():
            if parent_name in joint_angles:
                angle = max(MIMIC_LOWER, min(MIMIC_UPPER, joint_angles[parent_name] * multiplier + offset))
                p.resetJointState(self._ik_robot_id, mimic_pb_idx, angle)

    def _ik_compute_palm_frame(self, parsed_data):
        needed = [
            'RightHand_position_x', 'RightHand_position_y', 'RightHand_position_z',
            'RightDigit3MetaCarpal_position_x', 'RightDigit3MetaCarpal_position_y', 'RightDigit3MetaCarpal_position_z',
            'RightDigit2MetaCarpal_position_x', 'RightDigit2MetaCarpal_position_y', 'RightDigit2MetaCarpal_position_z',
            'RightDigit5MetaCarpal_position_x', 'RightDigit5MetaCarpal_position_y', 'RightDigit5MetaCarpal_position_z',
        ]
        if not all(k in parsed_data for k in needed):
            return None, None
        palm = np.array([parsed_data['RightHand_position_x'],
                         parsed_data['RightHand_position_y'],
                         parsed_data['RightHand_position_z']])
        mid  = np.array([parsed_data['RightDigit3MetaCarpal_position_x'],
                         parsed_data['RightDigit3MetaCarpal_position_y'],
                         parsed_data['RightDigit3MetaCarpal_position_z']])
        idx  = np.array([parsed_data['RightDigit2MetaCarpal_position_x'],
                         parsed_data['RightDigit2MetaCarpal_position_y'],
                         parsed_data['RightDigit2MetaCarpal_position_z']])
        pnk  = np.array([parsed_data['RightDigit5MetaCarpal_position_x'],
                         parsed_data['RightDigit5MetaCarpal_position_y'],
                         parsed_data['RightDigit5MetaCarpal_position_z']])
        fwd = mid - palm;         fwd /= np.linalg.norm(fwd)
        lat = pnk - idx;          lat /= np.linalg.norm(lat)
        nrm = np.cross(fwd, lat); nrm /= np.linalg.norm(nrm)
        lat = np.cross(nrm, fwd); lat /= np.linalg.norm(lat)
        return np.column_stack([lat, nrm, fwd]), palm

    def _ik_get_rokoko_tip(self, parsed_data, finger_name):
        pos_cols = self._ik_tip_position_mapping[finger_name]
        if all(col in parsed_data for col in pos_cols):
            return np.array([parsed_data[col] for col in pos_cols], dtype=float)
        digit_num = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5}[finger_name]
        alt = [f'RightDigit{digit_num}Tip_pos_{c}' for c in ['x', 'y', 'z']]
        if all(col in parsed_data for col in alt):
            return np.array([parsed_data[col] for col in alt], dtype=float)
        return None

    def _run_jparse_ik(self, parsed_data):
        """
        Jacobian-based IK per finger (same algorithm as nano_retargeting.jparse_ik_control).
        Returns radian joint angles in _ik_joint_names order.
        PyBullet state persists between calls so each frame seeds from the previous solution.
        """
        # Seed from current PyBullet state (continuity across frames)
        joint_angles = {
            jname: p.getJointState(self._ik_robot_id, pb_idx)[0]
            for pb_idx, jname in self._ik_pybullet_to_nano.items()
        }

        palm_link_info = p.getLinkState(self._ik_robot_id, self._ik_palm_link_id, computeForwardKinematics=True)
        inspire_palm_in_world = np.array(palm_link_info[0], dtype=float)

        palm_R, palm_world = self._ik_compute_palm_frame(parsed_data)
        HUMAN_FINGER_DIP_REF = 0.145
        HUMAN_THUMB_DIP_REF  = 0.09

        inspire_tips_in_world = {}
        for finger_name in self._ik_tip_position_mapping:
            if palm_R is not None:
                rokoko_tip = self._ik_get_rokoko_tip(parsed_data, finger_name)
                if rokoko_tip is None:
                    continue
                tip_local = palm_R.T @ (rokoko_tip - palm_world)
                if finger_name == 'thumb':
                    tip_nano = np.array([tip_local[0], -tip_local[2], tip_local[1]])
                    scale = self._ik_tip_distances.get('thumb', 0.1) / HUMAN_THUMB_DIP_REF
                else:
                    tip_nano = np.array([tip_local[0], tip_local[1], -tip_local[2]])
                    scale = self._ik_tip_distances.get(finger_name, 0.08) / HUMAN_FINGER_DIP_REF
                inspire_tips_in_world[finger_name] = tip_nano * scale
            else:
                rokoko_tip = self._ik_get_rokoko_tip(parsed_data, finger_name)
                if rokoko_tip is None:
                    continue
                inspire_tips_in_world[finger_name] = rokoko_tip + inspire_palm_in_world

        if not inspire_tips_in_world:
            self.get_logger().warn('No IK targets — returning current state', throttle_duration_sec=5.0)
            return [joint_angles.get(name, 0.0) for name in self._ik_joint_names]

        for finger_name, chain_info in self._ik_finger_chains.items():
            if finger_name not in inspire_tips_in_world:
                continue

            target_pos = inspire_tips_in_world[finger_name]
            current_ee_pos = np.array(p.getLinkState(self._ik_robot_id, chain_info['ee_link_idx'])[0])
            pos_error = target_pos - current_ee_pos

            # Build full Jacobian
            joint_positions, joint_velocities, joint_accelerations = [], [], []
            joint_index_to_col = {}
            col = 0
            for i in range(self._ik_num_joints):
                if p.getJointInfo(self._ik_robot_id, i)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    js = p.getJointState(self._ik_robot_id, i)
                    joint_positions.append(js[0])
                    joint_velocities.append(js[1])
                    joint_accelerations.append(0.0)
                    joint_index_to_col[i] = col
                    col += 1

            jacobian_linear, _ = p.calculateJacobian(
                bodyUniqueId=self._ik_robot_id,
                linkIndex=chain_info['ee_link_idx'],
                localPosition=[0, 0, 0],
                objPositions=joint_positions,
                objVelocities=joint_velocities,
                objAccelerations=joint_accelerations,
            )
            J_linear = np.array(jacobian_linear)

            # 2-DOF effective Jacobian: fold pip/dip mimic coupling into curl column
            pb_wiggle, pb_curl, pb_pip, pb_dip = chain_info['joint_indices']
            pip_mult = self._ik_mimic_joints[pb_pip][2]
            dip_mult = self._ik_mimic_joints[pb_dip][2]
            J_eff = np.column_stack([
                J_linear[:, joint_index_to_col[pb_wiggle]],
                J_linear[:, joint_index_to_col[pb_curl]]
                    + pip_mult * J_linear[:, joint_index_to_col[pb_pip]]
                    + dip_mult * J_linear[:, joint_index_to_col[pb_dip]],
            ])

            try:
                jp_solver = jparse.JParseCore(gamma=0.1)
                J_pinv = jp_solver.compute(
                    jacobian=J_eff,
                    singular_direction_gain_position=1.0,
                    position_dimensions=3,
                    return_nullspace=False,
                )
                max_step = 0.03
                error_norm = np.linalg.norm(pos_error)
                clamped = pos_error * (max_step / error_norm) if error_norm > max_step else pos_error
                delta_q = J_pinv @ clamped  # [delta_wiggle, delta_curl]

                wiggle_name = self._ik_pybullet_to_nano[pb_wiggle]
                curl_name   = self._ik_pybullet_to_nano[pb_curl]

                new_wiggle = p.getJointState(self._ik_robot_id, pb_wiggle)[0] + delta_q[0]
                new_curl   = p.getJointState(self._ik_robot_id, pb_curl)[0]   + delta_q[1]

                lo, hi = self._ik_joint_limits_rad[wiggle_name]
                joint_angles[wiggle_name] = max(lo, min(hi, new_wiggle))

                lo, hi = self._ik_joint_limits_rad[curl_name]
                safe_hi = self._ik_mcp_safe_upper.get(curl_name, hi)
                joint_angles[curl_name] = max(lo, min(safe_hi, new_curl))

                self._ik_jparse_log_count += 1
                if self._ik_jparse_log_count <= 20 or self._ik_jparse_log_count % 100 == 0:
                    self.get_logger().info(
                        f'{finger_name}: err={error_norm*1000:.1f}mm '
                        f'dq_max={np.degrees(np.max(np.abs(delta_q))):.1f}°'
                    )

            except Exception as e:
                self.get_logger().error(f'jparse IK failed for {finger_name}: {e}')

        # Write solution back to PyBullet (seeds next frame) and propagate mimics
        for pb_idx, jname in self._ik_pybullet_to_nano.items():
            if jname in joint_angles:
                p.resetJointState(self._ik_robot_id, pb_idx, joint_angles[jname])
        self._ik_apply_mimic_joints(joint_angles)

        return [joint_angles.get(name, 0.0) for name in self._ik_joint_names]

    def _jparse_ik_positions(self, row):
        """
        Run jparse IK on a pandas CSV row and return [0,1] normalized positions
        in the same 10-element servo order as extract_positions().

        Matches nano_hardware_control.py angle_to_proportion exactly:
          wiggle: 1.0 - t              (URDF lower=left → hamsa 1.0=left)
          curl:   max(0, 1.0 - t * CURL_GAIN)  (inverted + amplified so small
                  IK angles produce full servo travel; same CURL_GAIN=2.0)
        """
        parsed_data = {col: float(val) for col, val in row.items() if pd.notnull(val)}
        angles_rad = self._run_jparse_ik(parsed_data)

        positions = []
        for jname, angle in zip(self._ik_joint_names, angles_rad):
            lo, hi = self._ik_joint_limits_rad[jname]
            span = hi - lo
            t = (angle - lo) / span if span > 0 else 0.0
            t = max(0.0, min(1.0, t))
            if 'curl' in jname:
                prop = max(0.0, 1.0 - t * CURL_GAIN)
            else:
                prop = 1.0 - t
            positions.append(prop)
        return positions

    def _reset_ik_state(self):
        """Reset PyBullet joint states to zero so each new stream starts fresh."""
        for pb_idx in self._ik_pybullet_to_nano:
            p.resetJointState(self._ik_robot_id, pb_idx, 0.0)
        for mimic_pb_idx in self._ik_mimic_joints:
            p.resetJointState(self._ik_robot_id, mimic_pb_idx, 0.0)
        self._ik_jparse_log_count = 0

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream_csv(self):
        df = self.current_df
        if df is None:
            self.get_logger().info("Current dataframe is None!")
            return

        thresholds = self.current_thresholds if self.use_fsr else {}
        mode = 'jparse IK' if self.use_jparse_ik else 'direct'
        self.get_logger().info(
            f"Streaming {len(df)} frames (mode: {mode}, FSR gating: {'on' if thresholds else 'off'})."
        )
        self.is_open_pose = False

        # Per-finger hold state
        finger_held = [False] * 5
        if self.use_jparse_ik:
            last_positions = list(self._jparse_ik_positions(df.iloc[0]))
        else:
            last_positions = list(self.extract_positions(df.iloc[0]))

        # Fingers that have a force target — we stop the recording once all of these are held
        force_targeted = [i for i in range(5) if thresholds.get(f'fsr{i}', {}).get('grasp_force') is not None]

        for _, row in df.iloc[1:].iterrows():
            if self.stop_flag:
                return

            if self.use_jparse_ik:
                target = list(self._jparse_ik_positions(row))
            else:
                target = list(self.extract_positions(row))

            if thresholds and self._fsr_fresh():
                fsr = self.latest_fsr
                for fsr_idx, dof_indices in FSR_TO_DOF_INDICES.items():
                    grasp_force = thresholds.get(f'fsr{fsr_idx}', {}).get('grasp_force')
                    if grasp_force is None:
                        continue  # finger not expected to contact: follow recording as-is
                    if finger_held[fsr_idx]:
                        for di in dof_indices:
                            target[di] = last_positions[di]
                    elif fsr[fsr_idx] >= grasp_force:
                        finger_held[fsr_idx] = True
                        self.get_logger().info(f"fsr{fsr_idx} reached grasp force ({fsr[fsr_idx]:.0f} >= {grasp_force:.0f}), holding.")
                        for di in dof_indices:
                            target[di] = last_positions[di]

            last_positions = target
            self._publish_positions(target)
            time.sleep(self.frame_delay)

            # Stop early once every force-targeted finger is held — don't play release motion
            if force_targeted and all(finger_held[i] for i in force_targeted):
                self.get_logger().info("All force targets met — stopping recording early.")
                break

        # --- Extrapolation phase ---
        # Only for fingers with a grasp_force target that hasn't been reached yet.
        # Requires live FSR; aborts on stale data or timeout.
        needs_extrap = [
            fsr_idx for fsr_idx in range(5)
            if not finger_held[fsr_idx]
            and thresholds.get(f'fsr{fsr_idx}', {}).get('grasp_force') is not None
        ]

        if needs_extrap:
            self.get_logger().info(f"Recording ended; extrapolating fingers {needs_extrap}.")
            t_start = time.time()
            while needs_extrap and not self.stop_flag:
                if time.time() - t_start > EXTRAP_TIMEOUT_S:
                    self.get_logger().warn("Extrapolation timeout — stopping.")
                    break
                if not self._fsr_fresh():
                    self.get_logger().warn("FSR stale during extrapolation — stopping.")
                    break

                fsr = self.latest_fsr
                still_needs = []
                for fsr_idx in needs_extrap:
                    grasp_force = thresholds[f'fsr{fsr_idx}']['grasp_force']
                    if fsr[fsr_idx] >= grasp_force:
                        self.get_logger().info(f"fsr{fsr_idx} reached grasp force during extrapolation.")
                        finger_held[fsr_idx] = True
                    else:
                        flex_di = FSR_FLEX_DOF_INDEX[fsr_idx]
                        # Decrease position: in hamsa space curl 1=open→0=closed
                        last_positions[flex_di] = max(0.0, last_positions[flex_di] - EXTRAP_STEP)
                        still_needs.append(fsr_idx)
                needs_extrap = still_needs

                self._publish_positions(last_positions)
                time.sleep(self.frame_delay)

        self.current_df = None

    def start_stream(self):
        self.stop_flag = False
        if self.use_jparse_ik:
            self._reset_ik_state()
        self.active_thread = threading.Thread(target=self.stream_csv)
        self.active_thread.start()

    def stop_stream(self):
        self.stop_flag = True
        if self.active_thread and self.active_thread.is_alive():
            self.active_thread.join()

    # ------------------------------------------------------------------
    # Hand poses
    # ------------------------------------------------------------------

    def open_hand(self):
        if not self.is_open_pose:
            self._publish_positions([0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0])
            self.is_open_pose = True

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def emg_callback(self, msg):
        gesture = msg.data

        if self.active_thread and self.active_thread.is_alive():
            return

        if gesture == 0:
            self.last_gesture = 0
            self.candidate_gesture = None
            self.candidate_count = 0
            return

        if gesture != self.candidate_gesture:
            self.candidate_gesture = gesture
            self.candidate_count = 1
        else:
            self.candidate_count += 1

        if self.candidate_count >= self.required_stability and gesture != self.last_gesture:
            if gesture == 1:
                self.get_logger().info("Start streaming...")
                self.start_stream()
            elif gesture == 2:
                self.get_logger().info("Opening hand...")
                self.open_hand()
            self.last_gesture = gesture
            self.candidate_count = 0

    def start_pregrasp(self, detected_object):
        csv_file = self.csv_map[detected_object]
        file_path = os.path.join(get_package_share_directory('nano_hand'), 'rokoko_csv', csv_file)
        self.current_df = pd.read_csv(file_path)
        self.current_thresholds = self._load_fsr_thresholds(csv_file) if self.use_fsr else {}
        self.get_logger().info(f"Pre-grasp: loaded {csv_file}.")
        if not self.current_df.empty and not self.stop_flag:
            first_row = self.current_df.iloc[0]
            if self.use_jparse_ik:
                self._publish_positions(self._jparse_ik_positions(first_row))
            else:
                self._publish_positions(self.extract_positions(first_row))
        else:
            self.get_logger().info(f"Empty dataframe in {csv_file}!")

    def gemini_callback(self, msg):
        detected_object = msg.data
        if detected_object in self.csv_map:
            self.start_pregrasp(detected_object)
            self.is_open_pose = False
        elif detected_object == "open":
            self.open_hand()
        else:
            self.get_logger().warn(f"No CSV mapped for: {detected_object}")


def main(args=None):
    rclpy.init(args=args)
    node = EMGToNanoMultiCSV()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
