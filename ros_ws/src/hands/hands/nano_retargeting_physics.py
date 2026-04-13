#!/usr/bin/env python3
"""
Nano Hand Retargeting Node — Physics-based collision avoidance variant.

Key difference from nano_retargeting.py:
  Instead of resetJointState (kinematic teleport that ignores collisions),
  this node uses setJointMotorControl2 + stepSimulation so PyBullet's
  physics engine enforces contact constraints automatically.

  Gravity is disabled. Each callback:
    1. Set position targets on all independent joints via PD motor control.
    2. Drive mimic joints (PIP/DIP) via velocity control to track their parent.
    3. Step the simulation N times to let contacts resolve.
    4. Read back the settled joint positions and publish.

  Because physics resolves collisions, fingers that would interpenetrate
  instead push each other apart — no explicit collision detection code needed.
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


class NanoRetargetingPhysics(Node):
    def __init__(self):
        super().__init__('nano_retargeting_physics')
        self.marker_pub = self.create_publisher(MarkerArray, 'fingertip_targets', 10)

        self.declare_parameter('data_in_degrees', True)
        self.data_in_degrees = self.get_parameter('data_in_degrees').get_parameter_value().bool_value

        # Physics sim steps per control callback — more steps = better collision
        # resolution but higher CPU cost. 10 steps at ~30Hz callback ≈ 300 Hz sim.
        self.declare_parameter('sim_steps_per_update', 30)
        self.sim_steps = self.get_parameter('sim_steps_per_update').get_parameter_value().integer_value

        if self.data_in_degrees:
            self.get_logger().info('Data format: DEGREES')
        else:
            self.get_logger().info('Data format: RADIANS')

        # ── PyBullet (physics mode, not DIRECT) ──────────────────────────────
        self.physics_client_id = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0)
        p.setPhysicsEngineParameter(
            numSolverIterations=100,
            numSubSteps=4,
            contactERP=0.8,       # error reduction per step — higher = stiffer contacts
            globalCFM=0.0001,     # constraint force mixing — small = rigid contacts
        )

        share_directory = get_package_share_directory('hands')
        urdf_path = os.path.join(share_directory, 'urdf', 'nano', 'nano_hand_right.urdf')
        try:
            self.robot_id = p.loadURDF(
                urdf_path, useFixedBase=True,
                flags=(p.URDF_USE_SELF_COLLISION |
                       p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS),
            )
            self.get_logger().info(f'Loaded URDF: {urdf_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            raise

        # ── Discover joint indices by name ────────────────────────────────────
        self.num_joints = p.getNumJoints(self.robot_id)
        _jname_to_pb = {}
        for i in range(self.num_joints):
            name = p.getJointInfo(self.robot_id, i)[1].decode('utf-8')
            _jname_to_pb[name] = i

        # ── Independent (servo-driven) joints ────────────────────────────────
        self.joints = {
            'pinky_wiggle': {'pb_idx': _jname_to_pb['pinky_wiggle'], 'limits': [-0.17,  0.26 ]},
            'pinky_curl':   {'pb_idx': _jname_to_pb['pinky_curl'],   'limits': [0.0,    0.97 ]},
            'ring_wiggle':  {'pb_idx': _jname_to_pb['ring_wiggle'],  'limits': [-0.2,   0.07 ]},
            'ring_curl':    {'pb_idx': _jname_to_pb['ring_curl'],    'limits': [0.0,    0.83 ]},
            'middle_wiggle':{'pb_idx': _jname_to_pb['middle_wiggle'],'limits': [-0.13,  0.24 ]},
            'middle_curl':  {'pb_idx': _jname_to_pb['middle_curl'],  'limits': [0.0,    1.09 ]},
            'index_wiggle': {'pb_idx': _jname_to_pb['index_wiggle'], 'limits': [-0.17,  0.17 ]},
            'index_curl':   {'pb_idx': _jname_to_pb['index_curl'],   'limits': [0.0,    1.06 ]},
            'thumb_wiggle': {'pb_idx': _jname_to_pb['thumb_wiggle'], 'limits': [-0.78,  1.48 ]},
            'thumb_curl':   {'pb_idx': _jname_to_pb['thumb_curl'],   'limits': [0.0,    0.68 ]},
        }

        # ── Mimic joints ──────────────────────────────────────────────────────
        # Format: pb_idx: (parent_joint_name, parent_pb_idx, multiplier, offset)
        self.mimic_joints = {
            _jname_to_pb['pinky_curl_pip']:  ('pinky_curl',  _jname_to_pb['pinky_curl'],  2.0,  0.0),
            _jname_to_pb['pinky_curl_dip']:  ('pinky_curl',  _jname_to_pb['pinky_curl'],  2.23, 0.0),
            _jname_to_pb['ring_curl_pip']:   ('ring_curl',   _jname_to_pb['ring_curl'],   1.5,  0.0),
            _jname_to_pb['ring_curl_dip']:   ('ring_curl',   _jname_to_pb['ring_curl'],   1.96, 0.0),
            _jname_to_pb['middle_curl_pip']: ('middle_curl', _jname_to_pb['middle_curl'], 2.0,  0.0),
            _jname_to_pb['middle_curl_dip']: ('middle_curl', _jname_to_pb['middle_curl'], 2.18, 0.0),
            _jname_to_pb['index_curl_pip']:  ('index_curl',  _jname_to_pb['index_curl'],  2.0,  0.0),
            _jname_to_pb['index_curl_dip']:  ('index_curl',  _jname_to_pb['index_curl'],  2.23, 0.0),
            _jname_to_pb['thumb_curl_pip']:  ('thumb_curl',  _jname_to_pb['thumb_curl'],  0.83, 0.0),
            _jname_to_pb['thumb_curl_dip']:  ('thumb_curl',  _jname_to_pb['thumb_curl'],  2.0,  0.0),
        }

        # ── Rokoko CSV column mapping ─────────────────────────────────────────
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

        self.tip_position_mapping = {
            'thumb':  ['RightDigit1DistalPhalanx_position_x', 'RightDigit1DistalPhalanx_position_y', 'RightDigit1DistalPhalanx_position_z'],
            'index':  ['RightDigit2DistalPhalanx_position_x', 'RightDigit2DistalPhalanx_position_y', 'RightDigit2DistalPhalanx_position_z'],
            'middle': ['RightDigit3DistalPhalanx_position_x', 'RightDigit3DistalPhalanx_position_y', 'RightDigit3DistalPhalanx_position_z'],
            'ring':   ['RightDigit4DistalPhalanx_position_x', 'RightDigit4DistalPhalanx_position_y', 'RightDigit4DistalPhalanx_position_z'],
            'pinky':  ['RightDigit5DistalPhalanx_position_x', 'RightDigit5DistalPhalanx_position_y', 'RightDigit5DistalPhalanx_position_z'],
        }

        self.pybullet_to_nano = {v['pb_idx']: k for k, v in self.joints.items()}
        self.joint_names = [self.pybullet_to_nano[i] for i in sorted(self.pybullet_to_nano)]
        self.joint_limits_rad = {name: data['limits'] for name, data in self.joints.items()}

        # All movable joints (independent + mimic, fixed joints excluded)
        self.movable_joints = [
            i for i in range(self.num_joints)
            if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED
        ]

        self.find_link_indices()

        # ── Safe MCP upper limits (prevent mimic joints exceeding their limits) ─
        self._mcp_safe_upper = {}
        mimic_urdf_upper = {
            _jname_to_pb['pinky_curl_pip']:  1.41,
            _jname_to_pb['pinky_curl_dip']:  1.64,
            _jname_to_pb['ring_curl_pip']:   1.29,
            _jname_to_pb['ring_curl_dip']:   1.55,
            _jname_to_pb['middle_curl_pip']: 1.65,
            _jname_to_pb['middle_curl_dip']: 1.6,
            _jname_to_pb['index_curl_pip']:  1.65,
            _jname_to_pb['index_curl_dip']:  1.64,
            _jname_to_pb['thumb_curl_pip']:  0.66,
            _jname_to_pb['thumb_curl_dip']:  1.65,
        }
        for mimic_pb, (parent_name, _, multiplier, _) in self.mimic_joints.items():
            if multiplier > 0:
                safe = mimic_urdf_upper[mimic_pb] / multiplier
                existing = self._mcp_safe_upper.get(parent_name, float('inf'))
                self._mcp_safe_upper[parent_name] = min(existing, safe)
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

        # ── Motor control gains ───────────────────────────────────────────────
        # High stiffness so joints track targets quickly each sim step.
        # maxForce limits how hard a joint pushes — lower values = softer
        # collision response (fingers yield more to each other).
        self.POSITION_GAIN   = 0.5
        self.VELOCITY_GAIN   = 1.0
        self.MAX_FORCE_INDEP = 2.0   # independent joints — lower = yields more to collisions
        self.MAX_FORCE_MIMIC = 1.0   # mimic joints — even softer

        # Initialise all movable joints with position control at 0
        for pb_idx in self.movable_joints:
            p.setJointMotorControl2(
                self.robot_id, pb_idx,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                positionGain=self.POSITION_GAIN,
                velocityGain=self.VELOCITY_GAIN,
                force=self.MAX_FORCE_INDEP,
            )

        self.control_type = 'direct'
        self.latest_rokoko_data = None
        self._smoothed_targets = {}

        self.create_subscription(String, 'control_type', self.control_type_callback, 10)
        self.create_subscription(String, 'rokoko_ref_data', self.rokoko_data_callback, 10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        self.get_logger().info('Nano Retargeting Physics Node started')

    # ── Link discovery ────────────────────────────────────────────────────────

    def find_link_indices(self):
        self.fingertip_link_indices = {}
        self.palm_link_id = None
        tip_link_names = {
            'pinky_tip':  'pinky',
            'ring_tip':   'ring',
            'middle_tip': 'middle',
            'index_tip':  'index',
            'thumb_tip':  'thumb',
        }
        PALM_PROXY_CHILD = 'pinky_base'
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            link_name = info[12].decode('utf-8')
            if link_name in tip_link_names:
                self.fingertip_link_indices[tip_link_names[link_name]] = i
            if link_name == PALM_PROXY_CHILD:
                self.palm_link_id = i
        self.get_logger().info(f'Fingertip link indices: {self.fingertip_link_indices}')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_rokoko_tip(self, parsed_data, finger_name):
        pos_columns = self.tip_position_mapping[finger_name]
        if all(col in parsed_data for col in pos_columns):
            return np.array([parsed_data[col] for col in pos_columns], dtype=float)
        digit_num = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5}[finger_name]
        alt = [f'RightDigit{digit_num}Tip_pos_{c}' for c in ['x', 'y', 'z']]
        if all(col in parsed_data for col in alt):
            return np.array([parsed_data[col] for col in alt], dtype=float)
        return None

    def rotation_rokoko_to_nano(self, vec):
        rx, ry, rz = vec
        return [rz, -rx, ry]

    def _set_independent_targets(self, joint_angles):
        """Send position targets to all independent joints via motor control."""
        for jname, angle in joint_angles.items():
            pb_idx = self.joints[jname]['pb_idx']
            p.setJointMotorControl2(
                self.robot_id, pb_idx,
                p.POSITION_CONTROL,
                targetPosition=float(angle),
                positionGain=self.POSITION_GAIN,
                velocityGain=self.VELOCITY_GAIN,
                force=self.MAX_FORCE_INDEP,
            )

    def _set_mimic_targets(self, joint_angles):
        """Drive mimic joints to track their parent curl joints."""
        for mimic_pb, (parent_name, _, multiplier, offset) in self.mimic_joints.items():
            if parent_name in joint_angles:
                target = joint_angles[parent_name] * multiplier + offset
                # Clamp to mimic joint URDF limits
                info = p.getJointInfo(self.robot_id, mimic_pb)
                lo, hi = info[8], info[9]
                target = max(lo, min(hi, target))
                p.setJointMotorControl2(
                    self.robot_id, mimic_pb,
                    p.POSITION_CONTROL,
                    targetPosition=float(target),
                    positionGain=self.POSITION_GAIN,
                    velocityGain=self.VELOCITY_GAIN,
                    force=self.MAX_FORCE_MIMIC,
                )

    def _step_and_read(self):
        """Step physics N times and read back settled joint positions."""
        for _ in range(self.sim_steps):
            p.stepSimulation()

        joint_angles = {}
        for pb_idx, jname in self.pybullet_to_nano.items():
            state = p.getJointState(self.robot_id, pb_idx)
            joint_angles[jname] = state[0]
        return joint_angles

    def _publish_fk_markers(self):
        marker_array = MarkerArray()
        colors = {
            'thumb':  (1.0, 0.5, 0.0),
            'index':  (0.0, 1.0, 0.0),
            'middle': (0.0, 0.0, 1.0),
            'ring':   (1.0, 0.0, 1.0),
            'pinky':  (0.0, 1.0, 1.0),
        }
        for i, (finger_name, tip_idx) in enumerate(self.fingertip_link_indices.items()):
            link_state = p.getLinkState(self.robot_id, tip_idx, computeForwardKinematics=True)
            pos = np.array(link_state[0])
            pos_base = np.array([pos[0], -pos[1], -pos[2]])
            m = Marker()
            m.header.frame_id = 'base'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'fingertip_targets'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pos_base[0])
            m.pose.position.y = float(pos_base[1])
            m.pose.position.z = float(pos_base[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.01
            m.color.a = 1.0
            r, g, b = colors.get(finger_name, (1.0, 1.0, 1.0))
            m.color.r, m.color.g, m.color.b = r, g, b
            marker_array.markers.append(m)
        self.marker_pub.publish(marker_array)

    def publish_joint_state(self, joint_angles):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(joint_angles.get(name, 0.0)) for name in self.joint_names]
        self.joint_pub.publish(msg)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def control_type_callback(self, msg):
        self.control_type = msg.data

    def rokoko_data_callback(self, msg):
        parsed_data = self.parse_csv_data(msg.data)
        if not parsed_data:
            return

        if self.control_type == 'direct':
            joint_angles = self.direct_control(parsed_data)
        elif self.control_type == 'fingertip_ik':
            joint_angles = self.fingertip_ik_control(parsed_data)
        else:
            self.get_logger().warn(f'Unknown control type: {self.control_type}', once=True)
            return

        if joint_angles:
            self.publish_joint_state(joint_angles)
            self._publish_fk_markers()

    def parse_csv_data(self, csv_string):
        parsed = {}
        for line in csv_string.strip().split('\n'):
            parts = line.split(',')
            if len(parts) < 3:
                continue
            try:
                parsed[parts[1]] = float(parts[2])
            except ValueError:
                pass
        return parsed

    # ── Control methods ───────────────────────────────────────────────────────

    def direct_control(self, parsed_data):
        """
        Direct joint angle control with physics-based collision avoidance.
        Targets are set via motor control, then the sim is stepped so contacts
        resolve naturally — fingers push each other instead of interpenetrating.
        """
        joint_angles = {}
        for nano_joint, rokoko_col in self.joint_mapping.items():
            val = parsed_data.get(rokoko_col, 0.0)
            rad = math.radians(val) if self.data_in_degrees else val
            lo, hi = self.joint_limits_rad[nano_joint]
            safe_hi = self._mcp_safe_upper.get(nano_joint, hi)
            rad = max(lo, min(safe_hi, rad))
            joint_angles[nano_joint] = rad

        self._set_independent_targets(joint_angles)
        self._set_mimic_targets(joint_angles)
        return self._step_and_read()

    def fingertip_ik_control(self, parsed_data):
        """
        Fingertip IK with physics-based collision avoidance.
        Thumb uses direct angle mapping (stable); fingers 2-5 use PyBullet IK.
        After setting motor targets, physics is stepped for collision resolution.
        """
        # ── Thumb: direct (avoids IK branch-flipping on 2-DOF underdetermined problem)
        joint_angles = {}
        for nano_joint in ['thumb_curl', 'thumb_wiggle']:
            val = parsed_data.get(self.joint_mapping[nano_joint], 0.0)
            rad = math.radians(val) if self.data_in_degrees else val
            lo, hi = self.joint_limits_rad[nano_joint]
            safe_hi = self._mcp_safe_upper.get(nano_joint, hi)
            joint_angles[nano_joint] = max(lo, min(safe_hi, rad))

        self._set_independent_targets(joint_angles)
        self._set_mimic_targets(joint_angles)

        # Step once to place thumb before solving other fingers
        for _ in range(self.sim_steps):
            p.stepSimulation()

        # ── Fingers 2-5: IK with EMA-smoothed targets ────────────────────────
        if self.palm_link_id is None:
            return self._step_and_read()

        palm_cols = ['RightHand_position_x', 'RightHand_position_y', 'RightHand_position_z']
        if not all(col in parsed_data for col in palm_cols):
            self.get_logger().warn('No palm position data for IK', once=True)
            return self._step_and_read()

        palm_link_info = p.getLinkState(self.robot_id, self.palm_link_id, computeForwardKinematics=True)
        nano_palm_in_world = np.array(palm_link_info[0], dtype=float)
        rokoko_palm = np.array([parsed_data[col] for col in palm_cols], dtype=float)
        rokoko_palm_mapped = np.array(self._remap(rokoko_palm), dtype=float)

        EMA_ALPHA = 0.4

        # ── Diagnostic: show what position keys exist in parsed_data ──────────
        if not hasattr(self, '_pos_keys_logged'):
            pos_keys = [k for k in parsed_data if 'position' in k.lower() or '_pos_' in k]
            self.get_logger().info(f'[DIAG] Position keys in parsed_data: {pos_keys[:10]}')
            self._pos_keys_logged = True

        ik_targets = {}
        for finger_name in ['index', 'middle', 'ring', 'pinky']:
            rokoko_tip = self.get_rokoko_tip(parsed_data, finger_name)
            if rokoko_tip is None:
                expected = self.tip_position_mapping[finger_name]
                missing = [c for c in expected if c not in parsed_data]
                self.get_logger().warn(
                    f'No tip for {finger_name}. Missing: {missing}',
                    once=True
                )
                continue
            raw = np.array(self._remap(rokoko_tip), dtype=float)
            raw_world = (raw - rokoko_palm_mapped) + nano_palm_in_world
            if finger_name in self._smoothed_targets:
                smoothed = EMA_ALPHA * raw_world + (1 - EMA_ALPHA) * self._smoothed_targets[finger_name]
            else:
                smoothed = raw_world
            self._smoothed_targets[finger_name] = smoothed
            ik_targets[finger_name] = smoothed

        self.get_logger().info(f'IK targets built for: {list(ik_targets.keys())}', once=True)

        # Build IK limit arrays over all movable joints
        movable_limits = self._build_movable_limits()

        current_poses = [p.getJointState(self.robot_id, i)[0] for i in self.movable_joints]
        pb_to_movable = {pb: i for i, pb in enumerate(self.movable_joints)}

        # ── Diagnostic: compare IK targets to current FK positions ───────────
        if not hasattr(self, '_diag_count'):
            self._diag_count = 0
        if self._diag_count < 1:
            self._diag_count += 1
            self.get_logger().info(
                f'[DIAG] nano_palm={np.round(nano_palm_in_world,4)}, '
                f'rokoko_palm_mapped={np.round(rokoko_palm_mapped,4)}'
            )
            for finger_name, target in ik_targets.items():
                link_idx = self.fingertip_link_indices.get(finger_name)
                if link_idx is not None:
                    fk = p.getLinkState(self.robot_id, link_idx, computeForwardKinematics=True)
                    fk_pos = np.array(fk[0])
                    dist = np.linalg.norm(target - fk_pos)
                    self.get_logger().info(
                        f'[DIAG] {finger_name}: target={np.round(target,4)}, '
                        f'fk={np.round(fk_pos,4)}, dist={dist:.4f}m'
                    )

        finger_joint_pbs = {
            'index':  [self.joints['index_wiggle']['pb_idx'],  self.joints['index_curl']['pb_idx']],
            'middle': [self.joints['middle_wiggle']['pb_idx'], self.joints['middle_curl']['pb_idx']],
            'ring':   [self.joints['ring_wiggle']['pb_idx'],   self.joints['ring_curl']['pb_idx']],
            'pinky':  [self.joints['pinky_wiggle']['pb_idx'],  self.joints['pinky_curl']['pb_idx']],
        }

        for finger_name, target in ik_targets.items():
            link_idx = self.fingertip_link_indices.get(finger_name)
            if link_idx is None:
                continue
            try:
                ik_result = p.calculateInverseKinematics(
                    self.robot_id, link_idx, target,
                    lowerLimits=movable_limits['lower'],
                    upperLimits=movable_limits['upper'],
                    jointRanges=movable_limits['ranges'],
                    restPoses=current_poses,
                    maxNumIterations=200,
                    residualThreshold=1e-4,
                )
                if self._diag_count <= 3:
                    curl_raw = ik_result[pb_to_movable[finger_joint_pbs[finger_name][1]]]
                    wiggle_raw = ik_result[pb_to_movable[finger_joint_pbs[finger_name][0]]]
                    self.get_logger().info(
                        f'[DIAG IK] {finger_name}: curl_raw={curl_raw:.4f} wiggle_raw={wiggle_raw:.4f}'
                    )
                for pb_idx in finger_joint_pbs[finger_name]:
                    if pb_idx in pb_to_movable:
                        angle = ik_result[pb_to_movable[pb_idx]]
                        lo, hi = self.joints[self.pybullet_to_nano[pb_idx]]['limits']
                        # Use actual URDF limits (not safe_hi) — mimic joints are
                        # clamped independently in _set_mimic_targets
                        angle = max(lo, min(hi, angle))
                        joint_angles[self.pybullet_to_nano[pb_idx]] = angle
                        current_poses[pb_to_movable[pb_idx]] = angle
            except Exception as e:
                self.get_logger().warn(f'IK failed for {finger_name}: {e}', once=True)
        self._set_independent_targets(joint_angles)
        self._set_mimic_targets(joint_angles)
        return self._step_and_read()

    def _remap(self, vec):
        """Remap Rokoko character-space axes to PyBullet world axes.
        Rokoko: x=right, y=up, z=forward (character space)
        PyBullet nano hand: fingers extend in -z, spread in +x
          Rokoko +z (forward) → PyBullet -x  (new_x = -rz)
          Rokoko +x (right)   → PyBullet +y  (new_y =  rx)
          Rokoko +y (up)      → PyBullet -z  (new_z = -ry)  ← open fingers → -z (correct)
        """
        rx, ry, rz = vec
        return [-rz, rx, -ry]

    def _build_movable_limits(self):
        """Build lower/upper/range arrays over all movable joints for PyBullet IK."""
        mimic_limits = {
            pb: (p.getJointInfo(self.robot_id, pb)[8], p.getJointInfo(self.robot_id, pb)[9])
            for pb in self.mimic_joints
        }
        indep_limits = {v['pb_idx']: v['limits'] for v in self.joints.values()}
        lower, upper = [], []
        for pb_idx in self.movable_joints:
            if pb_idx in indep_limits:
                lo, hi = indep_limits[pb_idx]
            elif pb_idx in mimic_limits:
                lo, hi = mimic_limits[pb_idx]
            else:
                lo, hi = -3.14, 3.14
            lower.append(lo)
            upper.append(hi)
        ranges = [u - l for l, u in zip(lower, upper)]
        return {'lower': lower, 'upper': upper, 'ranges': ranges}


def main(args=None):
    rclpy.init(args=args)
    node = NanoRetargetingPhysics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
