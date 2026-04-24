#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import socket
import json
import gzip
import zlib


class RokokoListener(Node):
    def __init__(self):
        super().__init__('rokoko_listener')

        # Declare and get parameters
        self.declare_parameter('ip_address', '0.0.0.0')
        self.declare_parameter('port', 14043)
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('playback_mode', False)
        self.declare_parameter('csv_file', '')
        self.declare_parameter('loop_playback', False)

        self.ip_address = self.get_parameter('ip_address').get_parameter_value().string_value
        self.port = self.get_parameter('port').get_parameter_value().integer_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.playback_mode = self.get_parameter('playback_mode').get_parameter_value().bool_value
        self.csv_file = self.get_parameter('csv_file').get_parameter_value().string_value
        self.loop_playback = self.get_parameter('loop_playback').get_parameter_value().bool_value

        # Create publishers for raw and processed Rokoko data
        self.raw_publisher_ = self.create_publisher(String, 'rokoko_raw_data', 10)
        self.ref_publisher_ = self.create_publisher(String, 'rokoko_ref_data', 10)

        # Initialize mode-specific components
        if self.playback_mode:
            self.get_logger().info('='*80)
            self.get_logger().info('CSV PLAYBACK MODE')
            self.get_logger().info(f'CSV file: {self.csv_file}')
            self.get_logger().info(f'Loop playback: {self.loop_playback}')
            self.get_logger().info('='*80)

            # Load CSV data
            self.csv_data = self.load_csv_file(self.csv_file)
            self.csv_index = 0

            # Set up UDP socket (but don't use it)
            self.sock = None
        else:
            # Set up UDP socket for live mode
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.ip_address, self.port))
            self.sock.setblocking(False)  # Non-blocking mode

            self.get_logger().info(f'Rokoko Listener started on {self.ip_address}:{self.port}')
            self.get_logger().info(f'Publishing raw data to: /rokoko_raw_data')
            self.get_logger().info(f'Publishing processed data to: /rokoko_ref_data')

        # Create timer to check for UDP data or playback CSV
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.last_data = None
        self.packet_count = 0
        self.decode_errors = 0
        self.format_detected = False
        self.json_structure_logged = False
        self.sample_data_count = 0
        self.parsed_data_saved = False

    def try_decode_data(self, data):
        """Try multiple methods to decode the data"""

        # Method 1: Direct UTF-8 decode
        try:
            decoded = data.decode('utf-8')
            json_data = json.loads(decoded)
            return decoded, json_data, 'utf-8'
        except:
            pass

        # Method 2: Try gzip decompression
        try:
            decompressed = gzip.decompress(data)
            decoded = decompressed.decode('utf-8')
            json_data = json.loads(decoded)
            return decoded, json_data, 'gzip'
        except:
            pass

        # Method 3: Try zlib decompression
        try:
            decompressed = zlib.decompress(data)
            decoded = decompressed.decode('utf-8')
            json_data = json.loads(decoded)
            return decoded, json_data, 'zlib'
        except:
            pass

        # Method 4: Try latin-1 encoding
        try:
            decoded = data.decode('latin-1')
            json_data = json.loads(decoded)
            return decoded, json_data, 'latin-1'
        except:
            pass

        # Method 5: Try ignoring errors in UTF-8
        try:
            decoded = data.decode('utf-8', errors='ignore')
            json_data = json.loads(decoded)
            return decoded, json_data, 'utf-8-ignore'
        except:
            pass

        return None, None, None

    def log_json_structure(self, data, indent=0):
        """Recursively log the structure of JSON data"""
        prefix = '  ' * indent

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    self.get_logger().info(f'{prefix}{key}: (dict with {len(value)} keys)')
                    if indent < 3:  # Limit depth to avoid too much output
                        self.log_json_structure(value, indent + 1)
                elif isinstance(value, list):
                    self.get_logger().info(f'{prefix}{key}: (list with {len(value)} items)')
                    if len(value) > 0 and indent < 3:
                        self.get_logger().info(f'{prefix}  First item:')
                        self.log_json_structure(value[0], indent + 2)
                else:
                    # Show actual values for primitive types
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:50] + '...'
                    self.get_logger().info(f'{prefix}{key}: {value_str}')
        elif isinstance(data, list):
            self.get_logger().info(f'{prefix}List with {len(data)} items')
            if len(data) > 0:
                self.log_json_structure(data[0], indent)
        else:
            value_str = str(data)
            if len(value_str) > 50:
                value_str = value_str[:50] + '...'
            self.get_logger().info(f'{prefix}{value_str}')

    def load_csv_file(self, csv_path):
        """Load CSV file and return list of row dictionaries"""
        import csv
        import os

        # Expand user home directory (~) in path
        csv_path = os.path.expanduser(csv_path)

        if not os.path.exists(csv_path):
            self.get_logger().error(f'CSV file not found: {csv_path}')
            return []

        data = []
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert string values to float
                    data.append({k: float(v) for k, v in row.items()})

            self.get_logger().info(f'Loaded {len(data)} frames from CSV file')
            return data
        except Exception as e:
            self.get_logger().error(f'Error loading CSV file: {e}')
            return []

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles (in radians) to quaternion"""
        import math

        # Compute half angles
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return x, y, z, w

    def csv_row_to_rokoko_json(self, row):
        """Convert a CSV row to Rokoko JSON format"""
        timestamp = int(row.get('Timestamp', 0))

        # Build Rokoko JSON structure
        json_data = {
            'scene': {
                'timestamp': timestamp,
                'actors': [{
                    'body': {}
                }]
            }
        }

        body = json_data['scene']['actors'][0]['body']

        # Right hand mapping - matching the parse_rokoko_to_csv logic
        # Thumb
        if 'RightDigit1Carpometacarpal_flexion' in row:
            roll = row.get('RightDigit1Carpometacarpal_flexion', 0)
            pitch = row.get('RightDigit1Carpometacarpal_ulnarDeviation', 0)
            yaw = row.get('RightDigit1Carpometacarpal_pronation', 0)
            x, y, z, w = self.euler_to_quaternion(roll, pitch, yaw)
            body['rightThumbProximal'] = {'rotation': {'x': x, 'y': y, 'z': z, 'w': w}}

        if 'RightDigit1Metacarpophalangeal_flexion' in row:
            roll = row.get('RightDigit1Metacarpophalangeal_flexion', 0)
            pitch = row.get('RightDigit1Metacarpophalangeal_ulnarDeviation', 0)
            yaw = row.get('RightDigit1Metacarpophalangeal_pronation', 0)
            x, y, z, w = self.euler_to_quaternion(roll, pitch, yaw)
            body['rightThumbMedial'] = {'rotation': {'x': x, 'y': y, 'z': z, 'w': w}}

        if 'RightDigit1Interphalangeal_flexion' in row:
            roll = row.get('RightDigit1Interphalangeal_flexion', 0)
            pitch = row.get('RightDigit1Interphalangeal_ulnarDeviation', 0)
            yaw = row.get('RightDigit1Interphalangeal_pronation', 0)
            x, y, z, w = self.euler_to_quaternion(roll, pitch, yaw)
            body['rightThumbDistal'] = {'rotation': {'x': x, 'y': y, 'z': z, 'w': w}}

        # Index finger - note the negation for flexion
        if 'RightDigit2Metacarpophalangeal_flexion' in row:
            roll = -row.get('RightDigit2Metacarpophalangeal_flexion', 0)  # Negated
            pitch = row.get('RightDigit2Metacarpophalangeal_ulnarDeviation', 0)
            yaw = row.get('RightDigit2Metacarpophalangeal_pronation', 0)
            x, y, z, w = self.euler_to_quaternion(roll, pitch, yaw)
            body['rightIndexProximal'] = {'rotation': {'x': x, 'y': y, 'z': z, 'w': w}}

        if 'RightDigit2ProximalInterphalangeal_flexion' in row:
            roll = -row.get('RightDigit2ProximalInterphalangeal_flexion', 0)  # Negated
            pitch = row.get('RightDigit2ProximalInterphalangeal_ulnarDeviation', 0)
            yaw = row.get('RightDigit2ProximalInterphalangeal_pronation', 0)
            x, y, z, w = self.euler_to_quaternion(roll, pitch, yaw)
            body['rightIndexMedial'] = {'rotation': {'x': x, 'y': y, 'z': z, 'w': w}}

        if 'RightDigit2DistalInterphalangeal_flexion' in row:
            roll = -row.get('RightDigit2DistalInterphalangeal_flexion', 0)  # Negated
            pitch = row.get('RightDigit2Metacarpophalangeal_ulnarDeviation', 0)
            yaw = row.get('RightDigit2Metacarpophalangeal_pronation', 0)
            x, y, z, w = self.euler_to_quaternion(roll, pitch, yaw)
            body['rightIndexDistal'] = {'rotation': {'x': x, 'y': y, 'z': z, 'w': w}}

        # Middle finger
        if 'RightDigit3Metacarpophalangeal_flexion' in row:
            roll = -row.get('RightDigit3Metacarpophalangeal_flexion', 0)
            pitch = -row.get('RightDigit3Metacarpophalangeal_ulnarDeviation', 0)
            yaw = row.get('RightDigit3Metacarpophalangeal_pronation', 0)
            x, y, z, w = self.euler_to_quaternion(roll, pitch, yaw)
            body['rightMiddleProximal'] = {'rotation': {'x': x, 'y': y, 'z': z, 'w': w}}

        if 'RightDigit3ProximalInterphalangeal_flexion' in row:
            roll = -row.get('RightDigit3ProximalInterphalangeal_flexion', 0)
            pitch = -row.get('RightDigit3Metacarpophalangeal_ulnarDeviation', 0)
            yaw = row.get('RightDigit3Metacarpophalangeal_pronation', 0)
            x, y, z, w = self.euler_to_quaternion(roll, pitch, yaw)
            body['rightMiddleMedial'] = {'rotation': {'x': x, 'y': y, 'z': z, 'w': w}}

        if 'RightDigit3DistalInterphalangeal_flexion' in row:
            roll = -row.get('RightDigit3DistalInterphalangeal_flexion', 0)
            pitch = -row.get('RightDigit3Metacarpophalangeal_ulnarDeviation', 0)
            yaw = row.get('RightDigit3Metacarpophalangeal_pronation', 0)
            x, y, z, w = self.euler_to_quaternion(roll, pitch, yaw)
            body['rightMiddleDistal'] = {'rotation': {'x': x, 'y': y, 'z': z, 'w': w}}

        # Ring finger
        if 'RightDigit4Metacarpophalangeal_flexion' in row:
            roll = -row.get('RightDigit4Metacarpophalangeal_flexion', 0)
            pitch = -row.get('RightDigit4Metacarpophalangeal_ulnarDeviation', 0)
            yaw = row.get('RightDigit4Metacarpophalangeal_pronation', 0)
            x, y, z, w = self.euler_to_quaternion(roll, pitch, yaw)
            body['rightRingProximal'] = {'rotation': {'x': x, 'y': y, 'z': z, 'w': w}}

        if 'RightDigit4ProximalInterphalangeal_flexion' in row:
            roll = -row.get('RightDigit4ProximalInterphalangeal_flexion', 0)
            pitch = -row.get('RightDigit4Metacarpophalangeal_ulnarDeviation', 0)
            yaw = row.get('RightDigit4Metacarpophalangeal_pronation', 0)
            x, y, z, w = self.euler_to_quaternion(roll, pitch, yaw)
            body['rightRingMedial'] = {'rotation': {'x': x, 'y': y, 'z': z, 'w': w}}

        if 'RightDigit4DistalInterphalangeal_flexion' in row:
            roll = -row.get('RightDigit4DistalInterphalangeal_flexion', 0)
            pitch = -row.get('RightDigit4Metacarpophalangeal_ulnarDeviation', 0)
            yaw = row.get('RightDigit4Metacarpophalangeal_pronation', 0)
            x, y, z, w = self.euler_to_quaternion(roll, pitch, yaw)
            body['rightRingDistal'] = {'rotation': {'x': x, 'y': y, 'z': z, 'w': w}}

        # Little/Pinky finger
        if 'RightDigit5Metacarpophalangeal_flexion' in row:
            roll = -row.get('RightDigit5Metacarpophalangeal_flexion', 0)
            pitch = -row.get('RightDigit5Metacarpophalangeal_ulnarDeviation', 0)
            yaw = row.get('RightDigit5Metacarpophalangeal_pronation', 0)
            x, y, z, w = self.euler_to_quaternion(roll, pitch, yaw)
            body['rightLittleProximal'] = {'rotation': {'x': x, 'y': y, 'z': z, 'w': w}}

        if 'RightDigit5ProximalInterphalangeal_flexion' in row:
            roll = -row.get('RightDigit5ProximalInterphalangeal_flexion', 0)
            pitch = -row.get('RightDigit5Metacarpophalangeal_ulnarDeviation', 0)
            yaw = row.get('RightDigit5Metacarpophalangeal_pronation', 0)
            x, y, z, w = self.euler_to_quaternion(roll, pitch, yaw)
            body['rightLittleMedial'] = {'rotation': {'x': x, 'y': y, 'z': z, 'w': w}}

        if 'RightDigit5DistalInterphalangeal_flexion' in row:
            roll = -row.get('RightDigit5DistalInterphalangeal_flexion', 0)
            pitch = -row.get('RightDigit5Metacarpophalangeal_ulnarDeviation', 0)
            yaw = row.get('RightDigit5Metacarpophalangeal_pronation', 0)
            x, y, z, w = self.euler_to_quaternion(roll, pitch, yaw)
            body['rightLittleDistal'] = {'rotation': {'x': x, 'y': y, 'z': z, 'w': w}}

        # Add hand position if available
        if 'RightHand_position_x' in row:
            body['rightHand'] = {
                'position': {
                    'x': row.get('RightHand_position_x', 0),
                    'y': row.get('RightHand_position_y', 0),
                    'z': row.get('RightHand_position_z', 0)
                }
            }

        return json_data

    # ── Quaternion helpers ────────────────────────────────────────────────────

    def _quat_inv(self, q):
        """Inverse of a unit quaternion (w, x, y, z)"""
        w, x, y, z = q
        return (w, -x, -y, -z)

    def _quat_mul(self, q1, q2):
        """Hamilton product of two quaternions (w, x, y, z)"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return (
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        )

    def _rotate_vec_by_quat(self, q, v):
        """Rotate vector v=(vx,vy,vz) by unit quaternion q=(w,x,y,z).
        Uses sandwich product: q * [0,v] * q^-1, returns the vector part.
        To rotate world→local use q = palm_q_inv.
        """
        qv = (0.0, v[0], v[1], v[2])
        q_conj = (q[0], -q[1], -q[2], -q[3])
        result = self._quat_mul(self._quat_mul(q, qv), q_conj)
        return (result[1], result[2], result[3])

    def _relative_euler(self, body, parent_key, child_key):
        """
        Compute Euler angles of child relative to parent.
        Returns (roll, pitch, yaw) in radians, or None if keys missing.
        This gives the actual joint angle rather than the global bone orientation.
        """
        if parent_key not in body or child_key not in body:
            return None
        rp = body[parent_key]['rotation']
        rc = body[child_key]['rotation']
        qp = (rp['w'], rp['x'], rp['y'], rp['z'])
        qc = (rc['w'], rc['x'], rc['y'], rc['z'])
        q_rel = self._quat_mul(self._quat_inv(qp), qc)
        return self.quaternion_to_euler(q_rel[1], q_rel[2], q_rel[3], q_rel[0])

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles (roll, pitch, yaw) in radians"""
        import math

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def parse_rokoko_to_csv(self, json_data):
        """
        Parse Rokoko JSON v3 data and convert to structured joint format.
        Returns CSV-style string with anatomical joint names.

        Joint angles: computed as relative rotations (parent^-1 * child) so they
        are independent of the global arm/hand orientation in world space.

        Fingertip positions: published as both world-space (_pos_) and palm-local
        (_local_).  The palm-local columns rotate each tip vector by the inverse of
        the palm quaternion, making them independent of where the hand is pointing.
        nano_retargeting uses _local_ columns for orientation-independent IK.
        """

        try:
            # Extract timestamp
            timestamp = json_data.get('scene', {}).get('timestamp', 0)

            # Initialize joint dictionary
            joints = {}

            # Check if we have actors with body data
            if 'scene' in json_data and 'actors' in json_data['scene']:
                actors = json_data['scene']['actors']

                if len(actors) > 0:
                    actor = actors[0]  # Use first actor
                    body = actor.get('body', {})

                    # Parse right hand fingers using RELATIVE quaternions (parent^-1 * child).
                    # Global bone orientations include arm/wrist pose (~120-165 deg) and immediately
                    # clamp to joint limits. Relative rotations give the actual joint angle (~15-30 deg).

                    # Thumb (Digit 1)
                    # we only care for roll and yaw
                    # roll is rotation about x(flex, ext), yaw rotation about z (add,abd)
                    rel = self._relative_euler(body, 'rightHand', 'rightThumbProximal')
                    if rel:
                        roll, pitch, yaw = rel
                        joints['RightDigit1Carpometacarpal_flexion'] = -roll
                        joints['RightDigit1Carpometacarpal_ulnarDeviation'] = -yaw

                    rel = self._relative_euler(body, 'rightThumbProximal', 'rightThumbMedial')
                    if rel:
                        roll, pitch, yaw = rel
                        joints['RightDigit1Metacarpophalangeal_flexion'] = -roll
                        joints['RightDigit1Metacarpophalangeal_ulnarDeviation'] = -yaw

                    rel = self._relative_euler(body, 'rightThumbMedial', 'rightThumbDistal')
                    if rel:
                        roll, pitch, yaw = rel
                        joints['RightDigit1Interphalangeal_flexion'] = -roll

                    # Index (Digit 2)
                    rel = self._relative_euler(body, 'rightHand', 'rightIndexProximal')
                    if rel:
                        roll, pitch, yaw = rel
                        joints['RightDigit2Metacarpophalangeal_flexion'] = -roll
                        joints['RightDigit2Metacarpophalangeal_ulnarDeviation'] = -yaw

                    rel = self._relative_euler(body, 'rightIndexProximal', 'rightIndexMedial')
                    if rel:
                        roll, pitch, yaw = rel
                        joints['RightDigit2ProximalInterphalangeal_flexion'] = -roll

                    rel = self._relative_euler(body, 'rightIndexMedial', 'rightIndexDistal')
                    if rel:
                        roll, pitch, yaw = rel
                        joints['RightDigit2DistalInterphalangeal_flexion'] = -roll

                    # Middle (Digit 3)
                    rel = self._relative_euler(body, 'rightHand', 'rightMiddleProximal')
                    if rel:
                        roll, pitch, yaw = rel
                        joints['RightDigit3Metacarpophalangeal_flexion'] = -roll
                        joints['RightDigit3Metacarpophalangeal_ulnarDeviation'] = -yaw

                    rel = self._relative_euler(body, 'rightMiddleProximal', 'rightMiddleMedial')
                    if rel:
                        roll, pitch, yaw = rel
                        joints['RightDigit3ProximalInterphalangeal_flexion'] = -roll

                    rel = self._relative_euler(body, 'rightMiddleMedial', 'rightMiddleDistal')
                    if rel:
                        roll, pitch, yaw = rel
                        joints['RightDigit3DistalInterphalangeal_flexion'] = -roll

                    # Ring (Digit 4)
                    rel = self._relative_euler(body, 'rightHand', 'rightRingProximal')
                    if rel:
                        roll, pitch, yaw = rel
                        joints['RightDigit4Metacarpophalangeal_flexion'] = -roll
                        joints['RightDigit4Metacarpophalangeal_ulnarDeviation'] = -yaw

                    rel = self._relative_euler(body, 'rightRingProximal', 'rightRingMedial')
                    if rel:
                        roll, pitch, yaw = rel
                        joints['RightDigit4ProximalInterphalangeal_flexion'] = -roll

                    rel = self._relative_euler(body, 'rightRingMedial', 'rightRingDistal')
                    if rel:
                        roll, pitch, yaw = rel
                        joints['RightDigit4DistalInterphalangeal_flexion'] = -roll

                    # Little/Pinky (Digit 5)
                    rel = self._relative_euler(body, 'rightHand', 'rightLittleProximal')
                    if rel:
                        roll, pitch, yaw = rel
                        joints['RightDigit5Metacarpophalangeal_flexion'] = -roll
                        joints['RightDigit5Metacarpophalangeal_ulnarDeviation'] = -yaw

                    rel = self._relative_euler(body, 'rightLittleProximal', 'rightLittleMedial')
                    if rel:
                        roll, pitch, yaw = rel
                        joints['RightDigit5ProximalInterphalangeal_flexion'] = -roll

                    rel = self._relative_euler(body, 'rightLittleMedial', 'rightLittleDistal')
                    if rel:
                        roll, pitch, yaw = rel
                        joints['RightDigit5DistalInterphalangeal_flexion'] = -roll

                    # Fingertip positions — published in two forms:
                    #   _pos_   : world-space (kept for backward compat / CSV fallback)
                    #   _local_ : palm-local frame (palm_rot^-1 * (tip - palm_pos))
                    #             orientation-independent → used by fingertip_ik_control
                    digit_map = {'Thumb': 1, 'Index': 2, 'Middle': 3, 'Ring': 4, 'Little': 5}

                    palm_local_available = (
                        'rightHand' in body
                        and 'position' in body['rightHand']
                        and 'rotation' in body['rightHand']
                    )

                    if palm_local_available:
                        palm_p = body['rightHand']['position']
                        palm_r = body['rightHand']['rotation']
                        palm_q = (palm_r['w'], palm_r['x'], palm_r['y'], palm_r['z'])
                        palm_q_inv = self._quat_inv(palm_q)

                    for finger in ['Thumb', 'Index', 'Middle', 'Ring', 'Little']:
                        tip_key = f'right{finger}Tip'
                        if tip_key not in body:
                            continue
                        pos = body[tip_key]['position']
                        d = digit_map[finger]
                        # World-space positions (orientation-dependent)
                        joints[f'RightDigit{d}Tip_pos_x'] = pos['x']
                        joints[f'RightDigit{d}Tip_pos_y'] = pos['y']
                        joints[f'RightDigit{d}Tip_pos_z'] = pos['z']
                        # Palm-local positions (orientation-independent)
                        if palm_local_available:
                            rel_vec = (
                                pos['x'] - palm_p['x'],
                                pos['y'] - palm_p['y'],
                                pos['z'] - palm_p['z'],
                            )
                            local = self._rotate_vec_by_quat(palm_q_inv, rel_vec)
                            joints[f'RightDigit{d}DistalPhalanx_local_x'] = local[0]
                            joints[f'RightDigit{d}DistalPhalanx_local_y'] = local[1]
                            joints[f'RightDigit{d}DistalPhalanx_local_z'] = local[2]

                    # Include hand/palm position for IK calculations
                    if 'rightHand' in body and 'position' in body['rightHand']:
                        hand_pos = body['rightHand']['position']
                        joints['RightHand_position_x'] = hand_pos['x']
                        joints['RightHand_position_y'] = hand_pos['y']
                        joints['RightHand_position_z'] = hand_pos['z']

                    # Parse left hand fingers (for Inspire hand)
                    # Thumb (Digit 1)
                    if 'leftThumbProximal' in body:
                        rot = body['leftThumbProximal']['rotation']
                        roll, pitch, yaw = self.quaternion_to_euler(rot['x'], rot['y'], rot['z'], rot['w'])
                        joints['LeftDigit1Carpometacarpal_flexion'] = -roll
                        joints['LeftDigit1Carpometacarpal_ulnarDeviation'] = pitch

                    if 'leftThumbMedial' in body:
                        rot = body['leftThumbMedial']['rotation']
                        roll, pitch, yaw = self.quaternion_to_euler(rot['x'], rot['y'], rot['z'], rot['w'])
                        joints['LeftDigit1Metacarpophalangeal_flexion'] = -roll
                        joints['LeftDigit1Metacarpophalangeal_ulnarDeviation'] = pitch

                    if 'leftThumbDistal' in body:
                        rot = body['leftThumbDistal']['rotation']
                        roll, pitch, yaw = self.quaternion_to_euler(rot['x'], rot['y'], rot['z'], rot['w'])
                        joints['LeftDigit1Interphalangeal_flexion'] = -roll

                    # Index (Digit 2)
                    if 'leftIndexProximal' in body:
                        rot = body['leftIndexProximal']['rotation']
                        roll, pitch, yaw = self.quaternion_to_euler(rot['x'], rot['y'], rot['z'], rot['w'])
                        joints['LeftDigit2Metacarpophalangeal_flexion'] = -roll
                        joints['LeftDigit2Metacarpophalangeal_ulnarDeviation'] = pitch

                    if 'leftIndexMedial' in body:
                        rot = body['leftIndexMedial']['rotation']
                        roll, pitch, yaw = self.quaternion_to_euler(rot['x'], rot['y'], rot['z'], rot['w'])
                        joints['LeftDigit2ProximalInterphalangeal_flexion'] = -roll

                    if 'leftIndexDistal' in body:
                        rot = body['leftIndexDistal']['rotation']
                        roll, pitch, yaw = self.quaternion_to_euler(rot['x'], rot['y'], rot['z'], rot['w'])
                        joints['LeftDigit2DistalInterphalangeal_flexion'] = -roll

                    # Middle (Digit 3)
                    if 'leftMiddleProximal' in body:
                        rot = body['leftMiddleProximal']['rotation']
                        roll, pitch, yaw = self.quaternion_to_euler(rot['x'], rot['y'], rot['z'], rot['w'])
                        joints['LeftDigit3Metacarpophalangeal_flexion'] = -roll
                        joints['LeftDigit3Metacarpophalangeal_ulnarDeviation'] = -pitch

                    if 'leftMiddleMedial' in body:
                        rot = body['leftMiddleMedial']['rotation']
                        roll, pitch, yaw = self.quaternion_to_euler(rot['x'], rot['y'], rot['z'], rot['w'])
                        joints['LeftDigit3ProximalInterphalangeal_flexion'] = -roll

                    if 'leftMiddleDistal' in body:
                        rot = body['leftMiddleDistal']['rotation']
                        roll, pitch, yaw = self.quaternion_to_euler(rot['x'], rot['y'], rot['z'], rot['w'])
                        joints['LeftDigit3DistalInterphalangeal_flexion'] = -roll

                    # Ring (Digit 4)
                    if 'leftRingProximal' in body:
                        rot = body['leftRingProximal']['rotation']
                        roll, pitch, yaw = self.quaternion_to_euler(rot['x'], rot['y'], rot['z'], rot['w'])
                        joints['LeftDigit4Metacarpophalangeal_flexion'] = -roll
                        joints['LeftDigit4Metacarpophalangeal_ulnarDeviation'] = -pitch

                    if 'leftRingMedial' in body:
                        rot = body['leftRingMedial']['rotation']
                        roll, pitch, yaw = self.quaternion_to_euler(rot['x'], rot['y'], rot['z'], rot['w'])
                        joints['LeftDigit4ProximalInterphalangeal_flexion'] = -roll

                    if 'leftRingDistal' in body:
                        rot = body['leftRingDistal']['rotation']
                        roll, pitch, yaw = self.quaternion_to_euler(rot['x'], rot['y'], rot['z'], rot['w'])
                        joints['LeftDigit4DistalInterphalangeal_flexion'] = -roll

                    # Little/Pinky (Digit 5)
                    if 'leftLittleProximal' in body:
                        rot = body['leftLittleProximal']['rotation']
                        roll, pitch, yaw = self.quaternion_to_euler(rot['x'], rot['y'], rot['z'], rot['w'])
                        joints['LeftDigit5Metacarpophalangeal_flexion'] = -roll
                        joints['LeftDigit5Metacarpophalangeal_ulnarDeviation'] = -pitch

                    if 'leftLittleMedial' in body:
                        rot = body['leftLittleMedial']['rotation']
                        roll, pitch, yaw = self.quaternion_to_euler(rot['x'], rot['y'], rot['z'], rot['w'])
                        joints['LeftDigit5ProximalInterphalangeal_flexion'] = -roll

                    if 'leftLittleDistal' in body:
                        rot = body['leftLittleDistal']['rotation']
                        roll, pitch, yaw = self.quaternion_to_euler(rot['x'], rot['y'], rot['z'], rot['w'])
                        joints['LeftDigit5DistalInterphalangeal_flexion'] = -roll

                    # Left hand fingertip positions for IK control
                    for finger in ['Thumb', 'Index', 'Middle', 'Ring', 'Little']:
                        tip_key = f'left{finger}Tip'
                        if tip_key in body:
                            pos = body[tip_key]['position']
                            digit_num = {'Thumb': 1, 'Index': 2, 'Middle': 3, 'Ring': 4, 'Little': 5}[finger]
                            joints[f'LeftDigit{digit_num}Tip_pos_x'] = pos['x']
                            joints[f'LeftDigit{digit_num}Tip_pos_y'] = pos['y']
                            joints[f'LeftDigit{digit_num}Tip_pos_z'] = pos['z']

                    # Left hand/palm position for IK calculations
                    if 'leftHand' in body and 'position' in body['leftHand']:
                        hand_pos = body['leftHand']['position']
                        joints['LeftHand_position_x'] = hand_pos['x']
                        joints['LeftHand_position_y'] = hand_pos['y']
                        joints['LeftHand_position_z'] = hand_pos['z']

            # Convert to CSV format: timestamp,joint_name,value
            if joints:
                csv_lines = [f"{timestamp},{joint_name},{value}" for joint_name, value in joints.items()]
                return '\n'.join(csv_lines)

            return None

        except Exception as e:
            self.get_logger().warn(f'Error parsing Rokoko JSON: {e}')
            import traceback
            self.get_logger().warn(f'Traceback: {traceback.format_exc()}')
            return None

    def timer_callback(self):
        if self.playback_mode:
            # CSV playback mode
            self.playback_csv_data()
        else:
            # Live UDP mode
            self.receive_udp_data()

    def playback_csv_data(self):
        """Playback one frame from CSV data.

        CSV files store world-space positions and do not include the palm rotation
        quaternion, so true palm-local positions cannot be computed here.  Instead
        we publish palm-relative world-space offsets under the _local_ column names
        so that nano_retargeting can still use them (orientation-dependent, but at
        least position-independent).  For fully orientation-independent IK use live
        Rokoko data, which supplies the palm rotation quaternion.
        """
        if not self.csv_data or len(self.csv_data) == 0:
            return

        # Get current row
        if self.csv_index >= len(self.csv_data):
            if self.loop_playback:
                self.csv_index = 0
                self.get_logger().info('Looping playback...')
            else:
                self.get_logger().info('Playback finished')
                return

        row = self.csv_data[self.csv_index]
        self.csv_index += 1

        csv_lines = []
        timestamp = row.get('Timestamp', 0)

        # Publish all joint angles from the CSV row
        for joint_name, value in row.items():
            if joint_name != 'Timestamp':
                csv_lines.append(f'{timestamp},{joint_name},{value}')

        # Compute palm-relative (world-frame) tip offsets as _local_ columns.
        # These are NOT rotated into palm frame (CSV lacks palm quaternion) but
        # remove absolute-position dependence, which is a partial improvement.
        palm_x = row.get('RightHand_position_x')
        palm_y = row.get('RightHand_position_y')
        palm_z = row.get('RightHand_position_z')
        if palm_x is not None:
            for digit in range(1, 6):
                tx = row.get(f'RightDigit{digit}DistalPhalanx_position_x')
                ty = row.get(f'RightDigit{digit}DistalPhalanx_position_y')
                tz = row.get(f'RightDigit{digit}DistalPhalanx_position_z')
                if tx is not None:
                    csv_lines.append(f'{timestamp},RightDigit{digit}DistalPhalanx_local_x,{tx - palm_x}')
                    csv_lines.append(f'{timestamp},RightDigit{digit}DistalPhalanx_local_y,{ty - palm_y}')
                    csv_lines.append(f'{timestamp},RightDigit{digit}DistalPhalanx_local_z,{tz - palm_z}')

        ref_msg = String()
        ref_msg.data = '\n'.join(csv_lines)
        self.ref_publisher_.publish(ref_msg)

        # Log progress every 100 frames
        if self.csv_index % 100 == 0:
            self.get_logger().info(f'Playback progress: {self.csv_index}/{len(self.csv_data)} frames')

    def receive_udp_data(self):
        """Receive and process UDP data from Rokoko Studio"""
        try:
            # Try to receive data (non-blocking)
            data, addr = self.sock.recvfrom(65535)  # Max UDP packet size

            if data:
                self.packet_count += 1

                # Try to decode the data
                decoded_data, json_data, format_type = self.try_decode_data(data)

                if decoded_data and json_data:
                    if not self.format_detected:
                        self.get_logger().info(f'Data format detected: {format_type}')
                        self.format_detected = True

                    # Save complete JSON structure for the first packet to a file
                    if self.sample_data_count == 0:
                        self.sample_data_count += 1

                        import json as json_module
                        import os

                        # Save to home directory
                        output_file = os.path.expanduser('~/rokoko_packet_sample.json')

                        try:
                            with open(output_file, 'w') as f:
                                json_module.dump(json_data, f, indent=2)

                            self.get_logger().info('='*80)
                            self.get_logger().info(f'✓ First packet JSON saved to: {output_file}')
                            self.get_logger().info('='*80)
                            self.json_structure_logged = True
                        except Exception as e:
                            self.get_logger().error(f'Failed to save JSON to file: {e}')

                    # Publish raw JSON data
                    raw_msg = String()
                    raw_msg.data = decoded_data
                    self.raw_publisher_.publish(raw_msg)

                    # Parse and publish CSV-format reference data
                    csv_data = self.parse_rokoko_to_csv(json_data)
                    if csv_data:
                        ref_msg = String()
                        ref_msg.data = csv_data
                        self.ref_publisher_.publish(ref_msg)

                        # Save parsed data for first packet
                        if not self.parsed_data_saved:
                            self.parsed_data_saved = True
                            import os
                            output_file = os.path.expanduser('~/rokoko_parsed_data_sample.txt')

                            try:
                                with open(output_file, 'w') as f:
                                    f.write("# Parsed Rokoko Data - First Packet\n")
                                    f.write("# Format: timestamp,joint_name,value\n")
                                    f.write("#\n")
                                    f.write(csv_data)

                                self.get_logger().info('='*80)
                                self.get_logger().info(f'✓ Parsed data saved to: {output_file}')
                                self.get_logger().info('='*80)
                            except Exception as e:
                                self.get_logger().error(f'Failed to save parsed data: {e}')

                    if self.packet_count % 100 == 0:
                        self.get_logger().info(f'Received {self.packet_count} packets from {addr} (format: {format_type})')
                else:
                    self.decode_errors += 1
                    if self.decode_errors <= 5:
                        # Log first few bytes to help debug
                        self.get_logger().warn(f'Could not decode packet. First 20 bytes (hex): {data[:20].hex()}')
                        self.get_logger().warn(f'Packet size: {len(data)} bytes')

        except BlockingIOError:
            # No data available, this is normal in non-blocking mode
            pass
        except Exception as e:
            self.get_logger().error(f'Unexpected error: {e}')

    def destroy_node(self):
        if self.sock is not None:
            self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    rokoko_listener = RokokoListener()

    try:
        rclpy.spin(rokoko_listener)
    except KeyboardInterrupt:
        pass
    finally:
        rokoko_listener.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
