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


class EMGToNanoMultiCSV(Node):
    def __init__(self):
        super().__init__('emg_to_nano')
        self.declare_parameter('use_fsr', False)
        self.use_fsr = self.get_parameter('use_fsr').value

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

        self.get_logger().info(f"EMG to Nano Hand started (use_fsr={self.use_fsr}).")

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
    # Angle → position helpers
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
    # Streaming
    # ------------------------------------------------------------------

    def stream_csv(self):
        df = self.current_df
        if df is None:
            self.get_logger().info("Current dataframe is None!")
            return

        thresholds = self.current_thresholds if self.use_fsr else {}
        self.get_logger().info(f"Streaming {len(df)} frames (FSR gating: {'on' if thresholds else 'off'}).")
        self.is_open_pose = False

        # Per-finger hold state
        finger_held = [False] * 5
        last_positions = list(self.extract_positions(df.iloc[0]))

        # Fingers that have a force target — we stop the recording once all of these are held
        force_targeted = [i for i in range(5) if thresholds.get(f'fsr{i}', {}).get('grasp_force') is not None]

        for _, row in df.iloc[1:].iterrows():
            if self.stop_flag:
                return

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
                        last_positions[flex_di] = max(0.0, last_positions[flex_di] - EXTRAP_STEP)
                        still_needs.append(fsr_idx)
                needs_extrap = still_needs

                self._publish_positions(last_positions)
                time.sleep(self.frame_delay)

        self.current_df = None

    def start_stream(self):
        self.stop_flag = False
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
            self._publish_positions(self.extract_positions(self.current_df.iloc[0]))
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
