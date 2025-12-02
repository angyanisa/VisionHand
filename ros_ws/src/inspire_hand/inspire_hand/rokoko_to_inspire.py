#simialr to the EMG_to_inspire but this only reads a single csv file, good for a continuius task coming from a single csv file

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float32
import pandas as pd
import time
import json
import os
import numpy as np

DOF_ANGLE_SOURCES = {
    0: ['LeftDigit5Metacarpophalangeal_flexion', 'LeftDigit5ProximalInterphalangeal_flexion'],
    1: ['LeftDigit4Metacarpophalangeal_flexion', 'LeftDigit4ProximalInterphalangeal_flexion'],
    2: ['LeftDigit3Metacarpophalangeal_flexion', 'LeftDigit3ProximalInterphalangeal_flexion'],
    3: ['LeftDigit2Metacarpophalangeal_flexion', 'LeftDigit2ProximalInterphalangeal_flexion'],
    4: ['LeftDigit1Metacarpophalangeal_flexion'],
    5: ['LeftDigit1Carpometacarpal_flexion'],
}

CALIBRATION_FILE = "calibration.json"

class CSVToInspireHand(Node):
    def __init__(self, csv_path, frame_rate=100, default_force=400.0, default_speed=700.0):
        super().__init__('csv_to_inspire_hand')
        self.publisher = self.create_publisher(JointTrajectory, '/inspire_hand/joint_trajectory', 10)
        self.force_pub = self.create_publisher(Float32, '/inspire_hand/force', 10)
        self.speed_pub = self.create_publisher(Float32, '/inspire_hand/speed', 10)

        self.csv_data = pd.read_csv(csv_path)
        self.frame_delay = 1.0 / frame_rate
        self.default_force = default_force
        self.default_speed = default_speed

        # Per-finger closure limits (0 = fully closed, 1 = fully open)
        # Adjust these values as needed
        # Add two arrays for tuning
        self.min_closure = [0.1, 0.4, 0.5, 0.4, 0.4, 0.0]  # minimum closed position per finger
        self.max_open = [0.8, 0.7, 0.9, 1.0, 1.0, 0.0]     # maximum open position per finger


        if self.csv_data.empty:
            self.get_logger().error("CSV file is empty.")
            rclpy.shutdown()
            return

        if not os.path.exists(CALIBRATION_FILE):
            self.compute_optimized_calibration(csv_path)
        with open(CALIBRATION_FILE) as f:
            self.calibration = json.load(f)

        self.get_logger().info(f"Loaded CSV with {len(self.csv_data)} frames.")
        self.get_logger().info(f"Streaming at {frame_rate} FPS ({self.frame_delay:.3f}s per frame).")
        self.publish_trajectory()

    def compute_optimized_calibration(self, csv_path):
        df = pd.read_csv(csv_path)
        calibration = {}
        num_frames = len(df)

        for dof, joints in DOF_ANGLE_SOURCES.items():
            values = []
            for joint in joints:
                if joint in df.columns:
                    values.extend(df[joint].dropna().tolist())
            if values:
                arr = np.array(values)
                open_val = np.percentile(arr, 5)
                closed_val = np.percentile(arr, 95)

                start_mean = np.mean(arr[:int(0.1 * num_frames)])
                end_mean = np.mean(arr[-int(0.1 * num_frames):])
                invert = bool(end_mean < start_mean)
            else:
                open_val, closed_val, invert = 0.0, 1.0, False

            calibration[str(dof)] = {
                "csv_open": float(open_val),
                "csv_closed": float(closed_val),
                "invert": invert,
                "hand_open": 1000,
                "hand_closed": 0
            }

        temp_file = CALIBRATION_FILE + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(calibration, f, indent=2)
        os.replace(temp_file, CALIBRATION_FILE)
        self.get_logger().info(f"Optimized calibration file created safely: {CALIBRATION_FILE}")

    def normalize_with_calibration(self, angle, dof):
        cal = self.calibration[str(dof)]
        min_v, max_v = cal["csv_open"], cal["csv_closed"]
        if max_v - min_v == 0:
            norm = 0.0
        else:
            norm = (angle - min_v) / (max_v - min_v)
        norm = max(0.0, min(1.0, norm))
        if cal.get("invert", False):
            norm = 1.0 - norm
        return norm

    def extract_dof_positions(self, row):
        dof_positions = []
        for dof, joints in DOF_ANGLE_SOURCES.items():
            values = [row[j] for j in joints if j in row and pd.notnull(row[j])]
            if not values:
                dof_positions.append(0.0)
            else:
                avg = sum(values) / len(values)
                norm = self.normalize_with_calibration(avg, dof)
                dof_positions.append(norm)
        return dof_positions

    def publish_trajectory(self):
        for i, row in self.csv_data.iterrows():
            msg = JointTrajectory()
            msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

            point = JointTrajectoryPoint()
            raw_positions = self.extract_dof_positions(row)

            # Scale down positions to prevent over-closing
            scaled_positions = []
            for dof, raw_pos in enumerate(raw_positions):
                # Map normalized (0-1) → custom min-max range
                min_val = self.min_closure[dof]
                max_val = self.max_open[dof]
                scaled = min_val + (max_val - min_val) * raw_pos
                scaled_positions.append(scaled)


            point.positions = scaled_positions
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = int(self.frame_delay * 1e9)

            msg.points.append(point)
            self.publisher.publish(msg)
            self.force_pub.publish(Float32(data=self.default_force))
            self.speed_pub.publish(Float32(data=self.default_speed))

            self.get_logger().info(f"Published frame {i+1}/{len(self.csv_data)}: {point.positions}")
            time.sleep(self.frame_delay)

def main(args=None):
    rclpy.init(args=args)
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "rokoko_csv/bottle_lid.csv")
    node = CSVToInspireHand(file_path, frame_rate=100)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()