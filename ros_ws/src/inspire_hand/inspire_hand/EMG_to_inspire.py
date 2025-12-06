#!/usr/bin/env python3
#reads directly from the csv files


import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Int32, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import pandas as pd
import json
import os
import time
import threading
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

class EMGToInspireMultiCSV(Node):
    def __init__(self):
        super().__init__('emg_to_inspire')
        self.create_subscription(Int32, '/emg_gesture', self.emg_callback, 10)
        self.create_subscription(String, "/gemini/detected_object", self.gemini_callback, 10)
        self.pub = self.create_publisher(JointTrajectory, '/inspire_hand/joint_trajectory', 10)

        # Map object to recording CSV and last row index of the grasp
        self.csv_map = {
            "Bottle body": 'bottle_body.csv',
            "Bottle cap": 'bottle_lid.csv',
            "Mug body": 'mug_body.csv',
            "Mug handle": 'mug_handle.csv'
        }

        self.current_df = None
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

        self.max_open = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.min_closure = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # test 1: grasping cube
        # self.min_closure = [0.1, 0.4, 0.5, 0.4, 0.4, 0.0]
        # self.max_open = [0.8, 0.7, 0.9, 1.0, 1.0, 0.0]
        #test 2: pinching coin
        #self.min_closure = [0.1, 0.4, 0.2, 0.15, 0.4, 0.0]  # minimum closed position per finger
        #self.max_open = [0.8, 0.7, 0.9, 1.0, 0.5, 0.0]
        #test 3: sliding card
        # self.min_closure = [0.3, 0.3, 0.0, 0.0, 0.1, 0.0]  # minimum closed position per finger
        # self.max_open = [0.8, 0.7, 0.9, 0.80, 0.5, 0.0]     # maximum open position per finger
        self.get_logger().info("EMG to Inspire multi-CSV node started.")

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
            min_val, max_val = self.min_closure[dof], self.max_open[dof]
            scaled = min_val + (max_val - min_val) * norm
            dof_positions.append(scaled)
        return dof_positions

    def stream_csv(self):
        df = self.current_df
        if df is not None:
            self.get_logger().info(f"Streaming file with {len(df)} frames.")
            self.is_open_pose = False
            for i, row in df.iloc[1:].iterrows():
                if self.stop_flag:
                    break
                msg = JointTrajectory()
                msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
                point = JointTrajectoryPoint()
                point.positions = self.extract_positions(row)
                point.time_from_start.sec = 0
                point.time_from_start.nanosec = int(self.frame_delay * 1e9)
                msg.points.append(point)
                self.pub.publish(msg)
                time.sleep(self.frame_delay)
            self.current_df = None
        else:
            self.get_logger().info(f"Current dataframe is None!")

    def start_stream(self):
        self.stop_flag = False
        self.active_thread = threading.Thread(target=self.stream_csv)
        self.active_thread.start()

    def stop_stream(self):
        self.stop_flag = True
        if self.active_thread and self.active_thread.is_alive():
            self.active_thread.join()

    def open_hand(self):
        if not self.is_open_pose:
            msg = JointTrajectory()
            msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            point = JointTrajectoryPoint()
            point.positions = self.max_open
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = int(self.frame_delay * 1e9)
            msg.points.append(point)
            self.pub.publish(msg)
            self.is_open_pose = True

    def emg_callback(self, msg):
        gesture = msg.data

        # If already playing a gesture, ignore further input until finished
        if self.active_thread and self.active_thread.is_alive():
            return

        # Ignore zero unless no motion is active
        if gesture == 0:
            self.get_logger().info("Zero detected → idle")
            self.last_gesture = 0
            self.candidate_gesture = None
            self.candidate_count = 0
            return

        # Candidate stability filter
        if gesture != self.candidate_gesture:
            self.candidate_gesture = gesture
            self.candidate_count = 1
        else:
            self.candidate_count += 1

        # If stable enough and different from last executed
        if self.candidate_count >= self.required_stability and gesture != self.last_gesture:
            if gesture == 1:        # grab
                self.start_stream()
            elif gesture == 2:      # release
                self.open_hand()
            self.last_gesture = gesture
            self.candidate_count = 0
                

    def start_pregrasp(self, detected_object):
        csv_file = self.csv_map[detected_object]
        file_path = os.path.join(get_package_share_directory("inspire_hand"), "rokoko_csv", csv_file)
        self.current_df = pd.read_csv(file_path)
        self.get_logger().info(f"Streaming 1st row of {csv_file}.")
        if not self.current_df.empty and not self.stop_flag:
            row = self.current_df.iloc[0]
            msg = JointTrajectory()
            msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            point = JointTrajectoryPoint()
            point.positions = self.extract_positions(row)
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = int(self.frame_delay * 1e9)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
            msg.points.append(point)
            self.pub.publish(msg)
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
            self.get_logger().warn(f"No CSV mapped for unknown object")

        


def main(args=None):
    rclpy.init(args=args)
    node = EMGToInspireMultiCSV()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
