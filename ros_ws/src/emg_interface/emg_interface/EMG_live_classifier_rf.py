#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import socket
import threading
import time
import numpy as np
import joblib
from collections import deque
from std_msgs.msg import Int32
from EMG_feature_extractor import extract_features  # step 2
import os

WINDOW_SIZE = 100
WINDOW_INCREMENT = 50
NUM_CHANNELS = 8
UDP_IP = '127.0.0.1'
UDP_PORT = 12346

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.realpath(__file__))
# Construct full path to the model file in the same directory
model_path = os.path.join(script_dir, "emg_model_lda.pkl")

class EMGLiveClassifier(Node):
    def __init__(self):
        super().__init__('emg_live_classifier')

        # UDP setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.setblocking(False)

        # Load model + encoder
        # saved = joblib.load("emg_model_rf.pkl")
        saved = joblib.load(model_path)
        self.model = saved["model"]
        self.encoder = saved["encoder"]

        self.buffer = deque(maxlen=WINDOW_SIZE)

        # ROS Publisher
        self.gesture_pub = self.create_publisher(Int32, '/emg_gesture', 10)

        self.running = True
        self.thread = threading.Thread(target=self.udp_listener)
        self.thread.start()

        self.get_logger().info(f"Classifier started. Listening for EMG...")
        self.get_logger().info(f"Recognizing gestures: {list(self.encoder.classes_)}")

    def udp_listener(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)

                # print(f"Received {len(data)} bytes from {addr}")

                decoded = data.decode("utf-8").strip()
                parts = decoded.split()
                if len(parts) >= NUM_CHANNELS:
                    emg = [int(x) for x in parts[:NUM_CHANNELS]]
                    self.buffer.append(emg)

                    # When enough data is collected, classify
                    if len(self.buffer) == WINDOW_SIZE:
                        window_np = np.array(self.buffer)
                        features = np.array(extract_features(window_np)).reshape(1, -1)

                        pred_id = self.model.predict(features)[0]
                        gesture_name = self.encoder.inverse_transform([pred_id])[0]

                        # Publish integer gesture ID
                        self.gesture_pub.publish(Int32(data=int(pred_id)))
                        print(f"[Prediction] Gesture ID: {pred_id} | Name: {gesture_name}")

                        # Slide window
                        for _ in range(WINDOW_INCREMENT):
                            if self.buffer:
                                self.buffer.popleft()

            except BlockingIOError:
                time.sleep(0.01)

    def destroy_node(self):
        self.running = False
        self.thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = EMGLiveClassifier()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping classifier...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
