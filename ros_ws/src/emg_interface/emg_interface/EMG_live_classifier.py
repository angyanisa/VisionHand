# 4

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import socket
import threading
import time
import numpy as np
import joblib
import os
from collections import deque
from emg_interface.EMG_feature_extractor import extract_features  # from step 2

WINDOW_SIZE = 100
WINDOW_INCREMENT = 50
NUM_CHANNELS = 8
UDP_PORT = 12346

class EMGLiveClassifier(Node):
    def __init__(self):
        super().__init__('emg_live_classifier')

        self.pub = self.create_publisher(Int32, '/emg_gesture', 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', UDP_PORT))
        self.sock.setblocking(False)

        # Load model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'emg_model_lda.pkl')
        self.model = joblib.load(model_path)
        self.get_logger().info(f"Loaded model from: {model_path}")

        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.running = True
        self.thread = threading.Thread(target=self.udp_listener)
        self.thread.start()

        self.get_logger().info("Classifier started. Listening for EMG...")

    def udp_listener(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024)
                decoded = data.decode("utf-8").strip()
                parts = decoded.split()
                if len(parts) >= NUM_CHANNELS:
                    emg = [int(x) for x in parts[:NUM_CHANNELS]]
                    self.buffer.append(emg)

                    if len(self.buffer) == WINDOW_SIZE:
                        window_np = np.array(self.buffer)
                        features = np.array(extract_features(window_np)).reshape(1, -1)
                        pred = self.model.predict(features)[0]
                        print(f"[Prediction] Gesture: {pred}")

                        msg = Int32()
                        msg.data = int(pred)
                        self.pub.publish(msg)
                        
                        # Drop WINDOW_INCREMENT samples to simulate real-time windowing
                        for _ in range(WINDOW_INCREMENT):
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