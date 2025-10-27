#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import socket
import threading
import time
import os
import csv
import termios
import tty
import select
import sys

UDP_PORT = 12346
DATA_DIR = "data"

class EMGUDPCollector(Node):
    def __init__(self):
        super().__init__('emg_udp_collector')

        self.label = 0  # Start with gesture 0
        self.samples = []
        self.running = True

        os.makedirs(DATA_DIR, exist_ok=True)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', UDP_PORT))
        self.sock.setblocking(False)

        print("[INFO] Press keys 0–4 to change gesture label during recording. Ctrl+C to stop.")
        print(f"[INFO] Listening for EMG data on UDP port {UDP_PORT}...")

        # Start listener threads
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener)
        self.keyboard_thread.start()

        self.udp_thread = threading.Thread(target=self.udp_listener)
        self.udp_thread.start()

    def keyboard_listener(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    char = sys.stdin.read(1)
                    if char in ['0', '1', '2', '3', '4']:
                        self.label = int(char)
                        print(f"[LABEL] Set current label to {self.label}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def udp_listener(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024)
                decoded = data.decode("utf-8").strip()
                parts = decoded.split()
                if len(parts) >= 8:
                    emg = [int(x) for x in parts[:8]]
                    timestamp = time.time()
                    self.samples.append(emg + [self.label, timestamp])
            except BlockingIOError:
                time.sleep(0.01)

    def save_data(self):
        filename = os.path.join(DATA_DIR, f"emg_train_{int(time.time())}.csv")
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["emg0","emg1","emg2","emg3","emg4","emg5","emg6","emg7","label","timestamp"])
            writer.writerows(self.samples)
        print(f"[SAVE] Saved {len(self.samples)} samples to {filename}")

    def destroy_node(self):
        self.running = False
        self.udp_thread.join()
        self.keyboard_thread.join()
        self.save_data()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = EMGUDPCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[EXIT] Stopping and saving data...")
        node.destroy_node()  # Manual shutdown
        rclpy.shutdown()     # Only call once


if __name__ == '__main__':
    main()
