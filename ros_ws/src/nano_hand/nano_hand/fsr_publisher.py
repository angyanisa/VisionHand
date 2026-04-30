#!/usr/bin/env python3
"""
BLE FSR publisher node.

Connects to the XIAO-FSR device, subscribes to force notifications,
and publishes raw ADC readings to /fsr/data as Float32MultiArray.

Also publishes /fsr/connected (Bool) so other nodes can check BLE health
and reject stale readings.

Usage:
    ros2 run nano_hand fsr_publisher
"""

import asyncio
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray

from bleak import BleakClient, BleakScanner

DEVICE_NAME_PREFIX = "XIAO"
FSR_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

PACKET_FORMAT = "<II5H"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

NUM_SENSORS = 5
SCAN_TIMEOUT = 8.0
RECONNECT_DELAY = 3.0


class FSRPublisher(Node):
    def __init__(self):
        super().__init__("fsr_publisher")

        self.pub_data = self.create_publisher(Float32MultiArray, "/fsr/data", 10)
        self.pub_connected = self.create_publisher(Bool, "/fsr/connected", 10)

        # Publish connection status at 2 Hz so subscribers notice drops quickly
        self.create_timer(0.5, self._publish_connected)

        self._connected = False
        self._sample_count = 0

        # BLE runs in its own thread with its own event loop
        self._ble_thread = threading.Thread(target=self._ble_thread_main, daemon=True)
        self._ble_thread.start()

        self.get_logger().info("FSR publisher node started.")

    # ------------------------------------------------------------------
    # BLE
    # ------------------------------------------------------------------

    def _ble_thread_main(self):
        asyncio.run(self._ble_loop())

    async def _ble_loop(self):
        """Scan → connect → receive; reconnect on any failure."""
        while rclpy.ok():
            device = await self._scan()
            if device is None:
                await asyncio.sleep(RECONNECT_DELAY)
                continue

            self.get_logger().info(f"Connecting to '{device.name}' ({device.address})...")
            try:
                async with BleakClient(device) as client:
                    self._connected = True
                    self.get_logger().info("BLE connected.")
                    await client.start_notify(FSR_CHAR_UUID, self._handle_notification)
                    # Keep connection alive; BleakClient raises on disconnect
                    while client.is_connected and rclpy.ok():
                        await asyncio.sleep(0.5)
            except Exception as e:
                self.get_logger().warn(f"BLE error: {e}")
            finally:
                self._connected = False
                self.get_logger().info("BLE disconnected. Retrying...")
                await asyncio.sleep(RECONNECT_DELAY)

    async def _scan(self):
        self.get_logger().info(
            f"Scanning for device starting with '{DEVICE_NAME_PREFIX}'..."
        )
        devices = await BleakScanner.discover(timeout=SCAN_TIMEOUT)
        for d in devices:
            if d.name and d.name.startswith(DEVICE_NAME_PREFIX):
                return d
        self.get_logger().warn(
            f"No {DEVICE_NAME_PREFIX} device found. "
            f"Seen: {[d.name for d in devices if d.name]}"
        )
        return None

    def _handle_notification(self, _sender, data: bytearray):
        if len(data) != PACKET_SIZE:
            self.get_logger().warn(f"Unexpected packet size: {len(data)}")
            return

        _seq, _dev_ms, f0, f1, f2, f3, f4 = struct.unpack(PACKET_FORMAT, data)
        self._sample_count += 1

        msg = Float32MultiArray()
        msg.data = [float(f0), float(f1), float(f2), float(f3), float(f4)]
        self.pub_data.publish(msg)

        if self._sample_count % 100 == 0:
            self.get_logger().info(
                f"[#{self._sample_count}] FSR: {[f0, f1, f2, f3, f4]}"
            )

    # ------------------------------------------------------------------
    # Timers
    # ------------------------------------------------------------------

    def _publish_connected(self):
        msg = Bool()
        msg.data = self._connected
        self.pub_connected.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FSRPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
