#!/usr/bin/env python3
"""
Connects to a XIAO-FSR BLE device, subscribes to force notifications,
logs them to a CSV file, and prints human-readable status to the terminal.

Usage:
    python3 fsr_logger.py                  # writes to fsr_log_<timestamp>.csv
    python3 fsr_logger.py myrun.csv        # writes to myrun.csv
"""

import asyncio
import csv
import struct
import sys
import time
from datetime import datetime

from bleak import BleakClient, BleakScanner

DEVICE_NAME_PREFIX = "XIAO"
FSR_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

# Packet: uint32 seq, uint32 millis, uint16 fsr0..fsr4 = 18 bytes, little-endian
PACKET_FORMAT = "<II5H"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

# ---- ADC / divider constants (must match firmware wiring) ----
VREF = 3.3
ADC_MAX = 4095
R_FIXED = 10000.0  # 10k pulldown resistor

# How often to print a status block to the terminal (in samples).
# At ~100 Hz, every 25 samples = ~4 prints/second.
PRINT_EVERY = 25



def force_label(raw: int) -> str:
    """Bucket a raw ADC reading into a human-readable force category."""
    if raw < 50:
        return "none"
    elif raw < 400:
        return "light"
    elif raw < 1500:
        return "medium"
    elif raw < 3000:
        return "hard"
    else:
        return "very hard"



def compute_voltage(raw: int) -> float:
    return (raw * VREF) / ADC_MAX



def compute_resistance(voltage: float) -> float:
    """Back-calculate the FSR's resistance from the divider voltage. -1 = open."""
    if voltage < 0.01:
        return -1.0
    return R_FIXED * (VREF - voltage) / voltage



async def main():
    if len(sys.argv) > 1:
        out_path = sys.argv[1]
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"fsr_log_{ts}.csv"

    print(f"Scanning for a device whose name starts with '{DEVICE_NAME_PREFIX}'...")
    devices = await BleakScanner.discover(timeout=8.0)
    device = None
    for d in devices:
        if d.name and d.name.startswith(DEVICE_NAME_PREFIX):
            device = d
            break
    if device is None:
        print(f"Could not find any {DEVICE_NAME_PREFIX} device. Is the XIAO powered and advertising?")
        print("Named devices seen:")
        for d in devices:
            if d.name:
                print(f"  {d.address}  {d.name}")
        return

    print(f"Matched '{device.name}' ({device.address}). Connecting...")

    csv_file = open(out_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "host_time_s",
        "seq",
        "device_ms",
        "fsr0", "fsr1", "fsr2", "fsr3", "fsr4",
    ])
    csv_file.flush()

    sample_count = 0
    t_start = time.time()

    def handle_notification(_sender, data: bytearray):
        nonlocal sample_count
        if len(data) != PACKET_SIZE:
            print(f"  ! unexpected packet size: {len(data)} bytes")
            return
        seq, dev_ms, f0, f1, f2, f3, f4 = struct.unpack(PACKET_FORMAT, data)
        host_t = time.time()

        # Always write the raw values to CSV
        writer.writerow([f"{host_t:.6f}", seq, dev_ms, f0, f1, f2, f3, f4])
        sample_count += 1

        # Print a human-readable block every PRINT_EVERY samples
        if sample_count % PRINT_EVERY == 0:
            elapsed = host_t - t_start
            rate = sample_count / elapsed if elapsed > 0 else 0
            raws = [f0, f1, f2, f3, f4]

            print(f"\n[#{sample_count}  {rate:.1f} Hz  {dev_ms} ms]")
            print(f"{'FSR':<5}{'Raw':>6}{'Volts':>9}{'R_FSR(ohm)':>14}   Force")
            for i, raw in enumerate(raws):
                v = compute_voltage(raw)
                r = compute_resistance(v)
                r_str = "open" if r < 0 else f"{int(r)}"
                print(f"FSR{i:<2}{raw:>6}{v:>9.3f}{r_str:>14}   {force_label(raw)}")

    try:
        async with BleakClient(device) as client:
            print(f"Connected. Logging to {out_path}")
            print("Press Ctrl+C to stop.")
            await client.start_notify(FSR_CHAR_UUID, handle_notification)
            while True:
                await asyncio.sleep(1.0)
                csv_file.flush()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        csv_file.close()
        print(f"Wrote {sample_count} samples to {out_path}")



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

