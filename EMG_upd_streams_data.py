# sample_udp.py
# Streams EMG data over UDP in a format compatible with the ML-based classify.py, with smoothing filter

import struct
import socket
import time
import asyncio
import numpy as np
from gforce import DataNotifFlags, GForceProfile, NotifDataType
from collections import deque

UDP_IP = '127.0.0.1'
UDP_PORT = 12346
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

packet_cnt = 0
start_time = 0

# Smoothing filter: simple moving average
SMA_WINDOW = 5
emg_history = [deque(maxlen=SMA_WINDOW) for _ in range(8)]

def set_cmd_cb(resp):
    print("Command result: {}".format(resp))

def ondata(data):
    global packet_cnt, start_time
    if len(data) > 0:
        if data[0] == NotifDataType.NTF_EMG_ADC_DATA and len(data) == 129:
            for i in range(16):
                offset = 1 + i * 8
                emg = data[offset:offset+8]
                if len(emg) == 8:
                    # Update history for smoothing
                    for ch in range(8):
                        emg_history[ch].append(emg[ch])
                    # Compute smoothed value
                    smoothed = [int(np.mean(emg_history[ch])) if len(emg_history[ch]) == SMA_WINDOW else int(emg[ch]) for ch in range(8)]
                    emg_str = ' '.join(str(x) for x in smoothed)
                    sock.sendto(emg_str.encode('utf-8'), (UDP_IP, UDP_PORT))
                    print(f"Sent {emg_str}")
            packet_cnt += 1
            if packet_cnt % 100 == 0:
                if start_time == 0:
                    start_time = time.time()
                period = time.time() - start_time
                sample_rate = 100 * 16 / period
                print(f"UDP sample_rate: {sample_rate:.1f} Hz")
                start_time = time.time()

async def main():
    sampRate = 500
    channelMask = 0xFF
    dataLen = 128
    resolution = 8

    event = asyncio.Event()
    gForce = GForceProfile()
    device_address = "90:7B:C6:63:4B:D4"
    await gForce.connect(device_address)
    # print(f"Connected directly to {device_address}")
    await gForce.setEmgRawDataConfig(
        sampRate,
        channelMask,
        dataLen,
        resolution,
        cb=set_cmd_cb,
        timeout=1000,
    )
    await gForce.setDataNotifSwitch(DataNotifFlags.DNF_EMG_RAW, set_cmd_cb, 1000)
    await asyncio.sleep(1)
    await gForce.startDataNotification(ondata)
    
    print("Press enter to stop...")
    await asyncio.to_thread(lambda: input())

    print("Stopping...")
    await gForce.stopDataNotification()
    await asyncio.sleep(1)
    await gForce.setDataNotifSwitch(DataNotifFlags.DNF_OFF, set_cmd_cb, 1000)
    await gForce.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
