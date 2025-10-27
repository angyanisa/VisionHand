import asyncio
from _oymotion_windows_streamer import Gforce, DataSubscription

async def main():
    gforce = Gforce(emg=True, imu=False, shared_memory_items=[])
    await gforce.connect()
    print("Connected to OYMotion device!")

    await gforce.set_emg_raw_data_config()
    await gforce.set_subscription(DataSubscription.EMG_RAW)
    print("Subscribed to EMG_RAW data.")

    q = await gforce.start_streaming()
    print("Streaming EMG data...")
    for _ in range(10):
        emg_packets = await q.get()
        print(f"Received EMG packet: {emg_packets}")

    await gforce.stop_streaming()
    await gforce.disconnect()
    print("Disconnected from OYMotion device.")

if __name__ == "__main__":
    asyncio.run(main())
