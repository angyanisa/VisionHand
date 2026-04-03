# Hand Retargeting System Documentation

## Overview
This ROS2 system provides a complete pipeline for retargeting Rokoko glove data to various robotic hands (ORCA, Inspire, Leap) with multiple control methods.

## System Architecture

```
Rokoko Studio (UDP)
    ↓
rokoko_listener → /rokoko_raw_data
                → /rokoko_ref_data (CSV format)
    ↓
visualization_node → /hand_type
                   → /control_type
    ↓
retargeting_node (orca/inspire/leap) → /joint_states
```

## Nodes

### 1. visualization_node
- **Purpose**: Publishes system configuration
- **Topics Published**:
  - `/hand_type` (String): Which hand model to use
  - `/control_type` (String): Which control method to use
- **Parameters**:
  - `hand_type`: orca | inspire | leap (default: orca)
  - `control_type`: direct | fingertip_ik | jparse_ik (default: direct)

### 2. rokoko_listener
- **Purpose**: Receives Rokoko Studio UDP data and processes it
- **Topics Published**:
  - `/rokoko_raw_data` (String): Raw JSON from Rokoko
  - `/rokoko_ref_data` (String): Parsed CSV format data
- **Parameters**:
  - `ip_address`: IP to listen on (default: 0.0.0.0)
  - `port`: UDP port (default: 14043)
  - `publish_rate`: Hz (default: 30.0)
- **CSV Format**: `timestamp,hand,finger,joint,angle_x,angle_y,angle_z,pos_x,pos_y,pos_z`

### 3. Retargeting Nodes (orca_retargeting, inspire_retargeting, leap_retargeting)
- **Purpose**: Convert Rokoko data to specific hand joint angles
- **Topics Subscribed**:
  - `/control_type` (String): Control method to use
  - `/rokoko_ref_data` (String): Reference data from Rokoko
- **Topics Published**:
  - `/joint_states` (JointState): Retargeted joint positions
- **Control Methods**:
  - `direct`: Direct joint angle mapping
  - `fingertip_ik`: Fingertip position-based IK
  - `jparse_ik`: Advanced IK with joint constraints

## Usage

### Launch the Complete System

```bash
cd /home/shalika/Desktop/hand_control_ws
source install/setup.bash

# Launch with ORCA hand and direct control (default)
ros2 launch hands hand_retargeting_system.launch.py

# Launch with Inspire hand and fingertip IK
ros2 launch hands hand_retargeting_system.launch.py hand_type:=inspire control_type:=fingertip_ik

# Launch with Leap hand and JPARSE IK
ros2 launch hands hand_retargeting_system.launch.py hand_type:=leap control_type:=jparse_ik

# Custom IP and port for Rokoko
ros2 launch hands hand_retargeting_system.launch.py ip_address:=0.0.0.0 port:=14043
```

### Rokoko Studio Configuration

1. Open Rokoko Studio on Windows laptop
2. Go to Settings → Streaming
3. Configure:
   - **Forward IP**: `10.34.84.40` (your Linux machine IP)
   - **Port**: `14043`
   - **Format**: JSON v3
4. Start streaming

### Monitor Topics

```bash
# View hand configuration
ros2 topic echo /hand_type
ros2 topic echo /control_type

# View raw Rokoko data
ros2 topic echo /rokoko_raw_data

# View parsed CSV data
ros2 topic echo /rokoko_ref_data

# View retargeted joint states
ros2 topic echo /joint_states
```

### Check Node Status

```bash
# List running nodes
ros2 node list

# Get info about a specific node
ros2 node info /visualization_node
ros2 node info /rokoko_listener
ros2 node info /orca_retargeting
```

## Development

### Implementing Control Methods

Each retargeting node (`orca_retargeting.py`, `inspire_retargeting.py`, `leap_retargeting.py`) has three skeleton methods to implement:

1. **direct_joint_angle_control(parsed_data)** (`/home/shalika/Desktop/hand_control_ws/src/hands/hands/{hand}_retargeting.py`)
   - Map Rokoko finger angles directly to hand joints
   - Access data: `parsed_data['finger_name']['angles']` and `parsed_data['finger_name']['position']`

2. **fingertip_ik_control(parsed_data)**
   - Use fingertip positions to solve IK
   - Access positions: `parsed_data['finger_name']['position']`

3. **jparse_ik_control(parsed_data)**
   - Advanced IK with constraints
   - Include joint limits, collision avoidance, optimization

### CSV Data Format

Each line in `/rokoko_ref_data`:
```
timestamp,hand,finger,joint,angle_x,angle_y,angle_z,pos_x,pos_y,pos_z
```

Example:
```
1234567890,left,thumb,tip,0.5,-0.2,0.1,0.05,0.03,0.02
1234567890,left,index,tip,0.3,0.1,-0.1,0.08,0.01,0.04
```

### Modifying Rokoko JSON Parser

If Rokoko JSON v3 format differs, update the `parse_rokoko_to_csv()` method in:
`/home/shalika/Desktop/hand_control_ws/src/hands/hands/rokoko_listener.py:94`

## Troubleshooting

### No data from Rokoko
1. Check Rokoko Studio is streaming
2. Verify IP address: `hostname -I`
3. Test with: `sudo tcpdump -i any port 14043 -n`
4. Check firewall settings

### Data decoding errors
- Check Rokoko Studio format is set to JSON v3
- View logs: `ros2 node info /rokoko_listener`
- The listener tries multiple decompression methods automatically

### Wrong hand launches
- Verify `hand_type` parameter: `ros2 topic echo /hand_type`
- Check launch arguments are correct

## File Locations

- **Nodes**: `/home/shalika/Desktop/hand_control_ws/src/hands/hands/`
  - `visualization_node.py`
  - `rokoko_listener.py`
  - `orca_retargeting.py`
  - `inspire_retargeting.py`
  - `leap_retargeting.py`

- **Launch**: `/home/shalika/Desktop/hand_control_ws/src/hands/launch/`
  - `hand_retargeting_system.launch.py`

- **Setup**: `/home/shalika/Desktop/hand_control_ws/src/hands/setup.py`
