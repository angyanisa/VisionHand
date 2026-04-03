# CSV Playback Mode Guide

This guide explains how to use pre-recorded CSV data for hand control visualization instead of live Rokoko Studio streaming.

## Overview

The Rokoko listener now supports two modes:
1. **Live Mode** (default): Receives real-time data from Rokoko Studio via UDP
2. **Playback Mode**: Reads pre-recorded CSV data and converts it to Rokoko JSON format

## CSV Format

Your CSV file should have the format matching `move_fingers_1.csv`:
- **Header row** with column names: `Timestamp,LeftHand_position_x,...,RightDigit5DistalInterphalangeal_pronation`
- **Data rows** with numeric values for each timestep

Example structure:
```csv
Timestamp,RightHand_position_x,RightHand_position_y,RightHand_position_z,RightDigit1Carpometacarpal_flexion,...
0,0.3219397,1.303011,0.164579,43.20371,...
10,0.3204921,1.303023,0.1658307,43.00916,...
20,0.320184,1.303035,0.1660767,42.9484,...
```

## Usage

### Option 1: CSV Playback Only

Launch just the Rokoko listener in playback mode:

```bash
ros2 launch hands rokoko_csv_playback.launch.py \
    csv_file:=/path/to/your/file.csv \
    publish_rate:=30.0 \
    loop_playback:=false
```

**Parameters:**
- `csv_file`: Path to your CSV file (required)
- `publish_rate`: Playback rate in Hz (default: 30.0)
- `loop_playback`: Whether to loop playback when finished (default: false)

### Option 2: Full Pipeline with CSV Playback

Launch the complete hand retargeting and visualization system with CSV playback:

```bash
ros2 launch hands csv_playback_demo.launch.py \
    csv_file:=/home/shalika/Desktop/hand_control_ws/src/hands/hands/move_fingers_1.csv \
    hand_type:=orca \
    control_method:=direct \
    loop:=true \
    publish_rate:=30.0
```

**Parameters:**
- `csv_file`: Path to your CSV file (required)
- `hand_type`: Type of hand (`orca`, `inspire`, or `leap`)
- `control_method`: Control method (`direct`, `fingertip_ik`, or `jparse_ik`)
- `loop`: Whether to loop the playback
- `publish_rate`: Playback rate in Hz

This will launch:
- Rokoko listener in playback mode
- Hand retargeting node (for your chosen hand type)
- Visualization node
- RViz for 3D visualization

### Option 3: Manual Launch with Custom Pipeline

You can also manually configure the rokoko_listener node in your own launch file:

```python
from launch_ros.actions import Node

rokoko_node = Node(
    package='hands',
    executable='rokoko_listener',
    name='rokoko_listener',
    output='screen',
    parameters=[{
        'playback_mode': True,  # Enable CSV playback
        'csv_file': '/path/to/your/file.csv',
        'publish_rate': 30.0,
        'loop_playback': False
    }]
)
```

## Topics Published

The playback mode publishes the same topics as live mode:
- `/rokoko_raw_data` (std_msgs/String): Raw Rokoko JSON data
- `/rokoko_ref_data` (std_msgs/String): Parsed CSV-format joint data

## Example Workflows

### 1. Test CSV Playback
```bash
# Simple playback test
ros2 launch hands rokoko_csv_playback.launch.py \
    csv_file:=~/hand_control_ws/src/hands/hands/move_fingers_1.csv

# Monitor the published data
ros2 topic echo /rokoko_ref_data
```

### 2. Visualize Recorded Hand Motions
```bash
# Launch full pipeline with ORCA hand
ros2 launch hands csv_playback_demo.launch.py \
    csv_file:=~/hand_control_ws/src/hands/hands/move_fingers_1.csv \
    hand_type:=orca \
    loop:=true
```

### 3. Compare Different Hand Types
```bash
# Try with Inspire hand
ros2 launch hands csv_playback_demo.launch.py \
    csv_file:=~/hand_control_ws/src/hands/hands/move_fingers_1.csv \
    hand_type:=inspire

# Try with Leap hand
ros2 launch hands csv_playback_demo.launch.py \
    csv_file:=~/hand_control_ws/src/hands/hands/move_fingers_1.csv \
    hand_type:=leap
```

## Data Conversion Details

### CSV → JSON Conversion

The system automatically converts your CSV data to the Rokoko JSON format that downstream nodes expect:

**CSV Input:**
```csv
Timestamp,RightDigit1Carpometacarpal_flexion,RightDigit1Carpometacarpal_ulnarDeviation,...
0,43.20371,-15.67476,...
```

**JSON Output:**
```json
{
  "scene": {
    "timestamp": 0,
    "actors": [{
      "body": {
        "rightThumbProximal": {
          "rotation": {"x": x, "y": y, "z": z, "w": w}
        },
        ...
      }
    }]
  }
}
```

### Coordinate Transformations

The conversion handles:
- **Euler to Quaternion**: Joint angles (roll, pitch, yaw) → quaternions
- **Sign conventions**: Matches the original Rokoko parsing logic
  - Thumb: Direct mapping
  - Index/Middle/Ring/Pinky: Negated flexion values
- **Position data**: Hand and fingertip positions preserved

## Troubleshooting

### CSV File Not Found
```bash
[ERROR] CSV file not found: /path/to/file.csv
```
**Solution**: Check that the file path is correct and accessible

### No Data Published
```bash
[INFO] Playback finished
```
**Solution**:
- Check that your CSV file is not empty
- Use `loop_playback:=true` to repeat playback
- Verify CSV format matches expected structure

### Wrong Data Format
```bash
[ERROR] Error loading CSV file: ...
```
**Solution**:
- Ensure CSV has a header row with column names
- Check that all values are numeric (no missing data)
- Verify timestamp column exists

### Playback Too Fast/Slow
**Solution**: Adjust the `publish_rate` parameter:
```bash
# Slower playback (15 Hz)
publish_rate:=15.0

# Faster playback (60 Hz)
publish_rate:=60.0
```

## Recording New CSV Files

To create new CSV recordings for playback:

1. **Run live Rokoko capture with recording enabled**
2. **Use your existing recording system** to save the joint states
3. **Ensure the CSV format matches** the structure in `move_fingers_1.csv`

Required columns:
- `Timestamp`
- Hand positions: `RightHand_position_x/y/z`
- Finger joints: `RightDigit[1-5][Joint]_flexion/ulnarDeviation/pronation`

## Integration with Existing Code

The CSV playback mode integrates seamlessly with all existing nodes:
- ✅ Hand retargeting nodes (orca, inspire, leap)
- ✅ Visualization node
- ✅ IK controllers
- ✅ Any custom nodes that subscribe to `/rokoko_raw_data` or `/rokoko_ref_data`

No changes needed to downstream nodes!

## Performance Notes

- **Memory usage**: Entire CSV is loaded into memory at startup
- **Playback rate**: Independent of file length - controlled by `publish_rate` parameter
- **Large files**: For very large CSV files (>100k rows), consider splitting into smaller files

## Future Enhancements

Potential improvements:
- [ ] Support for Left hand data
- [ ] CSV format auto-detection
- [ ] Playback speed multiplier
- [ ] Seek to specific timestamp
- [ ] Real-time playback progress bar
- [ ] Export to different formats

## Support

For issues or questions:
1. Check that CSV format matches `move_fingers_1.csv`
2. Verify ROS2 topics are publishing: `ros2 topic list`
3. Check node logs: `ros2 topic echo /rosout`
