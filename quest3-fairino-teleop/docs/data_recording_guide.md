# Data Recording Guide

This guide walks through the complete episode recording workflow — from hardware setup through saving files to disk — and explains how the output is structured for imitation-learning training.

---

## Before You Record

### 1. Hardware checklist

- SO-101 plugged in and visible at `/dev/ttyACM0` (run `ls /dev/ttyACM*` to confirm)
- FR5 powered on and on the same subnet as the PC (`ping 192.168.58.2`)
- D405 camera plugged into a USB 3.0 port **directly on the motherboard** (not through a hub)
- FR5 in a safe home configuration — both arms will capture current positions as "home" when `teleop.py` starts

### 2. Write the task instruction

```bash
echo "pick up the red block and place it in the bin" > episode_instruction.txt
```

This file is read when you press `R` to start recording. Write it before launching `teleop.py` so the language label is ready. One line, plain text. It is stored verbatim in the episode JSON.

If you forget to write it before pressing `R`, the episode will be saved with an empty `language_instruction` field — you can edit the JSON later, but setting it in advance is cleaner.

---

## Starting Teleop

```bash
python teleop.py
```

On startup the system:
1. Tries to open the D405 camera (failure is non-fatal — teleoperation continues without video)
2. Opens the SO-101 serial port and disables torque on all motors (arm goes limp, reads freely)
3. Connects to the FR5 via Ethernet and reads current joint positions
4. Captures **home poses** for both arms — the current position of each arm is stored as the reference
5. Enters ServoJ mode on the FR5

You will see:

```
Connecting to hardware...
SO-101 home: ['12.3', '-5.1', '94.2', '0.0', '0.0', '0.0']
FR5 home:    ['12.3', '-5.1', '94.2', '0.0', '0.0', '0.0']
Ready. Space=E-STOP  R=record  H=re-home  Ctrl-C=quit
```

The first ServoJ command is always Δ=0 — the FR5 stays at its home position until you move the SO-101.

---

## Recording an Episode

### Step 1 — Position both arms

Move the SO-101 to the starting configuration for your task. The FR5 will track it. Once both arms are in the desired starting position, press `H` to re-home if needed (resets the delta reference without restarting).

### Step 2 — Start recording

Press `R`. You will see:

```
[REC] Recording started (episode 1716123456789)
  instruction: "pick up the red block and place it in the bin"
```

The episode ID is a millisecond timestamp. If the camera is attached, video recording begins at the same moment.

### Step 3 — Perform the task

Move the SO-101 naturally. Watch the FR5 follow. The heartbeat line prints every second:

```
[HB] cyc=125  sing=CLEAR  SO101 moved  23.1°  →  FR5 cmd moved  34.7°  errs=0
```

- `sing=CLEAR` — not near a singularity
- `sing=WARN` — speed is being reduced; move the arm away from the problematic configuration
- `sing=DANGER` — motion is blocked; back away and press `H` to re-home if needed

### Step 4 — Stop recording

Press `R` again to stop. You will see:

```
[REC] Stopped — saved to ./episodes/episode_1716123456789.csv
```

All four output files are flushed to disk at this moment.

---

## Output Files

Each episode produces up to four files in `./episodes/`:

### `episode_{id}.csv`

One row per control cycle at 125 Hz. Columns:

| Column | Type | Description |
|---|---|---|
| `timestamp` | float64 | Wall-clock time (seconds since epoch) |
| `so101_shoulder_pan` | float | Leader joint position (deg) |
| `so101_shoulder_lift` | float | Leader joint position (deg) |
| `so101_elbow_flex` | float | Leader joint position (deg) |
| `so101_wrist_flex` | float | Leader joint position (deg) |
| `so101_wrist_roll` | float | Leader joint position (deg) |
| `fr5_cmd_j1` .. `fr5_cmd_j6` | float | Commanded joint positions sent to FR5 (deg) |
| `fr5_actual_j1` .. `fr5_actual_j6` | float | Measured FR5 joint positions (deg) |
| `fr5_eef_x_mm` / `_y_mm` / `_z_mm` | float | TCP position in mm |
| `fr5_eef_rx_deg` / `_ry_deg` / `_rz_deg` | float | TCP Euler orientation (deg) |
| `gripper_norm` | float | SO-101 gripper [0.0=closed, 1.0=open] |
| `fr5_vel_j1` .. `fr5_vel_j6` | float | Actual FR5 joint velocities (deg/s) |

Actual FR5 state is read at ~21 Hz per quantity (positions, EEF, velocities are staggered one per qualifying cycle and cached) — every row has complete data.

### `episode_{id}.json`

Episode metadata:

```json
{
  "episode_id": 1716123456789,
  "start_time": 1716123456.789,
  "language_instruction": "pick up the red block and place it in the bin",
  "num_steps": 1250,
  "duration_s": 10.0,
  "camera_intrinsics": {
    "fx": 615.3, "fy": 615.2, "cx": 320.1, "cy": 240.0,
    "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    "width": 640, "height": 480,
    "distortion_model": "inverse_brown_conrady"
  },
  "camera_num_frames": 300
}
```

If no camera was attached, `camera_intrinsics` and `camera_num_frames` will be `null` / `0`.

### `episode_{id}_camera.mp4`

RGB video at 30 fps. Playback is at a fixed nominal rate — real frame timing is in the `.npy` sidecar. Do not use this file's frame numbers as ground-truth timestamps.

### `episode_{id}_camera_ts.npy`

NumPy array of shape `(N,)`, dtype `float64`. Entry `i` is the wall-clock timestamp (seconds since epoch) of frame `i` in the MP4.

---

## Syncing Camera to Joint Data

The CSV and MP4 share a common wall-clock (`time.time()`). To find the joint-state row closest to video frame `i`:

```python
import numpy as np
import pandas as pd

df      = pd.read_csv("episodes/episode_123.csv")
cam_ts  = np.load("episodes/episode_123_camera_ts.npy")

# For each camera frame, find the nearest CSV row index
csv_ts  = df["timestamp"].values
indices = np.searchsorted(csv_ts, cam_ts)
indices = np.clip(indices, 0, len(df) - 1)

# aligned_rows[i] is the joint state at camera frame i
aligned_rows = df.iloc[indices]
```

Frame-to-row alignment is typically within ±4 ms (half a 125 Hz cycle).

---

## Training Format Notes

The CSV + JSON + MP4 bundle maps directly to common imitation-learning formats:

| Framework | What to feed |
|---|---|
| **ACT / ALOHA** | `fr5_cmd_j1..6` at 125 Hz as joint-space actions; reshape to HDF5 |
| **Diffusion Policy** | Downsample to 10–25 Hz; use `fr5_eef_*` for EEF-delta actions |
| **π0 / flow matching** | `fr5_cmd_j1..6` at 20–50 Hz; `language_instruction` from JSON |
| **VLA** | EEF delta (7D: x/y/z + quat + gripper); `language_instruction` required |
| **GR00T / LeRobot** | Convert CSV to LeRobot HDF5 schema; camera frames are pre-aligned |

---

## Tips for Clean Demonstrations

- **Short, decisive motions** record better than slow drift — the rate limiter will round off very small movements
- **Re-home before each episode** (`H` key) to ensure a consistent starting reference
- **Check the heartbeat** for `errs=0` before recording — persistent errors mean a shaky episode
- **One task per episode** — keep the instruction and motion aligned; don't reuse episodes for multiple tasks
- **Camera stabilisation** — the wrist camera moves a lot; mount it as rigidly as possible to reduce blur

---

## Emergency Stop During Recording

If you press `Space` (E-STOP) while recording, the episode is **auto-saved** before the process exits. You will see:

```
[E-STOP] Space pressed — stopping motion.
[REC] Auto-saved episode to ./episodes/episode_1716123456789.csv
```

The recording is complete up to the moment of the stop. Check the robot state before restarting.

---

## Checking Saved Episodes

Quick sanity check for a saved episode:

```python
import pandas as pd, json, numpy as np

df   = pd.read_csv("episodes/episode_123.csv")
meta = json.load(open("episodes/episode_123.json"))

print(f"Steps: {len(df)}  Duration: {meta['duration_s']}s")
print(f"Instruction: {meta['language_instruction']}")
print(f"Camera frames: {meta['camera_num_frames']}")
print(df[["fr5_cmd_j1", "fr5_cmd_j2", "fr5_cmd_j3"]].describe())
```
