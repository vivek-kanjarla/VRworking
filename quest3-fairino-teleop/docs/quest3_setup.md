# Meta Quest 3 VR Teleoperation — Setup Guide

Real-time end-effector control of the FR5 using the Meta Quest 3 right controller.
Controller 6-DOF pose → FR5 TCP motion via inverse kinematics. Trigger drives gripper.

Two transport modes — pick one and configure `QUEST3_MODE` in `config.py`:

| Mode | How it works | Latency | Friction |
|---|---|---|---|
| **`oculus_reader`** | APK sideloaded via ADB, streams over TCP | ~30 ms | Low |
| **`vuer`** | Quest browser → HTTPS WebXR server, no APK | ~80–100 ms | Medium |
| **`udp`** | Custom Quest app sends UDP packets | ~20 ms | High (needs app) |

---

## Option A — Oculus Reader (APK / ADB)

### A1 — Enable Developer Mode
Open the **Meta Quest mobile app → Menu → Devices → [Quest 3] → Developer Mode → ON**.
Accept the prompt in the headset.

### A2 — Install ADB

```bash
# macOS
brew install android-platform-tools

# Ubuntu/Debian
sudo apt install adb

# Verify
adb version
```

### A3 — Sideload the APK (Quest 3 compatible fork)

> **Use `jborbik/oculus_reader` — not the original `rail-berkeley` repo.
> The original does not support Quest 3.**

```bash
git clone https://github.com/jborbik/oculus_reader
cd oculus_reader

# Connect Quest 3 via USB-C; accept "Allow USB debugging" prompt in headset
adb devices        # should list your Quest 3 serial number
adb install reader.apk
```

### A4 — Forward the port (every session)

```bash
adb forward tcp:5555 tcp:5555
```

> Re-run this after every USB reconnect or ADB restart.
> You can disconnect the cable after forwarding — but for stability keep it plugged in.

### A5 — Launch the app on Quest 3

Open **App Library → Unknown Sources → OculusReader**.
The app shows a small overlay and starts streaming immediately — no further interaction needed.

### A6 — Verify the stream

```bash
python - <<'EOF'
from quest3 import Quest3Reader
import time
with Quest3Reader() as q:
    for _ in range(30):
        time.sleep(0.1)
        s = q.get_state()
        print(f"valid={s.valid}  pos={s.pos.round(3)}  trig={s.trigger:.2f}")
EOF
```

Move the right controller — `pos` should change.

---

## Option B — Vuer (Quest browser, no APK)

Vuer starts a local HTTPS WebXR server on your PC. The Quest browser connects to it, enters an XR session, and streams controller pose over WebSocket. No sideloading needed.

### B1 — Install Vuer

```bash
pip install vuer
```

### B2 — Generate a self-signed SSL certificate

WebXR requires HTTPS. Do this once and keep the files.

```bash
mkdir -p ssl
openssl req -x509 -newkey rsa:4096 -nodes \
  -out ssl/cert.pem -keyout ssl/key.pem \
  -days 365 -subj "/CN=quest-teleop"
```

Verify paths match `config.py`:

```python
QUEST3_VUER_CERT = "./ssl/cert.pem"
QUEST3_VUER_KEY  = "./ssl/key.pem"
```

### B3 — Set Vuer mode in config.py

```python
QUEST3_MODE = "vuer"
```

### B4 — Find your PC's local IP address

```bash
# macOS / Linux
ipconfig getifaddr en0   # or: ip route get 1 | awk '{print $7}'
```

This PC and the Quest 3 must be on the **same WiFi network** (5 GHz recommended).

### B5 — Run teleop_vr.py

```bash
python teleop_vr.py
```

On startup you'll see:

```
[QUEST3] Vuer server started — open on Quest 3 browser:
         https://192.168.x.x:8012
         Accept the self-signed cert warning, then tap 'Enter VR'.
[QUEST3] Waiting for Quest 3 to connect to Vuer server...
```

### B6 — Connect from Quest 3 browser

1. Put on the headset and open the **Meta Quest Browser**.
2. Navigate to `https://<your-pc-ip>:8012`.
3. You'll see a security warning about the self-signed certificate.
   Tap **Advanced → Proceed** (you only need to do this once per cert).
4. Tap **Enter VR** on the Vuer page.
5. Grant controller tracking permission if prompted.

The terminal should print:

```
[QUEST3] Vuer server started — waiting...
```

and then the teleop loop will begin.

### B7 — Trust the cert on Quest (optional, avoids warning each time)

To avoid clicking through the warning on every session, install the cert into Quest's trust store:

```bash
# Copy cert to Quest
adb push ssl/cert.pem /sdcard/cert.pem

# On the Quest, go to:
# Settings → General → Device Certificates → Install
# Select cert.pem
```

---

## Running teleoperation

Once either transport is working:

```bash
python teleop_vr.py
```

Hold the controller **still** during startup — home poses are captured at launch.

### Controls

| Input | Action |
|---|---|
| Move controller | EEF translates (scaled by `VR_POSITION_SCALE`) |
| Rotate controller | EEF rotates (scaled by `VR_ROTATION_SCALE`) |
| Trigger ≥ 0.7 | Gripper close |
| Trigger ≤ 0.2 | Gripper open |
| `H` key | Re-home — reset reference to current poses |
| `R` key | Toggle episode recording |
| `Space` | Emergency stop |
| `Ctrl-C` | Graceful exit |

### Heartbeat output

```
[HB] cyc=125  sing=CLEAR  Quest moved  12.3mm  →  FR5 cmd moved  18.5°  trig=0.00  errs=0
```

- `Quest moved 0 mm` → controller stream frozen / not connected
- `Quest moved N mm` but `FR5 cmd moved 0°` → IK failing or singularity
- `errs=N` rising fast → communication issue

---

## Coordinate frame calibration

Default `VR_FRAME_ROTATION` maps:

| Quest 3 axis | FR5 TCP axis |
|---|---|
| X (right) | Y |
| Y (up) | Z (up) |
| Z (back) | −X |

If EEF moves in the wrong direction, adjust `VR_FRAME_ROTATION` in `config.py`.
It's a 3×3 matrix — use only ±1 and 0 entries.

---

## Tuning

| Parameter | Default | Effect |
|---|---|---|
| `VR_POSITION_SCALE` | 300 | mm of EEF motion per metre of controller motion. Start low, increase gradually. |
| `VR_ROTATION_SCALE` | 0.8 | EEF rotation gain. Reduce if wrist snaps feel aggressive. |
| `VR_MAX_DELTA_POS_MM` | 8.0 | Hard clamp per cycle — prevents lurches from tracking jumps. |
| `VR_MAX_DELTA_ROT_DEG` | 3.0 | Hard clamp per cycle on rotation. |
| `GRIPPER_CLOSE_THRESHOLD` | 0.7 | Trigger level to close gripper. |
| `GRIPPER_OPEN_THRESHOLD` | 0.2 | Trigger level to open gripper. |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `TimeoutError` (oculus_reader) | ADB forward missing or app not running | Re-run `adb forward tcp:5555 tcp:5555`, relaunch app |
| `TimeoutError` (vuer) | Quest browser not connected or cert not accepted | Navigate to URL, accept cert warning, tap Enter VR |
| `FileNotFoundError: ssl/cert.pem` | SSL cert not generated | Run the `openssl` command in B2 above |
| `ImportError: vuer not installed` | Package missing | `pip install vuer` |
| `Quest moved 0 mm` in heartbeat | Controller asleep or tracking lost | Wake controller, check battery |
| IK errors / `GetInverseKin failed` | EEF target unreachable | Reduce `VR_POSITION_SCALE`; press `H` to re-home |
| EEF axes inverted | Coordinate frame mismatch | Adjust `VR_FRAME_ROTATION` in config.py |
| Gripper not responding | Trigger threshold mismatch | Adjust `GRIPPER_*_THRESHOLD` in config.py |
| High latency (Vuer) | WiFi congestion | Use 5 GHz band; reduce background traffic |
| `triggerValue` inverted | Vuer version difference | Negate trigger: change `trigger=float(state.get("triggerValue",0))` to `1.0 - ...` in quest3.py `_run_vuer_server` |
