# FVR — Fairino FR5 VR Teleoperation

> **Branch: `working`** — verified end-to-end on the real FR5. Unified
> **SE(3) relative-transform** retargeting and **damped least-squares (DLS) IK**
> for smooth, jitter-free trajectories, decoupled thumbstick/hand wrist control,
> a dexterous mid-workspace mapping, and **DLS IK on hardware in both `validate`
> and `execute`** (`EXECUTE_USE_DLS_IK`). Forward/back axis calibrated to the arm.

Real-time VR teleoperation of a **Fairino FR5 cobot** using a **Meta Quest 3** controller.  
Controller pose streams via WebXR → One-Euro filtered → **SE(3) relative-transform retargeting** → **DLS IK (Pinocchio)** → ServoJ at **125 Hz**.

---

## Hardware

| Component | Details |
|---|---|
| Meta Quest 3 | VR controller, WiFi or USB (ADB) |
| Fairino FR5 cobot | 6-DOF industrial arm, Ethernet `192.168.58.2` |
| DH AG-160-95 gripper | Parallel gripper, flange-mounted |
| Intel RealSense D405 | Wrist-mounted RGB camera, 640×480 @ 30 fps |
| Linux PC | Ubuntu 22.04, ROS 2 Humble, Python 3.10 |

---

## Quick Start

### 1 — Install dependencies (one-time)

```bash
git clone https://github.com/vivek-kanjarla/FVR.git
cd FVR
bash linux_setup.sh
pip install -r requirements.txt
```

**Fairino SDK** (required for `validate`/`execute` — not on PyPI). The SDK ships a
precompiled `Robot` extension per Python version. For the venv's Python 3.10, drop
the matching `.so` into a `fairino` package on the venv path:

```bash
# from fairino-python-sdk-main/linux/
SP=~/teleop_env/lib/python3.10/site-packages/fairino
mkdir -p "$SP" && touch "$SP/__init__.py"
cp libfairino/Robot.cpython-310-x86_64-linux-gnu.so "$SP/"
~/teleop_env/bin/python3 -c "from fairino import Robot; print('OK', hasattr(Robot,'RPC'))"
```

Install ROS 2 Humble (required for RViz visualisation):

```bash
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt-get update && sudo apt-get install -y \
  ros-humble-desktop ros-humble-robot-state-publisher ros-humble-rviz2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### 2 — Get the FR5 URDF

```bash
mkdir -p ~/ros2_teleop_ws/src
git clone --depth 1 https://github.com/FAIR-INNOVATION/frcobot_ros2.git \
  ~/ros2_teleop_ws/src/frcobot_ros2-main
cd ~/ros2_teleop_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select fairino_description --symlink-install
```

### 3 — Generate SSL cert (one-time)

```bash
mkdir -p ssl
openssl req -x509 -newkey rsa:4096 -nodes \
  -out ssl/cert.pem -keyout ssl/key.pem -days 365 -subj "/CN=quest-teleop"
```

### 4 — Create Python virtual environment

```bash
python3 -m venv ~/teleop_env
~/teleop_env/bin/pip install numpy pandas pynput opencv-python vuer pybullet pyyaml pin
```

### 5 — Launch

```bash
chmod +x START_WIFI.sh
./START_WIFI.sh sim        # DLS IK → RViz/PyBullet only — no hardware
./START_WIFI.sh validate   # DLS IK → limit/collision gate → ServoJ to real FR5  (recommended on hardware)
./START_WIFI.sh execute    # DLS IK → ServoJ to real FR5, no gate  (EXECUTE_USE_DLS_IK=True)
```

All three modes share the **same front half** — VR → SE(3) retarget → DLS IK.
They differ only in what consumes the joint target (see
[Sim vs Validate vs Execute](#sim-vs-validate-vs-execute)).

### 6 — Connect Quest 3

1. Put the headset **on your head** and **face the robot** (this locks the reference frame)
2. Open **Meta Browser** → `https://<laptop-ip>:8012`
3. Accept the cert warning → tap **TAP ANYWHERE**
4. **Squeeze the right index-finger trigger** → arm homes to your hand position
5. Move wrist with trigger held → robot tracks your hand

---

## Controller Mapping

| Action | Control |
|---|---|
| **Home / enable tracking** | Right **index trigger** (hold) |
| **Freeze arm** | Release trigger |
| **Re-home from new position** | Squeeze trigger again |
| **Wrist tilt (J4)** | Right **thumbstick** up/down |
| **Wrist turn (J5)** | Right **thumbstick** left/right |
| **Close gripper** | Right **grip** button |
| **Open gripper** | Release grip |

### Axis mapping (Cartesian, SE(3))

Relative hand motion from home is mapped into the FR5 base frame by the constant
rotation `VR_FRAME_ROTATION` (`config.py`). The signs below are **calibrated to the
real arm** — in particular the forward/back axis is flipped so pushing forward
reaches forward (`config.py:VR_FRAME_ROTATION` row Y = `−Quest_Z`).

| Hand movement | EEF motion |
|---|---|
| Push **forward** (away from body) | EEF moves forward (reach out) |
| Pull **backward** (toward body) | EEF moves back |
| Move **right** / **left** | EEF moves laterally |
| Move **up** / **down** | EEF rises / lowers |

> The matrix is a reflection (`det = −1`) after the forward/back flip. That's safe:
> SE(3) retarget applies it to position as `X·dp` and to orientation by conjugation
> `X·dR·Xᵀ`, whose determinant equals `det(dR)` regardless of `det(X)`, so the
> orientation mapping stays a valid rotation.

---

## Architecture

```
Meta Quest 3 (WebXR / Meta Browser)
        │  HTTPS + WSS  (ports 8012 / 8013)
        ▼
quest3.py ── Quest3Reader
        │  ControllerState [pos, quat, trigger, grip]
        ▼
pose_filter.py ── One-Euro adaptive LPF
        │  filtered pos (3D) + filtered quat (SLERP on S³)
        ▼
mapper_vr.py ── vr_to_fr5()
        │  USE_SE3_RETARGET=True → retarget_se3.py (SE(3) relative transform)
        │  FR5 EEF target [x, y, z, rx, ry, rz]
        ▼
dls_ik.py ── DLS IK (Pinocchio, adaptive damping λ)
        │  joint angles [J1 … J6]  (+ thumbstick/hand wrist → J4/J5)
        ▼
sim_bridge.py ── 125 Hz loop (singularity check → rate limit → ServoJ)
        │
        ├── PyBullet + RViz2  (sim / validate)
        └── FR5 hardware      (validate / execute)
```

---

## Key Files

| File | Purpose |
|---|---|
| `config.py` | All tunable parameters — start here |
| `mapper_vr.py` | Pose retargeting + IK call (dispatches to SE(3) backend) |
| `retarget_se3.py` | **Unified SE(3) relative-transform retargeter** (spatial/body modes) |
| `dls_ik.py` | **Damped least-squares IK** (Pinocchio, adaptive λ) |
| `pose_filter.py` | One-Euro filter: position 3D + quaternion SLERP |
| `sim_bridge.py` | 125 Hz control loop with auto-return-to-home |
| `quest3.py` | Quest 3 transport layer (vuer / UDP / oculus_reader) |
| `singularity.py` | J3/J5 singularity detection and speed scaling |
| `axis_logger.py` | Real-time CSV + terminal axis diagnostic |
| `frame_audit.py` | Coordinate-frame audit tool |
| `watch_axes.py` | Live axis mapping terminal display |
| `START_WIFI.sh` | One-command launch for WiFi Quest 3 |
| `webxr/index.html` | WebXR page served to Meta Browser |

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `FR5_IP` | `192.168.58.2` | FR5 controller Ethernet IP |
| `QUEST3_MODE` | `"vuer"` | Transport: `vuer` / `oculus_reader` / `udp` |
| `VR_POSITION_SCALE` | `800.0` | mm of EEF per metre of hand movement (1:0.8 — reach without slamming the boundary) |
| `VR_ROTATION_SCALE` | `0.0` | Wrist rotation multiplier (0 = lock orientation, clean position-only tracking) |
| `USE_SE3_RETARGET` | `True` | Use the unified SE(3) relative-transform pipeline (vs. legacy delta mapping) |
| `SE3_RETARGET_MODE` | `"spatial"` | `spatial` = world-fixed control dirs; `body` = controller-relative |
| `EXECUTE_USE_DLS_IK` | `True` | In `execute`, use DLS IK (home EEF from DLS FK) instead of the FR5 onboard IK |
| `VR_MAX_DELTA_POS_MM` | `450.0` | Max EEF offset from home — keeps the arm in a dexterous mid-workspace bubble |
| `WRIST_GAIN_J4` / `_J5` | `1.0` | Hand pitch→J4, hand yaw→J5 (deg of joint per deg of hand rotation) |
| `JOYSTICK_J4_SCALE` / `_J5_SCALE` | `0.30` / `0.15` | Thumbstick → direct J4/J5 wrist control (°/cycle) |
| `FILTER_POS_MIN_CUTOFF` | `1.0` Hz | Position filter cutoff at rest |
| `FILTER_POS_BETA` | `10.0` | Position responsiveness to speed |
| `FILTER_ROT_MIN_CUTOFF` | `1.0` Hz | Rotation filter cutoff at rest |
| `FILTER_ROT_BETA` | `1.0` | Rotation responsiveness to speed |
| `AUTO_HOME_IDLE_S` | `8.0` s | Idle seconds before arm returns to safe home |
| `AUTO_HOME_SPEED` | `0.3` | Auto-return speed (fraction of rate limits) |

---

## Pose Filter — One-Euro (pose_filter.py)

Applied every cycle **before** delta mapping:

- **Position** — independent 1€ filter on X/Y/Z. At rest: ~0.4 mm residual jitter. During fast motion: near-transparent (adaptive cutoff rises with speed).
- **Orientation** — 1€ filter using **SLERP on S³** (not Euler-angle averaging). Handles quaternion sign-flips (shortest-path). No gimbal lock.

**Why One-Euro over EMA**: fixed-cutoff EMA cannot simultaneously suppress rest jitter and maintain low latency. One-Euro adapts `fc = min_cutoff + β·|ẋ|` so latency falls automatically during motion.

**Why SLERP**: quaternion components are constrained to the unit sphere S³. Component-wise lerp breaks the constraint and does not follow the geodesic. SLERP is the unique constant-angular-velocity geodesic on S³.

---

## SE(3) Relative-Transform Retargeting (retarget_se3.py)

Poses are handled as 4×4 homogeneous transforms in **SE(3)** rather than separate
translation deltas and Euler-angle deltas. Each cycle the retargeter computes the
controller's relative motion from the home pose and maps it into the FR5 base frame
through a single, consistent frame rotation `X`:

- **`spatial` mode** (default) — relative motion expressed in world-fixed control
  directions, matching the calibrated `VR_FRAME_ROTATION`.
- **`body` mode** — relative motion expressed in the controller's own frame
  (`T_homeⁱ⁻¹ · T_current`).

Rotation is scaled via the **SO(3) log-map** (correct for large angles, no gimbal
issues), and the same frame rotation is applied consistently to both position and
orientation. Translation output is identical to the legacy pipeline; the SE(3)
formulation fixes rotation scaling near gimbal and produces a smoother trajectory.
Toggle with `USE_SE3_RETARGET` in `config.py`.

---

## DLS Inverse Kinematics (dls_ik.py)

End-effector targets are solved with **damped least-squares IK** built on **Pinocchio**:

```
Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ · e
```

The damping factor **λ adapts** to the smallest singular value of the Jacobian
(`λ_min = 0.001`, `λ_max = 0.05`, `σ_thresh = 0.05`): near a singularity λ rises to
keep joint velocities bounded; in well-conditioned regions λ drops toward zero for
accurate, low-error tracking. This trades a small amount of position error for
**continuous, jitter-free joint trajectories** — no velocity spikes when the arm
approaches the elbow/wrist singularities.

The same solver drives **all three modes** (`sim`, `validate`, `execute`). When
`EXECUTE_USE_DLS_IK = True`, `execute` uses DLS IK directly on hardware instead of
the FR5's onboard IK. Because DLS solves in the Pinocchio (flange) frame, the home
EEF in that mode is seeded from `DLSIK.get_fk(start_joints)` — **not** the robot's
`GetActualTCPPose()`, which reports the configured tool tip (~100 mm tool offset on
this setup). Using model FK keeps `target = home + delta` in one frame so the arm
does not jump on home. Verified against the real arm: PyBullet FK and DLS FK agree
to sub-mm, and a FK→IK round-trip returns the seed joints exactly.

---

## Wrist Control — Thumbstick & Hand (decoupled from position)

Wrist joints are driven independently of EEF translation so position tracking stays clean:

- **Thumbstick** — UP/DOWN → **J4** (forearm tilt), LEFT/RIGHT → **J5**, with a
  dead-zone (`JOYSTICK_DEAD_ZONE = 0.08`) and per-cycle scaling.
- **Hand orientation** — hand **pitch → J4**, hand **yaw → J5** at `WRIST_GAIN_J4/J5`
  (1:1 by default). Targets are rate-limited against the previous wrist value, not the
  raw IK solution, so the wrist eases smoothly instead of snapping.

Set `VR_ROTATION_SCALE = 0.0` (default) to lock EEF orientation entirely and let IK
track position only — the cleanest mode for pick-and-place.

---

## Sim vs Validate vs Execute

All three modes run the identical front half — **VR → SE(3) retarget → DLS IK** —
and call the *same* `DLSIK` instance. They differ only in what consumes the joint target:

| Mode | Moves hardware? | Back half (endpoint) | Use when |
|---|---|---|---|
| `sim` | **No** | DLS IK → PyBullet + RViz only | Development, axis calibration, no robot |
| `validate` | **Yes** | DLS IK → collision/limit gate → ServoJ + PyBullet + RViz | **Default on hardware** — DLS *with* a safety net |
| `execute` | **Yes** | DLS IK → ServoJ (+ RViz joint publish), **no gate** | Lowest overhead once `validate` is trusted |

Notes:
- `validate` ≈ `execute` **plus** a per-command collision/limit check and sim visualization.
  Anything that fails the check prints `[SIM] BLOCKED` and is skipped — the arm doesn't move.
- `execute` has **no pre-screening**: every DLS solution goes straight to `ServoJ`.
- Set `EXECUTE_USE_DLS_IK = False` to fall back to the FR5's onboard IK in `execute`
  (legacy lean path, no model/URDF dependency).

---

## Running on Real Hardware

Verified workflow on the FR5 (the `working` branch is named for this — it's been
driven on the real arm):

1. **Network** — wire Ethernet to the FR5 and put your NIC on its subnet; the Quest
   stays on WiFi. You need **both** interfaces up:
   ```bash
   ping -c2 192.168.58.2                 # FR5 must reply
   ip -4 addr show                       # note your WiFi IP for START_WIFI.sh
   ```
2. **Preflight** — confirm the SDK imports and the model frame matches the arm:
   ```bash
   ~/teleop_env/bin/python3 check_network.py
   ~/teleop_env/bin/python3 check_hardware.py
   ```
   The DLS `get_fk(real_joints)` should match the robot's reported pose in X/Y and
   orientation; a constant Z offset is the tool/TCP offset and is expected (kept out
   of the loop — see [DLS Inverse Kinematics](#dls-inverse-kinematics-dls_ikpy)).
3. **Launch** — `./START_WIFI.sh validate` (set `LAPTOP_IP` in the script first).
   The stack connects to the FR5, enables servos, starts servo mode, and waits for
   the Quest. **The arm is now live** — keep the e-stop within reach.
4. **Drive** — Meta Browser → `https://<LAPTOP_IP>:8012` → hold trigger to home →
   move. Healthy teleop shows `got=251/251  ik=251 fail=0` in the `[DIAG]` line.
5. **Stop cleanly** so servo mode is stopped and the controller disconnects (avoids
   the CNDE reconnect cooldown):
   ```bash
   kill -INT <bridge-pid>     # or Ctrl-C in the foreground terminal
   ```

Safety: start with `FR5_SERVO_VEL` low (`15`). `VR_MAX_DELTA_POS_MM = 450` keeps the
arm in a mid-workspace bubble; re-home with your hand nearer your body if you see
`[VAL] EEF reach … > max 900mm` warnings (you're hitting the physical reach limit).

---

## Singularity Protection (singularity.py)

| Joint | Singularity | Warn | Danger | Effect |
|---|---|---|---|---|
| J3 | Elbow (arm straight) | < 8° | < 1° | Speed → 0.05 |
| J5 | Wrist (J4/J6 align) | < 15° | < 3° | Speed → 0.05 |

Scale floor of **0.05** (not 0.0) allows the arm to creep out of near-singular configurations rather than freezing.

---

## Auto Return to Home

When the trigger is released for **`AUTO_HOME_IDLE_S`** seconds (default 8 s), the arm glides back to the factory default joint configuration at `AUTO_HOME_SPEED` (30% of normal rate limits). This prevents the arm from being left in a stretched or singular configuration.

---

## Diagnostic Tools

**Live axis display** (run in second terminal while sim is active):
```bash
python3 watch_axes.py
```

**CSV axis log** (written automatically):
```bash
tail -f /tmp/axis_log.csv
# Columns: t, trigger, grip, q_right_m, q_up_m, q_back_m, qx,qy,qz,qw,
#          fr5_dx_mm, fr5_dy_mm, fr5_dz_mm, fr5_x, fr5_y, fr5_z, ...
```

**Coordinate frame audit**:
```bash
python3 frame_audit.py            # static analysis
python3 frame_audit.py --live     # live calibration with Quest connected
```

---

## Transport Modes

| Mode | Latency | Setup |
|---|---|---|
| `vuer` (default) | ~60 ms | Meta Browser → HTTPS server; SSL cert required |
| `oculus_reader` | ~20 ms | ADB + jborbik/oculus_reader APK |
| `udp` | ~20 ms | Custom UDP streamer |

---

## Known Issues

- **Controllers sleep after ~30 s idle** — squeeze any button to wake
- **`visible-blurred` state** — press Quest button to dismiss system overlay; session resumes immediately
- **CNDE single-client limit** — always use Ctrl-C to stop; force-kill requires ~30 s cooldown before reconnect
- **Re-home when arm drifts** — release trigger for 8 s (auto-return) then re-home from a comfortable position

---

## Dependencies

```
Python:  numpy  pandas  pynput  opencv-python  vuer  pybullet  pyyaml  pin (pinocchio)
ROS 2:   ros-humble-desktop  robot-state-publisher  rviz2
SDK:     fairino (*.whl — from Fairino, not on PyPI)
ADB:     android-tools-adb (for oculus_reader mode)
```

---

## License

MIT
