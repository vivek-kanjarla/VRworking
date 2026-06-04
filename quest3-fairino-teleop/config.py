"""
config.py — configuration for Meta Quest 3 → FR5 VR teleoperation.
Edit these values to match your physical setup before running teleop_vr.py.
"""

# ── FR5 (follower cobot) ──────────────────────────────────────────────────────
FR5_IP = "192.168.58.2"           # Default Fairino controller IP

# ── Control loop ──────────────────────────────────────────────────────────────
LOOP_HZ     = 125                 # ServoJ must run between 60–1000 Hz
LOOP_PERIOD = 1.0 / LOOP_HZ      # ~0.008 s between calls

# ── Safety ────────────────────────────────────────────────────────────────────
# Global fallback — used when per-joint limit not specified
MAX_DELTA_DEG_PER_CYCLE = 0.08    # 0.08 × 125Hz = 10°/s

# Per-joint rate limits (degrees/cycle) for [J1, J2, J3, J4, J5, J6]
# Raised ~2× so the robot keeps up with amplified hand motion (sim).
# REVERT to the commented hardware-safe values before running on real FR5.
# Hardware-safe caps (real robot) — halves per-cycle motion so a controller
# tracking-glitch correction ramps gently instead of lurching. Sim values
# (2× faster) were: [0.32, 0.24, 0.24, 0.60, 0.16, 0.40].
MAX_DELTA_PER_JOINT = [0.16, 0.12, 0.12, 0.30, 0.08, 0.20]
#                       J1     J2     J3     J4    J5    J6
#                       20°/s  15°/s  15°/s  37/s  10/s  25°/s

FR5_SERVO_VEL           = 15      # ServoJ velocity % — start low, tune up
FR5_FILTER_T            = 0.04    # ServoJ trajectory filter (seconds) — smooths between commands

# From GetJointSoftLimitDeg() on this controller, with 5° margin inside each limit.
FR5_JOINT_LIMITS = [
    (-170, 170),   # J1  hardware ±175°
    (-260,  80),   # J2  hardware (-265, 85)
    (-155, 155),   # J3  hardware ±160°
    (-260,  80),   # J4  hardware (-265, 85)
    (-170, 170),   # J5  hardware ±175°
    (-350, 350),   # J6  continuous wrist rotation — real robot parks at ~178°
]

# ── DH AG-160-95 gripper (flange-mounted, via Fairino SDK) ───────────────────
GRIPPER_INDEX     = 1      # gripper number configured on the FR5 controller
GRIPPER_TYPE      = 0      # 0=parallel gripper (AG-160-95), 1=rotating

GRIPPER_OPEN_PCT  = 100    # position % sent when opening  (0–100)
GRIPPER_CLOSE_PCT = 0      # position % sent when closing
GRIPPER_VEL_PCT   = 50     # movement speed    (0–100)
GRIPPER_FORCE_PCT = 50     # grip force        (0–100)
GRIPPER_MAXTIME_MS = 5000  # max travel time before timeout (ms)

# Quest 3 trigger analog mapped to gripper state via hysteresis:
#   trigger ≥ CLOSE_THRESHOLD → close
#   trigger ≤ OPEN_THRESHOLD  → open
#   between                   → hold current state
GRIPPER_OPEN_THRESHOLD  = 0.2   # trigger below this → open
GRIPPER_CLOSE_THRESHOLD = 0.7   # trigger above this → close

# ── Meta Quest 3 VR controller ───────────────────────────────────────────────
#
# Two transport modes:
#   "oculus_reader" — Oculus Reader APK (rail-berkeley/oculus_reader) via ADB
#                     Run: adb forward tcp:5555 tcp:5555
#   "udp"           — lightweight UDP packet streamer (custom app / HTS-compatible)
#
QUEST3_MODE        = "vuer"           # "oculus_reader" | "vuer" | "udp"
QUEST3_ADB_PORT    = 5555             # TCP port after adb forward (oculus_reader mode)
QUEST3_UDP_PORT    = 5005             # UDP port this PC listens on (udp mode)
QUEST3_ACTIVE_HAND = "right"          # which controller drives the arm: "right" | "left"

# Vuer transport (QUEST3_MODE = "vuer") — Quest 3 browser → HTTPS WebXR server
# Generate cert once: openssl req -x509 -newkey rsa:4096 -nodes \
#   -out ssl/cert.pem -keyout ssl/key.pem -days 365 -subj "/CN=quest-teleop"
QUEST3_VUER_PORT   = 8012             # HTTPS port the Quest browser connects to
QUEST3_VUER_CERT   = ""               # empty = HTTP mode (use adb reverse tcp:8012 tcp:8012)
QUEST3_VUER_KEY    = ""               # empty = HTTP mode

# Position scale: 1 metre of controller motion → this many mm of EEF motion.
# Start conservative (300) and increase once the motion feels right.
VR_POSITION_SCALE = 800.0    # mm / m  (1m hand = 0.8m EEF — amplified but stays in dexterous range)
#   600  = 1:0.6  fine control, lots of re-homing
#   800  = 1:0.8  good balance — reach without slamming the workspace boundary
#   1500 = 1:1.5  too aggressive: small moves hit the 900mm reach limit, arm contorts

# Rotation scale: multiplier on controller rotation → EEF rotation (deg).
# 1.0 = 1:1 mapping. Reduce if wrist snaps feel aggressive.
VR_ROTATION_SCALE = 0.0       # 0 = lock EEF orientation, IK tracks position only (clean translation)
#   0.0  = orientation held at home — cleanest position tracking, no wrist coupling
#   0.3  = gentle wrist control, minimal coupling
#   0.8+ = strong wrist control but hand tremor couples into arm motion

# Retargeting backend:
#   True  = unified SE(3) relative-transform pipeline (retarget_se3.py).
#           Rotation scaled via SO(3) log-map (correct for large angles),
#           single frame-rotation X applied consistently to pos + orientation.
#   False = legacy separate translation-delta + Euler-delta (mapper_vr inline).
# Both produce identical translation; SE(3) fixes rotation scaling near gimbal.
USE_SE3_RETARGET = True
# SE(3) relative-motion frame:
#   "spatial" = world-fixed control directions (matches calibration, default)
#   "body"    = controller-relative directions (T_home⁻¹·T_current)
SE3_RETARGET_MODE = "spatial"

# Execute (LIVE_ONLY) IK backend:
#   True  = DLS IK (dls_ik.py) — the SAME smooth solver as sim/validate.
#           Home EEF is taken from DLS/Pinocchio FK on the real start joints
#           (NOT the robot's get_eef_pose()) so the model frame matches and the
#           arm does not jump on home. No PyBullet, no validator — lean + smooth.
#   False = FR5 onboard IK via RPC (legacy lean path, no model dependency).
EXECUTE_USE_DLS_IK = True

# ── Thumbstick → direct wrist joint control ───────────────────────────────────
# Thumbstick UP/DOWN   → J4 (forearm tilt)
# Thumbstick LEFT/RIGHT→ J5 (wrist bend)
# Scale = degrees per cycle at full joystick deflection (±1.0)
# At 125 Hz: 0.3°/cycle × 125 = 37.5°/s for J4; 0.15°/cycle × 125 = 18.75°/s for J5
JOYSTICK_J4_SCALE  = 0.30   # °/cycle (matches MAX_DELTA_PER_JOINT[3])
JOYSTICK_J5_SCALE  = 0.15   # °/cycle (half of MAX_DELTA_PER_JOINT[4] for fine control)
JOYSTICK_J6_SCALE  = 0.20   # °/cycle — left X/Y → J6 roll; 0.20 × 125 Hz = 25°/s
#                              (= hardware-safe J6 rate cap; was 0.60 ≈ 75°/s, too fast)
JOYSTICK_DEAD_ZONE = 0.08   # ignore thumbstick deflections smaller than this

# ── Hand wrist orientation → J4/J5 (direct, decoupled from position) ──────────
# Robot joint degrees per degree of hand rotation from home orientation.
# 1.0 = 1:1 (tilt hand 30° → J4 moves 30°). Lower for finer control / less
# incidental wrist twist while translating. Negative = invert that joint.
#   hand PITCH (tilt up/down) → J4   |   hand YAW (turn left/right) → J5
WRIST_GAIN_J4 = -0.5   # flipped (J4 direction was inverted) + halved gain
WRIST_GAIN_J5 = 0.5    # halved gain to reduce incidental coupling

# Workspace clamps applied BEFORE IK — max total offset from home position
# Motion smoothness is controlled by MAX_DELTA_PER_JOINT, not these values
VR_MAX_DELTA_POS_MM  = 450.0  # max EEF translation from home (mm) — keeps arm in dexterous
#   mid-workspace bubble around home, away from the ~900mm reach boundary where
#   X/Y motion saturates. Re-home (release+squeeze trigger) to recenter the bubble.
VR_MAX_DELTA_ROT_DEG = 90.0   # max EEF rotation from home (deg, per axis)

# Coordinate frame rotation: Quest 3 OpenXR convention → FR5 TCP frame.
# Quest OpenXR axes: X=RIGHT  Y=UP  Z=BACK(toward user; negative=forward)
#
# Row i = which Quest axis (with sign) maps onto FR5 axis i.
# Use only ±1 and 0 — must be a proper rotation matrix (det = +1).
#
# The [AXMAP] log lines in sim_bridge tell you the live mapping.
# Confirmed by log: RIGHT→Y, UP→Z, FORWARD→X at scale VR_POSITION_SCALE.
#
# Common alternatives — uncomment the one that matches your robot orientation:
#
# Hand forward/back → robot reach (X);  hand right/left → robot lateral (Y);  up → Z
# Confirmed by 2-min capture (mapping_capture.py) + user preference.
#   Hand FORWARD → FR5 X+ (reach out)   Hand RIGHT → FR5 Y− (robot's own right)   Hand UP → FR5 Z+
VR_FRAME_ROTATION = [
    [ 1.0, 0.0,  0.0],   # FR5 X ← +Quest_X  (hand RIGHT/LEFT → robot lateral, FLIPPED)
    [ 0.0, 0.0, -1.0],   # FR5 Y ← −Quest_Z  (hand FWD/BACK   → robot reach)
    [ 0.0, 1.0,  0.0],   # FR5 Z ← +Quest_Y  (hand UP/DOWN    → robot Z)
]
# det = +1 (proper rotation): right/left flipped to match the real arm.
# (prior note) det = -1 (reflection): forward/back reversed vs the prior mapping to match the
# real arm. Safe under SE(3) retarget — X is applied to position as X@dp and to
# orientation by conjugation X@dR@Xᵀ, whose det = det(dR) regardless of det(X).
#
# [B] Y sign flip — arm goes LEFT when hand goes right (flip to fix):
#   FR5_Y ← -Quest_X
# VR_FRAME_ROTATION = [
#     [ 0.0, 0.0, -1.0],   # FR5 X ← -Quest_Z
#     [-1.0, 0.0,  0.0],   # FR5 Y ← -Quest_X  ← FLIPPED
#     [ 0.0, 1.0,  0.0],   # FR5 Z ← +Quest_Y
# ]
#
# [C] X sign flip — arm goes backward when hand goes forward (flip to fix):
# VR_FRAME_ROTATION = [
#     [ 0.0, 0.0,  1.0],   # FR5 X ← +Quest_Z  ← FLIPPED
#     [ 1.0, 0.0,  0.0],   # FR5 Y ← +Quest_X
#     [ 0.0, 1.0,  0.0],   # FR5 Z ← +Quest_Y
# ]
#
# [D] User faces robot SIDE (robot rotated 90° — right = forward, forward = left):
#   Hand RIGHT → FR5 X+   Hand UP → FR5 Z+   Hand FORWARD → FR5 -Y+
# VR_FRAME_ROTATION = [
#     [ 1.0, 0.0,  0.0],   # FR5 X ← +Quest_X
#     [ 0.0, 0.0,  1.0],   # FR5 Y ← +Quest_Z
#     [ 0.0, 1.0,  0.0],   # FR5 Z ← +Quest_Y
# ]

# ── Pose filter (One-Euro adaptive low-pass, applied before IK) ───────────────
#
# One-Euro adapts the cutoff frequency to signal speed:
#   fc = min_cutoff + beta × |speed|
# This gives jitter suppression at rest (low fc) and low latency during fast
# motion (high fc).  Latency ≈ 1 / (2π × fc) seconds at the current fc.
#
# POSITION parameters (speed in m/s):
#   pos_min_cutoff: cutoff at rest.  1 Hz → ~159 ms lag at rest (invisible
#                   when still).  2 Hz → ~80 ms (slightly less smooth).
#   pos_beta:       Hz per (m/s).   10 → fc = 6 Hz at 0.5 m/s hand speed
#                   → lag drops to ~26 ms during brisk motion.
#
# ROTATION parameters (speed in rad/s):
#   rot_min_cutoff: cutoff at rest.  1 Hz recommended.
#   rot_beta:       Hz per (rad/s). 1.0 → fc = 6 Hz at 5 rad/s wrist speed
#                   (~286°/s, a fast snap) → lag ~26 ms during fast rotation.
#
# d_cutoff:  derivative smoother cutoff.  1 Hz suits all Quest 3 use cases.
#            Raise to 5–10 Hz only if the adaptive cutoff feels sluggish to
#            respond when acceleration starts.
#
# Quick tuning guide:
#   Robot jittery at rest   → decrease pos_min_cutoff / rot_min_cutoff
#   Robot lags fast motion  → increase pos_beta / rot_beta
#   Jitter returns on move  → increase d_cutoff (derivative smoothing too slow)
FILTER_ENABLED        = True
FILTER_POS_MIN_CUTOFF = 2.5    # Hz  (was 1.0 → 159ms lag; 2.5 ≈ 64ms, less latency)
FILTER_POS_BETA       = 6.0    # Hz / (m/s)  (was 10 → jerky; gentler speed adaptation)
FILTER_ROT_MIN_CUTOFF = 2.5    # Hz  (was 1.0 → cuts wrist-orientation lag too)
FILTER_ROT_BETA       = 1.0    # Hz / (rad/s)
FILTER_D_CUTOFF       = 1.0    # Hz  (shared for pos + rot derivative smoother)

# ── RealSense cameras (ACT recording) ─────────────────────────────────────────
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30    # 30 / 60 / 90 supported; 30 is standard for training data

# Per-camera serials (run: rs-enumerate-devices, or scripts that print serials).
WRIST_CAM_SERIAL = "409122273756"   # Intel RealSense D405  (eye-in-hand)
SCENE_CAM_SERIAL = "420122071835"   # Intel RealSense D435I (eye-to-hand)
SCENE_CAM_DEPTH  = True             # capture aligned uint16 depth on the scene cam

# ── ACT episode recording ─────────────────────────────────────────────────────
# Quest 3 right-controller buttons (see quest3.py _BTN_NAMES).
RECORD_BUTTON  = "A_X"   # rising edge → start / stop an episode
SUCCESS_BUTTON = "B_Y"   # while stopped → toggle success flag on the last episode

# ── Data logging ──────────────────────────────────────────────────────────────
LOG_DIR = "./episodes"

# Write the task description here before pressing R to start recording.
# One line, plain text. Stored in the episode metadata JSON alongside the CSV.
# Example: "pick up the red block and place it in the bin"
INSTRUCTION_FILE = "./episode_instruction.txt"

# State reads are staggered: each qualifying cycle reads ONE of {joint positions,
# EEF pose, joint velocities} in rotation. LOG_STATE_DOWNSAMPLE controls how
# often a qualifying cycle occurs — e.g., N=2 means one read every 2 cycles,
# cycling through all three properties every 6 cycles (~21 Hz per property).
# This caps per-cycle overhead at one RPC call (~3 ms) and keeps the 8 ms
# ServoJ budget intact. Every CSV row still gets complete data via caching.
LOG_STATE_DOWNSAMPLE = 2
