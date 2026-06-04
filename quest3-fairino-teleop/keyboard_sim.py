"""
keyboard_sim.py — drive the FR5 sim with keyboard input (no VR hardware needed).

Starts:  robot_state_publisher  →  RViz2  →  PyBullet IK loop
         Joint states published to /joint_states → RViz animates the FR5 model.

Controls:
  W / S        EEF +X / -X  (forward / back from base)
  A / D        EEF +Y / -Y  (left / right)
  R / F        EEF +Z / -Z  (up / down)
  I / K        EEF pitch +  / -
  J / L        EEF yaw  +  / -
  U / O        EEF roll +  / -
  G            toggle gripper open / close
  H            reset EEF to home position
  Q / Ctrl-C   quit

Usage:
  python3 keyboard_sim.py
  python3 keyboard_sim.py --no-rviz          # PyBullet GUI only
  python3 keyboard_sim.py --no-gui           # RViz only, no PyBullet window
  python3 keyboard_sim.py --no-rviz --no-gui # headless, prints joint states only

Dependencies:  pybullet  (pip install pybullet)
               pynput    (pip install pynput)   ← already in requirements.txt
               ROS2 Humble  (optional, for RViz)
"""

import argparse
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import yaml

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROS2_SETUP = "/opt/ros/humble/setup.bash"
_WS_SETUP   = os.path.expanduser("~/ros2_teleop_ws/install/setup.bash")
_URDF_PATH  = os.path.abspath(os.path.join(
    _HERE, "..", "ros2_teleop_ws", "src", "frcobot_ros2-main",
    "fairino_description", "urdf", "fairino5_v6.urdf",
))

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sim_bridge import SimRobot
from vr_sim_launch import _RVIZ_CONFIG as _RVIZ_CONFIG_CONTENT
from config import FR5_JOINT_LIMITS

# ── step sizes per keypress ───────────────────────────────────────────────────
_STEP_MM  = 2.0    # mm per translation key event
_STEP_DEG = 1.5    # degrees per rotation key event

# Home joint angles — SRDF pos1 converted to degrees
_HOME_DEG = [120.8, -110.3, 108.3, -87.1, -82.2, 0.0]

# ── key → EEF delta mapping ───────────────────────────────────────────────────
# Each entry: (axis_index, sign)
#   axes 0-2  → position  [x, y, z] in mm
#   axes 3-5  → rotation  [rx, ry, rz] in degrees
_KEY_MAP = {
    "w": (0, +1), "s": (0, -1),   # X forward/back
    "a": (1, +1), "d": (1, -1),   # Y left/right
    "r": (2, +1), "f": (2, -1),   # Z up/down
    "i": (3, +1), "k": (3, -1),   # pitch
    "j": (4, +1), "l": (4, -1),   # yaw
    "u": (5, +1), "o": (5, -1),   # roll
}


# ── keyboard listener (non-blocking, pynput) ──────────────────────────────────

class KeyboardState:
    def __init__(self):
        self._held:  set[str]  = set()
        self._events: list[str] = []   # single-press events
        self._lock = threading.Lock()

    def on_press(self, key):
        try:
            ch = key.char.lower() if hasattr(key, "char") and key.char else None
        except AttributeError:
            ch = None
        if ch:
            with self._lock:
                self._held.add(ch)
                if ch in ("h", "g", "q"):
                    self._events.append(ch)

    def on_release(self, key):
        try:
            ch = key.char.lower() if hasattr(key, "char") and key.char else None
        except AttributeError:
            ch = None
        if ch:
            with self._lock:
                self._held.discard(ch)

    def get_held(self) -> set[str]:
        with self._lock:
            return set(self._held)

    def pop_events(self) -> list[str]:
        with self._lock:
            evts = self._events[:]
            self._events.clear()
        return evts


# ── ROS2 optional /joint_states publisher ─────────────────────────────────────

class _JointPub:
    def __init__(self):
        self._ok = False
        self._pub = None
        self._node = None
        try:
            import rclpy
            from sensor_msgs.msg import JointState

            if not rclpy.ok():
                rclpy.init(args=None)
            self._rclpy      = rclpy
            self._JointState = JointState
            self._node       = rclpy.create_node("fr5_keyboard_sim")
            self._pub        = self._node.create_publisher(JointState, "/joint_states", 10)

            def _spin():
                try:
                    rclpy.spin(self._node)
                except Exception:
                    pass

            threading.Thread(target=_spin, daemon=True, name="ros2-spin").start()
            self._ok = True
            print("[SIM] ROS2 /joint_states publisher active")
        except Exception as exc:
            print(f"[SIM] rclpy unavailable ({exc}) — no RViz output")

    def publish(self, joints_deg: list[float]):
        if not self._ok:
            return
        try:
            msg = self._JointState()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.name     = ["j1", "j2", "j3", "j4", "j5", "j6"]
            msg.position = [math.radians(j) for j in joints_deg]
            self._pub.publish(msg)
        except Exception:
            pass

    def close(self):
        if not self._ok:
            return
        try:
            self._node.destroy_node()
            self._rclpy.shutdown()
        except Exception:
            pass


# ── subprocess helpers (RSP + RViz) ───────────────────────────────────────────

_procs: list[subprocess.Popen] = []
_tmpfiles: list[str] = []

def _ros2_cmd(cmd: str) -> str:
    parts = [f"source {_ROS2_SETUP}"]
    if os.path.exists(_WS_SETUP):
        parts.append(f"source {_WS_SETUP}")
    parts.append(cmd)
    return " && ".join(parts)

def _spawn(label: str, cmd: str, ros2: bool = True) -> subprocess.Popen:
    shell_cmd = _ros2_cmd(cmd) if ros2 else cmd
    print(f"[LAUNCH] {label}")
    p = subprocess.Popen(shell_cmd, shell=True, executable="/bin/bash")
    _procs.append(p)
    return p

def _cleanup():
    for p in _procs:
        if p.poll() is None:
            try: p.send_signal(signal.SIGINT)
            except ProcessLookupError: pass
    time.sleep(0.8)
    for p in _procs:
        if p.poll() is None:
            try: p.terminate()
            except ProcessLookupError: pass
    for f in _tmpfiles:
        try: os.unlink(f)
        except OSError: pass

_stopping = False
def _sigint(sig, frame):
    global _stopping
    if _stopping: return
    _stopping = True
    print("\n[SIM] Shutting down.")
    _cleanup()
    sys.exit(0)

def _start_rsp():
    with open(_URDF_PATH) as fh:
        urdf = fh.read()
    params = {"robot_state_publisher": {"ros__parameters": {"robot_description": urdf}}}
    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
    yaml.dump(params, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    _tmpfiles.append(tmp.name)
    _spawn("robot_state_publisher",
           f"ros2 run robot_state_publisher robot_state_publisher "
           f"--ros-args --params-file {tmp.name}")
    time.sleep(2.0)   # wait for RSP to bind

def _start_rviz():
    cfg = tempfile.NamedTemporaryFile(suffix=".rviz", delete=False, mode="w")
    cfg.write(_RVIZ_CONFIG_CONTENT)
    cfg.close()
    _tmpfiles.append(cfg.name)
    _spawn("rviz2", f"rviz2 -d {cfg.name}")
    time.sleep(2.0)


# ── main simulation loop ───────────────────────────────────────────────────────

def run_sim(gui: bool):
    print(f"[SIM] Loading URDF: {_URDF_PATH}")
    robot = SimRobot(_URDF_PATH, gui=gui)
    robot.set_joints(_HOME_DEG)

    ros2 = _JointPub()
    ros2.publish(_HOME_DEG)

    try:
        from pynput import keyboard as _kb
    except ImportError:
        print("[SIM] pynput not installed — run: pip install pynput")
        robot.close(); ros2.close(); return

    kb = KeyboardState()
    listener = _kb.Listener(on_press=kb.on_press, on_release=kb.on_release)
    listener.start()

    current_joints = list(_HOME_DEG)
    home_eef       = robot.get_fk(current_joints)
    current_eef    = list(home_eef)
    gripper_open   = True

    _print_controls()
    _print_eef(current_eef, current_joints)

    dt      = 1.0 / 50   # 50 Hz keyboard poll
    deadline = time.monotonic()

    while not _stopping:
        deadline += dt
        held   = kb.get_held()
        events = kb.pop_events()

        # ── single-press events ───────────────────────────────────────────────
        for ev in events:
            if ev == "q":
                _sigint(None, None)
            elif ev == "h":
                current_eef    = list(home_eef)
                current_joints = list(_HOME_DEG)
                robot.set_joints(current_joints)
                ros2.publish(current_joints)
                print("\r[SIM] Reset to home" + " " * 40)
                _print_eef(current_eef, current_joints)
            elif ev == "g":
                gripper_open = not gripper_open
                print(f"\r[SIM] Gripper {'OPEN' if gripper_open else 'CLOSED'}" + " " * 30)

        # ── held keys → EEF delta ─────────────────────────────────────────────
        delta = [0.0] * 6
        for ch in held:
            if ch in _KEY_MAP:
                axis, sign = _KEY_MAP[ch]
                step = _STEP_MM if axis < 3 else _STEP_DEG
                delta[axis] += sign * step

        if any(d != 0 for d in delta):
            target_eef = [current_eef[i] + delta[i] for i in range(6)]

            try:
                target_joints = robot.get_inverse_kin(target_eef, current_joints)
            except IOError as exc:
                print(f"\r[SIM] IK failed: {exc}" + " "*20)
                _sleep_until(deadline)
                continue

            # Joint-limit check (rate-limit check is ServoJ-specific, skip in sim)
            blocked = False
            for i, (j, (lo, hi)) in enumerate(zip(target_joints, FR5_JOINT_LIMITS)):
                if j < lo or j > hi:
                    print(f"\r[SIM] j{i+1}={j:.1f}° out of limits [{lo},{hi}]°" + " "*10)
                    blocked = True
                    break
            if blocked:
                _sleep_until(deadline)
                continue

            # Self-collision check
            robot.set_joints(target_joints)
            if robot.check_collisions() > 0:
                print(f"\r[SIM] self-collision blocked" + " "*30)
                robot.set_joints(current_joints)
                _sleep_until(deadline)
                continue

            current_joints = target_joints
            current_eef    = target_eef
            ros2.publish(current_joints)
            _print_eef(current_eef, current_joints)

        _sleep_until(deadline)

    listener.stop()
    robot.close()
    ros2.close()


def _sleep_until(t: float):
    r = t - time.monotonic()
    if r > 0:
        time.sleep(r)


def _print_controls():
    print("""
┌─────────────────────────────────────────────────┐
│  FR5 Keyboard Sim                               │
│                                                 │
│  W/S   EEF +X/-X (forward/back)                │
│  A/D   EEF +Y/-Y (left/right)                  │
│  R/F   EEF +Z/-Z (up/down)                     │
│  I/K   pitch  +/-     U/O  roll +/-            │
│  J/L   yaw    +/-                               │
│  H     reset to home   G  toggle gripper        │
│  Q     quit                                     │
└─────────────────────────────────────────────────┘""")


def _print_eef(eef: list, joints: list):
    pos = eef[:3]; rot = eef[3:]
    j   = joints
    print(
        f"\r  EEF  x={pos[0]:+7.1f} y={pos[1]:+7.1f} z={pos[2]:+7.1f} mm  "
        f"rx={rot[0]:+6.1f} ry={rot[1]:+6.1f} rz={rot[2]:+6.1f}°  "
        f"J=[{j[0]:+6.1f} {j[1]:+6.1f} {j[2]:+6.1f} {j[3]:+6.1f} {j[4]:+6.1f} {j[5]:+6.1f}]",
        end="", flush=True,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    signal.signal(signal.SIGINT,  _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    parser = argparse.ArgumentParser(description="FR5 keyboard-driven simulation")
    parser.add_argument("--no-rviz", action="store_true", help="skip RViz2")
    parser.add_argument("--no-rsp",  action="store_true", help="skip robot_state_publisher")
    parser.add_argument("--no-gui",  action="store_true", help="no PyBullet GUI window")
    args = parser.parse_args()

    if not os.path.exists(_URDF_PATH):
        print(f"[SIM] ERROR: URDF not found: {_URDF_PATH}", file=sys.stderr)
        sys.exit(1)

    if not args.no_rsp and not args.no_rviz:
        _start_rsp()
    if not args.no_rviz:
        _start_rviz()

    run_sim(gui=not args.no_gui)
    _cleanup()


if __name__ == "__main__":
    main()
