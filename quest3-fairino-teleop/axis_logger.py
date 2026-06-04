"""
axis_logger.py — real-time axis-mapping log for Quest 3 → FR5 teleoperation.

Two outputs run simultaneously:
  1. /tmp/axis_log.csv  — one row per call, easy to tail or load in pandas
  2. Terminal [AXES] block — human-readable side-by-side every PRINT_HZ seconds

CSV columns:
  t, trig, grip,
  q_right, q_up, q_back,          # Quest controller position (metres, home-relative)
  q_qx, q_qy, q_qz, q_qw,        # Quest orientation (absolute, filtered)
  fr5_dx, fr5_dy, fr5_dz,         # FR5 EEF delta from home (mm)
  fr5_x, fr5_y, fr5_z,            # FR5 EEF absolute position (mm)
  fr5_rx, fr5_ry, fr5_rz,         # FR5 EEF orientation from home (deg)
  j1, j2, j3, j4, j5, j6         # Joint angles (deg)

Usage in sim_bridge.py:
    from axis_logger import AxisLogger
    self._axis_log = AxisLogger()          # create once in __init__
    # in run() loop when homed:
    self._axis_log.update(
        t=t0, trigger=state.trigger, grip=state.grip,
        quest_pos_raw=state.pos, quest_pos_filt=pos_filt, quest_quat=quat_filt,
        quest_home_pos=self._quest_home_pos, quest_home_quat=self._quest_home_quat,
        fr5_home_eef=self._home_eef, fr5_current_eef=last_target_eef,
        fr5_joints=self._current_joints,
    )
"""

from __future__ import annotations
import csv
import math
import os
import time

import numpy as np

CSV_PATH   = "/tmp/axis_log.csv"
PRINT_HZ   = 2.0          # terminal print frequency when homed (Hz)
CSV_EVERY  = 4            # write one CSV row every N update() calls (~31 Hz at 125 Hz loop)

_CSV_HEADER = [
    "t", "trigger", "grip",
    "q_right_m", "q_up_m", "q_back_m",
    "q_qx", "q_qy", "q_qz", "q_qw",
    "fr5_dx_mm", "fr5_dy_mm", "fr5_dz_mm",
    "fr5_x_mm",  "fr5_y_mm",  "fr5_z_mm",
    "fr5_rx_deg","fr5_ry_deg","fr5_rz_deg",
    "j1","j2","j3","j4","j5","j6",
    "joy_x","joy_y","joy_j4_target","joy_j5_target",
]

_DIR_LABEL = {
    "fr5_dx_mm": ("X", "+fwd", "−fwd"),
    "fr5_dy_mm": ("Y", "+right", "−right"),
    "fr5_dz_mm": ("Z", "+up", "−down"),
}


def _quat_to_euler_deg(q: np.ndarray):
    """ZYX Euler angles in degrees from unit quaternion [qx,qy,qz,qw]."""
    qx, qy, qz, qw = q
    roll  = math.degrees(math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
    sinp  = max(-1.0, min(1.0, 2*(qw*qy - qz*qx)))
    pitch = math.degrees(math.asin(sinp))
    yaw   = math.degrees(math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
    return roll, pitch, yaw


class AxisLogger:
    def __init__(self, csv_path: str = CSV_PATH, print_hz: float = PRINT_HZ):
        self._csv_path  = csv_path
        self._print_dt  = 1.0 / print_hz
        self._last_print = 0.0
        self._call_n    = 0
        self._t0        = time.monotonic()

        # Open CSV and write header (truncates previous file)
        self._fh  = open(csv_path, "w", newline="", buffering=1)
        self._csv = csv.writer(self._fh)
        self._csv.writerow(_CSV_HEADER)
        print(f"[AXLOG] Logging to {csv_path}  —  tail with: tail -f {csv_path}", flush=True)

    # ── public API ────────────────────────────────────────────────────────────

    def update(
        self,
        t:               float,
        trigger:         float,
        grip:            float,
        quest_pos_filt:  np.ndarray,
        quest_quat:      np.ndarray,
        quest_home_pos:  np.ndarray | None,
        quest_home_quat: np.ndarray | None,
        fr5_home_eef:    list[float] | None,
        fr5_current_eef: list[float] | None,
        fr5_joints:      list[float],
        joystick_x:      float = 0.0,   # thumbstick X → J5
        joystick_y:      float = 0.0,   # thumbstick Y → J4
        joy_j4:          float = 0.0,   # current J4 target from thumbstick
        joy_j5:          float = 0.0,   # current J5 target from thumbstick
    ) -> None:
        self._call_n += 1

        # ── deltas from home ──────────────────────────────────────────────────
        if quest_home_pos is not None:
            dq = quest_pos_filt - quest_home_pos   # Quest frame delta (m)
        else:
            dq = np.zeros(3)

        if fr5_home_eef is not None and fr5_current_eef is not None:
            home = np.array(fr5_home_eef[:6])
            curr = np.array(fr5_current_eef[:6])
            df   = curr - home           # FR5 EEF delta [dx,dy,dz,drx,dry,drz]
        else:
            df   = np.zeros(6)

        eef_abs = list(fr5_current_eef[:6]) if fr5_current_eef else [0]*6
        joints  = list(fr5_joints[:6]) if len(fr5_joints) >= 6 else fr5_joints + [0]*(6-len(fr5_joints))

        # (joystick data passed directly)

        # ── CSV write (throttled) ─────────────────────────────────────────────
        if self._call_n % CSV_EVERY == 0:
            row = [
                f"{t - self._t0:.3f}",
                f"{trigger:.3f}", f"{grip:.3f}",
                f"{dq[0]:+.4f}", f"{dq[1]:+.4f}", f"{dq[2]:+.4f}",
                f"{quest_quat[0]:+.4f}", f"{quest_quat[1]:+.4f}",
                f"{quest_quat[2]:+.4f}", f"{quest_quat[3]:+.4f}",
                f"{df[0]:+.1f}", f"{df[1]:+.1f}", f"{df[2]:+.1f}",
                f"{eef_abs[0]:+.1f}", f"{eef_abs[1]:+.1f}", f"{eef_abs[2]:+.1f}",
                f"{((df[3]+180)%360-180):+.2f}", f"{((df[4]+180)%360-180):+.2f}", f"{((df[5]+180)%360-180):+.2f}",
            ] + [f"{j:+.1f}" for j in joints] + [
                f"{joystick_x:+.3f}", f"{joystick_y:+.3f}",
                f"{joy_j4:+.1f}", f"{joy_j5:+.1f}",
            ]
            self._csv.writerow(row)

        if t - self._last_print >= self._print_dt:
            self._last_print = t
            self._print(trigger, grip, dq, quest_quat, df, eef_abs, joints,
                        joystick_x, joystick_y, joy_j4, joy_j5)

    # ── formatted terminal block ──────────────────────────────────────────────

    def _print(
        self,
        trigger:    float,
        grip:       float,
        dq:         np.ndarray,
        quat:       np.ndarray,
        df:         np.ndarray,
        eef_abs:    list[float],
        joints:     list[float],
        joystick_x: float = 0.0,
        joystick_y: float = 0.0,
        joy_j4:     float = 0.0,
        joy_j5:     float = 0.0,
    ) -> None:
        def _bar(v: float, scale: float = 300.0, width: int = 20) -> str:
            filled = int(abs(v) / scale * width)
            filled = min(filled, width)
            if v >= 0:
                return " " * width + "│" + "█" * filled + " " * (width - filled)
            else:
                return " " * (width - filled) + "█" * filled + "│" + " " * width

        # Euler angles from quaternion (ZYX, degrees) for display
        qx, qy, qz, qw = quat
        sinr_cosp = 2*(qw*qx + qy*qz)
        cosr_cosp = 1 - 2*(qx*qx + qy*qy)
        roll  = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
        sinp  = 2*(qw*qy - qz*qx)
        sinp  = max(-1.0, min(1.0, sinp))
        pitch = math.degrees(math.asin(sinp))
        siny  = 2*(qw*qz + qx*qy)
        cosy  = 1 - 2*(qy*qy + qz*qz)
        yaw   = math.degrees(math.atan2(siny, cosy))

        j = joints
        jy_bar = "█" * min(int(abs(joystick_y) * 10), 10)
        jx_bar = "█" * min(int(abs(joystick_x) * 10), 10)

        print(
            f"\n[AXES]───── RIGHT CONTROLLER (position) ─────────────── FR5 EEF ──────────────\n"
            f"  RIGHT  {dq[0]:+7.3f}m  {_bar(dq[0],.4)}  │  dY {df[1]:+7.1f}mm  {_bar(df[1])}\n"
            f"  UP     {dq[1]:+7.3f}m  {_bar(dq[1],.4)}  │  dZ {df[2]:+7.1f}mm  {_bar(df[2])}\n"
            f"  FWD    {dq[2]:+7.3f}m  {_bar(dq[2],.4)}  │  dX {df[0]:+7.1f}mm  {_bar(df[0])}\n"
            f"  trig={trigger:.2f}  grip={grip:.2f}"
            f"   EEF: [{eef_abs[0]:+.0f},{eef_abs[1]:+.0f},{eef_abs[2]:+.0f}]mm\n"
            f"\n[AXES]───── THUMBSTICK (wrist J4/J5) ────────────────────────────────────────\n"
            f"  Stick UP/DN  {joystick_y:+6.2f}  [{jy_bar:<10s}]  │  J4={joy_j4:+.1f}°  (target)\n"
            f"  Stick LT/RT  {joystick_x:+6.2f}  [{jx_bar:<10s}]  │  J5={joy_j5:+.1f}°  (target)\n"
            f"\n[AXES]───── JOINTS ───────────────────────────────────────────────────────────\n"
            f"  J1={j[0]:+.1f}°  J2={j[1]:+.1f}°  J3={j[2]:+.1f}°"
            f"   J4={j[3]:+.1f}°  J5={j[4]:+.1f}°  J6={j[5]:+.1f}°",
            flush=True,
        )

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
