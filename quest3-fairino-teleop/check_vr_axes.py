"""
check_vr_axes.py — print live controller delta so you can verify axis mapping.

Run:  python3 check_vr_axes.py
Move the right controller slowly in each cardinal direction after homing.
The printed FR5 delta columns should match your intended axis mapping.
Ctrl-C to quit.
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quest3 import Quest3Reader, ControllerState
from config import VR_FRAME_ROTATION, VR_POSITION_SCALE, VR_ROTATION_SCALE

_R = np.array(VR_FRAME_ROTATION, dtype=np.float64)

home_pos  = None
home_quat = None

def _quat_inv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

def _quat_mul(a, b):
    ax,ay,az,aw = a; bx,by,bz,bw = b
    return np.array([aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz])

with Quest3Reader() as q:
    print("Hold trigger ≥ 0.7 for 0.5 s to set home, then move controller.\n")
    print(f"{'Trigger':>8}  {'Quest ΔX':>9} {'Quest ΔY':>9} {'Quest ΔZ':>9}  "
          f"{'FR5 ΔX(mm)':>11} {'FR5 ΔY(mm)':>11} {'FR5 ΔZ(mm)':>11}")

    homing = False
    t_home = 0.0
    homed  = False

    while True:
        s = q.get_state()
        if not s.valid:
            time.sleep(0.05); continue

        t = time.monotonic()
        if s.trigger >= 0.7:
            if not homing: homing = True; t_home = t
            elif t - t_home >= 0.5 and not homed:
                home_pos = s.pos.copy(); home_quat = s.quat.copy()
                homed = True; print("HOME LOCKED\n")
        else:
            homing = False
            if s.trigger < 0.1: homed = False

        if homed:
            dq   = s.pos - home_pos
            dfr5 = _R @ dq * VR_POSITION_SCALE
            print(f"{s.trigger:8.2f}  {dq[0]:+9.4f} {dq[1]:+9.4f} {dq[2]:+9.4f}  "
                  f"{dfr5[0]:+11.1f} {dfr5[1]:+11.1f} {dfr5[2]:+11.1f}", end="\r")
        time.sleep(0.05)
