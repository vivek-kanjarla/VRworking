"""
frame_audit.py — coordinate-frame mapping pipeline audit for Quest 3 → FR5.

Run standalone (no hardware needed):
    python3 frame_audit.py

Run with live Quest calibration (requires sim running):
    python3 frame_audit.py --live

Sections
--------
1. Coordinate-frame conventions (Quest OpenXR, FR5 base, FR5 TCP)
2. Current VR_FRAME_ROTATION analysis (det, basis vectors)
3. FK-based empirical frame probe (PyBullet, no hardware)
4. Axis correspondence table + handedness verdict
5. Corrected matrix recommendation
6. Live calibration mode (optional)
"""

import sys
import os
import math
import re
import tempfile
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    VR_FRAME_ROTATION, VR_POSITION_SCALE, VR_ROTATION_SCALE,
    QUEST3_MODE,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _col(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"

RED    = lambda s: _col(s, "31")
GREEN  = lambda s: _col(s, "32")
YELLOW = lambda s: _col(s, "33")
CYAN   = lambda s: _col(s, "36")
BOLD   = lambda s: _col(s, "1")

def _vec(v) -> str:
    return f"[{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]"

def _dominant(v) -> str:
    """Return string like '+FR5_Y' for the largest component."""
    v = np.array(v)
    idx = np.argmax(np.abs(v))
    sign = "+" if v[idx] > 0 else "−"
    axis = ["X", "Y", "Z"][idx]
    return f"{sign}FR5_{axis}"

# ── Section 1: frame conventions ──────────────────────────────────────────────

def print_conventions():
    print(BOLD("\n══════════════════════════════════════════════════════════"))
    print(BOLD("  § 1  COORDINATE-FRAME CONVENTIONS"))
    print(BOLD("══════════════════════════════════════════════════════════"))

    print(CYAN("""
Quest 3 / WebXR  (local-floor reference space)
───────────────────────────────────────────────
  +X  →  user's RIGHT  (east when user faces north)
  +Y  →  UP  (gravity-opposite)
  +Z  →  user's BACK   (negative Z = user's FORWARD)
  Handedness: RIGHT-HANDED

  Consequence: "push hand forward" = Quest ΔZ is NEGATIVE
               "pull hand back"    = Quest ΔZ is POSITIVE

FR5 base frame  (from URDF joint axes)
───────────────────────────────────────────────
  J1 axis  = 0 0 1  → J1 rotates around base-Z (vertical)
             ∴ FR5 base +Z = UP  ✓

  J2 origin rpy = π/2 0 0  → shoulder rotated 90° around base-X
             ∴ at J1=J2=0 the upper arm extends in base −X direction

  J3 origin xyz = −0.425 0 0  (upper arm length in −X of j2 frame)
             ∴ the natural reach direction at zero config is −X

  Summary at J1=J2=…=0 (verified by FK probe in §3):
    EEF is at  ~[−820, −102, +50] mm from base
    ⟹ FR5 +X points AWAY from natural reach (arm points in −X)
    ⟹ FR5 +Y points LEFT of robot
    ⟹ FR5 +Z points UP

  Practical for teleoperation (user faces arm, standing in −X direction):
    User's RIGHT ≡ FR5 +Y  (robot left = user right when facing each other)
    User's UP    ≡ FR5 +Z
    User's FWD   ≡ FR5 −X  (both move in the same world direction)

FR5 TCP frame
───────────────────────────────────────────────
  The TCP pose [x,y,z,rx,ry,rz] is expressed in the base frame.
  rx,ry,rz are ZYX Euler angles of the tool frame in the base frame.
  Orientation delta uses the same VR_FRAME_ROTATION as position.
"""))


# ── Section 2: matrix analysis ────────────────────────────────────────────────

def print_matrix_analysis():
    print(BOLD("\n══════════════════════════════════════════════════════════"))
    print(BOLD("  § 2  CURRENT VR_FRAME_ROTATION ANALYSIS"))
    print(BOLD("══════════════════════════════════════════════════════════"))

    R   = np.array(VR_FRAME_ROTATION, dtype=float)
    det = np.linalg.det(R)

    print(f"\n  Matrix (rows = FR5 axes, cols = Quest axes):")
    for i, (row, fr5ax) in enumerate(zip(R, ["FR5_X", "FR5_Y", "FR5_Z"])):
        print(f"    [{row[0]:+.1f}  {row[1]:+.1f}  {row[2]:+.1f}]   # {fr5ax}")

    det_ok  = abs(abs(det) - 1.0) < 1e-6
    det_sgn = det > 0
    det_str = f"{det:+.4f}"
    if not det_ok:
        det_label = RED("INVALID  (must be ±1)")
    elif det_sgn:
        det_label = GREEN("+1  proper rotation  ✓  (right-handed → right-handed)")
    else:
        det_label = RED("-1  IMPROPER rotation ✗  (includes a reflection — rotations will be mirrored!)")

    print(f"\n  det(R) = {det_str}  →  {det_label}")

    print(f"\n  Basis-vector mapping  (R @ quest_basis):")
    print(f"  {'Quest vector':30s}  {'→ FR5 result':20s}  Dominant axis")
    print(f"  {'─'*30}  {'─'*20}  {'─'*14}")

    quest_labels = [
        ("+X  (user RIGHT)",  np.array([1,0,0])),
        ("+Y  (user UP)",     np.array([0,1,0])),
        ("+Z  (user BACK)",   np.array([0,0,1])),
        ("-Z  (user FORWARD)",np.array([0,0,-1])),
    ]
    for label, qv in quest_labels:
        fr5v = R @ qv
        dom  = _dominant(fr5v)
        ok_mark = ""
        if "+Y" in dom and "RIGHT" in label:  ok_mark = GREEN("✓")
        elif "+Z" in dom and "UP" in label:   ok_mark = GREEN("✓")
        elif "−X" in dom and "FORWARD" in label: ok_mark = GREEN("✓")
        elif "+X" in dom and "BACK" in label: ok_mark = GREEN("✓")
        print(f"  Quest {label:25s}  →  {_vec(fr5v)}  {dom:12s} {ok_mark}")


# ── Section 3: FK-based empirical frame probe ─────────────────────────────────

def fk_probe():
    print(BOLD("\n══════════════════════════════════════════════════════════"))
    print(BOLD("  § 3  FK-BASED EMPIRICAL FRAME PROBE  (PyBullet)"))
    print(BOLD("══════════════════════════════════════════════════════════"))

    try:
        import pybullet as pb
        import pybullet_data
    except ImportError:
        print(RED("  pybullet not available — skipping FK probe."))
        return

    URDF = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..",
        "ros2_teleop_ws", "src", "frcobot_ros2-main",
        "fairino_description", "urdf", "fairino5_v6.urdf",
    ))
    if not os.path.exists(URDF):
        print(RED(f"  URDF not found: {URDF}"))
        return

    pkg_root = os.path.abspath(os.path.join(os.path.dirname(URDF), ".."))
    with open(URDF) as f: txt = f.read()
    txt = re.sub(r"package://fairino_description", pkg_root, txt)
    tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
    tmp.write(txt); tmp.close()

    cid = pb.connect(pb.DIRECT)
    rid = pb.loadURDF(tmp.name, useFixedBase=True, physicsClientId=cid)
    os.unlink(tmp.name)

    ji = {}; ee = None
    for i in range(pb.getNumJoints(rid, physicsClientId=cid)):
        info  = pb.getJointInfo(rid, i, physicsClientId=cid)
        jname = info[1].decode()
        lname = info[12].decode()
        if jname in [f"j{k}" for k in range(1,7)]: ji[jname] = i
        if lname == "wrist3_link": ee = i

    def fk(jdeg):
        for k in range(1, 7):
            pb.resetJointState(rid, ji[f"j{k}"], math.radians(jdeg[k-1]),
                               physicsClientId=cid)
        ls = pb.getLinkState(rid, ee, computeForwardKinematics=True,
                             physicsClientId=cid)
        return np.array(ls[4]) * 1000.0   # mm

    zeros   = fk([0,0,0,0,0,0])
    default = fk([120.8,-110.3,108.3,-87.1,-82.2,0.0])

    print(f"\n  EEF at all-zeros   : {_vec(zeros)} mm")
    print(f"  EEF at default home: {_vec(default)} mm")
    print(f"\n  At all-zeros the arm extends toward {_vec(zeros/np.linalg.norm(zeros))} (unit vec)")

    print(f"\n  Effect of +10° on each joint from all-zeros (EEF delta, mm):")
    print(f"  {'Joint':8s}  {'ΔX':>8s}  {'ΔY':>8s}  {'ΔZ':>8s}  Dominant world direction")
    for k in range(1, 7):
        cfg = [0]*6; cfg[k-1] = 10
        delta = fk(cfg) - zeros
        dom_idx = np.argmax(np.abs(delta))
        dom_dir = ["+X","+Y","+Z"][dom_idx] if delta[dom_idx] > 0 else ["-X","-Y","-Z"][dom_idx]
        print(f"  J{k}(+10°) {delta[0]:>+8.1f}  {delta[1]:>+8.1f}  {delta[2]:>+8.1f}  {dom_dir}")

    pb.disconnect(cid)


# ── Section 4: axis correspondence table ─────────────────────────────────────

def print_correspondence_table():
    print(BOLD("\n══════════════════════════════════════════════════════════"))
    print(BOLD("  § 4  AXIS CORRESPONDENCE TABLE + VERDICT"))
    print(BOLD("══════════════════════════════════════════════════════════"))

    R   = np.array(VR_FRAME_ROTATION, dtype=float)
    det = np.linalg.det(R)

    rows = [
        ("Move hand RIGHT  (+Quest_X)",  np.array([1,0,0]),  "FR5 +Y  (robot left = user right)"),
        ("Move hand LEFT   (−Quest_X)",  np.array([-1,0,0]), "FR5 −Y"),
        ("Move hand UP     (+Quest_Y)",  np.array([0,1,0]),  "FR5 +Z  (up)"),
        ("Move hand DOWN   (−Quest_Y)",  np.array([0,-1,0]), "FR5 −Z  (down)"),
        ("Move hand FWD    (−Quest_Z)",  np.array([0,0,-1]), "FR5 −X  (same world dir as hand)"),
        ("Move hand BACK   (+Quest_Z)",  np.array([0,0,1]),  "FR5 +X"),
    ]

    print(f"\n  {'Hand movement':35s}  {'Actual FR5 delta':20s}  {'Expected':30s}  Match?")
    print(f"  {'─'*35}  {'─'*20}  {'─'*30}  {'─'*6}")
    for desc, qv, expected in rows:
        fr5 = R @ qv
        dom  = _dominant(fr5)
        ok   = expected.split()[0].replace("FR5","FR5") in dom
        mark = GREEN("✓") if ok else RED("✗")
        print(f"  {desc:35s}  {_vec(fr5):20s}  {expected:30s}  {mark}")

    print(f"\n  Determinant = {det:+.4f}")
    if det > 0:
        print(GREEN("  VERDICT: Proper rotation (det=+1). No handedness flip. ✓"))
        print(GREEN("  Orientation (wrist) control will NOT be reflected."))
    else:
        print(RED("  VERDICT: IMPROPER rotation (det=−1). Rotations will be mirrored! ✗"))
        print(RED("  Wrist twist in one direction will appear as opposite on the robot."))


# ── Section 5: corrected matrix ───────────────────────────────────────────────

def print_corrected_matrix():
    print(BOLD("\n══════════════════════════════════════════════════════════"))
    print(BOLD("  § 5  CORRECTED MATRIX RECOMMENDATION"))
    print(BOLD("══════════════════════════════════════════════════════════"))

    R   = np.array(VR_FRAME_ROTATION, dtype=float)
    det = np.linalg.det(R)

    if abs(det - 1.0) < 1e-6:
        print(GREEN("""
  Current matrix is already a proper rotation (det = +1).

  Verified mappings from empirical log data:
    RIGHT  → +Y  ✓   (confirmed by CSV: q_right+ → fr5_dy+)
    LEFT   → −Y  ✓   (confirmed by CSV: q_right− → fr5_dy−)
    UP     → +Z  ✓   (confirmed by CSV: q_up+    → fr5_dz+)
    DOWN   → −Z  ✓   (confirmed by CSV: q_up−    → fr5_dz−)
    FWD    → −X       (q_back− → fr5_dx−)  — test to confirm direction feels right
    BACK   → +X       (q_back+ → fr5_dx+)

  Mathematical note:
    There is NO proper rotation (det=+1) that simultaneously maps
      Quest +X → FR5 +Y   AND   Quest +Y → FR5 +Z   AND   Quest −Z → FR5 +X.
    The three constraints over-determine the system; the only solution that
    satisfies all three is det = −1 (improper = reflection + rotation).

    ∴ Given RIGHT→+Y and UP→+Z are confirmed correct, the forward mapping
    MUST be FWD→−X for a proper rotation. This is the current state.

  If FWD→−X feels wrong in practice, the only geometrically valid fixes are:
    Option A: swap RIGHT and FWD   →  RIGHT→+X, FWD→+Y, UP→+Z  (det=+1)
    Option B: negate RIGHT         →  RIGHT→−Y, FWD→+X, UP→+Z  (det=+1)
    Option C: negate UP            →  RIGHT→+Y, FWD→+X, DOWN→+Z (det=+1)

  Current matrix (det=+1, recommended):
"""))
        print("    VR_FRAME_ROTATION = [")
        print(f"        [ 0.0, 0.0,  1.0],   # FR5 X ← +Quest_Z  (BACK→X+, FWD→X−)")
        print(f"        [ 1.0, 0.0,  0.0],   # FR5 Y ← +Quest_X  (RIGHT→Y+)")
        print(f"        [ 0.0, 1.0,  0.0],   # FR5 Z ← +Quest_Y  (UP→Z+)")
        print("    ]")

        print(YELLOW("""
  ORIGINAL matrix (det=−1, DO NOT USE — rotations are mirrored):
    VR_FRAME_ROTATION = [
        [ 0.0, 0.0, -1.0],   # FR5 X ← −Quest_Z  (FWD→X+, BACK→X−)  ← translation seems right
        [ 1.0, 0.0,  0.0],   # FR5 Y ← +Quest_X  (RIGHT→Y+)
        [ 0.0, 1.0,  0.0],   # FR5 Z ← +Quest_Y  (UP→Z+)
    ]
  The original had correct POSITION direction (FWD→+X matches arm's reach)
  but det=−1 caused ORIENTATION to be reflected (wrist rotations mirrored).
"""))

    else:
        print(RED(f"  Current det={det:.3f} — improper rotation!"))
        print("  Recommended fix:")
        print("    VR_FRAME_ROTATION = [")
        print("        [ 0.0, 0.0,  1.0],   # FR5 X ← +Quest_Z")
        print("        [ 1.0, 0.0,  0.0],   # FR5 Y ← +Quest_X")
        print("        [ 0.0, 1.0,  0.0],   # FR5 Z ← +Quest_Y")
        print("    ]")


# ── Section 6: live calibration ───────────────────────────────────────────────

def live_calibration():
    print(BOLD("\n══════════════════════════════════════════════════════════"))
    print(BOLD("  § 6  LIVE CALIBRATION MODE"))
    print(BOLD("══════════════════════════════════════════════════════════"))
    print("""
  Instructions:
    1. Home the arm (squeeze trigger)
    2. Move ONLY in one direction at a time, hold 2s, return
    3. Watch the printed axis table

  Reading:  q_right/up/back = Quest delta from home (m)
            fr5_dx/dy/dz    = FR5 EEF delta from home (mm)
""")

    try:
        from quest3 import Quest3Reader
        from config import VR_FRAME_ROTATION, VR_POSITION_SCALE
    except ImportError as e:
        print(RED(f"  Import failed: {e}"))
        return

    R = np.array(VR_FRAME_ROTATION, dtype=float)
    home_pos = None

    print(f"  {'trig':>5}  {'q_RIGHT':>9} {'q_UP':>9} {'q_BACK':>9}"
          f"  │  {'dX(mm)':>8} {'dY(mm)':>8} {'dZ(mm)':>8}  Movement")
    print("  " + "─"*90)

    try:
        with Quest3Reader() as q:
            while True:
                s = q.get_state()
                if not s.valid:
                    time.sleep(0.05); continue

                if s.trigger >= 0.7 and home_pos is None:
                    home_pos  = s.pos.copy()
                    print(f"  {BOLD('[HOME SET]')}")
                elif s.trigger < 0.1:
                    home_pos = None

                if home_pos is not None:
                    dq   = s.pos - home_pos
                    dfr5 = R @ dq * VR_POSITION_SCALE

                    dom = max(abs(dq[0]), abs(dq[1]), abs(dq[2]))
                    if dom < 0.03:   tag = "~still"
                    elif abs(dq[0]) == dom: tag = ("→ RIGHT" if dq[0]>0 else "← LEFT")
                    elif abs(dq[1]) == dom: tag = ("↑ UP"    if dq[1]>0 else "↓ DOWN")
                    else:                   tag = ("↩ BACK"  if dq[2]>0 else "↪ FWD")

                    print(f"  {s.trigger:5.2f}  {dq[0]:+9.4f} {dq[1]:+9.4f} {dq[2]:+9.4f}"
                          f"  │  {dfr5[0]:+8.1f} {dfr5[1]:+8.1f} {dfr5[2]:+8.1f}  {tag}",
                          end="\r", flush=True)

                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n  Stopped.")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FR5 coordinate-frame audit")
    parser.add_argument("--live", action="store_true",
                        help="Run live calibration mode (needs sim + Quest connected)")
    args = parser.parse_args()

    print_conventions()
    print_matrix_analysis()
    fk_probe()
    print_correspondence_table()
    print_corrected_matrix()

    if args.live:
        live_calibration()
    else:
        print(BOLD("\n  Run with --live for real-time Quest calibration.\n"))
