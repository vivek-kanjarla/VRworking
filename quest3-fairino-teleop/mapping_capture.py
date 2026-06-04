"""
mapping_capture.py — capture VR controller vs FR5 EEF for a fixed window,
then report the empirical axis correspondence.

Reads /tmp/axis_log.csv (written by the running sim) for DURATION seconds,
then prints which Quest axis drives which FR5 axis, with sign and gain.

Usage:
    python3 mapping_capture.py [duration_seconds]
"""
import sys, time, os
import numpy as np

CSV = "/tmp/axis_log.csv"
DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0

def read_rows(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) < 13 or p[0] == "t":
                    continue
                try:
                    rows.append([float(x) for x in p[:13]])
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return rows

print(f"Capturing for {DURATION:.0f} s — move the controller now.")
print("Move in clean single-axis motions: right, left, forward, back, up, down.\n")

# Record the starting line count so we only analyze NEW data
start_n = len(read_rows(CSV))
t_start = time.monotonic()

# Live progress
while time.monotonic() - t_start < DURATION:
    elapsed = time.monotonic() - t_start
    n = len(read_rows(CSV)) - start_n
    print(f"  {elapsed:5.0f}s / {DURATION:.0f}s   samples captured: {max(0,n)}    ",
          end="\r", flush=True)
    time.sleep(1.0)

print("\n\nAnalyzing...\n")

rows = read_rows(CSV)[start_n:]
if len(rows) < 10:
    print("Not enough data — was the arm homed (trigger held)?")
    sys.exit(1)

arr = np.array(rows)
# Columns: t,trig,grip, q_right,q_up,q_back, qx,qy,qz,qw, dx,dy,dz
q = arr[:, 3:6]    # [right, up, back]  (metres)
d = arr[:, 10:13]  # [dx, dy, dz]       (mm)

# Work in deltas between consecutive samples to capture motion, not absolute pose
dq = np.diff(q, axis=0)   # quest velocity
dd = np.diff(d, axis=0)   # fr5 velocity

# Keep only samples where SOMETHING moved (reject noise/stationary)
mag = np.linalg.norm(dq, axis=1)
moving = mag > 0.002   # 2 mm/sample threshold
dq_m = dq[moving]
dd_m = dd[moving]

print(f"Total samples: {len(rows)}   moving samples: {len(dq_m)}\n")

if len(dq_m) < 10:
    print("Controller barely moved — try again with larger motions.")
    sys.exit(1)

# ── Least-squares fit: dd = M @ dq  →  M = dd^T dq (dq^T dq)^-1 ────────────────
# Solve M (3x3) mapping quest velocity → fr5 velocity
# dd_m: (N,3), dq_m: (N,3).  Want M such that dd_m ≈ dq_m @ M.T
M, residuals, rank, sv = np.linalg.lstsq(dq_m, dd_m, rcond=None)
# M is (3,3): row i = quest axis i, col j = fr5 axis j
# So dd[:,j] = sum_i dq[:,i] * M[i,j]

quest_names = ["RIGHT", "UP   ", "BACK "]
fr5_names   = ["X", "Y", "Z"]

print("EMPIRICAL MAPPING  (Quest motion → FR5 EEF motion, mm per metre):")
print(f"  {'Quest axis':12s}  →  {'FR5_X':>8s}  {'FR5_Y':>8s}  {'FR5_Z':>8s}   Dominant")
print("  " + "─"*60)
for i, qn in enumerate(quest_names):
    gains = M[i] * 1000.0 / 1.0   # mm per metre (d is mm, q is m, M already mm/m)
    dom_j = int(np.argmax(np.abs(M[i])))
    sign  = "+" if M[i, dom_j] > 0 else "−"
    print(f"  Quest {qn}  →  {M[i,0]*1000:8.0f}  {M[i,1]*1000:8.0f}  "
          f"{M[i,2]*1000:8.0f}   {sign}FR5_{fr5_names[dom_j]}")

print()
print("INTERPRETATION (what the robot does when you move your hand):")
hand_dirs = [
    ("RIGHT", 0, +1), ("LEFT", 0, -1),
    ("UP",    1, +1), ("DOWN", 1, -1),
    ("FORWARD (away)", 2, -1), ("BACKWARD (toward)", 2, +1),
]
for label, qi, qsign in hand_dirs:
    fr5_motion = M[qi] * qsign
    dom_j = int(np.argmax(np.abs(fr5_motion)))
    sign  = "+" if fr5_motion[dom_j] > 0 else "−"
    print(f"  Hand {label:20s} → robot moves {sign}{fr5_names[dom_j]}")

print()
print("FR5 axis reference (in RViz):")
print("  +X = robot reaches FORWARD/out (red arrow)")
print("  +Y = robot moves to its LEFT   (green arrow)")
print("  +Z = robot moves UP            (blue arrow)")
