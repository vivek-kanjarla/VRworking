"""
watch_axes.py — live terminal display of Quest controller → FR5 axis mapping.

Reads /tmp/axis_log.csv written by the running sim and shows:
  - Which Quest axis is active (RIGHT/LEFT/UP/DOWN/FWD/BACK)
  - Which FR5 axis responds and in which direction
  - A bar chart for intuitive reading

Run in a second terminal while the sim is running:
    python3 ~/Quest3-Fairino/watch_axes.py
"""
import os, sys, time, math

CSV = "/tmp/axis_log.csv"
DEAD = 0.015   # m — ignore deltas smaller than this (noise floor)

# ANSI
def bold(s):   return f"\033[1m{s}\033[0m"
def green(s):  return f"\033[32m{s}\033[0m"
def red(s):    return f"\033[31m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def cyan(s):   return f"\033[36m{s}\033[0m"
def grey(s):   return f"\033[90m{s}\033[0m"
def clear():   print("\033[2J\033[H", end="")

def bar(val: float, maxv: float = 300.0, width: int = 18) -> str:
    filled = min(int(abs(val) / maxv * width), width)
    if val >= 0:
        return grey("·"*width) + green("█"*filled) + grey("·"*(width-filled))
    else:
        return grey("·"*(width-filled)) + red("█"*filled) + grey("·"*width)

def dir_label(val: float, pos_name: str, neg_name: str) -> str:
    if abs(val) < DEAD * 600:   return grey("─ still ─")
    return green(f"▶ {pos_name}") if val > 0 else red(f"◀ {neg_name}")

def read_last_line(path: str):
    """Return the last non-empty line of a file, or None."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0: return None
            buf = b""
            pos = size - 1
            while pos >= 0:
                f.seek(pos)
                ch = f.read(1)
                if ch == b"\n" and buf:
                    break
                buf = ch + buf
                pos -= 1
            line = buf.decode(errors="ignore").strip()
            return line if line else None
    except Exception:
        return None

def parse(line: str):
    try:
        parts = line.split(",")
        if len(parts) < 25: return None
        v = [float(x) for x in parts]
        d = {
            "t":      v[0],
            "trig":   v[1],
            "grip":   v[2],
            "q_r":    v[3],   # Quest RIGHT (+) / LEFT (-)
            "q_u":    v[4],   # Quest UP (+) / DOWN (-)
            "q_b":    v[5],   # Quest BACK (+) / FWD (-)
            "dx":     v[10],  # FR5 delta X (mm)
            "dy":     v[11],  # FR5 delta Y (mm)
            "dz":     v[12],  # FR5 delta Z (mm)
            "ex":     v[13],  # FR5 EEF absolute X
            "ey":     v[14],
            "ez":     v[15],
            "j": [v[19], v[20], v[21], v[22], v[23], v[24]],
            # Left controller wrist (cols 25+)
            "l_trig":   v[25] if len(v) > 25 else 0.0,
            "l_valid":  int(v[26]) if len(v) > 26 else 0,
            "l_roll":   v[31] if len(v) > 31 else 0.0,
            "l_pitch":  v[32] if len(v) > 32 else 0.0,
            "l_yaw":    v[33] if len(v) > 33 else 0.0,
        }
        return d
    except Exception:
        return None

def dominant_quest(d) -> str:
    """Which Quest axis is largest?"""
    vals = {"RIGHT": d["q_r"], "UP": d["q_u"], "BACK": d["q_b"]}
    key  = max(vals, key=lambda k: abs(vals[k]))
    val  = vals[key]
    if abs(val) < DEAD: return grey("(still)")
    label = {"RIGHT": ("RIGHT","LEFT"), "UP": ("UP","DOWN"), "BACK": ("BACK","FWD")}
    return green(label[key][0]) if val > 0 else red(label[key][1])

def dominant_fr5(d) -> str:
    """Which FR5 axis changed most?"""
    vals = {"X": d["dx"], "Y": d["dy"], "Z": d["dz"]}
    key  = max(vals, key=lambda k: abs(vals[k]))
    val  = vals[key]
    if abs(val) < 2.0: return grey("(still)")
    sign = "+" if val > 0 else "−"
    return (green if val > 0 else red)(f"FR5 {sign}{key}")

def print_dashboard(d):
    clear()
    print(bold("═"*68))
    print(bold("  Quest 3 → FR5  LIVE AXIS MAP") +
          f"          t={d['t']:.1f}s  trig={d['trig']:.2f}")
    print(bold("═"*68))

    # ── Quest controller ──────────────────────────────────────────────────
    print(f"\n  {bold('QUEST CONTROLLER')}  (delta from home)\n")

    q_scale = 0.4  # 0.4 m max for bar

    def qbar(val):
        filled = min(int(abs(val)/q_scale*20), 20)
        if val >= 0:
            return "·"*20 + green("█"*filled) + "·"*(20-filled)
        else:
            return "·"*(20-filled) + red("█"*filled) + "·"*20

    for label, val, pos, neg in [
        ("RIGHT/LEFT", d["q_r"], "RIGHT →", "← LEFT"),
        ("UP / DOWN ", d["q_u"], "UP ↑",    "↓ DOWN"),
        ("FWD / BACK", d["q_b"], "BACK ↩",  "FWD ↪"),
    ]:
        arrow = green(pos) if val > DEAD else (red(neg) if val < -DEAD else grey("still"))
        print(f"  {label}  {val:+7.3f}m  {qbar(val)}  {arrow}")

    # ── FR5 EEF response ─────────────────────────────────────────────────
    print(f"\n  {bold('FR5 EEF DELTA')}  (mm from home)\n")

    fr5_scale = 300.0  # 300 mm max for bar

    for label, val, pos, neg in [
        ("X (fwd axis)", d["dx"], "+X →", "← −X"),
        ("Y (lat axis)", d["dy"], "+Y →", "← −Y"),
        ("Z (vertical)", d["dz"], "+Z ↑", "↓ −Z"),
    ]:
        arrow = green(pos) if val > 2 else (red(neg) if val < -2 else grey("still"))
        print(f"  {label}  {val:+7.1f}mm  {bar(val, fr5_scale)}  {arrow}")

    # ── Mapping verdict ───────────────────────────────────────────────────
    print(f"\n  {bold('ACTIVE MAPPING')}:")
    dq = dominant_quest(d)
    df = dominant_fr5(d)
    print(f"    Quest {dq}  →  {df}")

    # ── Confirmed correspondences table ──────────────────────────────────
    print(f"\n  {bold('AXIS CORRESPONDENCE')} (move each direction to fill):\n")
    print(f"  {'Quest':20s}  {'FR5 response':20s}  {'Sign':10s}  {'Status'}")
    print(f"  {'─'*20}  {'─'*20}  {'─'*10}  {'─'*8}")

    # Static table — update signs from data heuristic
    rows = [
        ("RIGHT (+q_right)", "+FR5_Y", d["q_r"]>DEAD and d["dy"]>2),
        ("LEFT  (−q_right)", "−FR5_Y", d["q_r"]<-DEAD and d["dy"]<-2),
        ("UP    (+q_up)",    "+FR5_Z", d["q_u"]>DEAD and d["dz"]>2),
        ("DOWN  (−q_up)",   "−FR5_Z", d["q_u"]<-DEAD and d["dz"]<-2),
        ("FWD   (−q_back)",  "−FR5_X", d["q_b"]<-DEAD and d["dx"]<-2),
        ("BACK  (+q_back)",  "+FR5_X", d["q_b"]>DEAD and d["dx"]>2),
    ]
    for label, fr5, active in rows:
        tag = bold(green("ACTIVE ▶")) if active else grey("waiting")
        print(f"  {label:20s}  {fr5:20s}  {'↔':10s}  {tag}")

    # ── EEF absolute position ─────────────────────────────────────────────
    print(f"\n  {bold('EEF absolute')}:  "
          f"X={d['ex']:+.0f}mm  Y={d['ey']:+.0f}mm  Z={d['ez']:+.0f}mm")

    # ── Joints ────────────────────────────────────────────────────────────
    j = d["j"]
    j3a, j5a = abs(j[2]), abs(j[4])
    sing_str = ""
    if j5a < 15: sing_str += red(f" ⚠ J5={j[4]:.1f}° near wrist-sing")
    if j3a < 8:  sing_str += yellow(f" ⚠ J3={j[2]:.1f}° near elbow-sing")

    # ── Left controller wrist section ─────────────────────────────────────────
    l_valid  = d.get("l_valid", 0)
    l_trig   = d.get("l_trig", 0.0)
    l_roll   = d.get("l_roll", 0.0)
    l_pitch  = d.get("l_pitch", 0.0)
    l_yaw    = d.get("l_yaw", 0.0)
    l_status = green("ACTIVE") if l_valid else grey("not tracked")

    def _wbar(v, scale=90, w=18):
        filled = min(int(abs(v)/scale*w), w)
        if v >= 0: return grey("·"*w) + cyan("█"*filled) + grey("·"*(w-filled))
        else:       return grey("·"*(w-filled)) + cyan("█"*filled) + grey("·"*w)

    print(f"\n  {bold('LEFT CONTROLLER')}  (wrist J4/J5/J6)  {l_status}"
          f"  trig={l_trig:.2f}\n")
    for lbl, val, jlbl, jval in [
        ("Roll ",  l_roll,  "J4", j[3]),
        ("Pitch",  l_pitch, "J5", j[4]),
        ("Yaw  ",  l_yaw,   "J6", j[5]),
    ]:
        arrow = cyan(f"▶ {'+' if val>=0 else '-'}") if abs(val) > 2 else grey("─ still ─")
        print(f"  {lbl}  {val:+7.1f}°  {_wbar(val)}  │  {jlbl}={jval:+.1f}°  {arrow}")

    print(f"\n  {bold('Joints')}:  "
          f"J1={j[0]:+.1f}° J2={j[1]:+.1f}° J3={j[2]:+.1f}°"
          f"  J4={j[3]:+.1f}° J5={j[4]:+.1f}° J6={j[5]:+.1f}°{sing_str}")
    print()
    print(grey("  Ctrl-C to quit"))

def main():
    if not os.path.exists(CSV):
        print(f"Waiting for {CSV} …")
        while not os.path.exists(CSV):
            time.sleep(0.5)

    # skip header
    last_t = -1.0
    print("Reading axis log… (move the controller with trigger held)")

    while True:
        line = read_last_line(CSV)
        if line and not line.startswith("t"):   # skip CSV header
            d = parse(line)
            if d and d["t"] != last_t:
                last_t = d["t"]
                print_dashboard(d)
        time.sleep(0.08)   # ~12 Hz refresh

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear()
        print("Stopped.")
