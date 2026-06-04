"""
diag_controller.py — real-time diagnostic for the Meta Quest 3 right controller.

Shows: tracking status, position (m), quaternion, trigger, grip.
Run this to verify controller data is flowing BEFORE launching the sim.

Usage:
    python3 diag_controller.py
    python3 diag_controller.py --hand left
"""

import argparse
import os
import subprocess
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from quest3 import _OculusReaderTransport, _rotmat_to_quat


def _parse_line(line: str, hand: str):
    """Parse one logcat line. Returns dict or None if no data."""
    if "wE9ryARX: " not in line:
        return None
    data = line.split("wE9ryARX: ", 1)[-1].strip()

    if not data or data == "&":
        return {"tracking": False}

    try:
        transforms_str, buttons_str = data.split("&", 1)
    except ValueError:
        return None

    transforms = {}
    for pair in transforms_str.split("|"):
        if ":" not in pair:
            continue
        hand_char, mat_str = pair.split(":", 1)
        vals = mat_str.split()
        if len(vals) == 16:
            mat = np.array(vals, dtype=np.float64).reshape(4, 4)
            transforms[hand_char.strip()] = mat

    key = hand[0]  # 'r' or 'l'
    if key not in transforms:
        return {"tracking": False}

    mat  = transforms[key]
    pos  = mat[:3, 3]
    quat = _rotmat_to_quat(mat[:3, :3])

    # Parse buttons
    trigger = grip = 0.0
    for token in buttons_str.split(","):
        token = token.strip()
        if f"{key}ightTrig" in token or (key == "r" and "rightTrig" in token):
            try: trigger = float(token.split()[-1])
            except: pass
        if f"{key}ightGrip" in token or (key == "r" and "rightGrip" in token):
            try: grip = float(token.split()[-1])
            except: pass
        if key == "l":
            if "leftTrig" in token:
                try: trigger = float(token.split()[-1])
                except: pass
            if "leftGrip" in token:
                try: grip = float(token.split()[-1])
                except: pass

    return {
        "tracking": True,
        "pos":  pos,
        "quat": quat,
        "trigger": trigger,
        "grip": grip,
    }


def _bar(val: float, width: int = 20) -> str:
    filled = int(val * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {val:.2f}"


def run(hand: str):
    print(f"[DIAG] Starting real-time Quest 3 {hand} controller diagnostic")
    print("[DIAG] Cover the proximity sensor (nose bridge) if headset is not worn")
    print("[DIAG] Ctrl-C to quit\n")

    # Check if app is already streaming — only launch if it isn't
    probe = subprocess.run(
        ["adb", "logcat", "-d", "-s", "wE9ryARX:I"],
        capture_output=True, text=True,
    )
    already_running = any(
        "wE9ryARX" in l and l.split("wE9ryARX: ", 1)[-1].strip() not in ("", "&")
        for l in probe.stdout.splitlines()[-20:]
    )

    if not already_running:
        print("[DIAG] Launching OculusReader app via ADB...")
        subprocess.run(
            ["adb", "shell", "am", "start",
             "-n", "com.rail.oculus.teleop/.MainActivity",
             "-c", "org.khronos.openxr.intent.category.IMMERSIVE_HMD"],
            capture_output=True,
        )
        time.sleep(1.5)
        subprocess.run(["adb", "logcat", "-c"], capture_output=True)
    else:
        print("[DIAG] OculusReader already streaming — attaching to live session")

    proc = subprocess.Popen(
        ["adb", "logcat", "-s", "wE9ryARX:I"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )

    no_data_ticks = 0
    last_valid_t  = None
    frame_count   = 0
    t_start       = time.monotonic()

    try:
        for raw_line in proc.stdout:
            if not raw_line.strip():
                continue

            parsed = _parse_line(raw_line, hand)
            if parsed is None:
                continue

            now = time.monotonic()

            if not parsed["tracking"]:
                no_data_ticks += 1
                if no_data_ticks % 20 == 1:   # print every ~20 frames (~0.16 s at 125 Hz)
                    elapsed = now - t_start
                    print(f"\r[DIAG] t={elapsed:6.1f}s  NOT TRACKING "
                          f"(cover the proximity sensor near the nose bridge)"
                          + " " * 10, end="", flush=True)
                continue

            no_data_ticks = 0
            frame_count  += 1
            last_valid_t  = now
            elapsed       = now - t_start

            pos  = parsed["pos"]
            quat = parsed["quat"]
            trig = parsed["trigger"]
            grip = parsed["grip"]

            fps_str = f"{frame_count / max(elapsed, 0.001):.0f} Hz" if elapsed > 1 else "..."

            print(
                f"\r[DIAG] t={elapsed:6.1f}s  {fps_str:>6}  "
                f"pos: x={pos[0]:+6.3f} y={pos[1]:+6.3f} z={pos[2]:+6.3f} m  "
                f"q: [{quat[0]:+5.2f} {quat[1]:+5.2f} {quat[2]:+5.2f} {quat[3]:+5.2f}]  "
                f"trig={trig:.2f}  grip={grip:.2f}",
                end="", flush=True,
            )

    except KeyboardInterrupt:
        print("\n[DIAG] Stopped.")
    finally:
        proc.terminate()
        proc.wait(timeout=2)


def main():
    parser = argparse.ArgumentParser(description="Quest 3 controller real-time diagnostic")
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    args = parser.parse_args()
    run(args.hand)


if __name__ == "__main__":
    main()
