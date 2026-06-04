"""
fr5_motion_test.py — isolate the FR5 control path from the teleop pipeline.

Moves ONLY the FR5: no mapper, no singularity check, no gripper.
Ramps joint 1 (shoulder pan) a few degrees with ServoJ, then reports whether
the robot physically moved.

  FR5 MOVES here      → control path works; bug is in the mapper or Quest input.
  FR5 does NOT move   → bug is in the FR5 SDK / controller layer
                        (servo mode, enable state, or a controller fault).

Run with the FR5 powered and reachable:
    python fr5_motion_test.py
"""

import time

from fr5 import FR5Controller

JOINT  = 0        # 0 = J1 shoulder pan — safest joint, large clearance
DEG    = 5.0      # total degrees to move
CYCLES = 250      # number of ServoJ commands (250 / 125 Hz ≈ 2 s)
HZ     = 125


def main():
    print("=" * 62)
    print("  FR5 ISOLATED MOTION TEST  —  no mapper, no Quest input")
    print("=" * 62)

    with FR5Controller() as robot:
        from fairino import Robot
        flag = getattr(Robot.RPC, "is_conect", "<attribute missing>")
        print(f"  Robot.RPC.is_conect = {flag!r}")

        # Raw return — if this is a ServerProxy/str instead of (0, [6 floats]),
        # the SDK is not dispatching and nothing below will move.
        raw = robot._robot.GetActualJointPosDegree(0)
        print(f"  GetActualJointPosDegree raw: {type(raw).__name__}  {raw!r}")

        start = robot.get_joint_positions()
        print(f"  Start joints (deg): {[f'{v:.2f}' for v in start]}")

        print("  ServoMoveStart()...")
        robot.start_servo_mode()
        print("  ServoMoveStart OK")

        step   = DEG / CYCLES
        target = list(start)
        period = 1.0 / HZ
        print(f"  Ramping J{JOINT + 1} by {DEG:+.1f}° over {CYCLES} cycles...")

        for i in range(CYCLES):
            t0 = time.monotonic()
            target[JOINT] += step
            try:
                robot.servo_j(target)
            except Exception as exc:
                print(f"  [cycle {i}] servo_j FAILED: {exc}")
                break
            if i < 3 or (i + 1) % 50 == 0:
                print(f"  [cycle {i + 1:3d}] sent J{JOINT + 1} = {target[JOINT]:.3f}°")
            slack = period - (time.monotonic() - t0)
            if slack > 0:
                time.sleep(slack)

        robot.stop_servo_mode()
        time.sleep(0.3)

        end   = robot.get_joint_positions()
        moved = end[JOINT] - start[JOINT]
        print(f"  End joints (deg):   {[f'{v:.2f}' for v in end]}")
        print(f"  J{JOINT + 1} moved {moved:+.2f}°  (commanded {DEG:+.1f}°)")
        print("-" * 62)
        if abs(moved) > 0.5:
            print("  RESULT: FR5 MOVED — control path works.")
            print("          Bug is in the mapper or Quest 3 input pipeline.")
        else:
            print("  RESULT: FR5 DID NOT MOVE — bug is in the FR5 SDK / controller layer.")
            print("          Check: servo mode active? robot enabled? controller fault?")
    print("=" * 62)


if __name__ == "__main__":
    main()
