"""
check_hardware.py — verify FR5 and D405 camera hardware for VR teleop.
Run with: python check_hardware.py
"""

import sys

from config import FR5_IP


# ── FR5 ───────────────────────────────────────────────────────────────────────

def check_fr5():
    print("=" * 55)
    print("  FR5 Follower Cobot (Ethernet / Fairino SDK)")
    print("=" * 55)

    try:
        from fairino import Robot
    except ImportError:
        print("  FAIL  Could not import fairino — install from Fairino .whl")
        return False

    print(f"  Connecting to FR5 at {FR5_IP}...", flush=True)
    try:
        robot = Robot.RPC(FR5_IP)
        Robot.RPC.is_conect = True
    except Exception as exc:
        print(f"  FAIL  Robot.RPC({FR5_IP!r}) raised: {exc}")
        print("         Run check_network.py first to verify connectivity.")
        return False

    try:
        err, model, sdk_ver, fw_ver = robot.GetSoftwareVersion()
        if err == 0:
            print(f"  Robot model         : {model}")
            print(f"  SDK version         : {sdk_ver}")
            print(f"  Firmware version    : {fw_ver}")
            if sdk_ver != fw_ver and sdk_ver.lstrip('v') not in fw_ver:
                print("  WARN  SDK/firmware mismatch — check Fairino release notes.")
        else:
            print(f"  WARN  GetSoftwareVersion returned error {err}")
    except Exception as exc:
        print(f"  WARN  Could not read version info: {exc}")

    all_ok = True

    try:
        ret = robot.GetActualJointPosDegree(0)
        if isinstance(ret, (list, tuple)) and len(ret) == 2:
            err, joints = ret
        else:
            err, joints = 0, ret
        if err == 0 and joints:
            labels = ["J1", "J2", "J3", "J4", "J5", "J6"]
            print("  PASS  Current joint positions (degrees):")
            for label, val in zip(labels, joints):
                print(f"          {label}: {val:>8.3f}°")
        else:
            print(f"  FAIL  GetActualJointPosDegree returned error code {err}")
            all_ok = False
    except Exception as exc:
        print(f"  FAIL  Could not read joint positions: {exc}")
        all_ok = False

    try:
        ret = robot.GetActualTCPPose(0)
        if isinstance(ret, (list, tuple)) and len(ret) == 2:
            err, pose = ret
        else:
            err, pose = 0, ret
        if err == 0 and pose:
            labels = ["x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg"]
            print("  PASS  Current TCP pose:")
            for label, val in zip(labels, pose):
                print(f"          {label}: {val:>8.2f}")
    except Exception as exc:
        print(f"  WARN  Could not read TCP pose: {exc}")

    return all_ok


# ── D405 Camera ───────────────────────────────────────────────────────────────

def check_camera():
    print()
    print("=" * 55)
    print("  Intel RealSense D405 (USB 3.0 wrist camera)")
    print("=" * 55)

    try:
        import pyrealsense2 as rs
    except ImportError:
        print("  FAIL  Could not import pyrealsense2")
        print("         Run linux_setup.sh or: sudo apt-get install librealsense2-dev")
        print("         Then: pip install pyrealsense2")
        return False

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("  FAIL  No RealSense devices found")
        print("         Is the D405 plugged into a USB 3.0 port?")
        print("         Run: lsusb | grep Intel  — to check USB enumeration")
        print("         Run linux_setup.sh to install udev rules if needed.")
        return False

    for dev in devices:
        name   = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        fw     = dev.get_info(rs.camera_info.firmware_version)
        print(f"  PASS  {name}  serial={serial}  fw={fw}")

    # Quick pipeline test — open, grab one frame, close
    try:
        cfg      = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline = rs.pipeline()
        profile  = pipeline.start(cfg)
        frames   = pipeline.wait_for_frames(timeout_ms=3000)
        color    = frames.get_color_frame()
        pipeline.stop()
        if color:
            print(f"  PASS  Color stream 640×480@30 — got frame {color.get_frame_number()}")
        else:
            print("  WARN  Pipeline started but no color frame received")
    except Exception as exc:
        print(f"  FAIL  Pipeline test failed: {exc}")
        print("         Try unplugging and replugging the USB cable.")
        return False

    return True


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fr5_ok    = check_fr5()
    camera_ok = check_camera()

    print()
    print("=" * 55)
    print("  Summary")
    print("=" * 55)
    print(f"  FR5     : {'PASS' if fr5_ok    else 'FAIL'}")
    print(f"  D405    : {'PASS' if camera_ok else 'FAIL'}")
    print("=" * 55)

    if fr5_ok and camera_ok:
        print("  All checks PASSED — ready to run teleop_vr.py")
        sys.exit(0)
    else:
        print("  One or more checks FAILED — see messages above.")
        sys.exit(1)
