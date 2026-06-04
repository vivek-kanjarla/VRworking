"""
vr_sim_launch.py — starts the complete FR5 VR sim bridge stack.

Launches (in order):
  1. robot_state_publisher  — publishes /tf from URDF + /joint_states
  2. rviz2                  — live FR5 model visualisation
  3. sim_bridge.py          — VR → sim → (hardware) main loop

All children are cleanly terminated on Ctrl-C.

Usage:
  python3 vr_sim_launch.py --mode sim
  python3 vr_sim_launch.py --mode validate
  python3 vr_sim_launch.py --mode execute --no-rviz
  python3 vr_sim_launch.py --mode sim --gui
  python3 vr_sim_launch.py --mode sim --no-rviz --no-rsp   # bare bridge only

Dependencies:
  ROS2 Humble must be installed at /opt/ros/humble
  Optional: sourced workspace at ~/ros2_teleop_ws for mesh resolution in RViz
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROS2_SETUP = "/opt/ros/humble/setup.bash"
_WS_SETUP   = os.path.expanduser("~/ros2_teleop_ws/install/setup.bash")
_URDF_PATH  = os.path.abspath(os.path.join(
    _HERE, "..",
    "ros2_teleop_ws", "src", "frcobot_ros2-main",
    "fairino_description", "urdf", "fairino5_v6.urdf",
))
# Python interpreter that has pybullet + all teleop deps installed
_PYTHON = os.path.join(_HERE, "..", "teleop_env", "bin", "python3")
_PYTHON = os.path.abspath(_PYTHON) if os.path.exists(os.path.abspath(_PYTHON)) else sys.executable

# Minimal RViz config — shows FR5 robot model + grid; no MoveIt dependency
_RVIZ_CONFIG = """\
Panels:
  - Class: rviz_common/Displays
    Name: Displays
  - Class: rviz_common/Views
    Name: Views
Visualization Manager:
  Class: ""
  Enabled: true
  Name: root
  Displays:
    - Class: rviz_default_plugins/Grid
      Name: Grid
      Value: true
      Enabled: true
    - Class: rviz_default_plugins/RobotModel
      Name: RobotModel
      Enabled: true
      Value: true
      Description Topic:
        Depth: 5
        Durability Policy: Transient Local
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
    - Class: rviz_default_plugins/TF
      Name: TF
      Enabled: false
      Value: false
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: base_link
    Frame Rate: 30
  Tools:
    - Class: rviz_default_plugins/Interact
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 2.5
      Focal Point:
        X: 0
        Y: 0
        Z: 0.5
      Name: Current View
      Pitch: 0.5
      Yaw: 0.785
    Saved: ~
Window Geometry:
  Height: 720
  Width: 1280
  X: 50
  Y: 50
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _ros2_shell(cmd: str) -> str:
    """Wrap a command with ROS2 source preamble for subprocess shell execution."""
    parts = [f"source {_ROS2_SETUP}"]
    if os.path.exists(_WS_SETUP):
        parts.append(f"source {_WS_SETUP}")
    parts.append(cmd)
    return " && ".join(parts)


def _patch_urdf(urdf_path: str) -> str:
    """Replace package:// URIs with absolute paths; write to a temp file."""
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(urdf_path), ".."))
    with open(urdf_path) as fh:
        content = fh.read()
    content = re.sub(r"package://fairino_description", pkg_root, content)
    tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
    tmp.write(content)
    tmp.close()
    return tmp.name


def _write_rsp_params(urdf_content: str) -> str:
    """Write robot_description YAML params file; return path."""
    import yaml
    params = {"robot_state_publisher": {"ros__parameters": {"robot_description": urdf_content}}}
    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
    yaml.dump(params, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    return tmp.name


def _write_rviz_config() -> str:
    """Write the embedded RViz config to a temp file; return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".rviz", delete=False, mode="w")
    tmp.write(_RVIZ_CONFIG)
    tmp.close()
    return tmp.name


# ── process management ────────────────────────────────────────────────────────

_procs: list[subprocess.Popen] = []
_tmpfiles: list[str]           = []


def _cleanup():
    for p in _procs:
        if p.poll() is None:
            try:
                p.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
    time.sleep(1.0)
    for p in _procs:
        if p.poll() is None:
            try:
                p.terminate()
            except ProcessLookupError:
                pass
    for f in _tmpfiles:
        try:
            os.unlink(f)
        except OSError:
            pass


_shutting_down = False

def _handle_sigint(sig, frame):
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    print("\n[LAUNCH] Ctrl-C — shutting down all processes...")
    _cleanup()
    sys.exit(0)


def _launch(label: str, cmd: str, ros2: bool = True,
            env: dict | None = None) -> subprocess.Popen:
    """Spawn a process. ros2=True wraps with ROS2 source preamble."""
    shell_cmd = _ros2_shell(cmd) if ros2 else cmd
    print(f"[LAUNCH] Starting {label}")
    p = subprocess.Popen(
        shell_cmd,
        shell=True,
        executable="/bin/bash",
        env={**os.environ, **(env or {})},
    )
    _procs.append(p)
    return p


def _wait_for_ros2_node(node_name: str, timeout: float = 10.0) -> bool:
    """Return True once `ros2 node list` shows the node, or False on timeout."""
    deadline = time.monotonic() + timeout
    check_cmd = _ros2_shell(f"ros2 node list 2>/dev/null | grep -q '{node_name}'")
    while time.monotonic() < deadline:
        r = subprocess.run(check_cmd, shell=True, executable="/bin/bash")
        if r.returncode == 0:
            return True
        time.sleep(0.5)
    return False


# ── launch logic ──────────────────────────────────────────────────────────────

def launch(mode: str, gui: bool, no_rviz: bool, no_rsp: bool, record: bool = False):
    signal.signal(signal.SIGINT,  _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    if not os.path.exists(_ROS2_SETUP):
        print(f"[LAUNCH] ERROR: ROS2 not found at {_ROS2_SETUP}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(_URDF_PATH):
        print(f"[LAUNCH] ERROR: URDF not found: {_URDF_PATH}", file=sys.stderr)
        sys.exit(1)

    # ── 1. robot_state_publisher ──────────────────────────────────────────────
    rsp_proc = None
    if not no_rsp and mode in ("sim", "validate", "execute"):
        # Use the ORIGINAL URDF (package:// URIs) so RViz resolves meshes via ament.
        # The workspace is sourced in _ros2_shell, so fairino_description is on the
        # ament index and resource_retriever can find the STL files.
        with open(_URDF_PATH) as fh:
            urdf_content = fh.read()

        try:
            params_yaml = _write_rsp_params(urdf_content)
            _tmpfiles.append(params_yaml)
            rsp_cmd = (
                "ros2 run robot_state_publisher robot_state_publisher "
                f"--ros-args --params-file {params_yaml}"
            )
        except ImportError:
            # yaml not installed — fall back to inline xacro approach
            rsp_cmd = (
                f"ros2 run robot_state_publisher robot_state_publisher "
                f"--ros-args -p robot_description:=\"$(cat {_URDF_PATH})\""
            )

        rsp_proc = _launch("robot_state_publisher", rsp_cmd)

        print("[LAUNCH] Waiting for robot_state_publisher to come up ...")
        if not _wait_for_ros2_node("/robot_state_publisher", timeout=12.0):
            print("[LAUNCH] Warning: robot_state_publisher did not appear in node list — "
                  "RViz TF may be empty. Continuing anyway.")
        else:
            print("[LAUNCH] robot_state_publisher ready")

    # ── 2. rviz2 ──────────────────────────────────────────────────────────────
    rviz_proc = None
    if not no_rviz and mode in ("sim", "validate", "execute"):
        rviz_cfg = _write_rviz_config()
        _tmpfiles.append(rviz_cfg)
        rviz_proc = _launch("rviz2", f"rviz2 -d {rviz_cfg}")
        # Give RViz 2 s to start before the joint state stream begins
        time.sleep(2.0)

    # ── 3. sim_bridge.py ──────────────────────────────────────────────────────
    bridge_flags = f"--mode {mode}"
    if gui:
        bridge_flags += " --gui"
    if record:
        bridge_flags += " --record"

    # Inject ROS2 Python + shared-lib paths so rclpy is importable inside the
    # teleop virtualenv. rclpy lives in two dirs and needs librcl_action.so etc.
    ros2_py = (
        "/opt/ros/humble/local/lib/python3.10/dist-packages"
        ":/opt/ros/humble/lib/python3.10/site-packages"
    )
    existing_pp = os.environ.get("PYTHONPATH", "")
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    bridge_env = {
        "PYTHONPATH":     f"{ros2_py}:{existing_pp}".rstrip(":"),
        "LD_LIBRARY_PATH": f"/opt/ros/humble/lib:{existing_ld}".rstrip(":"),
    }

    print(f"[LAUNCH] Using Python: {_PYTHON}")
    bridge_proc = _launch(
        f"sim_bridge [{mode}]",
        f"cd {_HERE} && {_PYTHON} sim_bridge.py {bridge_flags}",
        ros2=False,
        env=bridge_env,
    )

    # ── monitor loop ──────────────────────────────────────────────────────────
    print("[LAUNCH] All processes running. Press Ctrl-C to stop.\n")

    try:
        while True:
            # If the bridge exits, bring everything down
            rc = bridge_proc.poll()
            if rc is not None:
                print(f"[LAUNCH] sim_bridge exited (rc={rc}) — shutting down")
                break

            # Alert if RSP or RViz crash
            if rsp_proc and rsp_proc.poll() is not None:
                print(f"[LAUNCH] WARNING: robot_state_publisher exited (rc={rsp_proc.returncode})")
                rsp_proc = None

            if rviz_proc and rviz_proc.poll() is not None:
                print(f"[LAUNCH] WARNING: rviz2 exited (rc={rviz_proc.returncode})")
                rviz_proc = None

            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Launch FR5 VR sim bridge with RViz visualisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  sim       pure sim + RViz, no hardware\n"
            "  validate  sim validates, then executes on FR5\n"
            "  execute   direct VR→FR5 (no sim overhead)\n"
        ),
    )
    parser.add_argument(
        "--mode", choices=["sim", "validate", "execute"], default="sim",
        help="bridge mode (default: sim)",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="open PyBullet GUI window alongside RViz",
    )
    parser.add_argument(
        "--no-rviz", action="store_true",
        help="skip launching RViz2",
    )
    parser.add_argument(
        "--no-rsp", action="store_true",
        help="skip robot_state_publisher (use when it is already running)",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="enable ACT episode recording (dual cameras + Quest A/B toggles)",
    )
    args = parser.parse_args()

    launch(
        mode    = args.mode,
        gui     = args.gui,
        no_rviz = args.no_rviz,
        no_rsp  = args.no_rsp,
        record  = args.record,
    )


if __name__ == "__main__":
    main()
