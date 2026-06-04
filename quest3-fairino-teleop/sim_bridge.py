"""
sim_bridge.py — VR → Sim → (optionally) Hardware bridge for FR5 validation.

Modes (--mode flag):
  sim       SIM_ONLY:         VR → PyBullet IK → RViz JointState  (no hardware)
  validate  SIM_THEN_EXECUTE: VR → PyBullet IK → collision/limit check → FR5 ServoJ
  execute   LIVE_ONLY:        VR → FR5 IK → FR5 ServoJ  (mirrors teleop_vr.py)

Dependencies:
  pip install pybullet
  ROS2 Humble + rclpy  (optional — sim works without it, just no RViz output)

Usage:
  python sim_bridge.py --mode sim
  python sim_bridge.py --mode validate
  python sim_bridge.py --mode execute
  python sim_bridge.py --mode sim --gui        # open PyBullet GUI window
"""

import argparse
import math
import os
import re
import sys
import tempfile
import threading
import time
from enum import Enum, auto

import numpy as np

# ── project package on sys.path ───────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import (
    FR5_JOINT_LIMITS,
    LOOP_PERIOD,
    MAX_DELTA_PER_JOINT,
    QUEST3_MODE,
    GRIPPER_INDEX, GRIPPER_CLOSE_PCT, GRIPPER_OPEN_PCT,
    GRIPPER_VEL_PCT, GRIPPER_FORCE_PCT, GRIPPER_MAXTIME_MS, GRIPPER_TYPE,
    FILTER_ENABLED,
    FILTER_POS_MIN_CUTOFF, FILTER_POS_BETA,
    FILTER_ROT_MIN_CUTOFF, FILTER_ROT_BETA,
    FILTER_D_CUTOFF,
    JOYSTICK_J4_SCALE, JOYSTICK_J5_SCALE, JOYSTICK_J6_SCALE, JOYSTICK_DEAD_ZONE,
    WRIST_GAIN_J4, WRIST_GAIN_J5,
    EXECUTE_USE_DLS_IK,
    RECORD_BUTTON, SUCCESS_BUTTON,
)
from quest3 import Quest3Reader, ControllerState
from axis_logger import AxisLogger
from mapper_vr import (
    vr_to_fr5, _quat_mul, _quat_inv, _quat_to_rotvec, _rotvec_to_euler_deg,
)
from fr5_sim_validator import FR5SimValidator
from dls_ik import DLSIK, DLSIKConfig, DampingMode
from pose_filter import PoseFilter
import singularity


# ── constants ─────────────────────────────────────────────────────────────────

class Mode(Enum):
    SIM_ONLY         = auto()
    SIM_THEN_EXECUTE = auto()
    LIVE_ONLY        = auto()


_MODE_MAP = {
    "sim":      Mode.SIM_ONLY,
    "validate": Mode.SIM_THEN_EXECUTE,
    "execute":  Mode.LIVE_ONLY,
}

JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
_EE_LINK    = "wrist3_link"

# Auto-return-to-home: if trigger is released for this many seconds, glide
# the arm back to the default safe home position at reduced speed.
AUTO_HOME_IDLE_S  = 8.0    # seconds of trigger-off before auto-return starts
AUTO_HOME_SPEED   = 0.3    # fraction of MAX_DELTA_PER_JOINT used for the glide

_URDF_PATH = os.path.join(
    _HERE, "..",
    "ros2_teleop_ws", "src", "frcobot_ros2-main",
    "fairino_description", "urdf", "fairino5_v6.urdf",
)
_URDF_PATH = os.path.abspath(_URDF_PATH)

# Joint limits in radians (for PyBullet IK bounds)
_JL_RAD_LOW  = [math.radians(lo) for lo, _ in FR5_JOINT_LIMITS]
_JL_RAD_HIGH = [math.radians(hi) for _, hi in FR5_JOINT_LIMITS]
_JL_RANGES   = [hi - lo for lo, hi in zip(_JL_RAD_LOW, _JL_RAD_HIGH)]

# SRDF "pos1" in degrees — reasonable working home when no hardware is present
_SIM_DEFAULT_HOME_DEG = [120.8, -110.3, 108.3, -87.1, -82.2, 0.0]

# Grip analog threshold: above this → gripper closes
_GRIP_CLOSE_THRESH = 0.7
_GRIP_OPEN_THRESH  = 0.1

# Trigger analog threshold: hold above this for HOME_HOLD_S seconds → sets VR home
_TRIGGER_HOME_THRESH = 0.7
_HOME_HOLD_S         = 0.5


# ── URDF preprocessing ────────────────────────────────────────────────────────

def _patch_urdf(urdf_path: str) -> str:
    """
    Return a temp-file path where package://fairino_description URIs are
    replaced with absolute filesystem paths PyBullet can load directly.
    """
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(urdf_path), ".."))
    with open(urdf_path) as fh:
        content = fh.read()
    content = re.sub(r"package://fairino_description", pkg_root, content)
    tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
    tmp.write(content)
    tmp.close()
    return tmp.name


# ── PyBullet sim robot ────────────────────────────────────────────────────────

class SimRobot:
    """
    Wraps a PyBullet URDF so IK/FK can run without hardware.
    Implements get_inverse_kin() with the same signature as FR5Controller,
    so it can be passed directly to mapper_vr.vr_to_fr5() as the `robot` arg.
    """

    def __init__(self, urdf_path: str, gui: bool = False):
        import pybullet as pb
        import pybullet_data

        self._pb    = pb
        self._tmp_urdf = None

        conn = pb.GUI if gui else pb.DIRECT
        self._cid = pb.connect(conn)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._cid)
        pb.setGravity(0, 0, -9.81, physicsClientId=self._cid)

        patched = _patch_urdf(urdf_path)
        self._tmp_urdf = patched

        self._robot_id = pb.loadURDF(
            patched,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._cid,
        )

        # Map joint names j1..j6 to PyBullet joint indices
        nj = pb.getNumJoints(self._robot_id, physicsClientId=self._cid)
        self._joint_indices: list[int] = []
        self._ee_link_id: int | None = None

        for i in range(nj):
            info  = pb.getJointInfo(self._robot_id, i, physicsClientId=self._cid)
            jtype = info[2]
            jname = info[1].decode()
            lname = info[12].decode()   # child link name
            if jtype == pb.JOINT_REVOLUTE and jname in JOINT_NAMES:
                self._joint_indices.append(i)
            if lname == _EE_LINK:
                self._ee_link_id = i

        if len(self._joint_indices) != 6:
            raise RuntimeError(
                f"Expected 6 revolute joints named j1..j6 in URDF, "
                f"found {len(self._joint_indices)}: "
                f"check {urdf_path}"
            )
        if self._ee_link_id is None:
            # Fallback: j6's index is the last revolute, its child is wrist3_link
            self._ee_link_id = self._joint_indices[-1]

        # Disable damping so IK residuals converge without torque bias
        for ji in self._joint_indices:
            pb.changeDynamics(
                self._robot_id, ji,
                linearDamping=0.0, angularDamping=0.0,
                physicsClientId=self._cid,
            )

        self._lock = threading.Lock()

    # ── public interface matching FR5Controller ───────────────────────────────

    def get_inverse_kin(self, eef_pose: list[float], ref_joints: list[float]) -> list[float]:
        """
        eef_pose   — [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]  (FR5 TCP convention)
        ref_joints — seed joint angles (deg) — selects the nearest IK solution
        Returns 6 joint angles in degrees, raises IOError on failure.
        """
        pb = self._pb

        pos_m  = [eef_pose[0] / 1000.0, eef_pose[1] / 1000.0, eef_pose[2] / 1000.0]
        euler  = [math.radians(eef_pose[3]), math.radians(eef_pose[4]), math.radians(eef_pose[5])]
        orn    = pb.getQuaternionFromEuler(euler, physicsClientId=self._cid)
        rest   = [math.radians(j) for j in ref_joints]

        with self._lock:
            # Seed joint state so IK picks the solution closest to current config
            for idx, ji in enumerate(self._joint_indices):
                pb.resetJointState(
                    self._robot_id, ji, rest[idx],
                    physicsClientId=self._cid,
                )

            sol = pb.calculateInverseKinematics(
                self._robot_id,
                self._ee_link_id,
                pos_m,
                orn,
                lowerLimits    = _JL_RAD_LOW,
                upperLimits    = _JL_RAD_HIGH,
                jointRanges    = _JL_RANGES,
                restPoses      = rest,
                maxNumIterations  = 100,
                residualThreshold = 1e-4,
                physicsClientId   = self._cid,
            )

        sol = sol[:6]

        for i, (j, lo, hi) in enumerate(zip(sol, _JL_RAD_LOW, _JL_RAD_HIGH)):
            if not math.isfinite(j):
                raise IOError(f"PyBullet IK returned non-finite value for j{i+1}")
            if j < lo - 0.17 or j > hi + 0.17:   # 10° tolerance for IK overshoot
                raise IOError(
                    f"PyBullet IK j{i+1}={math.degrees(j):.1f}° out of limits "
                    f"[{math.degrees(lo):.0f}°, {math.degrees(hi):.0f}°]"
                )

        return [math.degrees(j) for j in sol]

    # ── visualization helpers ─────────────────────────────────────────────────

    def set_joints(self, joints_deg: list[float]):
        """Directly set joint state (for visualization, no dynamics)."""
        pb = self._pb
        with self._lock:
            for idx, ji in enumerate(self._joint_indices):
                pb.resetJointState(
                    self._robot_id, ji,
                    math.radians(joints_deg[idx]),
                    physicsClientId=self._cid,
                )

    def get_fk(self, joints_deg: list[float]) -> list[float]:
        """Forward kinematics: joints (deg) → [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]."""
        pb = self._pb
        with self._lock:
            for idx, ji in enumerate(self._joint_indices):
                pb.resetJointState(
                    self._robot_id, ji,
                    math.radians(joints_deg[idx]),
                    physicsClientId=self._cid,
                )
            ls = pb.getLinkState(
                self._robot_id, self._ee_link_id,
                computeForwardKinematics=True,
                physicsClientId=self._cid,
            )
        pos_w  = ls[4]
        orn_w  = ls[5]
        euler  = pb.getEulerFromQuaternion(orn_w, physicsClientId=self._cid)
        return [
            pos_w[0] * 1000.0, pos_w[1] * 1000.0, pos_w[2] * 1000.0,
            math.degrees(euler[0]), math.degrees(euler[1]), math.degrees(euler[2]),
        ]

    def check_collisions(self) -> int:
        """Return number of self-collision contact points after stepping detection."""
        pb = self._pb
        with self._lock:
            pb.performCollisionDetection(physicsClientId=self._cid)
            contacts = pb.getContactPoints(self._robot_id, physicsClientId=self._cid)
        return len(contacts) if contacts else 0

    def close(self):
        try:
            self._pb.disconnect(self._cid)
        except Exception:
            pass
        if self._tmp_urdf and os.path.exists(self._tmp_urdf):
            os.unlink(self._tmp_urdf)


# ── ROS2 JointState publisher (optional) ─────────────────────────────────────

class _ROS2Publisher:
    """
    Publishes sensor_msgs/JointState on /joint_states for RViz.
    Silently disabled if rclpy is not importable or ROS2 is not running.
    """

    def __init__(self):
        self._pub  = None
        self._node = None
        self._ok   = False

        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import JointState

            if not rclpy.ok():
                rclpy.init(args=None)

            self._rclpy      = rclpy
            self._JointState = JointState
            self._node       = rclpy.create_node("fr5_sim_bridge")
            self._pub        = self._node.create_publisher(JointState, "/joint_states", 10)

            def _spin_safe():
                try:
                    rclpy.spin(self._node)
                except Exception:
                    pass

            self._spin_thread = threading.Thread(
                target=_spin_safe, daemon=True, name="ros2-spin"
            )
            self._spin_thread.start()
            self._ok = True
            print("[SIM] ROS2 /joint_states publisher active — open RViz to visualise")
        except Exception as exc:
            print(f"[SIM] ROS2 not available ({exc}) — skipping RViz output")

    def publish(self, joints_deg: list[float]):
        if not self._ok:
            return
        try:
            msg = self._JointState()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.name         = JOINT_NAMES
            msg.position     = [math.radians(j) for j in joints_deg]
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


# (validate_motion is provided by FR5SimValidator — imported above)


# ── main bridge ───────────────────────────────────────────────────────────────

class SimBridge:
    """
    Reads Quest 3 controller pose at 125 Hz, runs IK in sim, and optionally
    forwards validated joint commands to the real FR5.

    Homing gesture:
      Hold trigger (≥ 0.7) for 0.5 s → locks current controller pose and FR5
      EEF pose as the reference "home". Subsequent motion is relative to this.
      Release trigger below 0.1 → clears home (must re-home before moving again).

    Gripper:
      Grip analog ≥ 0.7 → close gripper
      Grip analog ≤ 0.1 → open gripper
      In SIM_ONLY mode the gripper state is printed but no hardware command is sent.
    """

    def __init__(self, mode: Mode, gui: bool = False, record: bool = False):
        self.mode  = mode
        self._gui  = gui

        # ACT episode recording (opt-in via --record).
        self._record       = record
        self._logger       = None     # EpisodeLogger
        self._cameras      = None      # DualCamera
        self._prev_buttons = {}        # for rising-edge detection
        self._last_success = None
        self._log_n        = 0         # recording-cycle counter (downsamples slow reads)
        self._gripper_norm = None      # cached jaw position (read ~12 Hz, not 125)
        # Auto-home before each episode -> every demo starts from an identical pose
        # (removes the proprioceptive shortcut so the policy must use the cameras).
        self._homing_for_record = False
        self._record_home  = [-99.2, -116.9, 139.2, -159.2, -82.1, -5.9]

        self._sim_robot: SimRobot           | None = None
        self._fr5:       object             | None = None   # FR5Controller
        self._ros2:      _ROS2Publisher     | None = None
        self._quest:     Quest3Reader       | None = None

        self._current_joints: list[float]   = list(_SIM_DEFAULT_HOME_DEG)
        self._home_joints:    list[float]   | None = None
        self._home_eef:       list[float]   | None = None
        self._quest_home_pos: np.ndarray    | None = None
        self._quest_home_quat:np.ndarray    | None = None
        # Thumbstick accumulates absolute J4/J5 targets
        # J4/J5 angles at home — wrist orientation maps relative to these
        self._wrist_home_j4: float = _SIM_DEFAULT_HOME_DEG[3]
        self._wrist_home_j5: float = _SIM_DEFAULT_HOME_DEG[4]
        self._j6_target:     float = _SIM_DEFAULT_HOME_DEG[5]   # thumbstick-integrated J6 roll

        self._gripper_open = True
        self._grip_held    = False     # debounce: True while squeeze is held down
        self._running      = False
        self._validator:   FR5SimValidator | None = None

        # One-Euro pose filter — smooths raw Quest 3 controller pose before IK.
        # Instantiated regardless of FILTER_ENABLED so the object always exists;
        # the enabled flag is checked in the run loop.
        self._axis_logger   = AxisLogger()
        self._trigger_off_t = None   # wall-clock time when trigger was last released
        self._pose_filter   = PoseFilter(
            pos_min_cutoff = FILTER_POS_MIN_CUTOFF,
            pos_beta       = FILTER_POS_BETA,
            rot_min_cutoff = FILTER_ROT_MIN_CUTOFF,
            rot_beta       = FILTER_ROT_BETA,
            d_cutoff       = FILTER_D_CUTOFF,
        )
        self._filter_dbg_t = 0.0   # last time a [FILT] line was printed

        # DLS IK — replaces PyBullet's calculateInverseKinematics.
        # Created once here; shared with SimRobot via monkey-patch in start().
        self._dls_ik = DLSIK(
            joint_limits_deg     = list(FR5_JOINT_LIMITS),
            max_dq_deg_per_cycle = list(MAX_DELTA_PER_JOINT),
            cfg = DLSIKConfig(
                damping_mode    = DampingMode.VARIABLE_SELECTIVE,
                lambda_min      = 0.001,
                lambda_max      = 0.05,
                sigma_threshold = 0.05,
                barrier_gain    = 0.3,
                w_pos           = 1.0,
                w_rot           = 0.15,   # soft orientation hold → less J1 swing
                #                          during pure translation. Raise toward
                #                          0.8 only when wrist orientation matters.
            ),
        )
        self._ik_diag_t = 0.0   # last time an IK diag line was printed

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        if self.mode != Mode.LIVE_ONLY:
            if not os.path.exists(_URDF_PATH):
                raise FileNotFoundError(
                    f"URDF not found: {_URDF_PATH}\n"
                    "Set _URDF_PATH in sim_bridge.py to the absolute path of fairino5_v6.urdf"
                )
            print(f"[SIM] Loading URDF: {_URDF_PATH}")
            self._sim_robot = SimRobot(_URDF_PATH, gui=self._gui)
            # Patch SimRobot's IK with DLS — same interface, no other changes.
            # DLS: continuous Jacobian-based steps, adaptive damping, ~27 µs/call.
            # PyBullet IK: 100-iteration optimizer, branch-switching, ~270 µs/call.
            _dls = self._dls_ik
            self._sim_robot.get_inverse_kin = lambda eef, q: _dls.get_inverse_kin(eef, q)
            print(f"[SIM] IK: DLS (Pinocchio, λ_min={_dls.cfg.lambda_min}, "
                  f"λ_max={_dls.cfg.lambda_max}, σ_thresh={_dls.cfg.sigma_threshold})")
            self._sim_robot.set_joints(self._current_joints)
            self._ros2 = _ROS2Publisher()
            self._ros2.publish(self._current_joints)
        else:
            self._ros2 = _ROS2Publisher()

        if self.mode in (Mode.SIM_THEN_EXECUTE, Mode.LIVE_ONLY):
            from fr5 import FR5Controller
            self._fr5 = FR5Controller()
            print("[SIM] Connecting to FR5 hardware...")
            self._fr5.connect()

            # Retry loop — CNDE UDP stream may need a moment after reconnect
            for _attempt in range(8):
                try:
                    real_joints = self._fr5.get_joint_positions()
                    break
                except IOError as exc:
                    if _attempt < 7:
                        print(f"[SIM] Waiting for FR5 CNDE ({exc}) — retry {_attempt+1}/8")
                        time.sleep(2.0)
                    else:
                        raise
            self._current_joints = list(real_joints)
            self._home_joints    = list(real_joints)
            print(f"[SIM] FR5 home joints (deg): {[f'{j:.1f}' for j in real_joints]}")

            if self.mode == Mode.SIM_THEN_EXECUTE:
                self._fr5.start_servo_mode()
                # Sync sim to real robot, then use PyBullet FK for home EEF
                # so IK and home EEF are always in the same (PyBullet) frame
                if self._sim_robot:
                    self._sim_robot.set_joints(real_joints)
                    self._home_eef = self._sim_robot.get_fk(real_joints)
                    self._ros2.publish(real_joints)
                    print(f"[SIM] FR5 home EEF (PyBullet FK): {[f'{v:.1f}' for v in self._home_eef]}")
                self._validator = FR5SimValidator(sim_robot=self._sim_robot)
            elif self.mode == Mode.LIVE_ONLY:
                self._fr5.start_servo_mode()
                if EXECUTE_USE_DLS_IK:
                    # Home EEF from DLS/Pinocchio FK on the real joints, so the
                    # IK target (home + delta) stays in the model frame.
                    self._home_eef = self._dls_ik.get_fk(real_joints)
                    print(f"[SIM] FR5 home EEF (DLS FK): {[f'{v:.1f}' for v in self._home_eef]}")
                else:
                    self._home_eef = self._fr5.get_eef_pose()
                    print(f"[SIM] FR5 home EEF (onboard): {[f'{v:.1f}' for v in self._home_eef]}")
        else:
            # SIM_ONLY: derive home EEF from FK on the default joint config
            self._home_joints = list(self._current_joints)
            self._home_eef    = self._sim_robot.get_fk(self._home_joints)
            print(f"[SIM] Sim home EEF (mm/deg): {[f'{v:.1f}' for v in self._home_eef]}")

        # ── ACT recording: episode logger + dual cameras ─────────────────────
        if self._record:
            from logger import EpisodeLogger
            from camera import DualCamera
            self._logger  = EpisodeLogger()
            print("[REC] Starting cameras (wrist D405 + scene D435I)...")
            self._cameras = DualCamera()
            self._cameras.start()
            self._logger.set_cameras(self._cameras)
            print(f"[REC] Ready. Press '{RECORD_BUTTON}' to start/stop an episode, "
                  f"'{SUCCESS_BUTTON}' to mark the last episode successful.")

        print(f"[SIM] Opening Quest 3 ({QUEST3_MODE} transport)...")
        self._quest = Quest3Reader()
        self._quest.open()

        self._running = True
        print(
            f"[SIM] Bridge running — mode={self.mode.name}\n"
            "[SIM] Hold trigger ≥ 0.7 for 0.5 s to set home, then move freely."
        )

    def stop(self):
        self._running = False
        self._axis_logger.close()
        if self._logger and self._logger.recording:
            print("[REC] Bridge stopping — flushing in-progress episode.")
            self._logger.stop(success=None)
        if self._cameras:
            self._cameras.stop()
        if self._fr5:
            try:
                self._fr5.stop_servo_mode()
            except Exception:
                pass
            try:
                self._fr5.disconnect()
            except Exception:
                pass
        if self._quest:
            self._quest.close()
        if self._ros2:
            self._ros2.close()
        if self._sim_robot:
            self._sim_robot.close()

    # ── control loop ──────────────────────────────────────────────────────────

    def run(self):
        """125 Hz loop. Runs until Ctrl-C or self._running goes False."""
        if self.mode != Mode.LIVE_ONLY:
            ik_robot = self._sim_robot                       # SimRobot, DLS-patched
        elif EXECUTE_USE_DLS_IK:
            ik_robot = self._dls_ik                          # DLS IK on hardware
        else:
            ik_robot = self._fr5                             # FR5 onboard IK (RPC)

        homing             = False
        home_trigger_start = 0.0
        homed              = False
        last_sing_msg      = ""
        last_target_eef    = None

        deadline      = time.monotonic()
        diag_t        = time.monotonic()
        diag_valid    = 0
        diag_invalid  = 0
        diag_ik_ok    = 0
        diag_ik_fail  = 0

        while self._running:
            t0       = time.monotonic()
            deadline += LOOP_PERIOD

            state: ControllerState = self._quest.get_state()

            # ── ACT recording: button edges (start/stop, success) ─────────────
            if self._record:
                self._handle_record_buttons(state)

            # ── auto-home before an episode: ramp to the fixed start pose, then
            #    begin logging (so every demo's frame 0 is identical). No trigger
            #    needed; gentle hardware-safe per-cycle steps.
            if self._homing_for_record:
                cur = self._current_joints
                tgt = self._record_home
                if max(abs(c - t) for c, t in zip(cur, tgt)) < 0.8:
                    self._homing_for_record = False
                    print("[REC] at start pose — recording now")
                    self._logger.start()
                else:
                    step = [float(np.clip(t, c - MAX_DELTA_PER_JOINT[i], c + MAX_DELTA_PER_JOINT[i]))
                            for i, (c, t) in enumerate(zip(cur, tgt))]
                    if self._fr5:
                        try: self._fr5.servo_j(step)
                        except IOError: pass
                    self._current_joints = step
                    if self._sim_robot: self._sim_robot.set_joints(step)
                    if self._ros2:      self._ros2.publish(step)
                    _sleep_until(deadline)
                    continue

            # ── One-Euro filter: smooth raw pose before IK ────────────────────
            # Apply filter to the raw sensor values first — the filter must see
            # raw data to estimate signal speed correctly for the adaptive cutoff.
            # Both home capture and IK use the filtered pose so deltas are
            # computed entirely in the filtered domain (no mixed-frame artifacts).
            want_filt_dbg = FILTER_ENABLED and (t0 - self._filter_dbg_t >= 2.0)
            if FILTER_ENABLED and state.valid:
                pos_filt, quat_filt, filt_dbg = self._pose_filter.apply(
                    state.pos, state.quat, t0, debug=want_filt_dbg,
                )
            else:
                pos_filt  = state.pos
                quat_filt = state.quat
                filt_dbg  = None

            # ── live diagnostic every 2 s ─────────────────────────────────────
            if t0 - diag_t >= 2.0:
                j = self._current_joints
                joints_str = f"J=[{j[0]:+.1f},{j[1]:+.1f},{j[2]:+.1f},{j[3]:+.1f},{j[4]:+.1f},{j[5]:+.1f}]"
                eef_str = ("EEF=none" if last_target_eef is None else
                           f"EEF=[{last_target_eef[0]:+.1f},{last_target_eef[1]:+.1f},{last_target_eef[2]:+.1f}]mm")
                raw_str = (f"raw=[{state.pos[0]:+.3f},{state.pos[1]:+.3f},{state.pos[2]:+.3f}]m"
                           if state.valid else "STALE")
                print(
                    f"[DIAG] valid={state.valid}  trig={state.trigger:.2f}  homed={homed}"
                    f"  {raw_str}  {eef_str}  {joints_str}"
                    f"  got={diag_valid}/{diag_valid+diag_invalid}  ik={diag_ik_ok} fail={diag_ik_fail}",
                    flush=True,
                )

                # Per-axis mapping diagnostic — shows quest delta → FR5 delta so axis
                # mismatches are immediately visible without stopping the sim.
                # Quest frame: X=RIGHT  Y=UP  Z=BACK(toward user, negative=forward)
                # Move the controller in one direction at a time to read the mapping.
                if homed and self._quest_home_pos is not None and state.valid:
                    from mapper_vr import _R_VR2FR5
                    from config import VR_POSITION_SCALE
                    dq = pos_filt - self._quest_home_pos          # delta in Quest frame (m)
                    df = _R_VR2FR5 @ dq * VR_POSITION_SCALE       # delta in FR5 frame (mm)
                    print(
                        f"[AXMAP] quest Δ(RIGHT={dq[0]:+.3f}m  UP={dq[1]:+.3f}m  BACK={dq[2]:+.3f}m) "
                        f"→ FR5 Δ(X={df[0]:+.1f}mm  Y={df[1]:+.1f}mm  Z={df[2]:+.1f}mm)",
                        flush=True,
                    )

                if filt_dbg is not None:
                    print(filt_dbg.log_line(), flush=True)
                    self._filter_dbg_t = t0
                # DLS IK diagnostics every 2 s
                if t0 - self._ik_diag_t >= 2.0:
                    print(self._dls_ik.last_diag.log_line(), flush=True)
                    self._ik_diag_t = t0
                diag_t = t0; diag_valid = 0; diag_invalid = 0
                diag_ik_ok = 0; diag_ik_fail = 0

            if not state.valid:
                diag_invalid += 1
                _sleep_until(deadline)
                continue

            diag_valid += 1

            # ── gripper: each squeeze TOGGLES open/close ──────────────────────
            # Edge-triggered on the grip button, independent of the trigger/homing
            # so you can grasp/release whether or not the arm is being driven.
            if state.grip >= _GRIP_CLOSE_THRESH and not self._grip_held:
                self._grip_held    = True
                self._gripper_open = not self._gripper_open
                self._cmd_gripper(close=not self._gripper_open)
            elif state.grip <= _GRIP_OPEN_THRESH and self._grip_held:
                self._grip_held = False          # released → ready for next squeeze

            # ── J6 wrist roll via LEFT controller X (CW) / Y (CCW) ────────────
            # Independent of the right trigger; integrates while held. When idle
            # j6_target tracks the actual J6 so it never jumps when you start.
            btns = state.buttons or {}
            j6_roll = (1.0 if btns.get("X") else -1.0 if btns.get("Y") else 0.0)
            if j6_roll != 0.0:
                self._j6_target = float(np.clip(
                    self._j6_target + j6_roll * JOYSTICK_J6_SCALE,
                    FR5_JOINT_LIMITS[5][0], FR5_JOINT_LIMITS[5][1]))
            else:
                self._j6_target = self._current_joints[5]
            # Never let the roll target race ahead of the actual J6 (e.g. if the
            # arm can't keep up or a button sticks): clamp to one rate-cap step
            # from the current J6 so it can never exceed the per-cycle limit.
            self._j6_target = float(np.clip(
                self._j6_target,
                self._current_joints[5] - MAX_DELTA_PER_JOINT[5],
                self._current_joints[5] + MAX_DELTA_PER_JOINT[5]))

            # ── homing: first squeeze sets reference, then it stays set ─────────
            # Deadman: trigger must stay above threshold to keep arm moving.
            # Release → arm holds last position. Re-squeeze → resumes from new home.
            if state.trigger >= _TRIGGER_HOME_THRESH:
                if not homed:
                    # Lock home on first trigger press — use FILTERED pose so the
                    # home reference is in the same filtered domain as later deltas.
                    self._quest_home_pos  = pos_filt.copy()
                    self._quest_home_quat = quat_filt.copy()
                    # Lock J4/J5 baseline at home — wrist orientation is relative to this
                    self._wrist_home_j4 = self._current_joints[3]
                    self._wrist_home_j5 = self._current_joints[4]
                    self._j6_target     = self._current_joints[5]   # seed J6 roll at home
                    if self._fr5 and self.mode == Mode.SIM_THEN_EXECUTE:
                        self._home_joints    = self._fr5.get_joint_positions()
                        self._current_joints = list(self._home_joints)
                        # Use PyBullet FK for home EEF to keep IK in same frame
                        if self._sim_robot:
                            self._sim_robot.set_joints(self._home_joints)
                            self._home_eef = self._sim_robot.get_fk(self._home_joints)
                    elif self._fr5 and self.mode == Mode.LIVE_ONLY:
                        self._home_joints    = self._fr5.get_joint_positions()
                        self._current_joints = list(self._home_joints)
                        self._home_eef       = self._fr5.get_eef_pose()
                    elif self._sim_robot:
                        self._home_eef = self._sim_robot.get_fk(self._current_joints)
                    homed = True
                    if self._validator:
                        self._validator.reset()
                    print(f"[SIM] Home locked  EEF={[f'{v:.1f}' for v in self._home_eef]}")
                    print("[SIM] Move wrist to control EEF — release trigger to pause")
            else:
                if homed:
                    # Re-home on next trigger press from new position
                    homed = False

            if not homed or self._quest_home_pos is None:
                # Trigger not held: arm holds position, BUT left X/Y can still
                # roll J6 in place (J6-only command, button-held = deadman).
                if j6_roll != 0.0 and self._fr5:
                    j = list(self._current_joints)
                    j[5] = self._j6_target
                    ok = (self.mode != Mode.SIM_THEN_EXECUTE
                          or self._validator is None
                          or self._validator.check(j, self._current_joints).ok)
                    if ok:
                        try:
                            self._fr5.servo_j(j)
                            self._current_joints = j
                            if self._sim_robot: self._sim_robot.set_joints(j)
                            if self._ros2:      self._ros2.publish(j)
                        except IOError:
                            pass
                # Keep logging while recording even when not driving, so the
                # final gripper open/close (and any holds) are captured until
                # you press A to stop. Arm holds → command = current joints.
                if self._logger and self._logger.recording:
                    self._log_frame(time.time(), list(self._current_joints),
                                    last_target_eef, state)
                _sleep_until(deadline)
                continue

            # ── singularity scale ─────────────────────────────────────────────
            level, sing_scale, sing_msg = singularity.check(self._current_joints)
            if sing_msg and sing_msg != last_sing_msg:
                print(f"[SING] {sing_msg}")
                last_sing_msg = sing_msg
            elif not sing_msg:
                last_sing_msg = ""

            # ── Auto-return to safe home after idle ───────────────────────────
            # When the trigger is released (homed=False) the arm freezes. If it
            # stays frozen for AUTO_HOME_IDLE_S seconds we glide it back to the
            # factory default joint configuration so it never stays stretched.
            # The glide uses a capped rate so the motion is slow and safe.
            if not homed:
                if self._trigger_off_t is None:
                    self._trigger_off_t = t0          # start idle clock
                elif t0 - self._trigger_off_t >= AUTO_HOME_IDLE_S:
                    target_deg = list(_SIM_DEFAULT_HOME_DEG)
                    dist = max(abs(a - b) for a, b in
                               zip(target_deg, self._current_joints))
                    if dist > 0.5:                     # still away from home
                        lim  = [l * AUTO_HOME_SPEED for l in MAX_DELTA_PER_JOINT]
                        step = []
                        for tgt, cur, lm in zip(target_deg,
                                                self._current_joints, lim):
                            d = float(np.clip(tgt - cur, -lm, lm))
                            step.append(cur + d)
                        self._current_joints = step
                        if self._sim_robot and self.mode != Mode.LIVE_ONLY:
                            self._sim_robot.set_joints(self._current_joints)
                            self._ros2.publish(self._current_joints)
                        if abs(dist - round(dist)) < 0.1:  # print once per degree
                            print(f"[AUTO-HOME] returning… dist={dist:.1f}°",
                                  flush=True)
                    else:
                        print("[AUTO-HOME] reached safe home ✓", flush=True)
                        self._trigger_off_t = None     # reset so we only print once
                _sleep_until(deadline)
                continue
            else:
                self._trigger_off_t = None             # trigger held → cancel idle clock

            # ── IK (sim or real robot depending on mode) ──────────────────────
            try:
                target_joints = vr_to_fr5(
                    quest_pos        = pos_filt,
                    quest_quat       = quat_filt,
                    quest_home_pos   = self._quest_home_pos,
                    quest_home_quat  = self._quest_home_quat,
                    fr5_home_eef     = self._home_eef,
                    prev_fr5_joints  = self._current_joints,
                    robot            = ik_robot,
                    sing_scale       = sing_scale,
                )

                # ── Hand wrist orientation → J4 / J5, DECOUPLED from position ──
                # The right controller's rotation RELATIVE TO HOME is mapped
                # directly to J4 and J5 as a joint offset (absolute: hand back to
                # home orientation → J4/J5 back to home). This is independent of
                # the IK (which drives J1/J2/J3/J6 for position), so wrist tilt
                # does NOT scramble translation.
                #   hand PITCH (tilt up/down) → J4
                #   hand YAW   (turn L/R)     → J5
                if self._quest_home_quat is not None:
                    qd  = _quat_mul(quat_filt, _quat_inv(self._quest_home_quat))
                    eul = _rotvec_to_euler_deg(_quat_to_rotvec(qd))  # [rx,ry,rz] deg
                    j4_t = self._wrist_home_j4 + eul[0] * WRIST_GAIN_J4
                    j5_t = self._wrist_home_j5 + eul[1] * WRIST_GAIN_J5
                    # Rate-limit against the PREVIOUS wrist value (not the IK's J4),
                    # so the wrist command fully owns J4/J5 and the IK's use of
                    # those joints for position is overridden cleanly.
                    prev_j4 = self._current_joints[3]
                    prev_j5 = self._current_joints[4]
                    j4_t = float(np.clip(j4_t,
                        prev_j4 - MAX_DELTA_PER_JOINT[3],
                        prev_j4 + MAX_DELTA_PER_JOINT[3]))
                    j5_t = float(np.clip(j5_t,
                        prev_j5 - MAX_DELTA_PER_JOINT[4],
                        prev_j5 + MAX_DELTA_PER_JOINT[4]))
                    target_joints[3] = float(np.clip(j4_t,
                        FR5_JOINT_LIMITS[3][0], FR5_JOINT_LIMITS[3][1]))
                    target_joints[4] = float(np.clip(j5_t,
                        FR5_JOINT_LIMITS[4][0], FR5_JOINT_LIMITS[4][1]))

                    # J6 roll comes from LEFT-controller X/Y (integrated above,
                    # before the homing gate) — apply it over the IK's J6.
                    target_joints[5] = self._j6_target
            except IOError as exc:
                diag_ik_fail += 1
                print(f"[SIM] IK failed: {exc}")
                _sleep_until(deadline)
                continue

            diag_ik_ok += 1

            # Track EEF for DIAG (FK on result joints)
            if self._sim_robot:
                try:
                    last_target_eef = self._sim_robot.get_fk(target_joints)
                except Exception:
                    pass

            # ── axis logger — VR vs EEF side-by-side CSV + terminal ───────────
            self._axis_logger.update(
                t               = t0,
                trigger         = state.trigger,
                grip            = state.grip,
                quest_pos_filt  = pos_filt,
                quest_quat      = quat_filt,
                quest_home_pos  = self._quest_home_pos,
                quest_home_quat = self._quest_home_quat,
                fr5_home_eef    = self._home_eef,
                fr5_current_eef = last_target_eef,
                fr5_joints      = target_joints,
                joystick_x      = state.joystick_x,
                joystick_y      = state.joystick_y,
                joy_j4          = self._wrist_home_j4,
                joy_j5          = self._wrist_home_j5,
            )

            # ── update sim visualization ──────────────────────────────────────
            if self._sim_robot and self.mode != Mode.LIVE_ONLY:
                self._sim_robot.set_joints(target_joints)
                self._ros2.publish(target_joints)

            # ── hardware execution ────────────────────────────────────────────
            if self.mode == Mode.SIM_THEN_EXECUTE:
                result = self._validator.check(target_joints, self._current_joints)
                for w in result.warnings:
                    print(f"[VAL] {w}")
                if not result.ok:
                    print(f"[SIM] BLOCKED ({result.reason})")
                    _sleep_until(deadline)
                    continue
                try:
                    self._fr5.servo_j(target_joints)
                except IOError as exc:
                    print(f"[FR5] ServoJ error (skipping cycle): {exc}")
                    _sleep_until(deadline)
                    continue

            elif self.mode == Mode.LIVE_ONLY:
                try:
                    self._fr5.servo_j(target_joints)
                    if self._ros2:
                        self._ros2.publish(target_joints)
                except IOError as exc:
                    print(f"[FR5] ServoJ error (skipping cycle): {exc}")
                    _sleep_until(deadline)
                    continue

            self._current_joints = target_joints

            # ── ACT recording: log this commanded frame ───────────────────────
            if self._logger and self._logger.recording:
                self._log_frame(time.time(), target_joints, last_target_eef, state)

            _sleep_until(deadline)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _handle_record_buttons(self, state: "ControllerState"):
        """A/X rising edge → start/stop episode; B/Y rising edge → mark success."""
        btns = state.buttons or {}
        def edge(name: str) -> bool:
            now  = bool(btns.get(name, False))
            prev = bool(self._prev_buttons.get(name, False))
            self._prev_buttons[name] = now
            return now and not prev

        if edge(RECORD_BUTTON):
            if self._logger.recording:
                self._logger.stop(success=self._last_success)
                self._last_success = None
            elif not self._homing_for_record:
                # Don't start logging yet — first auto-ramp to the fixed home pose
                # (handled in the run loop); logging begins once home is reached.
                self._last_success = None
                if not self._gripper_open:        # start every episode gripper-open
                    self._gripper_open = True
                    self._cmd_gripper(close=False)
                self._homing_for_record = True
                print("[REC] homing to start pose before recording ...")
        if edge(SUCCESS_BUTTON):
            if self._logger.recording:
                self._last_success = True
                print("[REC] success flag armed — will mark this episode successful on stop.")
            elif self._logger._last_path:
                self._mark_last_success(True)

    def _mark_last_success(self, success: bool):
        import json
        p = os.path.join(self._logger._last_path, "meta.json")
        try:
            m = json.load(open(p)); m["success"] = success
            json.dump(m, open(p, "w"), indent=2)
            print(f"[REC] {self._logger._last_path} success={success}")
        except Exception as exc:
            print(f"[REC] could not set success: {exc}")

    def _log_frame(self, ts: float, target_joints, last_target_eef, state):
        """Read FR5 state (cheap CNDE-cache reads) and append one ACT row."""
        self._log_n += 1
        fr5_actual = list(target_joints)     # sim fallback = commanded
        fr5_vel    = None
        fr5_eef    = last_target_eef
        if self._fr5:
            # All flag=0 reads hit the CNDE cache (cheap, non-blocking).
            try:    fr5_actual = self._fr5.get_joint_positions()
            except Exception: pass
            try:    fr5_vel = self._fr5.get_joint_velocities()
            except Exception: pass
            try:    fr5_eef = self._fr5.get_eef_pose()
            except Exception: pass
            # Gripper position may be a blocking RPC — read ~12 Hz and cache.
            if self._log_n % 10 == 1:
                self._gripper_norm = self._fr5.get_gripper_position()
        self._logger.log(
            timestamp    = ts,
            fr5_actual   = fr5_actual,
            fr5_cmd      = target_joints,
            gripper_cmd  = 0 if self._gripper_open else 1,
            fr5_eef      = fr5_eef,
            gripper_norm = self._gripper_norm,
            fr5_vel      = fr5_vel,
            vr_trigger   = state.trigger,
            vr_grip      = state.grip,
        )

    def _cmd_gripper(self, close: bool):
        label = "CLOSE" if close else "OPEN"
        if self.mode == Mode.SIM_ONLY:
            print(f"[GRIP] {label} (sim only)")
            return
        if self._fr5:
            pct = GRIPPER_CLOSE_PCT if close else GRIPPER_OPEN_PCT
            self._fr5.send_gripper(
                GRIPPER_INDEX, pct, GRIPPER_VEL_PCT, GRIPPER_FORCE_PCT,
                GRIPPER_MAXTIME_MS, 0, GRIPPER_TYPE,
            )
            print(f"[GRIP] {label}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ── timing ────────────────────────────────────────────────────────────────────

def _sleep_until(deadline: float):
    remaining = deadline - time.monotonic()
    if remaining > 0.0:
        time.sleep(remaining)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FR5 VR Sim Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  sim       -- SIM_ONLY: PyBullet IK + RViz, no hardware\n"
            "  validate  -- SIM_THEN_EXECUTE: sim validates, then sends to FR5\n"
            "  execute   -- LIVE_ONLY: direct VR→FR5 (same as teleop_vr.py)\n"
        ),
    )
    parser.add_argument(
        "--mode", choices=["sim", "validate", "execute"], default="sim",
        help="control mode (default: sim)",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="open PyBullet GUI window (requires display; ignored in execute mode)",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="enable ACT episode recording (dual cameras + Quest A/B button toggles)",
    )
    args = parser.parse_args()

    mode = _MODE_MAP[args.mode]
    print(f"[SIM] mode={mode.name}  gui={args.gui}  record={args.record}")

    with SimBridge(mode=mode, gui=args.gui, record=args.record) as bridge:
        try:
            bridge.run()
        except KeyboardInterrupt:
            print("\n[SIM] Shutting down.")


if __name__ == "__main__":
    main()
