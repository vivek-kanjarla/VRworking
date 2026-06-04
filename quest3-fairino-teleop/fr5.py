"""
fr5.py — controls the FR5 follower cobot via the Fairino SDK (ServoJ mode).
"""

import threading
import time

from fairino import Robot
from config import FR5_IP, FR5_SERVO_VEL, FR5_FILTER_T


class FR5Controller:
    def __init__(self):
        self._robot    = None
        # Single lock serialises all RPC calls — xmlrpc.client is not thread-safe.
        # Both the 125 Hz ServoJ loop and the 5 Hz gripper thread acquire this.
        self._rpc_lock = threading.Lock()

    def connect(self):
        self._robot = Robot.RPC(FR5_IP)
        Robot.RPC.is_conect = True   # force XML-RPC mode; writing never raises AttributeError

        self._robot.StopMove()
        self._robot.ResetAllError()
        time.sleep(0.3)
        self._robot.Mode(0)
        self._robot.RobotEnable(1)
        time.sleep(0.5)   # servo drives need ~500 ms to fully energise after enable

    def start_servo_mode(self):
        time.sleep(0.1)
        with self._rpc_lock:
            err = self._robot.ServoMoveStart()
        if err not in (0, None):
            raise IOError(f"ServoMoveStart failed with error {err}")

    def stop_servo_mode(self):
        try:
            with self._rpc_lock:
                self._robot.ServoMoveEnd()
        except Exception:
            pass

    def reset_errors(self):
        with self._rpc_lock:
            self._robot.ResetAllError()

    def disconnect(self):
        self.stop_servo_mode()
        self._robot = None

    def get_joint_positions(self) -> list[float]:
        with self._rpc_lock:
            raw = self._robot.GetActualJointPosDegree(0)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise IOError(
                f"GetActualJointPosDegree returned unexpected value: {raw!r} "
                "(CNDE may not be connected — kill suspended teleop.py and retry)"
            )
        err, joints = raw
        if err != 0:
            raise IOError(f"GetActualJointPosDegree failed with error {err}")
        return list(joints)

    def get_eef_pose(self) -> list[float]:
        """Return [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] — TCP pose in flange frame."""
        with self._rpc_lock:
            raw = self._robot.GetActualTCPPose(0)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise IOError(f"GetActualTCPPose returned unexpected value: {raw!r}")
        err, pose = raw
        if err != 0:
            raise IOError(f"GetActualTCPPose failed with error {err}")
        return list(pose)

    def get_inverse_kin(self, eef_pose: list[float], ref_joints: list[float]) -> list[float]:
        """
        Compute inverse kinematics for a target TCP pose.

        eef_pose   — [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        ref_joints — reference joint angles (deg) used to select the nearest IK solution

        Returns list of 6 joint angles in degrees, or raises IOError on failure.
        """
        eef_pose   = [float(v) for v in eef_pose]
        ref_joints = [float(v) for v in ref_joints]
        with self._rpc_lock:
            raw = self._robot.GetInverseKinRef(0, eef_pose, ref_joints)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise IOError(f"GetInverseKinRef returned unexpected value: {raw!r}")
        err, joints = raw
        if err != 0:
            raise IOError(f"GetInverseKinRef failed with error {err} (target pose may be unreachable)")
        return [float(j) for j in joints]

    def get_joint_velocities(self) -> list[float]:
        """Return [v1..v6] — actual joint velocities in deg/s."""
        with self._rpc_lock:
            raw = self._robot.GetActualJointSpeedsDegree(0)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise IOError(f"GetActualJointSpeedsDegree returned unexpected value: {raw!r}")
        err, vels = raw
        if err != 0:
            raise IOError(f"GetActualJointSpeedsDegree failed with error {err}")
        return list(vels)

    def get_gripper_position(self) -> float | None:
        """
        Observed jaw position normalized to 0..1 (raw SDK position / 100).

        Best-effort — returns None if no gripper is configured. NOTE: on the
        AG-160-95 config used here MoveGripper sends 0=close / 100=open, so the
        reported position is ~0 when closed and ~1.0 when open. Returned raw
        (not inverted); flip at export if you want 1.0 = closed.
        """
        try:
            with self._rpc_lock:
                raw = self._robot.GetGripperCurPosition()
        except Exception:
            return None
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            err, pos = raw
            if err != 0:
                return None
        else:
            pos = raw
        try:
            return float(pos) / 100.0
        except Exception:
            return None

    def servo_j(self, joints_deg: list[float]):
        # Coerce to plain Python float — xmlrpc.client cannot marshal
        # numpy.float64 (raises TypeError). This is the Python→XML-RPC
        # boundary, so normalise here regardless of what the caller passes.
        joints_deg = [float(j) for j in joints_deg]
        with self._rpc_lock:
            err = self._robot.ServoJ(
                joints_deg, [0] * 6, FR5_SERVO_VEL, 0, 0.008, FR5_FILTER_T, 0
            )
        if err not in (0, None):
            raise IOError(f"ServoJ failed with error {err}")

    def activate_gripper(self, index: int) -> int:
        with self._rpc_lock:
            return self._robot.ActGripper(index, 1)

    def send_gripper(self, index, pct, vel, force, maxtime, blocking, gtype) -> int:
        with self._rpc_lock:
            return self._robot.MoveGripper(
                index, pct, vel, force, maxtime, blocking, gtype, 0, 0, 0
            )

    def stop(self):
        try:
            with self._rpc_lock:
                self._robot.StopMotion()
        except Exception:
            pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.stop()
        self.disconnect()
