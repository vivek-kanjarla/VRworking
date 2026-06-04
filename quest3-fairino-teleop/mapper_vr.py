"""
mapper_vr.py — maps Meta Quest 3 controller pose to FR5 EEF commands.

Strategy: delta-based Cartesian control.
  1. Record Quest 3 controller pose at "home" (startup or H re-home).
  2. Each cycle compute the delta (translation + rotation) from home.
  3. Apply coordinate-frame rotation (Quest OpenXR → FR5 TCP frame).
  4. Scale the delta and add it to FR5's home EEF pose.
  5. Clamp per-axis deltas to VR_MAX_DELTA_POS_MM / VR_MAX_DELTA_ROT_DEG.
  6. Call FR5 inverse kinematics (GetInverseKin) to convert EEF target → joints.
  7. Apply per-joint rate limiting matching the existing MAX_DELTA_PER_JOINT caps.

Rotation representation:
  Quest 3 reports quaternions. We compute the rotation delta as:
      q_delta = q_now * q_home^-1
  Convert q_delta to axis-angle, rotate the axis into the FR5 frame,
  then extract Euler angles (rx, ry, rz in degrees) to add to FR5 home orientation.

All numpy types are coerced to plain Python floats at the IK call boundary
so xmlrpc.client can marshal them.
"""

import math
import time

import numpy as np

from config import (
    VR_POSITION_SCALE,
    VR_ROTATION_SCALE,
    VR_MAX_DELTA_POS_MM,
    VR_MAX_DELTA_ROT_DEG,
    VR_FRAME_ROTATION,
    MAX_DELTA_DEG_PER_CYCLE,
    MAX_DELTA_PER_JOINT,
    FR5_JOINT_LIMITS,
    USE_SE3_RETARGET,
    SE3_RETARGET_MODE,
)

# Pre-compute the 3×3 frame rotation matrix Quest → FR5
_R_VR2FR5 = np.array(VR_FRAME_ROTATION, dtype=np.float64)  # shape (3, 3)

# ── SE(3) retargeter (module-level singleton; home re-set each call) ──────────
_se3_rt = None
if USE_SE3_RETARGET:
    from retarget_se3 import SE3Retargeter, SE3RetargetConfig, RetargetMode
    _se3_rt = SE3Retargeter(SE3RetargetConfig(
        frame_rotation = _R_VR2FR5,
        pos_scale      = VR_POSITION_SCALE,
        rot_scale      = VR_ROTATION_SCALE,
        workspace_mm   = VR_MAX_DELTA_POS_MM,
        max_rot_deg    = VR_MAX_DELTA_ROT_DEG,
        mode = (RetargetMode.BODY if SE3_RETARGET_MODE == "body"
                else RetargetMode.SPATIAL),
    ))


# ── Quaternion helpers ────────────────────────────────────────────────────────

def _quat_inv(q: np.ndarray) -> np.ndarray:
    """Invert a unit quaternion [qx, qy, qz, qw]."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [qx, qy, qz, qw]."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion to rotation vector (axis * angle_rad)."""
    q = q / (np.linalg.norm(q) + 1e-12)
    # clamp w to [-1, 1] for numerical safety
    w = float(np.clip(q[3], -1.0, 1.0))
    angle = 2.0 * np.arccos(abs(w))
    axis  = q[:3]
    norm  = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.zeros(3)
    if w < 0:
        angle = -angle
    return (axis / norm) * angle


def _rotvec_to_euler_deg(rvec: np.ndarray) -> np.ndarray:
    """
    Convert rotation vector to ZYX Euler angles in degrees.
    These match the rx, ry, rz convention used by the FR5 TCP pose.
    """
    angle = np.linalg.norm(rvec)
    if angle < 1e-9:
        return np.zeros(3)

    axis = rvec / angle
    c, s = np.cos(angle), np.sin(angle)
    t    = 1.0 - c
    x, y, z = axis

    # Rotation matrix from axis-angle
    R = np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])

    # ZYX (roll=Rz, pitch=Ry, yaw=Rx) decomposition — same as FR5 Euler convention
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        rx = np.arctan2( R[2, 1],  R[2, 2])
        ry = np.arctan2(-R[2, 0],  sy)
        rz = np.arctan2( R[1, 0],  R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2],  R[1, 1])
        ry = np.arctan2(-R[2, 0],  sy)
        rz = 0.0

    return np.degrees(np.array([rx, ry, rz]))


# ── Main mapping function ─────────────────────────────────────────────────────

def vr_to_fr5(
    quest_pos:       np.ndarray,         # current controller position (m)
    quest_quat:      np.ndarray,         # current controller orientation [qx,qy,qz,qw]
    quest_home_pos:  np.ndarray,         # controller position at home
    quest_home_quat: np.ndarray,         # controller orientation at home
    fr5_home_eef:    list[float],        # FR5 TCP pose at home [x,y,z,rx,ry,rz] (mm, deg)
    prev_fr5_joints: list[float],        # last commanded FR5 joint angles (deg) — IK seed + rate limiter
    robot,                               # FR5Controller or SimRobot — get_inverse_kin()
    sing_scale:      float = 1.0,        # singularity speed scale [0, 1]
    pose_filter      = None,             # optional PoseFilter — applied before delta mapping
    filter_t:        float | None = None,# time.monotonic() for the filter; auto if None
) -> list[float]:
    """
    Convert Quest 3 controller pose to FR5 6-joint command.

    Filtering note:
      If `pose_filter` is provided it is applied to (quest_pos, quest_quat)
      BEFORE any delta computation. This is the correct injection point: the
      filter must see the raw controller pose so it can estimate speed
      accurately, and the filtered pose must be used for both the delta
      calculation and the home-relative mapping.

      When called from a control loop, pass the current loop timestamp as
      `filter_t` (time.monotonic()) so the filter uses the true inter-frame
      dt rather than re-querying the clock internally.

    Returns a list of 6 joint angles in degrees, or raises IOError if IK fails.
    """
    # ── 0. Apply pose filter (One-Euro) before any delta computation ──────────
    # The filter must run on the raw sensor values — not on derived quantities
    # such as deltas — so it can observe the true signal speed and set the
    # adaptive cutoff correctly.
    if pose_filter is not None:
        t = filter_t if filter_t is not None else time.monotonic()
        quest_pos, quest_quat, _ = pose_filter.apply(quest_pos, quest_quat, t)

    # ── 1-6. Retarget controller pose → FR5 EEF target ───────────────────────
    if _se3_rt is not None:
        # Unified SE(3) relative-transform pipeline (retarget_se3.py).
        # Re-set home each call (cheap) so this stays drop-in compatible with
        # the stateless vr_to_fr5 signature — home args are passed in every cycle.
        _se3_rt.set_home(quest_home_pos, quest_home_quat, fr5_home_eef)
        target_eef = _se3_rt.compute_target(quest_pos, quest_quat)
        # Singularity speed-scale acts on the per-cycle delta from home.
        if sing_scale < 1.0:
            home = np.array(fr5_home_eef, dtype=np.float64)
            tgt  = np.array(target_eef, dtype=np.float64)
            target_eef = list(home + (tgt - home) * sing_scale)
    else:
        # Legacy separate translation-delta + Euler-delta pipeline.
        delta_pos_quest = (quest_pos - quest_home_pos) * VR_POSITION_SCALE
        delta_pos_fr5 = _R_VR2FR5 @ delta_pos_quest
        delta_pos_fr5 = np.clip(
            delta_pos_fr5 * sing_scale, -VR_MAX_DELTA_POS_MM, VR_MAX_DELTA_POS_MM)
        q_delta = _quat_mul(quest_quat, _quat_inv(quest_home_quat))
        rvec    = _quat_to_rotvec(q_delta)
        rvec_fr5 = _R_VR2FR5 @ rvec
        euler_delta = _rotvec_to_euler_deg(rvec_fr5) * VR_ROTATION_SCALE
        euler_delta = np.clip(
            euler_delta * sing_scale, -VR_MAX_DELTA_ROT_DEG, VR_MAX_DELTA_ROT_DEG)
        home = np.array(fr5_home_eef, dtype=np.float64)
        target_eef = [
            float(home[0] + delta_pos_fr5[0]),
            float(home[1] + delta_pos_fr5[1]),
            float(home[2] + delta_pos_fr5[2]),
            float(home[3] + euler_delta[0]),
            float(home[4] + euler_delta[1]),
            float(home[5] + euler_delta[2]),
        ]

    # 7. Inverse kinematics — uses prev_fr5_joints as the seed to pick the
    #    nearest solution and avoid solution jumps near singularities.
    fr5_joints = robot.get_inverse_kin(target_eef, prev_fr5_joints)

    # 8. Clamp joint targets to hard limits
    joints_clamped = [
        max(lo, min(hi, j))
        for j, (lo, hi) in zip(fr5_joints, FR5_JOINT_LIMITS)
    ]

    # 9. Per-joint rate limiting with singularity scale
    limits = [lim * sing_scale for lim in MAX_DELTA_PER_JOINT]
    result = []
    for target, prev, lim in zip(joints_clamped, prev_fr5_joints, limits):
        d = float(np.clip(target - prev, -lim, lim))
        result.append(float(prev + d))

    # 10. Wrap J6 to ±180° — prevents continuous wrist spinning when orientation
    #     mapping accumulates. IK may drive J6 past ±180° unnecessarily.
    j6 = result[5]
    if j6 > 180.0:
        result[5] = j6 - 360.0
    elif j6 < -180.0:
        result[5] = j6 + 360.0

    return result
