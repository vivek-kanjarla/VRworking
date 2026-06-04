"""
retarget_se3.py — unified SE(3) relative-transform pose retargeting.

Replaces the separate translation-delta + quaternion-delta pipeline in
mapper_vr.py with a single homogeneous-transform formulation.

═══════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULATION
═══════════════════════════════════════════════════════════════════════════

Poses are 4×4 homogeneous transforms in SE(3):

        ┌         ┐
   T =  │  R   p  │     R ∈ SO(3),  p ∈ ℝ³
        │  0   1  │
        └         ┘

Frames
------
  W  : Quest world frame (WebXR local-floor reference)
  C  : controller frame
  B  : FR5 base frame
  E  : FR5 end-effector (TCP) frame

Inputs each cycle
-----------------
  T_W_Chome : controller pose at home, in Quest world W   (locked on trigger)
  T_W_Ccur  : controller pose now, in Quest world W        (filtered Quest pose)
  T_B_Ehome : FR5 EEF pose at home, in FR5 base B          (from FK at home)

Relative controller motion
---------------------------
Two conventions (RETARGET_MODE selects):

  SPATIAL (default — world-fixed control directions):
      ΔT_W = T_W_Ccur · T_W_Chome⁻¹
      "move right in the room" is always world-right, regardless of how the
      controller is twisted.  Matches the calibrated VR_FRAME_ROTATION mapping.

  BODY (controller-relative directions — the spec's T_home⁻¹·T_current):
      ΔT_C = T_W_Chome⁻¹ · T_W_Ccur
      "move right" follows the controller's own axes as you rotate it.

Decoupled extraction (avoids screw-coupling)
--------------------------------------------
A naïve homogeneous product T_cur·T_home⁻¹ couples rotation into translation
through the lever arm (p_home): rotating the controller in place produces a
spurious EEF translation.  For teleoperation we want rotation about the EEF
point and translation independent of it, so we extract the two channels:

      ΔR = R_cur · R_homeᵀ            (spatial relative rotation, SO(3))
      Δp = p_cur − p_home             (world-frame translation, ℝ³)

These reassemble into a decoupled SE(3) motion applied at the EEF.

Frame change  W → B  (constant rotation X = VR_FRAME_ROTATION)
--------------------------------------------------------------
X is a pure rotation (no translation), so re-expressing the motion in the
FR5 base frame is the adjoint of a rotation:

      Δp_B = X · Δp_W                 (rotate the translation vector)
      ΔR_B = X · ΔR_W · Xᵀ            (conjugate the rotation — change of basis)

Equivalently, on the Lie algebra:  ω_B = X · ω_W   where  ω = log₃(ΔR).

Motion scaling via the Lie algebra log-map
-------------------------------------------
Translation scaling is a plain scalar.  Rotation scaling is done on the
tangent space (twist), NOT by multiplying Euler angles — Euler scaling is
wrong for large angles and breaks near gimbal lock:

      ω      = log₃(ΔR_B)             (axis·angle, ℝ³, the SO(3) tangent vector)
      ω_s    = rot_scale · ω
      ΔR_s   = exp₃(ω_s)              (back to SO(3) — geodesic-correct scaling)

Workspace scaling (translation)
-------------------------------
Uniform clamp on the translation NORM (preserves direction), unlike the old
per-axis clip which could distort the commanded direction at the corners:

      if |Δp_B·s| > r_max :  Δp_B ← Δp_B · r_max / |Δp_B·s|

Final target (applied at the EEF point, decoupled)
--------------------------------------------------
      p_target = p_Ehome + scale_pos · Δp_B          (clamped)
      R_target = ΔR_s · R_Ehome

Output:  [x,y,z (mm), rx,ry,rz (deg, ZYX)]  — identical format to the old
mapper, so the DLS IK and ServoJ streaming downstream are unchanged.

═══════════════════════════════════════════════════════════════════════════
STABILITY vs PREVIOUS IMPLEMENTATION
═══════════════════════════════════════════════════════════════════════════
  · Rotation scaling: OLD multiplied ZYX Euler angles (discontinuous near
    ±90° pitch, wrong magnitude for large angles).  NEW scales the SO(3)
    log-map — continuous, correct geodesic magnitude, no gimbal artefacts.
  · Frame consistency: OLD applied VR_FRAME_ROTATION to the position vector
    AND (separately) to the rotation axis-vector — two code paths that could
    drift apart.  NEW uses ONE rotation X via vector-rotate + conjugation,
    provably consistent.
  · Translation: NEW reproduces the OLD mapping exactly (X·(p_cur−p_home)),
    so the calibration carries over with zero regression.

References
----------
  Murray, Li, Sastry (1994) "A Mathematical Introduction to Robotic
    Manipulation" — Ch.2 SE(3), twists, exp/log maps.
  Lynch & Park (2017) "Modern Robotics" — Ch.3 rigid-body motions.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pinocchio as pin


class RetargetMode(Enum):
    SPATIAL = auto()   # ΔT = T_cur · T_home⁻¹  (world-fixed directions, default)
    BODY    = auto()   # ΔT = T_home⁻¹ · T_cur  (controller-relative directions)


@dataclass
class SE3RetargetConfig:
    frame_rotation:   np.ndarray = field(            # X: Quest W → FR5 B  (3×3)
        default_factory=lambda: np.eye(3))
    pos_scale:        float = 1500.0   # mm of EEF per metre of hand
    rot_scale:        float = 0.4      # rotation gain on the log-map twist
    workspace_mm:     float = 900.0    # max translation norm from home (mm)
    max_rot_deg:      float = 90.0     # max rotation magnitude from home (deg)
    mode:             RetargetMode = RetargetMode.SPATIAL


@dataclass
class FrameDebug:
    """Diagnostic snapshot of one retargeting cycle (for visualization)."""
    dp_world_mm:   np.ndarray = field(default_factory=lambda: np.zeros(3))
    dp_fr5_mm:     np.ndarray = field(default_factory=lambda: np.zeros(3))
    rot_angle_deg: float = 0.0          # |log3(ΔR_B)| after scaling
    rot_axis:      np.ndarray = field(default_factory=lambda: np.zeros(3))
    twist_v:       np.ndarray = field(default_factory=lambda: np.zeros(3))  # m
    twist_w:       np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad
    clamped:       bool = False

    def log_line(self) -> str:
        return (
            f"[SE3] Δp_W=[{self.dp_world_mm[0]:+.0f},{self.dp_world_mm[1]:+.0f},"
            f"{self.dp_world_mm[2]:+.0f}]mm → "
            f"Δp_B=[{self.dp_fr5_mm[0]:+.0f},{self.dp_fr5_mm[1]:+.0f},"
            f"{self.dp_fr5_mm[2]:+.0f}]mm  "
            f"∠={self.rot_angle_deg:.1f}°"
            f"{'  [clamped]' if self.clamped else ''}"
        )


# ── Quaternion → rotation matrix ([qx,qy,qz,qw] convention) ───────────────────

def _quat_to_R(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / (np.linalg.norm(q) + 1e-12)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ])


# ── SE(3) retargeter ──────────────────────────────────────────────────────────

class SE3Retargeter:
    """
    Unified SE(3) relative-transform retargeting.

    Usage:
        rt = SE3Retargeter(SE3RetargetConfig(frame_rotation=R, pos_scale=1500))
        rt.set_home(quest_pos, quest_quat, fr5_home_eef)   # on trigger press
        eef_target = rt.compute_target(quest_pos, quest_quat)  # each cycle
    """

    def __init__(self, cfg: SE3RetargetConfig | None = None):
        self.cfg = cfg or SE3RetargetConfig()
        self._X  = np.asarray(self.cfg.frame_rotation, dtype=float)   # W→B rotation

        # Home transforms (set in set_home)
        self._T_W_Chome: pin.SE3 | None = None   # controller home in Quest world
        self._T_B_Ehome: pin.SE3 | None = None   # EEF home in FR5 base
        self.last_debug = FrameDebug()

    # ── home capture ──────────────────────────────────────────────────────────

    def set_home(
        self,
        quest_pos:    np.ndarray,        # controller position (m), Quest world
        quest_quat:   np.ndarray,        # controller quat [qx,qy,qz,qw]
        fr5_home_eef: list[float],       # EEF [x,y,z,rx,ry,rz] mm/deg, FR5 base
    ) -> None:
        R_c = _quat_to_R(quest_quat)
        self._T_W_Chome = pin.SE3(R_c, np.asarray(quest_pos, dtype=float))

        p_e = np.asarray(fr5_home_eef[:3], dtype=float) / 1000.0     # mm → m
        R_e = pin.rpy.rpyToMatrix(*np.radians(fr5_home_eef[3:6]))    # ZYX Euler
        self._T_B_Ehome = pin.SE3(R_e, p_e)

    def is_homed(self) -> bool:
        return self._T_W_Chome is not None and self._T_B_Ehome is not None

    # ── per-cycle retargeting ───────────────────────────────────────────────────

    def compute_target(
        self,
        quest_pos:  np.ndarray,
        quest_quat: np.ndarray,
        debug:      bool = False,
    ) -> list[float]:
        """
        Map current controller pose to an FR5 EEF target.

        Returns [x,y,z,rx,ry,rz] in mm and degrees (ZYX), ready for the IK.
        """
        if not self.is_homed():
            raise RuntimeError("SE3Retargeter.set_home() not called")

        cfg = self.cfg

        # ── 1. Current controller transform (Quest world) ─────────────────────
        R_cur = _quat_to_R(quest_quat)
        p_cur = np.asarray(quest_pos, dtype=float)
        T_W_Ccur = pin.SE3(R_cur, p_cur)

        R_home = self._T_W_Chome.rotation
        p_home = self._T_W_Chome.translation

        # ── 2. Relative motion (decoupled spatial / body) ─────────────────────
        if cfg.mode == RetargetMode.SPATIAL:
            # World-fixed directions.  ΔR = R_cur·R_homeᵀ ; Δp = p_cur − p_home.
            dR_world = R_cur @ R_home.T
            dp_world = p_cur - p_home
        else:  # BODY  — controller-relative (spec's T_home⁻¹·T_cur)
            dR_world = R_home.T @ R_cur
            dp_world = R_home.T @ (p_cur - p_home)

        # ── 3. Frame change W → B  (single rotation X) ────────────────────────
        dp_fr5 = self._X @ dp_world                    # rotate translation vector
        dR_fr5 = self._X @ dR_world @ self._X.T        # conjugate rotation

        # ── 4. Motion scaling on the Lie algebra ──────────────────────────────
        # Translation: scalar gain (m → mm with pos_scale).
        dp_scaled = dp_fr5 * cfg.pos_scale             # mm

        # Rotation: scale the SO(3) log-map (NOT Euler angles).
        omega = pin.log3(dR_fr5)                       # axis·angle, rad
        omega_scaled = omega * cfg.rot_scale
        # Clamp rotation magnitude
        ang = np.linalg.norm(omega_scaled)
        max_ang = math.radians(cfg.max_rot_deg)
        if ang > max_ang:
            omega_scaled *= max_ang / ang
        dR_scaled = pin.exp3(omega_scaled)             # back to SO(3)

        # ── 5. Workspace clamp (uniform on translation norm) ──────────────────
        clamped = False
        norm = np.linalg.norm(dp_scaled)
        if norm > cfg.workspace_mm:
            dp_scaled *= cfg.workspace_mm / norm
            clamped = True

        # ── 6. Apply decoupled motion at the EEF point ────────────────────────
        p_target = self._T_B_Ehome.translation * 1000.0 + dp_scaled    # mm
        R_target = dR_scaled @ self._T_B_Ehome.rotation

        rpy_deg = np.degrees(pin.rpy.matrixToRpy(R_target))

        # ── 7. Diagnostics ────────────────────────────────────────────────────
        if debug:
            self.last_debug = FrameDebug(
                dp_world_mm   = dp_world * 1000.0,
                dp_fr5_mm     = dp_scaled.copy(),
                rot_angle_deg = float(np.degrees(np.linalg.norm(omega_scaled))),
                rot_axis      = (omega_scaled / (np.linalg.norm(omega_scaled)+1e-12)),
                twist_v       = dp_fr5.copy(),
                twist_w       = omega.copy(),
                clamped       = clamped,
            )

        return [
            float(p_target[0]), float(p_target[1]), float(p_target[2]),
            float(rpy_deg[0]),  float(rpy_deg[1]),  float(rpy_deg[2]),
        ]


# ── Self-test / regression vs legacy mapper ──────────────────────────────────

def _selftest() -> None:
    """
    Verify SE(3) retargeter reproduces the legacy translation mapping and
    correctly scales rotation via the log-map.
        python3 retarget_se3.py
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import VR_FRAME_ROTATION, VR_POSITION_SCALE, VR_ROTATION_SCALE

    X = np.array(VR_FRAME_ROTATION, dtype=float)
    cfg = SE3RetargetConfig(
        frame_rotation = X,
        pos_scale      = VR_POSITION_SCALE,
        rot_scale      = VR_ROTATION_SCALE if VR_ROTATION_SCALE > 0 else 0.4,
        mode           = RetargetMode.SPATIAL,
    )
    rt = SE3Retargeter(cfg)

    HOME_EEF = [266.6, -247.8, 562.8, 179.1, 7.8, -149.3]
    q_id = np.array([0., 0., 0., 1.])
    home_pos = np.array([0.1, 0.8, -0.2])

    rt.set_home(home_pos, q_id, HOME_EEF)

    print("\n=== SE(3) Retargeter self-test ===\n")
    print("Translation: SE(3) vs legacy  X·(p_cur−p_home)·scale")
    print(f"{'hand move (m)':22s}  {'SE3 Δp (mm)':28s}  {'legacy Δp (mm)':28s}  match")
    print("─"*90)
    for label, dp in [
        ("right +0.1 (X)", [0.1, 0, 0]),
        ("up    +0.1 (Y)", [0, 0.1, 0]),
        ("fwd   -0.1 (Z)", [0, 0, -0.1]),
    ]:
        cur = home_pos + np.array(dp)
        tgt = rt.compute_target(cur, q_id, debug=True)
        se3_dp = np.array(tgt[:3]) - np.array(HOME_EEF[:3])
        legacy = X @ np.array(dp) * VR_POSITION_SCALE
        ok = np.allclose(se3_dp, legacy, atol=0.5)
        print(f"  {label:20s}  [{se3_dp[0]:+7.1f},{se3_dp[1]:+7.1f},{se3_dp[2]:+7.1f}]  "
              f"[{legacy[0]:+7.1f},{legacy[1]:+7.1f},{legacy[2]:+7.1f}]  "
              f"{'✅' if ok else '❌'}")

    # Rotation scaling test
    print("\nRotation: log-map scaling (rot_scale=0.5)")
    cfg.rot_scale = 0.5
    rt2 = SE3Retargeter(cfg)
    rt2.set_home(home_pos, q_id, HOME_EEF)
    # 40° rotation about world Z
    ang = math.radians(40)
    q_rot = np.array([0, 0, math.sin(ang/2), math.cos(ang/2)])
    tgt = rt2.compute_target(home_pos, q_rot, debug=True)
    d = rt2.last_debug
    print(f"  Input 40° rot → scaled output {d.rot_angle_deg:.1f}° "
          f"(expect 20° = 40°×0.5)  {'✅' if abs(d.rot_angle_deg-20)<1 else '❌'}")

    print(f"\n  Debug line: {d.log_line()}")
    print("\n=== self-test complete ===\n")


if __name__ == "__main__":
    _selftest()
