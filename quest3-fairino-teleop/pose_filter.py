"""
pose_filter.py — One-Euro adaptive low-pass filter for Quest 3 controller pose.

WHY ONE-EURO INSTEAD OF EXPONENTIAL SMOOTHING (EMA):
  EMA uses a fixed smoothing coefficient α, which means a fixed cutoff frequency.
  You face an unavoidable trade-off: choose a low α (aggressive smoothing) and
  you suppress jitter at rest but introduce unacceptable lag during fast motion;
  choose a high α (light smoothing) and fast motion is tracked but jitter
  passes through at rest. No fixed α wins both.

  One-Euro (Casiez, Roussel, Vogel — CHI 2012) resolves this by making the
  cutoff frequency adaptive:

      fc = min_cutoff + beta * |dx_filtered|

  At rest the derivative |dx| ≈ 0, so fc ≈ min_cutoff (maximum smoothing).
  During fast motion |dx| is large, so fc rises and the filter becomes
  nearly transparent. Latency is proportional to 1/fc, so latency is
  automatically minimised wherever it matters — during motion.

WHY SLERP IS REQUIRED FOR QUATERNION INTERPOLATION:
  Unit quaternions lie on the unit sphere S³ ⊂ ℝ⁴. Component-wise linear
  interpolation (q_f = α·q_raw + (1-α)·q_prev) is wrong for two reasons:

    1. The result is not unit-length. Renormalising after the fact changes the
       interpolation path unpredictably and is not geodesic on S³.

    2. Linear ℝ⁴ interpolation does not follow the shortest arc on S³ — it
       passes through the interior of the sphere, not along its surface.

  SLERP (Spherical Linear intERPolation) is the unique geodesic on S³: it
  moves at constant angular velocity along the shortest great-circle arc
  between two orientations and maintains the unit constraint exactly.

  Additionally q and −q represent the same physical rotation. Before
  SLERPing we negate q_raw when dot(q_prev, q_raw) < 0 so we always take
  the shorter arc — without this, rotations that cross the ±180° boundary
  produce 360° jumps in the filtered output.

Reference:
  Casiez G., Roussel N., Vogel D. (2012). "1€ Filter: A Simple Speed-based
  Low-pass Filter for Noisy Input in Interactive Systems." CHI 2012.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field

import numpy as np


# ── Low-level math ────────────────────────────────────────────────────────────

def _alpha(cutoff: float, dt: float) -> float:
    """
    Smoothing coefficient for a first-order IIR low-pass at `cutoff` Hz.

    Derived from the bilinear-transform of the continuous-time RC filter:
        α = (2π·fc·T) / (2π·fc·T + 1)   where T = dt

    At fc → ∞: α → 1 (pass-through, zero lag).
    At fc → 0: α → 0 (freeze output, infinite lag).
    """
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between unit quaternions q0 and q1.

    Returns the unit quaternion at parameter t ∈ [0, 1]:
      t = 0 → q0,  t = 1 → q1.

    Sign correction: if dot(q0, q1) < 0 we negate q1 before interpolating.
    This ensures we always travel the shorter arc on S³.  Without it, rapid
    wrist rotations that cross the antipodal boundary produce 360° jumps.

    Near-parallel fallback: when |dot| > 0.9995 the arc is tiny and
    arccos(dot) is numerically unstable, so we use NLERP (normalised lerp)
    which is an excellent approximation for small angles.
    """
    dot = float(np.dot(q0, q1))

    # Ensure shortest-path: if dot < 0, q0 and q1 are on opposite hemispheres
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = min(dot, 1.0)   # guard arccos domain

    if dot > 0.9995:
        # Nearly parallel — NLERP is accurate and avoids arccos instability
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)

    theta_0 = math.acos(dot)          # angle between q0 and q1
    theta   = theta_0 * t             # fractional angle
    sin_0   = math.sin(theta_0)

    return (math.sin(theta_0 - theta) / sin_0) * q0 + (math.sin(theta) / sin_0) * q1


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of [qx,qy,qz,qw] quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ])


# ── Per-axis scalar One-Euro filter ──────────────────────────────────────────

class _OneEuroScalar:
    """One-Euro filter for a single real-valued signal."""

    __slots__ = ("min_cutoff", "beta", "d_cutoff",
                 "_x_prev", "_dx_prev", "_initialized")

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff  = min_cutoff
        self.beta        = beta
        self.d_cutoff    = d_cutoff
        self._x_prev     = 0.0
        self._dx_prev    = 0.0
        self._initialized = False

    def reset(self) -> None:
        self._initialized = False
        self._dx_prev     = 0.0

    def apply(self, x: float, dt: float) -> float:
        if not self._initialized:
            self._x_prev     = x
            self._initialized = True
            return x

        # Estimate raw derivative
        dx_raw = (x - self._x_prev) / dt

        # Smooth the derivative (fixed d_cutoff)
        a_d          = _alpha(self.d_cutoff, dt)
        dx_smoothed  = a_d * dx_raw + (1.0 - a_d) * self._dx_prev

        # Adaptive signal cutoff driven by signal speed
        cutoff   = self.min_cutoff + self.beta * abs(dx_smoothed)
        a        = _alpha(cutoff, dt)
        x_out    = a * x + (1.0 - a) * self._x_prev

        self._x_prev  = x_out
        self._dx_prev = dx_smoothed
        return x_out

    @property
    def effective_cutoff(self) -> float:
        """Current adaptive cutoff in Hz (for diagnostics)."""
        return self.min_cutoff + self.beta * abs(self._dx_prev)

    @property
    def lag_ms(self) -> float:
        """
        Group delay of the current filter configuration in milliseconds.

        For a first-order IIR H(z) = α/(1-(1-α)z⁻¹), the group delay at DC is
            τ = (1-α) / (α · fs)
        This gives the expected latency introduced by the filter at the
        current speed.  At rest (fc = min_cutoff = 1 Hz, fs = 125 Hz):
            α ≈ 0.048  →  τ ≈ (0.952)/(0.048·125) ≈ 159 ms
        During fast motion (fc = 10 Hz, fs = 125 Hz):
            α ≈ 0.335  →  τ ≈ (0.665)/(0.335·125) ≈  16 ms
        """
        fc = self.effective_cutoff
        dt = 1.0 / 125.0          # nominal sample period
        a  = _alpha(fc, dt)
        if a >= 1.0:
            return 0.0
        return (1.0 - a) / (a * 125.0) * 1000.0


# ── 3-D position filter ───────────────────────────────────────────────────────

class OneEuroFilter3D:
    """
    One-Euro filter applied independently to each component of a 3-vector.

    Position components are independent in Cartesian space, so per-axis
    filtering is both correct and efficient.  The effective cutoff is driven
    by the speed of each individual axis, not the 3-D norm, so the filter
    adapts separately to x/y/z motion — appropriate since a user often moves
    along a single axis at a time.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self._f = [_OneEuroScalar(min_cutoff, beta, d_cutoff) for _ in range(3)]

    def reset(self) -> None:
        for f in self._f:
            f.reset()

    def apply(self, v: np.ndarray, dt: float) -> np.ndarray:
        return np.array([self._f[i].apply(float(v[i]), dt) for i in range(3)])

    @property
    def lag_ms(self) -> float:
        """Max lag across axes (worst-case)."""
        return max(f.lag_ms for f in self._f)

    @property
    def effective_cutoff(self) -> float:
        return max(f.effective_cutoff for f in self._f)


# ── Quaternion One-Euro filter using SLERP ───────────────────────────────────

class OneEuroFilterQuat:
    """
    One-Euro filter for unit quaternions.

    The derivative signal is the angular speed |ω| extracted from the
    incremental rotation q_delta = q_raw × q_prev⁻¹, which lives on S³
    and has a well-defined magnitude:

        angle = 2·arccos(|q_delta.w|)   [radians]
        |ω|   = angle / dt              [rad/s]

    The One-Euro derivative smoother is applied to |ω|, giving the adaptive
    cutoff.  The signal smoother then uses SLERP (not component-wise lerp)
    to move the filtered quaternion toward the raw measurement by the
    computed α — staying on the unit sphere throughout.

    Rapid wrist rotation edge cases:
      - |ω| > 2π rad/s (>360°/s): cutoff rises well above min_cutoff →
        filter becomes nearly transparent → raw pass-through.  This is
        correct: fast intentional motion should not be filtered.
      - Sign flip (q → −q across 180°): handled by dot-product check in
        SLERP; the filter always takes the shorter arc.
      - Gimbal lock: irrelevant — the entire pipeline stays in quaternion
        space; Euler angles only appear at the FR5 IK boundary.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff   = min_cutoff
        self.beta         = beta
        self.d_cutoff     = d_cutoff
        self._q_prev      : np.ndarray | None = None
        self._omega_prev  = 0.0          # smoothed angular speed (rad/s)
        self._initialized = False

    def reset(self) -> None:
        self._initialized = False
        self._q_prev      = None
        self._omega_prev  = 0.0

    def apply(self, q: np.ndarray, dt: float) -> np.ndarray:
        q = q / (np.linalg.norm(q) + 1e-12)   # ensure unit quaternion

        if not self._initialized:
            self._q_prev     = q.copy()
            self._initialized = True
            return q

        # ── Step 1: estimate angular speed (derivative in rotation space) ─────
        # q_delta = q_raw × q_prev⁻¹  (conjugate inverse for unit quaternion)
        q_prev_inv = np.array([-self._q_prev[0], -self._q_prev[1],
                                -self._q_prev[2],  self._q_prev[3]])
        q_delta    = _quat_mul(q, q_prev_inv)

        # Extract rotation angle: angle = 2·arccos(|w|)
        w_clamped  = float(np.clip(abs(q_delta[3]), 0.0, 1.0))
        angle_raw  = 2.0 * math.acos(w_clamped)    # rad, in [0, π]
        omega_raw  = angle_raw / dt                 # rad/s

        # ── Step 2: smooth the derivative (fixed d_cutoff) ───────────────────
        a_d         = _alpha(self.d_cutoff, dt)
        omega       = a_d * omega_raw + (1.0 - a_d) * self._omega_prev

        # ── Step 3: adaptive cutoff driven by angular speed ───────────────────
        cutoff = self.min_cutoff + self.beta * omega

        # ── Step 4: SLERP toward raw measurement by α ────────────────────────
        # This is the One-Euro filter step, but on S³ instead of ℝ.
        # α = 1 → accept raw entirely (no smoothing, no lag).
        # α = 0 → freeze at previous value (maximum smoothing, maximum lag).
        a          = _alpha(cutoff, dt)
        q_filtered = _slerp(self._q_prev, q, a)

        self._q_prev     = q_filtered
        self._omega_prev = omega
        return q_filtered

    @property
    def omega_deg_s(self) -> float:
        """Current smoothed angular speed in deg/s (for diagnostics)."""
        return math.degrees(self._omega_prev)

    @property
    def effective_cutoff(self) -> float:
        return self.min_cutoff + self.beta * self._omega_prev

    @property
    def lag_ms(self) -> float:
        fc = self.effective_cutoff
        dt = 1.0 / 125.0
        a  = _alpha(fc, dt)
        if a >= 1.0:
            return 0.0
        return (1.0 - a) / (a * 125.0) * 1000.0


# ── Debug snapshot ────────────────────────────────────────────────────────────

@dataclass
class FilterDebug:
    """Diagnostic snapshot returned by PoseFilter.apply()."""
    raw_pos:        np.ndarray = field(default_factory=lambda: np.zeros(3))
    filt_pos:       np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_quat:       np.ndarray = field(default_factory=lambda: np.array([0.,0.,0.,1.]))
    filt_quat:      np.ndarray = field(default_factory=lambda: np.array([0.,0.,0.,1.]))
    pos_delta_mm:   float = 0.0   # |raw - filtered| in mm
    ang_err_deg:    float = 0.0   # angle between raw and filtered quaternion (deg)
    omega_deg_s:    float = 0.0   # current angular speed estimate (deg/s)
    pos_cutoff_hz:  float = 0.0   # effective position cutoff (Hz)
    rot_cutoff_hz:  float = 0.0   # effective rotation cutoff (Hz)
    pos_lag_ms:     float = 0.0   # estimated position filter lag (ms)
    rot_lag_ms:     float = 0.0   # estimated rotation filter lag (ms)
    compute_us:     float = 0.0   # filter CPU time (microseconds)

    def log_line(self) -> str:
        return (
            f"[FILT] "
            f"Δpos={self.pos_delta_mm:5.2f}mm  "
            f"Δang={self.ang_err_deg:5.2f}°  "
            f"ω={self.omega_deg_s:6.1f}°/s  "
            f"fc_pos={self.pos_cutoff_hz:.2f}Hz  "
            f"fc_rot={self.rot_cutoff_hz:.2f}Hz  "
            f"lag_pos={self.pos_lag_ms:.0f}ms  "
            f"lag_rot={self.rot_lag_ms:.0f}ms  "
            f"cpu={self.compute_us:.1f}µs"
        )


# ── Public combined filter ────────────────────────────────────────────────────

class PoseFilter:
    """
    Combined One-Euro filter for Quest 3 controller pose (position + orientation).

    Usage in the 125 Hz control loop:
        filter = PoseFilter(pos_min_cutoff=1.0, pos_beta=10.0,
                            rot_min_cutoff=1.0, rot_beta=1.0, d_cutoff=1.0)

        # each cycle:
        pos_f, quat_f, dbg = filter.apply(state.pos, state.quat, t_now)
        # pass pos_f, quat_f to vr_to_fr5() instead of raw values

    Parameters
    ----------
    pos_min_cutoff : Hz
        Position cutoff at rest.  Lower → more smoothing, more lag at rest.
        Recommended: 1.0–3.0 Hz.
    pos_beta :
        Position speed coefficient.  Higher → cutoff rises faster with hand
        speed, reducing lag during motion.  Units: Hz / (m/s).
        Recommended: 5–15 for Quest 3 at 125 Hz.
    rot_min_cutoff : Hz
        Rotation cutoff at rest.  Recommended: 1.0–3.0 Hz.
    rot_beta :
        Rotation speed coefficient.  Units: Hz / (rad/s).
        Recommended: 0.5–2.0 for Quest 3 wrist motion.
    d_cutoff : Hz
        Cutoff for the derivative smoother (controls how quickly the adaptive
        cutoff responds to speed changes).  Rarely needs tuning; 1.0 Hz is
        suitable for all Quest 3 use cases.
    """

    def __init__(
        self,
        pos_min_cutoff: float = 1.0,
        pos_beta:       float = 10.0,
        rot_min_cutoff: float = 1.0,
        rot_beta:       float = 1.0,
        d_cutoff:       float = 1.0,
    ) -> None:
        self._pos_filter  = OneEuroFilter3D(pos_min_cutoff,  pos_beta, d_cutoff)
        self._quat_filter = OneEuroFilterQuat(rot_min_cutoff, rot_beta, d_cutoff)
        self._t_prev: float | None = None

    def reset(self) -> None:
        """
        Reset filter state.

        Call this if there is a large discontinuity in the input (e.g. the
        headset was removed and repositioned).  Do NOT call between normal
        homing events — the filter handles those continuously.
        """
        self._pos_filter.reset()
        self._quat_filter.reset()
        self._t_prev = None

    def apply(
        self,
        pos:   np.ndarray,
        quat:  np.ndarray,
        t:     float,
        debug: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, FilterDebug | None]:
        """
        Filter position and orientation for one control cycle.

        Parameters
        ----------
        pos  : shape (3,)  — controller position in metres (OpenXR frame)
        quat : shape (4,)  — controller orientation [qx, qy, qz, qw]
        t    : wall-clock time from time.monotonic()
        debug: if True, populate and return a FilterDebug snapshot

        Returns
        -------
        pos_filtered  : shape (3,)
        quat_filtered : shape (4,)
        dbg           : FilterDebug or None
        """
        if self._t_prev is None:
            self._t_prev = t

        dt = t - self._t_prev
        # Guard: clamp dt to [1/1000, 0.1] s.
        # Upper bound: prevents enormous derivative spike after a data gap (e.g.
        #   headset taken off).  Lower bound: prevents division-by-zero.
        dt = max(1e-3, min(dt, 0.1))
        self._t_prev = t

        t_start = time.monotonic()

        pos_f  = self._pos_filter.apply(pos,  dt)
        quat_f = self._quat_filter.apply(quat, dt)

        compute_us = (time.monotonic() - t_start) * 1e6

        if not debug:
            return pos_f, quat_f, None

        # ── Build debug snapshot ──────────────────────────────────────────────
        pos_delta_mm = float(np.linalg.norm(pos - pos_f) * 1000.0)

        # Angular error: angle between raw and filtered quaternion
        dot = float(np.clip(abs(np.dot(quat / (np.linalg.norm(quat) + 1e-12), quat_f)), 0.0, 1.0))
        ang_err_deg = math.degrees(2.0 * math.acos(dot))

        dbg = FilterDebug(
            raw_pos       = pos.copy(),
            filt_pos      = pos_f.copy(),
            raw_quat      = quat.copy(),
            filt_quat     = quat_f.copy(),
            pos_delta_mm  = pos_delta_mm,
            ang_err_deg   = ang_err_deg,
            omega_deg_s   = self._quat_filter.omega_deg_s,
            pos_cutoff_hz = self._pos_filter.effective_cutoff,
            rot_cutoff_hz = self._quat_filter.effective_cutoff,
            pos_lag_ms    = self._pos_filter.lag_ms,
            rot_lag_ms    = self._quat_filter.lag_ms,
            compute_us    = compute_us,
        )
        return pos_f, quat_f, dbg
