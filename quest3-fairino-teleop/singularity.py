"""
singularity.py — detects proximity to singular configurations on the FR5.

Checks two singularities that can be hit during teleoperation:

  Wrist singularity  — J5 ≈ 0°  (J4 and J6 axes align, infinite IK solutions)
  Elbow singularity  — J3 ≈ 0°  (arm near full extension/fold)

Returns a severity level and a velocity scale factor (0.0–1.0) that the
teleop loop multiplies against MAX_DELTA_DEG_PER_CYCLE before sending ServoJ.
"""

from enum import Enum

# ── Thresholds (degrees) ──────────────────────────────────────────────────────
WRIST_WARN_DEG   = 15.0   # |J5| below this → WARN
WRIST_DANGER_DEG =  3.0   # |J5| below this → DANGER (tightened: true wrist lock at ~0°)

ELBOW_WARN_DEG   =  8.0   # |J3| below this → WARN
ELBOW_DANGER_DEG =  1.0   # |J3| below this → DANGER (tightened: J3=3° is still controllable)


class Level(Enum):
    CLEAR  = "CLEAR"
    WARN   = "WARN"
    DANGER = "DANGER"


def _scale(value: float, warn: float, danger: float) -> float:
    """
    Linear scale 1.0 → 0.05 between warn and danger thresholds.
    Floor is 0.05 (not 0.0) so the arm can always creep out of a near-singular
    pose rather than freezing completely with no escape route.
    """
    if value >= warn:
        return 1.0
    if value <= danger:
        return 0.05   # minimum — slow crawl so user can escape, not a full freeze
    return 0.05 + 0.95 * (value - danger) / (warn - danger)


def check(joints_deg: list[float]) -> tuple[Level, float, str]:
    """
    Evaluate singularity proximity for given joint positions.

    Returns:
        level       — worst Level across all checks
        scale       — multiply MAX_DELTA_DEG_PER_CYCLE by this (0.0 = full stop)
        message     — human-readable description of the active constraint
    """
    j3  = abs(joints_deg[2])   # J3: elbow — singular when ≈ 0° (arm straight)
    j5  = abs(joints_deg[4])   # J5: wrist — singular when ≈ 0° (J4/J6 axes align)
    # NOTE: wrist singularity is J5 (index 4), NOT J4 (index 3). J4 at 0° is a
    # normal working pose; only J5≈0° collapses the wrist into a gimbal-lock state.

    wrist_scale = _scale(j5, WRIST_WARN_DEG, WRIST_DANGER_DEG)
    elbow_scale = _scale(j3, ELBOW_WARN_DEG, ELBOW_DANGER_DEG)

    scale = min(wrist_scale, elbow_scale)

    parts = []
    if j5 <= WRIST_DANGER_DEG:
        parts.append(f"WRIST SINGULARITY (J5={joints_deg[4]:.1f}°) — motion blocked")
        level = Level.DANGER
    elif j5 <= WRIST_WARN_DEG:
        parts.append(f"wrist near singular (J5={joints_deg[4]:.1f}°, limit={WRIST_WARN_DEG}°)")
        level = Level.WARN
    elif j3 <= ELBOW_DANGER_DEG:
        parts.append(f"ELBOW SINGULARITY (J3={joints_deg[2]:.1f}°) — motion blocked")
        level = Level.DANGER
    elif j3 <= ELBOW_WARN_DEG:
        parts.append(f"elbow near singular (J3={joints_deg[2]:.1f}°, limit={ELBOW_WARN_DEG}°)")
        level = Level.WARN
    else:
        level = Level.CLEAR

    message = " | ".join(parts) if parts else ""
    return level, scale, message
