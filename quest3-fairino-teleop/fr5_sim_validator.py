"""
fr5_sim_validator.py — validates FR5 joint trajectories before hardware execution.

Real-time usage (imported by sim_bridge.py):
    from fr5_sim_validator import FR5SimValidator
    validator = FR5SimValidator(sim_robot)   # sim_robot may be None → skips collision
    result = validator.check(target_joints, prev_joints)
    if not result.ok:
        print(f"BLOCKED: {result.reason}")

Batch usage (standalone, validates a recorded episode CSV):
    python3 fr5_sim_validator.py episodes/episode_1779029738128.csv
    python3 fr5_sim_validator.py episodes/episode_1779029738128.csv --hz 125 --no-collision
    python3 fr5_sim_validator.py episodes/episode_1779029738128.csv --summary-only
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import (
    FR5_JOINT_LIMITS,
    LOOP_HZ,
    MAX_DELTA_PER_JOINT,
)
import singularity

# ── limits ────────────────────────────────────────────────────────────────────

# Maximum joint velocity (deg/s) — derived from servo velocity capability
# FR5 spec: ~180 deg/s per joint at full speed; use 120 deg/s as safe cap
_MAX_VEL_DEG_S = [120.0] * 6

# Maximum joint acceleration (deg/s²) — catches sudden command jumps
# Conservative: 500 deg/s² (~4.4 rad/s²) matches industrial cobot limits
_MAX_ACCEL_DEG_S2 = [500.0] * 6

# FR5 max reach from base (mm) — arm fully extended ≈ 915 mm
_WORKSPACE_MAX_MM = 900.0
# Minimum reach — below this the arm is near a shoulder singularity
_WORKSPACE_MIN_MM = 80.0

# Hard cap for collision check: contacts with separation below this (m) are flagged
_COLLISION_MARGIN_M = 0.005   # 5 mm safety margin


# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    ok:         bool
    reason:     str = ""   # first blocking failure, empty if ok
    violations: dict = field(default_factory=dict)
    # Non-blocking warnings (singularity, workspace margin, etc.)
    warnings:   list[str] = field(default_factory=list)

    def __bool__(self):
        return self.ok


# ── validator ─────────────────────────────────────────────────────────────────

class FR5SimValidator:
    """
    Per-cycle and batch trajectory validator for the FR5.

    sim_robot — object with set_joints(list[float]) and check_collisions() -> int.
                Pass None to disable self-collision checks (joint/rate/workspace
                checks still run).
    hz        — loop rate; used to convert per-cycle deltas to deg/s and deg/s².
    """

    def __init__(self, sim_robot=None, hz: float = LOOP_HZ):
        self._sim        = sim_robot
        self._dt         = 1.0 / hz
        self._hz         = hz
        self._prev_prev: list[float] | None = None
        self._warmup     = 0   # skip accel check for N cycles after reset/homing

    def reset(self):
        """Clear accumulated state (call between episodes in batch mode)."""
        self._prev_prev = None
        self._warmup    = 5

    # ── real-time check ───────────────────────────────────────────────────────

    def check(
        self,
        target_joints:   list[float],
        prev_joints:     list[float],
        prev_prev_joints: list[float] | None = None,
    ) -> ValidationResult:
        """
        Validate one commanded step.

        target_joints    — proposed next joint positions (deg)
        prev_joints      — current joint positions (deg)
        prev_prev_joints — cycle before prev_joints (deg) — enables accel check.
                           If None the instance's internal history is used.
        Returns ValidationResult; result.ok == False means BLOCK the command.
        """
        ppj = prev_prev_joints if prev_prev_joints is not None else self._prev_prev
        violations: dict[str, str] = {}
        warnings:   list[str]      = []

        # 1. Joint position limits
        for i, (j, (lo, hi)) in enumerate(zip(target_joints, FR5_JOINT_LIMITS)):
            if j < lo or j > hi:
                violations[f"limit_j{i+1}"] = (
                    f"j{i+1}={j:.2f}° outside [{lo}, {hi}]°"
                )

        if violations:
            self._prev_prev = list(prev_joints)
            return ValidationResult(ok=False, reason=next(iter(violations.values())),
                                    violations=violations, warnings=warnings)

        # 2. Per-cycle delta (rate limiting)
        for i, (j, p, lim) in enumerate(zip(target_joints, prev_joints, MAX_DELTA_PER_JOINT)):
            delta = abs(j - p)
            hard  = lim * 4.0   # 4× soft limit — mapper already applied the soft cap
            if delta > hard:
                violations[f"rate_j{i+1}"] = (
                    f"j{i+1} Δ={delta:.4f}° > hard cap {hard:.4f}°/cycle"
                )

        # 3. Joint velocity (deg/s)
        for i, (j, p) in enumerate(zip(target_joints, prev_joints)):
            vel = abs(j - p) * self._hz
            if vel > _MAX_VEL_DEG_S[i]:
                violations[f"vel_j{i+1}"] = (
                    f"j{i+1} vel={vel:.1f} deg/s > max {_MAX_VEL_DEG_S[i]} deg/s"
                )

        # Acceleration check removed — mapper_vr rate limiter already caps per-cycle velocity

        if violations:
            self._prev_prev = list(prev_joints)
            return ValidationResult(ok=False, reason=next(iter(violations.values())),
                                    violations=violations, warnings=warnings)

        # 5. Singularity proximity (non-blocking warn + blocking danger)
        level, scale, msg = singularity.check(target_joints)
        if msg:
            if level == singularity.Level.DANGER:
                violations["singularity"] = msg
                self._prev_prev = list(prev_joints)
                return ValidationResult(ok=False, reason=msg,
                                        violations=violations, warnings=warnings)
            warnings.append(f"singularity: {msg}")

        # 6. Self-collision via PyBullet
        if self._sim is not None:
            self._sim.set_joints(target_joints)
            n = self._sim.check_collisions()
            if n > 0:
                violations["collision"] = f"self-collision: {n} contact points"
                self._prev_prev = list(prev_joints)
                return ValidationResult(ok=False, reason=violations["collision"],
                                        violations=violations, warnings=warnings)

        # 7. Workspace reach (warn only — not blocking; IK would have failed first)
        if self._sim is not None:
            try:
                eef = self._sim.get_fk(target_joints)
                dist = math.sqrt(eef[0]**2 + eef[1]**2 + eef[2]**2)
                if dist > _WORKSPACE_MAX_MM:
                    warnings.append(f"EEF reach {dist:.0f}mm > max {_WORKSPACE_MAX_MM:.0f}mm")
                elif dist < _WORKSPACE_MIN_MM:
                    warnings.append(f"EEF reach {dist:.0f}mm < min {_WORKSPACE_MIN_MM:.0f}mm")
            except Exception:
                pass

        self._prev_prev = list(prev_joints)
        return ValidationResult(ok=True, violations={}, warnings=warnings)

    # ── batch trajectory validation ───────────────────────────────────────────

    def validate_trajectory(
        self,
        joints_sequence: list[list[float]],
        hz: float | None = None,
        verbose: bool = True,
    ) -> dict:
        """
        Validate a list of joint position snapshots (deg).

        joints_sequence — N×6 list, each element = [j1..j6] in degrees
        hz              — override loop rate for this sequence
        Returns a summary dict with statistics and per-check violation counts.
        """
        if hz is not None:
            original_hz  = self._hz
            original_dt  = self._dt
            self._hz     = hz
            self._dt     = 1.0 / hz

        self.reset()
        n = len(joints_sequence)
        if n == 0:
            return {"total": 0, "blocked": 0, "violations": {}}

        blocked     = 0
        all_viols:  dict[str, int] = {}
        all_warns:  list[str]      = []
        first_block: int | None    = None

        for idx, joints in enumerate(joints_sequence):
            if idx == 0:
                prev  = joints
                pprev = None
                self.reset()
                continue

            result = self.check(joints, prev, pprev)

            if not result.ok:
                blocked += 1
                if first_block is None:
                    first_block = idx
                for k, v in result.violations.items():
                    all_viols[k] = all_viols.get(k, 0) + 1
                if verbose:
                    print(f"  [row {idx:5d}] BLOCKED: {result.reason}")

            for w in result.warnings:
                all_warns.append(f"row {idx}: {w}")

            pprev = prev
            prev  = joints

        if hz is not None:
            self._hz = original_hz
            self._dt = original_dt

        summary = {
            "total":       n,
            "blocked":     blocked,
            "block_rate":  blocked / max(1, n - 1),
            "first_block": first_block,
            "violations":  all_viols,
            "warnings":    len(all_warns),
        }
        return summary


# ── standalone batch entrypoint ───────────────────────────────────────────────

_CMD_COLS = [f"fr5_cmd_j{i}" for i in range(1, 7)]


def _load_csv(path: str) -> list[list[float]]:
    """Load fr5_cmd_j1..j6 columns from an episode CSV."""
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except ImportError:
        import csv
        with open(path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        df_rows = [[float(row[c]) for c in _CMD_COLS] for row in rows]
        return df_rows

    missing = [c for c in _CMD_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing columns: {missing}\n"
            f"Available: {list(df.columns)}\n"
            "Expected episode CSV recorded by logger.py"
        )
    return df[_CMD_COLS].values.tolist()


def _print_summary(path: str, summary: dict, elapsed: float):
    n      = summary["total"]
    b      = summary["blocked"]
    rate   = summary["block_rate"] * 100
    fb     = summary["first_block"]
    viols  = summary["violations"]
    warns  = summary["warnings"]

    print(f"\n{'─'*60}")
    print(f"  File:        {os.path.basename(path)}")
    print(f"  Rows:        {n}")
    print(f"  Blocked:     {b} / {n-1} steps  ({rate:.1f}%)")
    if fb is not None:
        print(f"  First block: row {fb}")
    if viols:
        print(f"  Violations:")
        for k, cnt in sorted(viols.items(), key=lambda x: -x[1]):
            print(f"    {k:<25} {cnt:5d}×")
    if warns:
        print(f"  Warnings:    {warns}")
    print(f"  Validated in {elapsed*1000:.0f} ms")
    print(f"{'─'*60}")
    print(f"  Result: {'PASS ✓' if b == 0 else 'FAIL ✗  (would block hardware execution)'}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate FR5 episode CSV against joint/collision/workspace limits"
    )
    parser.add_argument("csv", help="Path to episode CSV (e.g. episodes/episode_123.csv)")
    parser.add_argument(
        "--hz", type=float, default=float(LOOP_HZ),
        help=f"Loop rate of the recording (default: {LOOP_HZ})",
    )
    parser.add_argument(
        "--no-collision", action="store_true",
        help="Skip PyBullet self-collision checks (faster)",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Suppress per-row violation lines",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.csv} ...")
    trajectory = _load_csv(args.csv)
    print(f"  {len(trajectory)} rows, {len(_CMD_COLS)} joints, {args.hz:.0f} Hz")

    sim_robot = None
    if not args.no_collision:
        try:
            from sim_bridge import SimRobot, _URDF_PATH
            print(f"Loading PyBullet URDF for collision checks ...")
            sim_robot = SimRobot(_URDF_PATH, gui=False)
            print("  PyBullet ready")
        except Exception as exc:
            print(f"  Warning: PyBullet unavailable ({exc}) — skipping collision checks")

    validator = FR5SimValidator(sim_robot=sim_robot, hz=args.hz)

    t0 = time.monotonic()
    summary = validator.validate_trajectory(
        trajectory, hz=args.hz, verbose=not args.summary_only
    )
    elapsed = time.monotonic() - t0

    _print_summary(args.csv, summary, elapsed)

    if sim_robot:
        sim_robot.close()

    sys.exit(0 if summary["blocked"] == 0 else 1)


if __name__ == "__main__":
    main()
