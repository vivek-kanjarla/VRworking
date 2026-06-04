# Motion Tuning Guide

All motion parameters live in `config.py`. This guide explains what each one does, how they interact, and a recommended tuning procedure for dialing in smooth, responsive teleoperation.

---

## Quick Reference

| Parameter | Default | What it controls |
|---|---|---|
| `FR5_SERVO_VEL` | 15 | ServoJ velocity % — how fast the robot moves toward each target |
| `FR5_FILTER_T` | 0.04 | ServoJ filter time — smooths trajectory between commands |
| `MAX_DELTA_PER_JOINT` | `[0.16, 0.12, 0.12, 0.30, 0.08, 0.20]` | Hard cap on degrees/cycle per joint |
| `JOINT_AMP` | `[1.5, 2.0, 2.0, 3.0, 1.5]` | Amplification: 1° SO-101 → N° FR5 |
| `JOINT_SCALE` | `[-1, 1, 1, 1, 1]` | Direction: +1 = same, −1 = reversed |
| `LOOP_HZ` | 125 | ServoJ command rate |

---

## Parameter Details

### `FR5_SERVO_VEL` — Velocity percentage

```python
FR5_SERVO_VEL = 15   # 15% of maximum joint speed
```

This is the `vel` argument passed directly to `ServoJ`. It sets the joint-space velocity limit as a percentage of the FR5's maximum rated speed. The FR5 planner will not command a joint to move faster than this percentage.

**Effect:**
- Too low (< 5) → robot feels sluggish; visible lag even for fast SO-101 movements
- Too high (> 40) → robot snaps aggressively when the rate limiter releases pent-up motion; overshoot becomes visible

**Practical range:** 10–25 for teleoperation. Start at 15 and increase if the arm is clearly too slow. Decrease if you see oscillation or jerky catch-up.

---

### `FR5_FILTER_T` — Trajectory filter time

```python
FR5_FILTER_T = 0.04   # 40 ms smoothing window
```

The Fairino SDK uses this as a smoothing time constant on the internal trajectory planner. A larger value blends recent commands more — the robot follows a smoother path but lags further behind the SO-101.

**Effect:**
- Too large (> 0.15 s) → noticeable lag; the FR5 is still catching up after the SO-101 has stopped
- Too small (< 0.01 s) → commands are passed through nearly raw; any jitter in the SO-101 read is amplified on the FR5

**Practical range:** 0.02–0.08 s. Lower values (0.02–0.04) are suitable for smooth human motion. Raise to 0.06–0.08 if the arm is vibrating.

**Interaction with `FR5_SERVO_VEL`:** These two work together. High velocity + low filter = fast and potentially jerky. Low velocity + high filter = smooth but sluggish. The sweet spot is usually moderate velocity with a short filter.

---

### `MAX_DELTA_PER_JOINT` — Per-joint rate limit

```python
MAX_DELTA_PER_JOINT = [0.16, 0.12, 0.12, 0.30, 0.08, 0.20]
#                       J1    J2    J3    J4    J5   J6
#                       20°/s 15°/s 15°/s 37°/s 10°/s 25°/s
```

Maximum degrees per control cycle (at 125 Hz). This is a hard cap applied in the mapper before the ServoJ call. It acts as a final safety net against sudden jumps from noisy reads, large amplitude changes, or re-homing.

**Interpreting the values:**
- Value × LOOP_HZ = maximum degrees per second
- `0.16 × 125 = 20°/s` for J1 (shoulder pan)

**Effect:**
- Too low → the FR5 never catches up with fast SO-101 movements; motion looks delayed and choppy
- Too high → a noisy read can cause a large single-cycle jump; the robot may jerk

**Tuning guideline:**
1. Identify which joint feels slowest — compare observed motion with the SO-101
2. Multiply current limit by 1.3–1.5 for that joint
3. Test; if the joint overshoots or jerks, back off 20%

The wrist joints (J4, J6) tolerate higher limits because they have less inertia and shorter lever arms. Shoulder joints (J1, J2) should be lower to feel safe near the operator.

---

### `JOINT_AMP` — Amplitude multiplier

```python
JOINT_AMP = [1.5, 2.0, 2.0, 3.0, 1.5]
#            pan  lift  elbow wrist roll
```

The SO-101 and FR5 have different workspace sizes. `JOINT_AMP[i] = 2.0` means 1° of SO-101 motion produces 2° of FR5 motion for that joint. This is applied in the mapper as a scalar on the delta.

**Effect:**
- Too low → the FR5 workspace is under-utilised; the operator has to move the SO-101 to extreme positions for normal FR5 motion
- Too high → small SO-101 movements cause large FR5 motions; control is twitchy and hard to use precisely

**When to adjust:**
- After changing the physical SO-101 mounting, range of motion, or homing position
- When the FR5 keeps hitting limits before the SO-101 does (reduce amp) or vice versa (increase amp)
- Typical useful range: 1.0–4.0 per joint

**Important:** `JOINT_AMP` amplifies the rate-limited delta. It multiplies the SO-101 delta BEFORE the rate limiter, so the rate limiter still caps the final command — high amp does not bypass `MAX_DELTA_PER_JOINT`.

---

### `JOINT_SCALE` — Direction flip

```python
JOINT_SCALE = [-1.0, 1.0, 1.0, 1.0, 1.0]
#              pan   lift  elbow wrist roll
```

`+1` = SO-101 and FR5 joints move in the same direction. `-1` = reversed. This handles differences in servo mounting orientation between the two arms.

**When to change:** If moving the SO-101 shoulder left causes the FR5 to move right, flip that joint's scale from `1.0` to `-1.0` (or vice versa). Do this one joint at a time while running `teleop.py` and checking the heartbeat.

---

### `LOOP_HZ` — Control loop frequency

```python
LOOP_HZ = 125   # Hz — must be between 60 and 1000
```

This sets both the ServoJ call rate and the `LOOP_PERIOD` sleep. The Fairino FR5 requires ServoJ to be called at a steady rate in the 60–1000 Hz range.

**Do not raise this without verifying the loop stays within budget.** Each cycle must complete SO-101 read + mapper + ServoJ within `1/LOOP_HZ` seconds. At 125 Hz, the budget is 8 ms. The SO-101 read takes ~1–2 ms, the ServoJ RPC takes ~2–3 ms, leaving ~3 ms of headroom. At 250 Hz the budget is 4 ms — too tight for XML-RPC.

**When to lower:** If `errs=N` in the heartbeat is rising fast, the loop may be timing out. Lower LOOP_HZ to 100 or 80 to give more headroom.

---

## Tuning Procedure

Start with the defaults. Follow this order when something feels wrong:

### 1. Direction wrong on a joint?

Change `JOINT_SCALE` for that joint only. Flip the sign. Test immediately.

### 2. FR5 motion too small for SO-101 range?

Increase `JOINT_AMP` for the slow joint. Raise by 0.5 at a time. Test across the full SO-101 range before committing.

### 3. Motion too slow / FR5 can't keep up?

First raise `MAX_DELTA_PER_JOINT` for the joint that lags. If that's already high, raise `FR5_SERVO_VEL` by 5 units. Do not skip the per-joint check — often only one or two joints are the bottleneck.

### 4. Motion jerky or oscillating?

Lower `FR5_SERVO_VEL` first. If still jerky, raise `FR5_FILTER_T` from 0.04 → 0.06 → 0.08. Stop when the oscillation disappears; do not over-filter or you'll introduce lag.

### 5. Occasional single-cycle jumps?

These are usually noisy SO-101 reads. Lower `MAX_DELTA_PER_JOINT` for the affected joint by 30%. The rate limiter clips the spike.

### 6. Everything smooth, but workspace feels small?

Raise `JOINT_AMP` uniformly by 10–15%. Check all joints reach their desired range. Verify the FR5 does not regularly hit `FR5_JOINT_LIMITS` in normal use.

---

## Current Default Values — Explanation

```python
FR5_SERVO_VEL = 15        # balanced: responsive without overshoot
FR5_FILTER_T  = 0.04      # 40 ms: smooth for typical hand speed
MAX_DELTA_PER_JOINT = [0.16, 0.12, 0.12, 0.30, 0.08, 0.20]
# J1: 20°/s — shoulder pan, long lever arm → conservative
# J2: 15°/s — shoulder lift, most weight-bearing → most conservative
# J3: 15°/s — elbow, similar to J2
# J4: 37°/s — wrist flex, light and fast
# J5: 10°/s — frozen joint; small limit keeps it from drifting during re-home
# J6: 25°/s — wrist roll, light
JOINT_AMP = [1.5, 2.0, 2.0, 3.0, 1.5]
# shoulder pan (J1): 1.5× — modest workspace difference
# shoulder lift (J2): 2.0× — FR5 range is ~2× the SO-101's
# elbow (J3): 2.0× — same reasoning
# wrist flex (J4): 3.0× — SO-101 wrist has limited range; FR5 wrist needs full range
# wrist roll (J6): 1.5× — moderate amplification
```

---

## Parameter Interaction Summary

```
Fast but jerky:     ↓ FR5_SERVO_VEL  or  ↑ FR5_FILTER_T
Slow / lags behind: ↑ MAX_DELTA_PER_JOINT  then  ↑ FR5_SERVO_VEL
Workspace too small:↑ JOINT_AMP
Direction wrong:    flip JOINT_SCALE sign
Noisy single jumps: ↓ MAX_DELTA_PER_JOINT (rate limiter clips spikes)
```
