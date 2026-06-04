# Joint Mapping: SO-101 → FR5 Explained

This document walks through every line of `mapper.py` and the config values it uses.
No assumed background — start from scratch.

---

## The Problem We're Solving

The SO-101 has **5 joints**. The FR5 has **6 joints**. Their joint angles are in
completely different physical ranges. Their "zero" positions are different. Some
joints move in opposite directions because of how the motors are mounted.

The mapper's job is to answer: **"given where the SO-101 is right now, where should
the FR5 go?"**

---

## Why Delta Mapping (Not Absolute Mapping)

The naive approach would be: read the SO-101 angle for joint 1, scale it, send that
angle to FR5 joint 1. This is called **absolute mapping** and it has a critical flaw.

**The problem:** When you start the program, the SO-101 might be at 120° and the FR5
at 45°. The very first command would slam the FR5 from 45° to wherever 120° maps to.
That's a sudden, potentially large movement with no warning.

**Delta mapping fixes this.** Instead of sending absolute angles, we only send the
*change* from where the SO-101 started. The idea:

> "The SO-101 moved 10° from its starting position. Move the FR5 10° (scaled) from
> its starting position."

At startup, the change is 0°, so the first command is always "stay where you are."
The FR5 doesn't move until the human actually moves the SO-101.

---

## The Code, Line by Line

```python
ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
```

SO-101 joint positions come in as a Python `dict` like:
```python
{"shoulder_pan": 183.2, "elbow_flex": 94.1, "wrist_roll": 210.5, ...}
```

Dicts in Python 3.7+ preserve insertion order, but the order depends on how the
dict was built. `ORDER` guarantees we always extract joints in the same fixed
sequence when converting to a numpy array. If we didn't enforce this, `delta[0]`
might sometimes be `shoulder_pan` and sometimes something else.

---

```python
so101_now  = np.array([so101_deg[k]  for k in ORDER])
so101_base = np.array([so101_home[k] for k in ORDER])
```

`so101_now` — the SO-101's current joint angles this cycle, as a 5-element array.
`so101_base` — the SO-101's angles at the moment the program started (home pose).

Both are in degrees, extracted in the same `ORDER`, so index 0 is always
`shoulder_pan`, index 1 is always `shoulder_lift`, and so on.

---

```python
delta = (so101_now - so101_base) * np.array(JOINT_SCALE) * np.array(JOINT_AMP)
```

This is the core of the mapper. Three things happen here:

### Step 1: `so101_now - so101_base` — compute the change

This gives the displacement from home for each joint, in degrees.

```
Example:
  so101_now  = [183°, 120°, 94°, 200°, 210°]
  so101_base = [180°, 120°, 94°, 200°, 210°]   ← captured at startup
  raw_delta  = [  3°,   0°,  0°,   0°,   0°]   ← shoulder_pan moved 3°
```

At startup, `so101_now == so101_base`, so `delta = [0, 0, 0, 0, 0]` and the FR5
doesn't move. Exactly what we want.

---

### Step 2: `* np.array(JOINT_SCALE)` — flip directions

```python
JOINT_SCALE = [-1.0, 1.0, 1.0, 1.0, 1.0]
```

Physical reality: when you rotate the SO-101's shoulder pan to the left, the FR5's
joint 1 needs to go to the right to produce the same real-world motion. This is
because of how the motors are oriented relative to each other.

`JOINT_SCALE` corrects for this. `-1.0` means "flip this joint's direction."
`+1.0` means "same direction."

```
raw_delta  = [3°, 0°, 0°, 0°, 0°]
JOINT_SCALE= [-1,  1,  1,  1,  1]
scaled     = [-3°, 0°, 0°, 0°, 0°]   ← shoulder_pan flipped
```

If a joint moves the wrong way in real life, change its `JOINT_SCALE` entry to `-1`.

---

### Step 3: `* np.array(JOINT_AMP)` — amplify (gear ratio)

```python
JOINT_AMP = [1.5, 2.0, 2.0, 3.0, 1.5]
```

The SO-101 is a small desktop arm with a limited range of motion. The FR5 is an
industrial cobot with a much larger workspace. If we mapped 1:1, the FR5 would never
move far enough to use its full workspace.

`JOINT_AMP` is a multiplier: a value of `2.0` means "1° of SO-101 movement becomes
2° of FR5 movement." Think of it as a virtual gear ratio.

```
scaled  = [-3°, 0°, 0°, 0°, 0°]
JOINT_AMP=[ 1.5, 2,  2,  3,  1.5]
delta   = [-4.5°, 0°, 0°, 0°, 0°]
```

The shoulder_pan moved 3° on the SO-101, and the FR5 will move 4.5° (because amp=1.5).

Tune these values if the FR5 moves too little (increase amp) or overshoots (decrease).

---

## Building the Target Position

```python
target = [
    fr5_home[0] + delta[0],  # J1 ← shoulder_pan
    fr5_home[1] + delta[1],  # J2 ← shoulder_lift
    fr5_home[2] + delta[2],  # J3 ← elbow_flex
    fr5_home[3] + delta[3],  # J4 ← wrist_flex
    prev_fr5_deg[4],         # J5 — frozen
    fr5_home[5] + delta[4],  # J6 ← wrist_roll
]
```

`fr5_home` is the FR5's actual joint angles at startup (read from the robot).

We add the scaled delta to the FR5's home pose. So:

> FR5 target = (FR5 where it started) + (how much SO-101 moved, scaled)

Notice **J5 is skipped** — the SO-101 only has 5 joints, so there's no input for
FR5's forearm roll (J5). It's frozen at whatever angle it was at startup
(`prev_fr5_deg[4]` — previous cycle's value, which chains back to the initial home).

Also notice `delta` has 5 elements but FR5 has 6 joints. The indexing skips J5:
- `delta[0]` → J1, `delta[1]` → J2, `delta[2]` → J3, `delta[3]` → J4, `delta[4]` → J6

---

## Joint Limit Clamping

```python
target_clamped = [
    max(lo, min(hi, t))
    for t, (lo, hi) in zip(target, FR5_JOINT_LIMITS)
]
```

`FR5_JOINT_LIMITS` defines the safe operating range for each joint, e.g.:
```python
FR5_JOINT_LIMITS = [
    (-170, 170),  # J1
    (-260,  80),  # J2
    ...
]
```

`max(lo, min(hi, t))` is the standard clamp idiom — it constrains `t` to `[lo, hi]`.

**Crucially, clamping happens BEFORE rate limiting.** Why does order matter?

Imagine J2 is at -258° (near its -260° limit) and the target says -270°. Clamping
snaps it to -260°. The rate limiter then sees a small delta (−2°) rather than a
large jump (−12°), and handles it smoothly. If we rate-limited first, we'd send -259°,
then -260° the next cycle — fine. But if the target was unclamped and wildly out of
range, the rate limiter would dutifully walk toward it for many cycles before the
clamp could help. Clamp first, rate-limit second.

---

## Rate Limiting

```python
sing_scale = (delta_limit / MAX_DELTA_DEG_PER_CYCLE) if delta_limit is not None else 1.0
limits = [lim * sing_scale for lim in MAX_DELTA_PER_JOINT]

result = []
for t, p, lim in zip(target_clamped, prev_fr5_deg, limits):
    d = np.clip(t - p, -lim, lim)
    result.append(p + d)
```

Even after clamping, the target might require a large jump in a single cycle —
e.g., if the SO-101 was moved quickly. Sending a large step to ServoJ can cause
jerky motion or trigger the robot's fault system.

The rate limiter caps how much each joint can move **per cycle**:

```python
MAX_DELTA_PER_JOINT = [0.08, 0.06, 0.06, 0.20, 0.04, 0.10]
#                      J1     J2     J3     J4    J5    J6
```

At 125 Hz:
- J1 can move at most 0.08°/cycle × 125 = **10°/s**
- J2/J3: **7.5°/s**
- J4: **25°/s** (wrist — less inertia, safe to move faster)
- J6: **12.5°/s**

`np.clip(t - p, -lim, lim)` — computes desired movement `(t - p)`, then clips it
to `[-lim, lim]`. Then `p + d` gives the actual command to send.

**Singularity scaling:** When the robot is near a singular configuration,
`delta_limit` comes in as a reduced value (from `singularity.py`). The `sing_scale`
factor shrinks all the per-joint limits proportionally, slowing down all joints
uniformly as the robot approaches danger.

---

## Full Example

Setup:
- SO-101 shoulder_pan home: 180°, current: 185°
- FR5 J1 home: 0°, JOINT_SCALE[0] = -1.0, JOINT_AMP[0] = 1.5
- FR5 J1 previous: 0°, rate limit: 0.08°/cycle

Calculation:
```
raw_delta    = 185 - 180 = 5°
scaled_delta = 5 × (-1.0) × 1.5 = -7.5°
target J1    = 0 + (-7.5) = -7.5°
clamped J1   = -7.5° (within limits)
d            = clip(-7.5 - 0, -0.08, 0.08) = -0.08°
command J1   = 0 + (-0.08) = -0.08°
```

The FR5 takes a small step toward -7.5°. Each cycle it advances by 0.08°, reaching
the target after ~94 cycles (~0.75 seconds at 125 Hz). This is intentional —
smooth motion is safer than instant jumps.

---

## Summary

| Step | What happens | Why |
|---|---|---|
| `so101_now - so101_base` | Compute displacement from home | Avoid startup drift |
| `× JOINT_SCALE` | Flip directions where needed | Hardware mounting differences |
| `× JOINT_AMP` | Amplify movement | SO-101 range << FR5 workspace |
| `fr5_home + delta` | Target = FR5 home + scaled SO-101 delta | Both arms referenced to their own home |
| J5 frozen | No SO-101 counterpart | 5-DOF leader → 6-DOF follower |
| Joint limit clamp | Hard safety bounds | Protect hardware |
| Rate limit per joint | Max degrees per cycle | Smooth motion, fault prevention |
| Singularity scale | Slow down near singular configs | Prevent runaway near IK degeneracy |
