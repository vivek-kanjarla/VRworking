# Fairino FR5 SDK Guide

How the Fairino Python SDK works, why we use it the way we do, and what every
method in `fr5.py` is actually doing under the hood.

---

## What Is the Fairino SDK?

Fairino ships a Python package called `fairino`. At its core it wraps an
**XML-RPC client** — a simple HTTP-based remote procedure call protocol. When your
Python code calls `robot.ServoJ(...)`, it sends an XML-formatted HTTP request to
the FR5 controller over Ethernet, the controller executes the command, and returns
a result in XML.

```
Your PC  ──── TCP/IP Ethernet ────  FR5 Controller (192.168.58.2)
   │                                        │
robot.ServoJ([...])  ──────────────►  executes joint motion
   │                                        │
(err, result)        ◄──────────────  returns status
```

Port: **20003** (Fairino XML-RPC port, verified in `check_network.py`).

### CNDE vs XML-RPC

The Fairino SDK has two transport modes:
- **CNDE** (C-based native driver) — faster, lower latency, used when available
- **XML-RPC** — pure Python fallback, fully functional for teleoperation

Our `fr5.py` always uses `Robot.RPC(ip)` which starts with XML-RPC and upgrades to
CNDE if the native driver is installed. The `is_connect` flag workaround in
`connect()` suppresses a spurious warning when CNDE isn't available.

---

## Connection Lifecycle

### `Robot.RPC(ip)` — connect

```python
self._robot = Robot.RPC(FR5_IP)
```

Creates the connection object and opens the TCP socket. After this, all method
calls on `self._robot` go to the controller.

### Startup sequence in `connect()`

```python
self._robot.StopMove()       # cancel any in-progress motion
self._robot.ResetAllError()  # clear any latched fault codes
time.sleep(0.3)              # give controller time to settle
self._robot.Mode(0)          # set to automatic mode (0=auto, 1=manual)
self._robot.RobotEnable(1)   # enable drives (1=on, 0=off)
```

**Why `StopMove()` first?** If a previous `teleop.py` was killed mid-motion (e.g.,
Ctrl-C during ServoJ), the controller may still think it's executing a command.
`StopMove()` clears that state.

**Why `ResetAllError()`?** The controller latches fault codes. If the previous
session hit a joint limit or lost communication, those faults prevent new motion
commands until explicitly cleared.

**Why `Mode(0)`?** The FR5 has a physical teach pendant that can switch the robot
into manual/teach mode. `Mode(0)` programmatically forces automatic mode so our
commands are accepted.

---

## ServoJ Mode — The Key Concept

ServoJ is the mode that makes real-time teleoperation possible. It's worth
understanding deeply.

### Normal motion vs ServoJ

Normally, you send the FR5 a target position (e.g., `MoveJ`) and the controller
plans a trajectory internally and executes it. You can't interrupt it smoothly —
it runs until done.

**ServoJ is different.** It puts the controller in a mode where it expects a
stream of target positions arriving at a fixed rate (60–1000 Hz). Each call says
"be at these joint angles in the next time step." The controller executes each
command and immediately waits for the next one. This gives you fine-grained,
real-time control — exactly what teleoperation needs.

### The three-call lifecycle

```
ServoMoveStart()          ← tell controller to enter streaming mode
  │
  ├── ServoJ([...])       ← send target, must arrive at LOOP_HZ rate
  ├── ServoJ([...])       ← next target
  ├── ServoJ([...])       ← ...repeat at 125 Hz
  │
ServoMoveEnd()            ← exit streaming mode
```

**ServoMoveStart** must be called once to enter the mode. **ServoMoveEnd** must be
called to exit it. If your process dies without calling `ServoMoveEnd`, the
controller may stay in streaming mode and refuse normal motion commands until it
times out or you call `StopMove()` on reconnect.

### Why the frequency matters

The FR5's internal trajectory filter expects commands at a consistent rate. Too
slow → the robot stops and waits (motion stalls). Too fast → commands queue up and
lag builds. The window is 60–1000 Hz. We run at **125 Hz** (8ms per cycle), which
gives enough headroom for USB serial reads and the Python overhead.

---

## ServoJ Parameters Explained

```python
self._robot.ServoJ(
    joints_deg,   # [J1, J2, J3, J4, J5, J6] target joint angles in degrees
    [0] * 6,      # joint velocities — pass zeros, controller ignores these
    FR5_SERVO_VEL,# velocity % — overall speed cap as a percentage of max
    0,            # acceleration % — 0 = use default
    0.008,        # lookahead time (seconds) — hardcoded to match 125 Hz period
    FR5_FILTER_T, # trajectory filter time constant (seconds)
    0,            # gain — 0 = default proportional gain
)
```

### `FR5_SERVO_VEL = 2`

The velocity percentage caps how fast joints can actually move to reach the target.
Start at 2% when tuning — the robot moves slowly, giving you time to check the
mapping is correct. Increase gradually up to 10–20% for normal operation. High
values risk fault-triggering if a large jump command arrives.

### `FR5_FILTER_T = 0.12`

The trajectory filter smooths the incoming command stream. Without it, even tiny
jitter in the SO-101 readings would cause visible vibration in the FR5.

`0.12` seconds means the filter blends the last ~0.12s of commands. The trade-off:
- **Higher value** → smoother motion, more lag (the FR5 trails behind the SO-101)
- **Lower value** → less lag, more jitter visible in motion

`0.12s` is a good starting point. For tasks requiring precise following, reduce to
`0.06`. For smooth demos, increase to `0.2`.

### `0.008` lookahead time

Should match your loop period (`1/LOOP_HZ`). Tells the controller how far ahead
to look when planning the micro-trajectory between commands. Hardcoded here because
it's tightly coupled to the 125 Hz rate.

---

## The RPC Lock

```python
self._rpc_lock = threading.Lock()
```

`xmlrpc.client` (Python's built-in XML-RPC library) is **not thread-safe**. If two
threads call methods on the same connection simultaneously, the XML packets get
interleaved and the controller receives garbage.

We have two threads that both need to talk to the FR5:
1. **Main loop** — calls `ServoJ` at 125 Hz
2. **Gripper thread** — calls `MoveGripper` at ~5 Hz

Every single RPC call is wrapped in `with self._rpc_lock:`. This ensures only one
thread talks to the controller at a time. The gripper thread will wait at the lock
boundary until the main loop releases it after its `ServoJ` call.

---

## Read Methods

### `GetActualJointPosDegree(0)`

Returns the actual encoder-measured joint angles. The `0` flag means "return
measured values" (as opposed to commanded values). Returns `(err, [J1..J6])`.

This differs from the commanded positions (`fr5_cmd` in the logger) — there will
be small tracking errors, especially at higher speeds.

### `GetActualTCPPose(0)`

Returns the end-effector (TCP = Tool Center Point) pose in the robot's base frame.
Format: `[x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]`.

- `x, y, z` — Cartesian position in **millimeters** from the robot base origin
- `rx, ry, rz` — Euler angles in **degrees** (rotation around X, Y, Z axes in
  the Fairino convention — typically ZYX intrinsic, but confirm with your
  firmware's coordinate system docs)

The `0` flag selects the **base frame** (robot's own coordinate system).
Use `1` if you want the world frame (only differs if you've configured a world
offset on the controller).

### `GetActualJointSpeedsDegree(0)`

Returns actual joint velocities in deg/s. Derived from encoder differentials
inside the controller — smoother than computing `Δposition / Δtime` yourself.

---

## Error Handling Pattern

Every Fairino SDK call returns `(error_code, data)` where `error_code == 0` means
success. Our wrapper methods follow this pattern:

```python
raw = self._robot.GetActualJointPosDegree(0)
if not isinstance(raw, (list, tuple)) or len(raw) != 2:
    raise IOError(...)      # SDK returned something unexpected
err, joints = raw
if err != 0:
    raise IOError(...)      # controller reported an error
return list(joints)
```

The `isinstance` check catches edge cases where the CNDE transport returns the
data in a slightly different format than XML-RPC. It's defensive, not paranoid.

Common error codes:
| Code | Meaning |
|---|---|
| `0` | Success |
| `-1` | Generic failure (check controller logs) |
| `73` | `MoveGripper` blocked by active ServoJ session |
| Non-zero | Check Fairino error code table in controller manual |

---

## Gripper Complication — Why It Needs a Pause

The FR5 controller internally serialises `ServoMoveStart` and `MoveGripper` — they
cannot run on the same connection at the same time (error 73). But our gripper
connects as a **second** `Robot.RPC()` instance:

```python
# Main loop connection (fr5.py)
self._robot = Robot.RPC(FR5_IP)

# Gripper connection (gripper.py)
self._rpc = Robot.RPC(FR5_IP)
```

Even with two separate connections, the controller still blocks `MoveGripper` while
any connection has `ServoMoveStart` active. This is a firmware-level restriction,
not a Python-level one.

The solution (in `gripper.py`) is the pause/resume handshake:
1. Gripper thread signals main loop: "I need to send a command"
2. Main loop calls `ServoMoveEnd()` on its connection
3. Main loop signals gripper thread: "ServoJ stopped, go ahead"
4. Gripper thread sends `MoveGripper()`
5. Gripper thread signals main loop: "done"
6. Main loop calls `ServoMoveStart()` and resumes

Total pause: ~400ms. Joint motion freezes, gripper moves, motion resumes.

---

## Context Manager Pattern

```python
with SO101Reader() as arm, FR5Controller() as robot:
    ...
```

`FR5Controller` implements `__enter__` and `__exit__`. This guarantees that no
matter how the program exits (normal, exception, Ctrl-C), `stop()` and
`disconnect()` are called. `stop()` sends `StopMotion()` and `ServoMoveEnd()` —
this is critical to leave the controller in a clean state for the next session.

Without this, a crash could leave the FR5 in ServoJ mode indefinitely, requiring
a manual controller reboot or `teleop.py` reconnect to clear.

---

## Quick Reference

| Method | What it does | When called |
|---|---|---|
| `Robot.RPC(ip)` | Open XML-RPC connection | Once at startup |
| `StopMove()` | Cancel any in-progress motion | Startup cleanup |
| `ResetAllError()` | Clear latched fault codes | Startup + after gripper pause |
| `Mode(0)` | Force automatic mode | Startup |
| `RobotEnable(1)` | Enable drives | Startup + after gripper pause |
| `ServoMoveStart()` | Enter streaming mode | Once before main loop |
| `ServoJ(...)` | Send one joint target | Every cycle at 125 Hz |
| `ServoMoveEnd()` | Exit streaming mode | On shutdown / gripper pause |
| `StopMotion()` | Emergency stop | On exception / exit |
| `GetActualJointPosDegree(0)` | Read encoder angles | Per cycle (logging) |
| `GetActualTCPPose(0)` | Read TCP pose | Per cycle (logging) |
| `GetActualJointSpeedsDegree(0)` | Read joint velocities | Per cycle (logging) |
| `ActGripper(index, 1)` | Activate gripper | Once at gripper init |
| `MoveGripper(...)` | Open / close gripper | On state change, outside ServoJ |
