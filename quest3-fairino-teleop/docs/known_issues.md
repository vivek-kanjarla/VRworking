# Known Issues & Fixes

A running log of bugs found, their root causes, and the fixes applied.

---

## 1. `Robot.RPC.is_conect` Must Be Force-Written

**File:** `fr5.py`

**Symptom:**
The Fairino SDK returns a `<ServerProxy object>` from every RPC call instead of executing the command. `GetActualJointPosDegree()` returns a proxy object, `ServoJ()` silently does nothing. No exception is raised.

**Root cause:**
`Robot.RPC.is_conect` (note: single 'n' — typo in the SDK) is a class-level flag that the SDK checks before dispatching XML-RPC calls. If it is not `True`, every method returns a proxy wrapper instead of making the actual network call. The flag is only set by the SDK's internal CNDE handshake, which may not run in all process configurations (e.g., when two `Robot.RPC()` instances exist, or when the SDK version differs from what the controller expects).

An earlier attempt to fix this using:
```python
try:
    if not Robot.RPC.is_conect:
        Robot.RPC.is_conect = True
except AttributeError:
    pass
```
did not work — if the attribute didn't exist yet, the `AttributeError` was caught and the flag was never set.

**Fix (current):**
```python
Robot.RPC.is_conect = True   # unconditional write — never raises AttributeError
```
Writing to a non-existent class attribute creates it. This is safe: if the SDK has already set it, we overwrite with the same value; if not, we create it.

---

## 2. Double CNDE Session — Gripper Creating Second `Robot.RPC()`

**File:** `gripper.py`

**Symptom:**
When `teleop.py` starts, both `FR5Controller` and `DHGripperController` called `Robot.RPC(ip)`. Two simultaneous CNDE connections to the same controller caused the Fairino controller to print `<ServerProxy ...>` log lines and the teleop loop to get proxy objects back from every RPC call — effectively the same symptom as issue 1.

**Root cause:**
`DHGripperController.__init__` opened its own `Robot.RPC()` connection, creating a second XML-RPC session to the same controller. The controller only supports one active CNDE session at a time.

**Fix:**
Gripper now shares the main `FR5Controller` connection. `gripper.start(robot)` takes the existing `FR5Controller` reference, and all gripper RPCs go through `robot.activate_gripper()` and `robot.send_gripper()` proxy methods on `FR5Controller`. No second `Robot.RPC()` is ever created.

---

## 3. FR5 Follower Never Moves — `numpy.float64` XML-RPC Marshal Error

**File:** `mapper.py`, `fr5.py`

**Symptom:**
The SO-101 heartbeat shows `SO101 moved N°` but `FR5 cmd moved 0°`. The FR5 stays completely still. No exception is printed. `fr5_motion_test.py` (which uses plain Python floats) moves the FR5 fine.

**Root cause:**
`so101_to_fr5()` uses NumPy arithmetic throughout (`np.array`, `np.clip`). The result list contained `numpy.float64` values. `xmlrpc.client` marshals arguments by exact Python type. `numpy.float64` is not registered and has no `__dict__`, so the marshaller raised:

```
TypeError: cannot marshal numpy.float64 objects
```

This exception was being swallowed by the teleop loop's inner `try/except Exception` (introduced for fault tolerance), so it never appeared in the terminal. The cycle was silently skipped 125 times per second, giving the appearance of the FR5 doing nothing.

**Fix — mapper.py:**
```python
result.append(float(p + d))   # coerce numpy.float64 → plain Python float
```

**Fix — fr5.py `servo_j()`:**
```python
joints_deg = [float(j) for j in joints_deg]  # normalise at XML-RPC boundary
```
The mapper fix is the primary fix. The `fr5.py` coercion is a belt-and-suspenders defence at the XML-RPC call site.

**Lesson:** Any value that crosses the Python → `xmlrpc.client` boundary must be a plain Python type (`int`, `float`, `str`, `list`, `dict`). NumPy scalars are not accepted.

---

## 4. SO-101 Leader Arm Stiff / Positions Not Updating

**File:** `so101.py`

**Symptom:**
Moving the SO-101 arm physically has no effect — joints resist movement and encoder readings stay frozen. The teleop loop sees zero delta and sends no motion commands to the FR5.

**Root cause:**
Feetech STS3215 servos boot with torque **enabled** (position-control mode). They actively fight any external force to hold their last commanded position. `SO101Reader.open()` never disabled torque, so the arm was always stiff regardless of the teleop state.

**Fix:**
Write `0` to `ADDR_TORQUE_ENABLE` (register 40) for all motors immediately after opening the port. Each write result is checked and a warning is printed if it fails, so a stuck motor is caught rather than silently ignored:
```python
for mid in self._motor_ids + [SO101_GRIPPER_ID]:
    result, error = self._packet.write1ByteTxRx(self._port, mid, ADDR_TORQUE_ENABLE, 0)
    if result != COMM_SUCCESS:
        print(f"[SO-101] WARNING: torque-disable failed on motor {mid} ...")
```

---

## 5. SO-101 Encoder Wraparound (Large Position Values ~2880°)

**File:** `so101.py`

**Symptom:**
Occasional readings like `2880°`, `2891°` for a joint that should be near `0°`. These appear when a motor is moved past its zero crossing going negative.

**Root cause:**
`read2ByteTxRx` returns an unsigned 16-bit integer. When the STS3215 encoder wraps past zero going negative, the raw value becomes something like `32768` (0x8000), which converts to `32768/4096 × 360 ≈ 2880°`. The correct interpretation is signed 16-bit.

**Fix:**
```python
def _signed(raw: int) -> int:
    return raw if raw < 32768 else raw - 65536
```
Applied to every `read2ByteTxRx` result before conversion to degrees.

---

## 6. Serial Bus Corruption Between Reads

**File:** `so101.py`

**Symptom:**
Sporadic joint read failures immediately after the gripper is read, with `IOError: Motor N read failed`. Occasional position jumps with no physical cause.

**Root cause:**
`read_gripper_deg()` could leave stale response bytes in the serial port receive buffer. The next call to `read_positions_deg()` would pick up these stale bytes as the start of a new packet, corrupting the read.

**Fix:**
```python
def read_positions_deg(self):
    self._port.clearPort()   # flush any stale bytes from the previous gripper read
    ...
```

---

## 7. D405 Camera USB State Corruption

**File:** `camera.py`

**Symptom:**
After killing a Python process that held an active RealSense pipeline (e.g., `Ctrl+C` mid-stream), the next `pipeline.start()` raises:
```
RuntimeError: xioctl(VIDIOC_S_FMT) failed, errno=16 Last Error: Device or resource busy
```
or the camera disappears from `lsusb`.

**Root cause:**
If the pipeline is not cleanly stopped via `pipeline.stop()`, the kernel's UVC driver leaves the USB device in a streaming state.

**Fixes:**
- `camera.py`: `D405Camera.stop()` always calls `pipeline.stop()` in a `try/except`.
- `teleop.py`: `_camera_cleanup()` context manager guarantees `camera.stop()` even if the robot connection fails before the main loop starts.

**Recovery:**
Unplug and replug the USB-C cable. Use a USB 3.0 port directly on the motherboard (not through a hub) and a data-capable cable.

---

## 8. D405 Dual-Stream Pipeline Instability

**File:** `camera.py`

**Symptom:**
Enabling both color and depth streams simultaneously causes `wait_for_frames()` to time out consistently even with 2000 ms timeout and 15-frame warmup.

**Root cause:**
Unknown — single-stream (color-only) works reliably on the same hardware. Possibly USB bandwidth negotiation or UVC driver contention.

**Fix:**
Reverted to color-only pipeline. Color intrinsics (including distortion model) are still saved in the episode JSON. Depth frames are not recorded.

---

## 9. D405 Distortion Model Not Saved

**File:** `camera.py`

**Symptom:**
`dist_coeffs` saved in episode JSON, but no model name. Applying those coefficients with `cv2.undistort()` would silently use the wrong formula.

**Root cause:**
D405 color uses `inverse_brown_conrady` — coefficients are applied in the inverse direction from OpenCV's standard Brown-Conrady. Without the model name, callers cannot select the correct undistortion.

**Fix:**
`"distortion_model": str(ci.model).split(".")[-1]` added to saved intrinsics. Value will be `"inverse_brown_conrady"` for D405 color.

---

## 10. FR5 Motion Too Slow (Rate Limiter Too Conservative)

**Config:** `config.py`

**Symptom:**
The FR5 moves but extremely slowly — operator may think it is not responding.

**Root cause:**
Early defaults `MAX_DELTA_PER_JOINT = [0.08, ...]` at 125 Hz gave max speeds of 10°/s for most joints. Reaching a 60° target takes 6 seconds — operators perceive this as no response.

**Fix:**
```python
MAX_DELTA_PER_JOINT = [0.16, 0.12, 0.12, 0.30, 0.08, 0.20]
```
Also raised `FR5_SERVO_VEL` from 5 → 15 and reduced `FR5_FILTER_T` from 0.08 → 0.04 for more responsive motion.

See **docs/tuning_guide.md** for a full explanation of how these parameters interact.
