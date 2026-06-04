"""
gripper.py — maps Quest 3 trigger input to DH AG-160-95 via Fairino SDK.

MoveGripper is blocked with error 73 while ServoMoveStart is active on any
connection. The fix: when a gripper state change is needed, the background
thread signals the main loop, which pauses ServoJ, sends the gripper command,
then resumes. Joint motion freezes for ~200ms per open/close — acceptable.

Handshake:
  gripper thread  →  sets _cmd_ready, blocks on _servo_paused
  main loop       →  sees _cmd_ready, calls pause_for_gripper()
                     → ServoMoveEnd, sets _servo_paused, waits on _cmd_done
  gripper thread  →  sends MoveGripper, sets _cmd_done, clears _servo_paused
  main loop       →  resumes ServoMoveStart
"""

import threading
import time

from config import (
    GRIPPER_INDEX,
    GRIPPER_OPEN_PCT, GRIPPER_CLOSE_PCT,
    GRIPPER_TYPE, GRIPPER_VEL_PCT, GRIPPER_FORCE_PCT, GRIPPER_MAXTIME_MS,
    GRIPPER_OPEN_THRESHOLD, GRIPPER_CLOSE_THRESHOLD,
)


class DHGripperController:
    POLL_HZ = 5

    def __init__(self):
        self._robot     = None
        self._thread    = None
        self._stop_evt  = threading.Event()
        self._state     = None      # "open" | "closed" | None

        self._norm      = 0.0
        self._norm_valid = False   # True after first update_normalized() call
        self._norm_lock  = threading.Lock()

        # Handshake events
        self._cmd_ready    = threading.Event()   # gripper → main: I need to send
        self._servo_paused = threading.Event()   # main → gripper: servo stopped, go ahead
        self._cmd_done     = threading.Event()   # gripper → main: command sent, resume

        self._target_pct   = 0                   # position to send when paused

    # ── called from main loop ─────────────────────────────────────────────────

    def get_normalized(self) -> float | None:
        """Return current trigger value [0.0, 1.0], or None if not yet set."""
        with self._norm_lock:
            return self._norm if self._norm_valid else None

    def update_normalized(self, norm: float):
        """Update with a normalised trigger value in [0.0, 1.0]."""
        with self._norm_lock:
            self._norm       = max(0.0, min(1.0, norm))
            self._norm_valid = True

    def wants_pause(self) -> bool:
        return self._cmd_ready.is_set()

    def pause_for_gripper(self, robot):
        """
        Called by main loop when wants_pause() is True.
        Stops servo mode, clears faults, lets gripper send, then resumes.
        Blocks for ~400ms total.
        """
        robot.stop_servo_mode()
        time.sleep(0.2)
        with robot._rpc_lock:
            robot._robot.ResetAllError()
        time.sleep(0.1)

        self._servo_paused.set()
        self._cmd_done.wait(timeout=2.0)
        self._cmd_done.clear()
        self._servo_paused.clear()
        self._cmd_ready.clear()

        with robot._rpc_lock:
            robot._robot.RobotEnable(1)
        robot.start_servo_mode()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self, robot):
        self._robot = robot

        err = robot.activate_gripper(GRIPPER_INDEX)
        if err != 0:
            print(f"[GRIPPER] ActGripper failed (err={err}) — check gripper config on FR5 controller")
            self._robot = None
            return
        time.sleep(0.5)
        print(f"[GRIPPER] DH AG-160-95 activated (index={GRIPPER_INDEX})")

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        self._servo_paused.set()   # unblock any pending handshake
        if self._thread:
            self._thread.join(timeout=2)
        self._robot = None

    # ── background thread ─────────────────────────────────────────────────────

    def _loop(self):
        interval = 1.0 / self.POLL_HZ
        while not self._stop_evt.is_set():
            t0 = time.monotonic()

            with self._norm_lock:
                norm  = self._norm
                valid = self._norm_valid

            if not valid:
                self._stop_evt.wait(timeout=interval)
                continue

            desired = None
            if norm >= GRIPPER_CLOSE_THRESHOLD:
                desired = "closed"
            elif norm <= GRIPPER_OPEN_THRESHOLD:
                desired = "open"

            if desired and desired != self._state:
                pct = GRIPPER_OPEN_PCT if desired == "open" else GRIPPER_CLOSE_PCT
                self._target_pct = pct

                self._cmd_ready.set()
                self._servo_paused.wait(timeout=3.0)

                if self._stop_evt.is_set():
                    break

                try:
                    err = self._robot.send_gripper(
                        GRIPPER_INDEX, pct,
                        GRIPPER_VEL_PCT, GRIPPER_FORCE_PCT,
                        GRIPPER_MAXTIME_MS,
                        1,
                        GRIPPER_TYPE,
                    )
                    if err == 0:
                        self._state = desired
                        print(f"[GRIPPER] {desired.upper()}  (trig={norm:.2f})")
                    else:
                        print(f"[GRIPPER] MoveGripper error {err}")
                except Exception as exc:
                    print(f"[GRIPPER] Exception: {exc}")
                finally:
                    self._cmd_done.set()

            remaining = interval - (time.monotonic() - t0)
            if remaining > 0:
                self._stop_evt.wait(timeout=remaining)
