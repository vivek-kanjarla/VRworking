"""
teleop_vr.py — Meta Quest 3 VR teleoperation loop.

The Quest 3 right controller (configurable) drives the FR5 end-effector using
delta-based Cartesian control:
  - Controller position delta → EEF translation
  - Controller orientation delta → EEF rotation
  - Trigger analog → gripper open/close

Controls:
  Space    — emergency stop
  R        — toggle episode recording
  H        — re-home (capture current poses as new reference)
  Ctrl-C   — graceful exit

Setup checklist (see docs/quest3_setup.md for full instructions):
  1. Sideload Oculus Reader APK and run `adb forward tcp:5555 tcp:5555`
     (or configure UDP mode in config.py)
  2. Confirm FR5 is reachable: python check_network.py
  3. Run: python teleop_vr.py
"""

import time
import threading
from contextlib import contextmanager

import numpy as np
from pynput import keyboard

from config import (
    LOOP_HZ, LOOP_PERIOD,
    LOG_DIR, INSTRUCTION_FILE, LOG_STATE_DOWNSAMPLE,
    GRIPPER_OPEN_THRESHOLD, GRIPPER_CLOSE_THRESHOLD,
    MAX_DELTA_DEG_PER_CYCLE,
)
from quest3 import Quest3Reader
from fr5 import FR5Controller
from mapper_vr import vr_to_fr5
from logger import EpisodeLogger
from singularity import check as singularity_check, Level
from gripper import DHGripperController
from camera import D405Camera


@contextmanager
def _camera_cleanup(camera):
    try:
        yield
    finally:
        if camera:
            camera.stop()


class VRTeleopSession:
    def __init__(self):
        self._stop_event   = threading.Event()
        self._rehome_event = threading.Event()
        self._estop        = False
        self._logger       = EpisodeLogger()
        self._fr5_current  = [0.0] * 6
        self._sing_level   = Level.CLEAR
        self._cycle        = 0
        self._comm_errors  = 0
        self._state_cache  = {"actual": None, "eef": None, "vel": None}

        # Gripper state for VR trigger mapping (separate from DHGripperController
        # which handles the SO-101 path — here we drive it from trigger analog)
        self._gripper_state = None    # "open" | "closed" | None

    # ── keyboard ──────────────────────────────────────────────────────────────

    def _on_press(self, key):
        if key == keyboard.Key.space:
            print("\n[E-STOP] Space pressed — stopping motion.")
            self._estop = True
            self._stop_event.set()
        elif hasattr(key, "char") and key.char in ("r", "R"):
            if self._logger.recording:
                path = self._logger.stop()
                print(f"[REC] Stopped — saved to {path}")
            else:
                self._state_cache = {"actual": None, "eef": None, "vel": None}
                self._logger.start()
                instr = self._logger._instruction
                instr_str = (
                    f'  instruction: "{instr}"' if instr
                    else f"  (no instruction — write to {INSTRUCTION_FILE} before pressing R)"
                )
                print(f"[REC] Recording started (episode {self._logger._episode_id})\n{instr_str}")
        elif hasattr(key, "char") and key.char in ("h", "H"):
            self._rehome_event.set()

    # ── gripper trigger mapping ───────────────────────────────────────────────

    def _update_gripper_from_trigger(self, trigger: float, gripper_ctrl: DHGripperController):
        """Map Quest trigger analog to gripper open/close via hysteresis."""
        if trigger >= GRIPPER_CLOSE_THRESHOLD:
            desired = "closed"
        elif trigger <= GRIPPER_OPEN_THRESHOLD:
            desired = "open"
        else:
            return  # hysteresis band — hold current state

        if desired != self._gripper_state:
            # Pass normalised extremes so gripper.py threshold logic triggers:
            #   open   → 0.0 (≤ GRIPPER_OPEN_THRESHOLD)
            #   closed → 1.0 (≥ GRIPPER_CLOSE_THRESHOLD)
            norm = 0.0 if desired == "open" else 1.0
            gripper_ctrl.update_normalized(norm)
            self._gripper_state = desired

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        print("Connecting to hardware...")
        print(f"  Tip: write task description to {INSTRUCTION_FILE} before pressing R.")
        print(f"  Quest 3 mode: {__import__('config').QUEST3_MODE}  hand: {__import__('config').QUEST3_ACTIVE_HAND}")

        camera = D405Camera()
        try:
            camera.start()
            self._logger.set_camera(camera)
        except Exception as exc:
            print(f"[CAMERA] Could not start D405 ({exc}) — continuing without camera.")
            camera = None

        with _camera_cleanup(camera), Quest3Reader() as quest, FR5Controller() as robot:
            # Capture home poses
            quest_state       = quest.get_state()
            quest_home_pos    = quest_state.pos.copy()
            quest_home_quat   = quest_state.quat.copy()

            self._fr5_current = robot.get_joint_positions()
            fr5_home_joints   = list(self._fr5_current)
            fr5_home_eef      = robot.get_eef_pose()   # [x,y,z,rx,ry,rz] mm/deg

            gripper_ctrl = DHGripperController()
            gripper_ctrl.start(robot)

            print(f"Quest home pos (m):   {[f'{v:.3f}' for v in quest_home_pos]}")
            print(f"FR5 home joints (°):  {[f'{v:.1f}' for v in fr5_home_joints]}")
            print(f"FR5 home EEF (mm/°):  {[f'{v:.1f}' for v in fr5_home_eef]}")

            robot.start_servo_mode()
            print("Ready. Space=E-STOP  R=record  H=re-home  Ctrl-C=quit")

            listener = keyboard.Listener(on_press=self._on_press)
            listener.start()

            try:
                while not self._stop_event.is_set():
                    t0 = time.monotonic()

                    try:
                        # Re-home
                        if self._rehome_event.is_set():
                            self._rehome_event.clear()
                            s = quest.get_state()
                            if s.valid:
                                quest_home_pos    = s.pos.copy()
                                quest_home_quat   = s.quat.copy()
                            self._fr5_current = robot.get_joint_positions()
                            fr5_home_joints   = list(self._fr5_current)
                            fr5_home_eef      = robot.get_eef_pose()
                            self._sing_level  = Level.CLEAR
                            print(f"\n[RE-HOME] New home captured — FR5 J1..J6: "
                                  f"{[f'{v:.1f}' for v in fr5_home_joints]}")

                        quest_state = quest.get_state()

                        if not quest_state.valid:
                            # No fresh Quest data — hold position
                            robot.servo_j(self._fr5_current)
                            self._cycle += 1
                            elapsed = time.monotonic() - t0
                            if elapsed < LOOP_PERIOD:
                                time.sleep(LOOP_PERIOD - elapsed)
                            continue

                        # Gripper from trigger
                        self._update_gripper_from_trigger(
                            quest_state.trigger, gripper_ctrl
                        )
                        if gripper_ctrl.wants_pause():
                            gripper_ctrl.pause_for_gripper(robot)
                            self._fr5_current = robot.get_joint_positions()

                        # Singularity check on current FR5 position
                        level, scale, msg = singularity_check(self._fr5_current)
                        if level != self._sing_level:
                            if level == Level.DANGER:
                                print(f"\n[SINGULARITY DANGER] {msg} — motion blocked")
                            elif level == Level.WARN:
                                print(f"\n[SINGULARITY WARN]   {msg} — speed reduced to {scale*100:.0f}%")
                            elif level == Level.CLEAR:
                                print("\n[SINGULARITY] Clear")
                            self._sing_level = level

                        if level == Level.DANGER:
                            fr5_cmd = list(self._fr5_current)
                        else:
                            fr5_cmd = vr_to_fr5(
                                quest_state.pos,
                                quest_state.quat,
                                quest_home_pos,
                                quest_home_quat,
                                fr5_home_eef,
                                self._fr5_current,
                                robot,
                                sing_scale=scale,
                            )

                        log_time = time.time()
                        robot.servo_j(fr5_cmd)
                        self._fr5_current = fr5_cmd
                        self._cycle += 1

                        # Staggered state reads (same strategy as teleop.py)
                        if self._logger.recording and self._cycle % LOG_STATE_DOWNSAMPLE == 0:
                            slot = (self._cycle // LOG_STATE_DOWNSAMPLE) % 3
                            if slot == 0:
                                try:
                                    self._state_cache["actual"] = robot.get_joint_positions()
                                except Exception:
                                    pass
                            elif slot == 1:
                                try:
                                    self._state_cache["eef"] = robot.get_eef_pose()
                                except Exception:
                                    pass
                            else:
                                try:
                                    self._state_cache["vel"] = robot.get_joint_velocities()
                                except Exception:
                                    pass

                        self._logger.log(
                            log_time,
                            {
                                "quest_pos_x":  float(quest_state.pos[0]),
                                "quest_pos_y":  float(quest_state.pos[1]),
                                "quest_pos_z":  float(quest_state.pos[2]),
                                "quest_quat_x": float(quest_state.quat[0]),
                                "quest_quat_y": float(quest_state.quat[1]),
                                "quest_quat_z": float(quest_state.quat[2]),
                                "quest_quat_w": float(quest_state.quat[3]),
                            },
                            fr5_cmd,
                            fr5_actual=self._state_cache["actual"],
                            fr5_eef=self._state_cache["eef"],
                            gripper_norm=quest_state.trigger,
                            fr5_vel=self._state_cache["vel"],
                        )

                        # Heartbeat
                        if self._cycle % LOOP_HZ == 0:
                            pos_drift = np.linalg.norm(quest_state.pos - quest_home_pos) * 1000  # mm
                            fr5_drift = max(abs(c - h) for c, h in zip(fr5_cmd, fr5_home_joints))
                            print(f"[HB] cyc={self._cycle}  sing={level.value}  "
                                  f"Quest moved {pos_drift:5.1f}mm  →  "
                                  f"FR5 cmd moved {fr5_drift:6.1f}°  "
                                  f"trig={quest_state.trigger:.2f}  errs={self._comm_errors}")

                    except Exception as exc:
                        self._comm_errors += 1
                        if self._comm_errors == 1 or self._comm_errors % 50 == 0:
                            print(f"[WARN] cycle skipped (error #{self._comm_errors}): {exc!r}")
                        time.sleep(LOOP_PERIOD)
                        continue

                    elapsed = time.monotonic() - t0
                    sleep   = LOOP_PERIOD - elapsed
                    if sleep > 0:
                        time.sleep(sleep)

            except KeyboardInterrupt:
                print("\n[EXIT] Ctrl-C — shutting down.")
            finally:
                gripper_ctrl.stop()
                robot.stop_servo_mode()
                listener.stop()
                if self._logger.recording:
                    path = self._logger.stop()
                    if path:
                        print(f"[REC] Auto-saved episode to {path}")

        if self._estop:
            print("[E-STOP] Motion halted. Check robot state before restarting.")
        print("VR teleop session ended.")


if __name__ == "__main__":
    VRTeleopSession().run()
