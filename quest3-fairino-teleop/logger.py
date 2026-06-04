"""
logger.py — ACT episode recorder for VR teleoperation.

Each episode is saved to its own directory:

  episodes/episode_NNN/
    data.csv             125 Hz control rows (state / action / eef / vel / vr)
    meta.json            instruction, duration, success, intrinsics, metrics
    wrist_cam.mp4        D405 RGB 640×480 @30
    scene_cam.mp4        D435I RGB 640×480 @30
    wrist_cam_ts.npy     per-frame host timestamps (float64, aligns with data.csv)
    scene_cam_ts.npy
    scene_cam_depth.npz  aligned uint16 depth, key 'depth' shape (N,480,640)

ACT mapping (see export_lerobot.py):
  observation.state  = fr5_actual_j1..6                  (6)
  action             = fr5_cmd_j1..6 + gripper_cmd       (7, gripper binary)
  observation.images.wrist_cam / scene_cam               (video)
  observation.eef_pose / gripper_norm / joint_vel        (extra, optional)
"""

import json
import os
import time

import numpy as np
import pandas as pd

from config import LOG_DIR, INSTRUCTION_FILE


class EpisodeLogger:
    def __init__(self):
        self._recording   = False
        self._rows        = []
        self._cameras     = None        # DualCamera (optional)
        self._epi_index   = None
        self._epi_dir     = None
        self._start_time  = 0.0
        self._instruction = ""
        self._last_path   = None        # dir of the most recently saved episode

    @property
    def recording(self) -> bool:
        return self._recording

    def set_cameras(self, cameras) -> None:
        """Attach a DualCamera (or any obj with start/stop_recording + intrinsics)."""
        self._cameras = cameras

    # ── episode lifecycle ───────────────────────────────────────────────────────

    def start(self):
        self._epi_index   = self._next_episode_index()
        self._epi_dir     = os.path.join(LOG_DIR, f"episode_{self._epi_index:03d}")
        self._rows        = []
        self._start_time  = time.time()
        self._instruction = self._read_instruction()
        self._recording   = True
        if self._cameras:
            self._cameras.start_recording()
        print(f"[LOGGER] ● REC  episode_{self._epi_index:03d}  «{self._instruction}»")

    def log(
        self,
        timestamp:    float,
        fr5_actual:   list[float],          # observation.state (6)
        fr5_cmd:      list[float],          # commanded joints (6)
        gripper_cmd:  int,                  # action[6] — 1=close, 0=open
        fr5_eef:      list[float] | None = None,   # [x,y,z mm, rx,ry,rz deg]
        gripper_norm: float | None = None,        # observed jaw position 0..1
        fr5_vel:      list[float] | None = None,   # joint velocities (6)
        vr_trigger:   float | None = None,
        vr_grip:      float | None = None,
    ):
        if not self._recording:
            return

        row: dict = {
            "timestamp":     timestamp,
            "episode_index": self._epi_index,
            "frame_index":   len(self._rows),
            "task_index":    0,
            "next.done":     False,
        }
        for i, v in enumerate(fr5_actual[:6], start=1):
            row[f"fr5_actual_j{i}"] = v
        for i, v in enumerate(fr5_cmd[:6], start=1):
            row[f"fr5_cmd_j{i}"] = v
        row["gripper_cmd"] = int(gripper_cmd)

        if fr5_eef is not None and len(fr5_eef) == 6:
            for k, v in zip(
                ["fr5_eef_x_mm", "fr5_eef_y_mm", "fr5_eef_z_mm",
                 "fr5_eef_rx_deg", "fr5_eef_ry_deg", "fr5_eef_rz_deg"], fr5_eef):
                row[k] = v
        if gripper_norm is not None:
            row["gripper_norm"] = gripper_norm
        if fr5_vel is not None:
            for i, v in enumerate(fr5_vel[:6], start=1):
                row[f"fr5_vel_j{i}"] = v
        if vr_trigger is not None:
            row["vr_trigger"] = vr_trigger
        if vr_grip is not None:
            row["vr_grip"] = vr_grip

        self._rows.append(row)

    def stop(self, success: bool | None = None) -> str | None:
        """Flush the episode to disk. Returns the episode dir, or None if empty."""
        self._recording = False
        cam = self._cameras.stop_recording() if self._cameras else {"wrist": [], "scene": []}

        if not self._rows:
            print("[LOGGER] ○ stopped — no data, nothing saved.")
            return None

        os.makedirs(self._epi_dir, exist_ok=True)
        self._rows[-1]["next.done"] = True
        df = pd.DataFrame(self._rows)

        # Global frame index continuing across episodes (LeRobot-style).
        df.insert(0, "index", self._global_index_offset() + np.arange(len(df)))
        df.to_csv(os.path.join(self._epi_dir, "data.csv"), index=False)

        wrist_n = self._save_camera("wrist_cam", cam.get("wrist", []), depth=False)
        scene_n = self._save_camera("scene_cam", cam.get("scene", []), depth=True)

        metrics  = self._metrics(df)
        duration = float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])
        meta = {
            "episode_index":        self._epi_index,
            "instruction":          self._instruction,
            "start_time":           self._start_time,
            "duration_s":           round(duration, 3),
            "num_steps":            int(len(df)),
            "control_hz":           round(len(df) / duration, 1) if duration > 0 else None,
            "success":              success,                    # null until annotated
            "camera_intrinsics":    self._cameras.intrinsics if self._cameras else None,
            "camera_frames":        {"wrist_cam": wrist_n, "scene_cam": scene_n},
            "quality_metrics":      metrics,
        }
        with open(os.path.join(self._epi_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self._last_path = self._epi_dir
        if wrist_n == 0 or scene_n == 0:
            bad = "wrist_cam" if wrist_n == 0 else ""
            bad += (" + " if bad and scene_n == 0 else "") + ("scene_cam" if scene_n == 0 else "")
            print(f"[LOGGER] ⚠️⚠️  WARNING: {bad} recorded 0 FRAMES — camera dropped! "
                  f"This episode is UNUSABLE. Restart the bridge to recover the camera, "
                  f"then re-record (delete episode_{self._epi_index:03d}).")
        print(f"[LOGGER] grasps={metrics['grasps']} (regrasps={metrics['regrasps']}) "
              f"pauses={metrics['pauses']} smoothness={metrics['smoothness']} "
              f"dur={duration:.1f}s  cams: wrist={wrist_n} scene={scene_n}")
        print(f"[LOGGER] Episode {self._epi_index:03d} saved → {self._epi_dir}/")
        return self._epi_dir

    # ── metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def _metrics(df: pd.DataFrame) -> dict:
        g = df["gripper_cmd"].to_numpy()
        closes = int(np.sum((g[1:] == 1) & (g[:-1] == 0)))      # open→close edges
        grasps   = closes
        regrasps = max(0, closes - 1)

        # Pauses: spans where joint speed stays below 2°/s for ≥ 0.4 s.
        pauses = 0
        smoothness = None
        vel_cols = [c for c in df.columns if c.startswith("fr5_vel_j")]
        if vel_cols and len(df) > 2:
            spd = np.linalg.norm(df[vel_cols].to_numpy(float), axis=1)   # deg/s
            t   = df["timestamp"].to_numpy(float)
            dt  = np.diff(t); dt[dt <= 0] = 1e-3
            slow = spd < 2.0
            i = 0
            while i < len(slow):
                if slow[i]:
                    j = i
                    while j < len(slow) and slow[j]:
                        j += 1
                    if (t[min(j, len(t) - 1)] - t[i]) >= 0.4:
                        pauses += 1
                    i = j
                else:
                    i += 1
            # Smoothness: dimensionless normalized jerk (lower = smoother).
            accel = np.diff(spd) / dt
            jerk  = np.diff(accel) / dt[1:]
            njerk = float(np.sqrt(np.mean(jerk ** 2)) * (t[-1] - t[0]) ** 2 / (spd.mean() + 1e-6))
            smoothness = round(njerk, 2)

        return {"grasps": grasps, "regrasps": regrasps,
                "pauses": pauses, "smoothness": smoothness}

    # ── file writers ────────────────────────────────────────────────────────────

    def _save_camera(self, name: str, frames: list, depth: bool) -> int:
        if not frames:
            return 0
        import cv2

        ts = np.array([f[0] for f in frames], dtype=np.float64)
        np.save(os.path.join(self._epi_dir, f"{name}_ts.npy"), ts)

        h, w = frames[0][1].shape[:2]
        from config import CAMERA_FPS
        path   = os.path.join(self._epi_dir, f"{name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw     = cv2.VideoWriter(path, fourcc, CAMERA_FPS, (w, h))
        if not vw.isOpened():
            print(f"[LOGGER] could not open video writer for {path} — skipping mp4.")
        else:
            for f in frames:
                vw.write(f[1])
            vw.release()

        if depth:
            d = np.stack([f[2] for f in frames]).astype(np.uint16)   # (N,H,W)
            np.savez_compressed(os.path.join(self._epi_dir, f"{name}_depth.npz"), depth=d)

        return len(frames)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _next_episode_index(self) -> int:
        os.makedirs(LOG_DIR, exist_ok=True)
        idxs = [int(d.split("_")[1]) for d in os.listdir(LOG_DIR)
                if d.startswith("episode_") and d.split("_")[1].isdigit()]
        return (max(idxs) + 1) if idxs else 0

    def _global_index_offset(self) -> int:
        total = 0
        for d in os.listdir(LOG_DIR):
            mp = os.path.join(LOG_DIR, d, "meta.json")
            if d.startswith("episode_") and os.path.exists(mp):
                try:
                    total += json.load(open(mp)).get("num_steps", 0)
                except Exception:
                    pass
        return total

    def _read_instruction(self) -> str:
        try:
            with open(INSTRUCTION_FILE) as f:
                return f.read().strip()
        except Exception:
            return ""
