"""
camera.py — Intel RealSense capture for ACT episode recording.

`RealSenseCamera` opens one device by serial and runs a background thread that
reads frames at CAMERA_FPS. When recording is active it buffers
`(timestamp, bgr[, depth_u16])` tuples. Timestamps are host wall-clock
(`time.time()`), the SAME clock the control loop writes into data.csv, so frames
and control rows align by a simple nearest-timestamp join offline.

`DualCamera` manages the two-camera rig used for ACT:
  - wrist : D405  color           (eye-in-hand)
  - scene : D435I color + depth   (eye-to-hand, aligned uint16 depth)

Intrinsics come from each device's on-chip factory calibration — no checkerboard.
They are written into the episode meta.json by the logger.
"""

import threading
import time

import numpy as np

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    WRIST_CAM_SERIAL, SCENE_CAM_SERIAL, SCENE_CAM_DEPTH,
)


class RealSenseCamera:
    def __init__(self, serial: str | None = None, name: str = "cam", depth: bool = False):
        self.name        = name
        self._serial     = serial
        self._want_depth = depth

        self._pipeline   = None
        self._align      = None
        self._intrinsics = None
        self._depth_scale = None

        self._thread     = None
        self._stop_evt   = threading.Event()

        self._recording   = False
        self._frames      = []   # (ts, bgr_uint8[, depth_uint16])
        self._frames_lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _open_pipeline(self, warmup: bool = True):
        """(Re)open the RealSense pipeline and read intrinsics. Used at start and
        on reconnect after a USB disconnect."""
        import pyrealsense2 as rs

        cfg = rs.config()
        if self._serial:
            cfg.enable_device(self._serial)
        cfg.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
        if self._want_depth:
            cfg.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)

        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(cfg)

        if self._want_depth:
            self._align = rs.align(rs.stream.color)
            self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        ci = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._intrinsics = {
            "serial":           self._serial,
            "width":            ci.width,
            "height":           ci.height,
            "fx":               ci.fx,
            "fy":               ci.fy,
            "cx":               ci.ppx,
            "cy":               ci.ppy,
            "distortion_model": str(ci.model).split(".")[-1],
            "dist_coeffs":      list(ci.coeffs),
            "depth_scale_m":    self._depth_scale,
        }

        if warmup:
            for _ in range(5):
                try:
                    self._pipeline.wait_for_frames(timeout_ms=1000)
                except Exception:
                    pass

        d = " + depth" if self._want_depth else ""
        print(
            f"[CAMERA] {self.name} ({self._serial or 'default'}) ready — "
            f"{CAMERA_WIDTH}×{CAMERA_HEIGHT}{d} @ {CAMERA_FPS} fps  "
            f"fx={ci.fx:.1f} fy={ci.fy:.1f} cx={ci.ppx:.1f} cy={ci.ppy:.1f}"
        )

    def _reopen(self) -> bool:
        """Attempt to recover the pipeline after a USB disconnect. Returns success."""
        try:
            if self._pipeline:
                try: self._pipeline.stop()
                except Exception: pass
            self._pipeline = None
            self._open_pipeline(warmup=False)
            print(f"[CAMERA] {self.name} reconnected.")
            return True
        except Exception:
            self._pipeline = None
            return False

    def start(self):
        self._open_pipeline(warmup=True)
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"cam-{self.name}")
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None

    # ── recording control ──────────────────────────────────────────────────────

    def start_recording(self):
        with self._frames_lock:
            self._frames    = []
            self._recording = True

    def stop_recording(self) -> list:
        """Return [(ts, bgr[, depth_u16]), ...] captured during the episode."""
        with self._frames_lock:
            self._recording = False
            frames = list(self._frames)
            self._frames = []
        return frames

    @property
    def intrinsics(self) -> dict | None:
        return self._intrinsics

    # ── background capture thread ───────────────────────────────────────────────

    def _loop(self):
        consecutive_errors = 0
        while not self._stop_evt.is_set():
            try:
                framesets = self._pipeline.wait_for_frames(timeout_ms=1000)
                if self._align is not None:
                    framesets = self._align.process(framesets)

                color = framesets.get_color_frame()
                if not color:
                    continue

                consecutive_errors = 0
                ts  = time.time()
                img = np.asanyarray(color.get_data()).copy()   # H×W×3 BGR uint8

                rec = (ts, img)
                if self._want_depth:
                    df = framesets.get_depth_frame()
                    depth = (np.asanyarray(df.get_data()).copy()
                             if df else np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH), np.uint16))
                    rec = (ts, img, depth)

                with self._frames_lock:
                    if self._recording:
                        self._frames.append(rec)

            except Exception as exc:
                if self._stop_evt.is_set():
                    break
                consecutive_errors += 1
                if consecutive_errors == 1 or consecutive_errors % 30 == 0:
                    print(f"[CAMERA] {self.name} frame error #{consecutive_errors}: {exc}")
                # USB disconnect, pipeline drop, OR sustained frame timeouts
                # (bandwidth/power starvation) → try to recover instead of
                # spinning forever (a dead thread silently zeroes recordings).
                exc_str = str(exc).lower()
                if ("disconnect" in exc_str or "before start" in exc_str
                        or "errno=5" in exc_str or self._pipeline is None
                        or consecutive_errors >= 10):
                    while not self._stop_evt.is_set():
                        time.sleep(1.0)
                        if self._reopen():
                            consecutive_errors = 0
                            break

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# Back-compat: teleop_vr.py / older EpisodeLogger expect a single D405 camera.
class D405Camera(RealSenseCamera):
    def __init__(self):
        super().__init__(serial=WRIST_CAM_SERIAL, name="wrist_cam", depth=False)


class DualCamera:
    """
    Wrist (D405 color) + scene (D435I color + aligned depth) rig for ACT recording.
    Mirrors the start/stop/start_recording/stop_recording interface so the logger
    can treat the pair as one unit.
    """

    def __init__(self):
        self.wrist = RealSenseCamera(WRIST_CAM_SERIAL, name="wrist_cam", depth=False)
        self.scene = RealSenseCamera(SCENE_CAM_SERIAL, name="scene_cam", depth=SCENE_CAM_DEPTH)

    def start(self):
        # Scene first (heavier: color+depth) so it has the longest warm-up.
        self.scene.start()
        self.wrist.start()

    def stop(self):
        self.wrist.stop()
        self.scene.stop()

    def start_recording(self):
        self.wrist.start_recording()
        self.scene.start_recording()

    def stop_recording(self) -> dict:
        """{'wrist': [(ts,bgr)...], 'scene': [(ts,bgr,depth)...]}"""
        return {
            "wrist": self.wrist.stop_recording(),
            "scene": self.scene.stop_recording(),
        }

    @property
    def intrinsics(self) -> dict:
        return {"wrist_cam": self.wrist.intrinsics, "scene_cam": self.scene.intrinsics}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
