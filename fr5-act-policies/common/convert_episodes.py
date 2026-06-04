"""
convert_episodes.py — raw teleop episodes  →  LeRobot v3.0 dataset.

The recorder writes one folder per episode:

    episodes/episode_000/
        data.csv            ~100 Hz log: so101 leader, fr5 cmd/actual joints,
                            gripper_norm, velocities, eef pose
        wrist_cam.mp4       ~30 Hz wrist camera
        wrist_cam_ts.npy    one timestamp per video frame
        scene_cam.*         (unused by the policy)
        meta.json           language instruction, camera intrinsics, ...

The ACT pipeline (training/dataset.py) consumes a LeRobot v3.0 dataset:

    <out>/meta/info.json
    <out>/meta/tasks.parquet
    <out>/meta/episodes/chunk-000/file-000.parquet
    <out>/data/chunk-000/file-000.parquet
    <out>/videos/observation.images.wrist_cam/chunk-000/file-<ep>.mp4

The CSV runs at ~100 Hz but the policy is trained/run at the 30 Hz camera
rate, so we resample the CSV onto the camera timestamps: exactly one dataset
row per video frame (nearest-in-time CSV sample). This keeps `frame_index`
aligned with the video and with the optional pre-extracted frames.

Usage:
    python tools/convert_episodes.py \
        --episodes episodes \
        --out ../so101-fr5-teleop/lerobot_dataset \
        --extract-frames

    # quick pipeline smoke test (few frames per episode, self-contained out dir)
    python tools/convert_episodes.py --out _smoke_dataset --extract-frames --max-frames 40
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import cv2
except ImportError:  # only needed for --extract-frames
    cv2 = None


CODEBASE_VERSION = "v3.0"
FPS = 30  # policy control rate; matches the camera, not the CSV

# CSV column -> dataset feature mapping
STATE_COLS = [f"fr5_actual_j{i}" for i in range(1, 7)]          # observation.state (6)
CMD_COLS = [f"fr5_cmd_j{i}" for i in range(1, 7)]               # action joints (6)
GRIPPER_COL = "gripper_cmd"                                     # action[6] (binary cmd; this dataset's gripper_norm is empty)
EEF_COLS = ["fr5_eef_x_mm", "fr5_eef_y_mm", "fr5_eef_z_mm",
            "fr5_eef_rx_deg", "fr5_eef_ry_deg", "fr5_eef_rz_deg"]  # observation.eef_pose (6)

CAMERAS    = ["wrist_cam", "scene_cam"]                 # both cameras as observations
VIDEO_KEYS = {c: f"observation.images.{c}" for c in CAMERAS}
VIDEO_KEY  = VIDEO_KEYS["wrist_cam"]                    # wrist drives the resample timeline


def _nearest_indices(csv_ts: np.ndarray, cam_ts: np.ndarray) -> np.ndarray:
    """For each camera timestamp, the index of the nearest CSV row."""
    pos = np.searchsorted(csv_ts, cam_ts)
    pos = np.clip(pos, 1, len(csv_ts) - 1)
    left, right = csv_ts[pos - 1], csv_ts[pos]
    choose_left = (cam_ts - left) <= (right - cam_ts)
    return pos - choose_left.astype(int)


def _load_episode(ep_dir: Path, max_frames: int | None):
    """Resample one raw episode onto its camera timestamps.

    Returns (frame_dict, task_str) or None if the episode can't be used.
    """
    csv_path = ep_dir / "data.csv"
    ts_path = ep_dir / "wrist_cam_ts.npy"
    if not csv_path.exists() or not ts_path.exists():
        print(f"  [skip] {ep_dir.name}: missing data.csv or wrist_cam_ts.npy")
        return None

    df = pd.read_csv(csv_path)
    cam_ts = np.load(ts_path).astype(np.float64)

    # actual-joint / eef columns have a couple of NaN rows at startup — fill them
    # so nearest-neighbour resampling never lands on a NaN.
    fill_cols = STATE_COLS + EEF_COLS
    df[fill_cols] = df[fill_cols].ffill().bfill()

    csv_ts = df["timestamp"].to_numpy(np.float64)
    idx = _nearest_indices(csv_ts, cam_ts)

    if max_frames is not None:
        idx = idx[:max_frames]

    rows = df.iloc[idx].reset_index(drop=True)
    n = len(rows)

    state = rows[STATE_COLS].to_numpy(np.float32)
    action = np.concatenate(
        [rows[CMD_COLS].to_numpy(np.float32),
         rows[[GRIPPER_COL]].to_numpy(np.float32)],
        axis=1,
    )
    eef = rows[EEF_COLS].to_numpy(np.float32)

    frame = {
        "observation.state": list(state),
        "action": list(action),
        "observation.eef_pose": list(eef),
        "frame_index": np.arange(n, dtype=np.int64),
        "timestamp": (cam_ts[:n] - cam_ts[0]).astype(np.float32),
        # which video frame each row maps to (here: identity, one row per frame)
        "video_frame": np.arange(n, dtype=np.int64),
    }

    meta_path = ep_dir / "meta.json"
    task = "pick up the block and place it in the bin"
    if meta_path.exists():
        _m = json.loads(meta_path.read_text())
        task = _m.get("language_instruction") or _m.get("instruction") or task

    return frame, task


def _extract_frames(video: Path, n_frames: int, out_dir: Path):
    """Dump the first n_frames of `video` as JPEGs: 000000.jpg, 000001.jpg, ..."""
    if cv2 is None:
        raise RuntimeError("--extract-frames needs opencv-python (cv2) installed")
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    written = 0
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), frame)
        written += 1
    cap.release()
    return written


def convert(episodes_dir: Path, out_dir: Path, extract_frames: bool,
            max_frames: int | None):
    ep_dirs = sorted(p for p in episodes_dir.glob("episode_*") if p.is_dir())
    if not ep_dirs:
        raise SystemExit(f"no episode_* folders under {episodes_dir}")

    print(f"found {len(ep_dirs)} episode(s) under {episodes_dir}")

    data_dir = out_dir / "data" / "chunk-000"
    ep_meta_dir = out_dir / "meta" / "episodes" / "chunk-000"
    video_dirs   = {c: out_dir / "videos" / VIDEO_KEYS[c] / "chunk-000" for c in CAMERAS}
    frames_roots = {c: out_dir / "frames" / VIDEO_KEYS[c] for c in CAMERAS}
    for d in (data_dir, ep_meta_dir, *video_dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    ep_records: list[dict] = []
    tasks: dict[str, int] = {}
    cursor = 0  # running absolute row index across episodes

    for ep_idx, ep_dir in enumerate(ep_dirs):
        loaded = _load_episode(ep_dir, max_frames)
        if loaded is None:
            continue
        frame, task = loaded
        n = len(frame["frame_index"])
        task_index = tasks.setdefault(task, len(tasks))

        for i in range(n):
            all_rows.append({
                "observation.state": frame["observation.state"][i],
                "action": frame["action"][i],
                "observation.eef_pose": frame["observation.eef_pose"][i],
                "timestamp": float(frame["timestamp"][i]),
                "frame_index": int(frame["frame_index"][i]),
                "episode_index": ep_idx,
                "index": cursor + i,
                "task_index": task_index,
            })

        # copy + extract every camera's video so dataset.py can read frames
        for cam in CAMERAS:
            src_video = ep_dir / f"{cam}.mp4"
            dst_video = video_dirs[cam] / f"file-{ep_idx:03d}.mp4"
            if src_video.exists():
                shutil.copyfile(src_video, dst_video)
                if extract_frames:
                    got = _extract_frames(dst_video, n, frames_roots[cam] / f"ep-{ep_idx:03d}")
                    print(f"  {ep_dir.name} [{cam}]: extracted {got}/{n} frames")
            else:
                print(f"  [warn] {ep_dir.name}: no {cam}.mp4")

        ep_records.append({
            "episode_index": ep_idx,
            "dataset_from_index": cursor,
            "dataset_to_index": cursor + n,
            "length": n,
            "tasks": [task],
            "task": task,
        })
        cursor += n
        print(f"  {ep_dir.name} -> ep {ep_idx}: {n} frames "
              f"[{cursor - n}:{cursor})  task={task!r}")

    if not all_rows:
        raise SystemExit("no frames produced — nothing written")

    total_frames = cursor
    total_episodes = len(ep_records)

    # ---- data/chunk-000/file-000.parquet ----
    data_df = pd.DataFrame(all_rows)
    pq.write_table(pa.Table.from_pandas(data_df, preserve_index=False),
                   data_dir / "file-000.parquet")

    # ---- meta/episodes/chunk-000/file-000.parquet ----
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(ep_records), preserve_index=False),
                   ep_meta_dir / "file-000.parquet")

    # ---- meta/tasks.parquet ----
    tasks_df = pd.DataFrame(
        {"task": list(tasks.keys()), "task_index": list(tasks.values())}
    ).set_index("task_index")
    pq.write_table(pa.Table.from_pandas(tasks_df.reset_index(), preserve_index=False),
                   out_dir / "meta" / "tasks.parquet")

    # ---- meta/info.json ----
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "fairino_fr5",
        "fps": FPS,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_chunks": 1,
        "chunks_size": 1000,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [len(STATE_COLS)],
                                   "names": STATE_COLS},
            "action": {"dtype": "float32", "shape": [len(CMD_COLS) + 1],
                       "names": CMD_COLS + [GRIPPER_COL]},
            "observation.eef_pose": {"dtype": "float32", "shape": [len(EEF_COLS)],
                                     "names": EEF_COLS},
            VIDEO_KEY: {"dtype": "video", "shape": [480, 640, 3]},
        },
    }
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    print(f"\nwrote LeRobot dataset to {out_dir}")
    print(f"  episodes={total_episodes}  frames={total_frames}  "
          f"tasks={len(tasks)}  fps={FPS}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="episodes",
                    help="folder containing episode_* directories")
    ap.add_argument("--out", default="../so101-fr5-teleop/lerobot_dataset",
                    help="output LeRobot dataset root")
    ap.add_argument("--extract-frames", action="store_true",
                    help="pre-extract aligned JPEG frames for fast training I/O")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames per episode (for quick pipeline tests)")
    args = ap.parse_args()

    convert(Path(args.episodes), Path(args.out),
            args.extract_frames, args.max_frames)


if __name__ == "__main__":
    main()
