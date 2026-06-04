"""
export_lerobot.py — convert recorded episodes → ACT / LeRobot training arrays.

Aligns the 125 Hz control rows to camera frames by nearest host-timestamp join
(both use time.time()), resamples to --fps, and emits per-episode ACT tensors:

  observation.state            (T, 6)   FR5 actual joint positions
  action                       (T, 7)   FR5 cmd joints (6) + gripper_cmd (binary)
  observation.eef_pose         (T, 6)   x,y,z mm + rx,ry,rz deg          (extra)
  observation.gripper_norm     (T, 1)   observed jaw position            (extra)
  observation.joint_vel        (T, 6)   joint velocities                 (extra)
  next.done                    (T,)     episode-end flag
  timestamp / frame_index / episode_index / task_index
  wrist_frame_idx / scene_frame_idx     row → mp4 frame map (decode on demand)

Videos (wrist_cam.mp4, scene_cam.mp4) and scene_cam_depth.npz are referenced in
place. By default only success==true episodes are exported.

Output:  <out>/episode_NNN.npz  +  <out>/dataset_info.json

If `lerobot` is installed this also prints the recommended packaging command;
the npz layout maps 1:1 onto LeRobotDataset features.

Usage:
  python3 export_lerobot.py --out ./act_dataset
  python3 export_lerobot.py --out ./act_dataset --fps 30 --include-failed
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from config import LOG_DIR

STATE_COLS  = [f"fr5_actual_j{i}" for i in range(1, 7)]
CMD_COLS    = [f"fr5_cmd_j{i}"    for i in range(1, 7)]
EEF_COLS    = ["fr5_eef_x_mm", "fr5_eef_y_mm", "fr5_eef_z_mm",
               "fr5_eef_rx_deg", "fr5_eef_ry_deg", "fr5_eef_rz_deg"]
VEL_COLS    = [f"fr5_vel_j{i}"    for i in range(1, 7)]


def _nearest_idx(src_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """For each query timestamp, the index of the nearest src timestamp."""
    pos = np.searchsorted(src_ts, query_ts)
    pos = np.clip(pos, 1, len(src_ts) - 1)
    left, right = src_ts[pos - 1], src_ts[pos]
    return np.where(np.abs(query_ts - left) <= np.abs(query_ts - right), pos - 1, pos)


def export_episode(epi_dir: str, out_dir: str, fps: float) -> dict | None:
    meta = json.load(open(os.path.join(epi_dir, "meta.json")))
    df   = pd.read_csv(os.path.join(epi_dir, "data.csv"))
    if df.empty:
        return None

    t = df["timestamp"].to_numpy(float)
    # Resample the control timeline to the requested fps.
    t0, t1 = t[0], t[-1]
    n = max(1, int(round((t1 - t0) * fps)))
    grid = t0 + np.arange(n) / fps
    ridx = _nearest_idx(t, grid)
    g    = df.iloc[ridx].reset_index(drop=True)

    state  = g[STATE_COLS].to_numpy(np.float32)
    action = np.concatenate(
        [g[CMD_COLS].to_numpy(np.float32),
         g["gripper_cmd"].to_numpy(np.float32)[:, None]], axis=1)        # (T,7)

    out = {
        "observation.state":     state,
        "action":                action,
        "timestamp":             (grid - t0).astype(np.float32),
        "frame_index":           np.arange(n, dtype=np.int64),
        "episode_index":         np.full(n, meta["episode_index"], np.int64),
        "task_index":            np.zeros(n, np.int64),
        "next.done":             np.array([i == n - 1 for i in range(n)]),
    }
    if all(c in g for c in EEF_COLS):
        out["observation.eef_pose"] = g[EEF_COLS].to_numpy(np.float32)
    if "gripper_norm" in g:
        out["observation.gripper_norm"] = g["gripper_norm"].to_numpy(np.float32)[:, None]
    if all(c in g for c in VEL_COLS):
        out["observation.joint_vel"] = g[VEL_COLS].to_numpy(np.float32)

    # Map each grid row to a camera mp4 frame index via the camera ts arrays.
    for cam in ("wrist_cam", "scene_cam"):
        tsp = os.path.join(epi_dir, f"{cam}_ts.npy")
        if os.path.exists(tsp):
            cam_ts = np.load(tsp)
            out[f"{cam}_frame_idx"] = _nearest_idx(cam_ts, grid).astype(np.int64)

    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f"episode_{meta['episode_index']:03d}.npz")
    np.savez_compressed(npz, **out)
    return {
        "episode_index": meta["episode_index"],
        "instruction":   meta.get("instruction", ""),
        "success":       meta.get("success"),
        "num_frames":    int(n),
        "source_dir":    epi_dir,
        "wrist_cam":     os.path.join(epi_dir, "wrist_cam.mp4"),
        "scene_cam":     os.path.join(epi_dir, "scene_cam.mp4"),
        "scene_depth":   os.path.join(epi_dir, "scene_cam_depth.npz"),
        "npz":           npz,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=LOG_DIR, help="recorded episodes root")
    ap.add_argument("--out", default="./act_dataset", help="output dataset dir")
    ap.add_argument("--fps", type=float, default=30.0, help="resample rate")
    ap.add_argument("--include-failed", action="store_true",
                    help="also export episodes with success != true")
    args = ap.parse_args()

    eps = [d for d in sorted(os.listdir(args.dir))
           if d.startswith("episode_") and
           os.path.exists(os.path.join(args.dir, d, "meta.json"))]
    if not eps:
        print(f"No episodes in {args.dir}")
        return

    exported = []
    skipped  = 0
    for d in eps:
        epi_dir = os.path.join(args.dir, d)
        meta = json.load(open(os.path.join(epi_dir, "meta.json")))
        if not args.include_failed and meta.get("success") is not True:
            skipped += 1
            print(f"skip {d} (success={meta.get('success')}) — label it or use --include-failed")
            continue
        info = export_episode(epi_dir, args.out, args.fps)
        if info:
            exported.append(info)
            print(f"✓ {d} → {info['num_frames']} frames @ {args.fps} fps")

    if not exported:
        print(f"\nNothing exported ({skipped} skipped). Label episodes with annotate_episodes.py.")
        return

    dataset = {
        "fps":          args.fps,
        "features": {
            "observation.state": {"shape": [6], "names": [f"j{i}" for i in range(1, 7)]},
            "action":            {"shape": [7], "names": [f"j{i}" for i in range(1, 7)] + ["gripper"]},
            "observation.images.wrist_cam": {"shape": [480, 640, 3], "video": "wrist_cam.mp4"},
            "observation.images.scene_cam": {"shape": [480, 640, 3], "video": "scene_cam.mp4"},
        },
        "episodes":     exported,
        "total_frames": int(sum(e["num_frames"] for e in exported)),
    }
    json.dump(dataset, open(os.path.join(args.out, "dataset_info.json"), "w"), indent=2)
    print(f"\nExported {len(exported)} episodes, {dataset['total_frames']} frames → {args.out}/")
    try:
        import lerobot  # noqa: F401
        print("lerobot detected — wrap these npz + mp4 in a LeRobotDataset for training.")
    except ImportError:
        print("(install `lerobot` to package this into a LeRobotDataset; the npz keys map 1:1.)")


if __name__ == "__main__":
    main()
