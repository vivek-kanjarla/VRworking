"""
deploy_fr5.py — run a trained policy on the real FR5, HARDWARE-SAFE.

Adapted from common/deploy.py for the Quest3-Fairino teleop stack, with safety
guards so a bad/out-of-distribution prediction cannot jerk the arm:

  * velocity rate-limit  — each joint may move at most MAX_VEL_DEG_S * dt per
                           cycle (derived from the hardware-safe per-cycle caps)
  * joint-limit clamp    — every command clamped inside FR5_JOINT_LIMITS
  * NaN/finite guard      — non-finite predictions are skipped (arm holds)
  * binary gripper        — action[6] >= 0.5 -> CLOSE, else OPEN  (our convention;
                            note gripper_norm was empty, action[6] = gripper_cmd)
  * --dry-run             — read obs + run policy + print commands, NEVER moves
                            the arm (validate the full pipeline safely first)
  * clean shutdown        — stop_servo_mode + disconnect on exit / error / Ctrl-C

Usage:
    # SAFE validation first (no motion):
    .venv/bin/python common/deploy_fr5.py --checkpoint policies/act/checkpoints_fr5/best.pt --dry-run
    # Live (arm moves):
    .venv/bin/python common/deploy_fr5.py --checkpoint policies/act/checkpoints_fr5/best.pt --steps 300
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

REPO_ROOT   = Path(__file__).resolve().parent.parent
TELEOP_ROOT = Path.home() / "Quest3-Fairino"          # our teleop repo (fr5.py, config.py, camera.py)
sys.path.insert(0, str(TELEOP_ROOT))

from config import (                                    # noqa: E402
    FR5_JOINT_LIMITS, MAX_DELTA_PER_JOINT, LOOP_HZ,
    GRIPPER_INDEX, GRIPPER_TYPE,
    GRIPPER_OPEN_PCT, GRIPPER_CLOSE_PCT,
    GRIPPER_VEL_PCT, GRIPPER_FORCE_PCT, GRIPPER_MAXTIME_MS,
)

POLICY_HZ      = 30
POLICY_PERIOD  = 1.0 / POLICY_HZ
GRIPPER_THRESH = 0.5                                     # binary action[6]
# Hardware-safe joint velocity caps (deg/s) = per-125Hz-cycle delta * 125.
MAX_VEL_DEG_S  = [d * LOOP_HZ for d in MAX_DELTA_PER_JOINT]
# Default home = mean episode-start pose (policy starts in-distribution).
HOME_JOINTS    = [-99.2, -116.9, 139.2, -159.2, -82.1, -5.9]

_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_policy_module(policy: str):
    path = REPO_ROOT / "policies" / policy / "model.py"
    if not path.exists():
        raise SystemExit(f"unknown policy '{policy}': {path} not found")
    spec = importlib.util.spec_from_file_location(f"policy_{policy}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_policy(ckpt_path: str, device, temporal_ensemble: float | None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    policy = ckpt.get("policy", "act")
    # Enable temporal ensembling at inference (smoother + reactive vs open-loop
    # 100-step chunks). Training is unaffected; this is inference-only.
    if temporal_ensemble is not None:
        cfg["model"]["temporal_ensemble_coeff"] = temporal_ensemble
    mod = _load_policy_module(policy)
    model = mod.build_model(cfg, ckpt["stats"], device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded {policy}  epoch={ckpt.get('epoch')}  val_l1={ckpt.get('val_l1')}  "
          f"temporal_ensemble={cfg['model'].get('temporal_ensemble_coeff')}")
    return model, cfg


def _cam_serial(name: str) -> str:
    from config import WRIST_CAM_SERIAL, SCENE_CAM_SERIAL
    return {"wrist_cam": WRIST_CAM_SERIAL, "scene_cam": SCENE_CAM_SERIAL}[name]


class LiveCamera:
    """Most-recent-frame view of one RealSense camera (auto-reconnect inherited)."""
    def __init__(self, name="wrist_cam"):
        from camera import RealSenseCamera
        self.name = name
        self._cam = RealSenseCamera(_cam_serial(name), name=name, depth=False)

    def start(self):
        self._cam.start(); self._cam.start_recording()

    def latest_bgr(self):
        with self._cam._frames_lock:
            return self._cam._frames[-1][1].copy() if self._cam._frames else None

    def stop(self):
        try: self._cam.stop_recording()
        except Exception: pass
        self._cam.stop()


def frame_to_tensor(bgr, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return _IMG_TRANSFORM(t).to(device).unsqueeze(0)


def rate_limit(cmd, current, dt):
    """Clamp each joint to a safe per-cycle step, then to joint limits."""
    out = []
    for i in range(6):
        step = MAX_VEL_DEG_S[i] * dt
        v = float(np.clip(cmd[i], current[i] - step, current[i] + step))
        lo, hi = FR5_JOINT_LIMITS[i]
        out.append(float(np.clip(v, lo, hi)))
    return out


def go_home(robot, target, dry, tol=0.6, timeout_s=25.0):
    """Slowly ramp the arm to `target` at the safe velocity caps before the
    policy takes over. Same rate-limit as the policy, so motion is gentle."""
    target = [float(np.clip(t, *FR5_JOINT_LIMITS[i])) for i, t in enumerate(target)]
    print(f"[HOME] moving to {[round(v,1) for v in target]} (gentle ramp) ...")
    if dry:
        cur = robot.get_joint_positions()
        print(f"[HOME] (dry) current={[round(v,1) for v in cur]} "
              f"max_err={max(abs(c-t) for c,t in zip(cur,target)):.1f}deg — would ramp here")
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        t0  = time.monotonic()
        cur = robot.get_joint_positions()
        if max(abs(c - t) for c, t in zip(cur, target)) < tol:
            print("[HOME] reached.")
            return
        try:
            robot.servo_j(rate_limit(target, cur, POLICY_PERIOD))
        except IOError as e:
            print(f"[HOME] ServoJ error: {e}")
        _sleep(t0)
    print("[HOME] timeout — not fully reached; continuing anyway")


class GripperHandler:
    """Binary gripper. action[6] >= 0.5 -> close, else open. Edge-triggered.
    FR5 can't move the gripper during servo, so each toggle briefly drops servo."""
    def __init__(self, robot, dry):
        self._robot = robot; self._dry = dry; self._state = None

    def update(self, a6):
        want = "closed" if a6 >= GRIPPER_THRESH else "open"
        if want == self._state:
            return
        pct = GRIPPER_CLOSE_PCT if want == "closed" else GRIPPER_OPEN_PCT
        if self._dry:
            print(f"[GRIPPER] (dry) -> {want.upper()} pct={pct}"); self._state = want; return
        self._robot.stop_servo_mode(); time.sleep(0.2)
        with self._robot._rpc_lock: self._robot._robot.ResetAllError()
        time.sleep(0.05)
        err = self._robot.send_gripper(GRIPPER_INDEX, pct, GRIPPER_VEL_PCT,
                                       GRIPPER_FORCE_PCT, GRIPPER_MAXTIME_MS, 1, GRIPPER_TYPE)
        print(f"[GRIPPER] {want.upper()}" if err == 0 else f"[GRIPPER] error {err}")
        if err == 0: self._state = want
        time.sleep(0.1)
        with self._robot._rpc_lock: self._robot._robot.RobotEnable(1)
        self._robot.start_servo_mode()


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  | DRY-RUN (no motion)" if args.dry_run else f"device: {device}  | LIVE")
    model, cfg = load_policy(args.checkpoint, device,
                             None if args.no_ensemble else args.temporal_ensemble)
    use_image = cfg["dataset"]["use_image"] and not args.no_image
    task = [args.task] if args.task else None

    cameras = {}   # {name: LiveCamera}
    try:
        if use_image:
            cam_names = cfg["dataset"].get("cameras", ["wrist_cam"])
            for name in cam_names:
                lc = LiveCamera(name); lc.start(); cameras[name] = lc
            print(f"[CAM] opening {cam_names}, waiting for first frames ...")
            ok = False
            for _ in range(60):
                if all(c.latest_bgr() is not None for c in cameras.values()):
                    ok = True; break
                time.sleep(0.1)
            if not ok:
                print("[CAM] not all cameras produced frames — aborting")
                for c in cameras.values(): c.stop()
                return

        from fr5 import FR5Controller
        with FR5Controller() as robot:
            if not args.dry_run:
                robot.start_servo_mode()
                robot.activate_gripper(GRIPPER_INDEX); time.sleep(0.5)
            gripper = GripperHandler(robot, args.dry_run)

            if args.home:
                target = ([float(x) for x in args.home_joints.split(",")]
                          if args.home_joints else HOME_JOINTS)
                if len(target) != 6:
                    raise SystemExit("--home-joints needs 6 comma-separated values")
                go_home(robot, target, args.dry_run)

            print(f"\nrunning {args.steps} steps @ {POLICY_HZ} Hz  "
                  f"(cams={list(cameras) or 'NONE'})  Ctrl-C to stop\n")
            model.reset()
            last_imgs = {}
            t_log = time.monotonic()
            for step in range(args.steps):
                t0 = time.monotonic()

                joints = robot.get_joint_positions()
                obs = torch.tensor(joints, dtype=torch.float32).unsqueeze(0).to(device)
                obs_images = None
                if cameras:
                    for name, c in cameras.items():
                        bgr = c.latest_bgr()
                        if bgr is not None:
                            last_imgs[name] = frame_to_tensor(bgr, device)
                    # use last good frame per camera; only pass cams we have a frame for
                    obs_images = {n: last_imgs[n] for n in cameras if n in last_imgs}

                action = model.predict(obs, obs_images, task=task)
                action = (action[0] if action.dim() == 2 else action).cpu().numpy()

                if not np.all(np.isfinite(action)):
                    print("[SAFE] non-finite action — holding"); _sleep(t0); continue

                raw_j = action[:6].tolist()
                safe_j = rate_limit(raw_j, joints, max(POLICY_PERIOD, time.monotonic() - t0))
                a6 = float(action[6])

                gripper.update(a6)
                if args.dry_run:
                    if time.monotonic() - t_log >= 1.0:
                        clamp = [round(s - r, 1) for s, r in zip(safe_j, raw_j)]
                        print(f"  step {step+1}/{args.steps} raw={[round(v,1) for v in raw_j]} "
                              f"safe={[round(v,1) for v in safe_j]} clamped_by={clamp} grip={a6:.2f}")
                        t_log = time.monotonic()
                else:
                    try:
                        robot.servo_j(safe_j)
                    except IOError as e:
                        print(f"[FR5] ServoJ error: {e}"); _sleep(t0); continue
                    if time.monotonic() - t_log >= 2.0:
                        print(f"  step {step+1}/{args.steps} j={[round(v,1) for v in safe_j]} grip={a6:.2f}")
                        t_log = time.monotonic()
                _sleep(t0)
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        for c in cameras.values(): c.stop()
    print("done")


def _sleep(t0):
    dt = time.monotonic() - t0
    if dt < POLICY_PERIOD: time.sleep(POLICY_PERIOD - dt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--dry-run", action="store_true", help="no motion; print commands (validate first!)")
    p.add_argument("--no-image", action="store_true")
    p.add_argument("--temporal-ensemble", type=float, default=0.01,
                   help="ACT temporal-ensemble coeff at inference (smoother). Default 0.01")
    p.add_argument("--no-ensemble", action="store_true", help="disable temporal ensembling")
    p.add_argument("--home", action="store_true",
                   help="gently ramp to the home pose before the policy starts (in-distribution start)")
    p.add_argument("--home-joints", default="",
                   help="override home as 6 comma-separated deg, e.g. '-99.2,-116.9,139.2,-159.2,-82.1,-5.9'")
    p.add_argument("--task", default="pick up the object and place it in the bin")
    run(p.parse_args())


if __name__ == "__main__":
    main()
