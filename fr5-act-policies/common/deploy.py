"""
deploy.py — run a trained policy on the FR5 robot.

The checkpoint records which policy produced it, so deploy auto-loads the
matching policies/<policy>/model.py — no per-model deploy script needed.

Usage:
    python common/deploy.py --checkpoint policies/act/checkpoints/best.pt
    python common/deploy.py --checkpoint policies/act/checkpoints/best.pt --steps 150 --no-image
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
TELEOP_ROOT = REPO_ROOT.parent / "so101-fr5-teleop"
sys.path.insert(0, str(TELEOP_ROOT))

from fr5 import FR5Controller
from config import (
    GRIPPER_INDEX, GRIPPER_TYPE,
    GRIPPER_OPEN_PCT, GRIPPER_CLOSE_PCT,
    GRIPPER_VEL_PCT, GRIPPER_FORCE_PCT, GRIPPER_MAXTIME_MS,
)


def _load_policy_module(policy: str):
    path = REPO_ROOT / "policies" / policy / "model.py"
    if not path.exists():
        raise SystemExit(f"unknown policy '{policy}': {path} not found")
    spec = importlib.util.spec_from_file_location(f"policy_{policy}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


POLICY_HZ           = 30
POLICY_PERIOD       = 1.0 / POLICY_HZ
GRIPPER_OPEN_THRESH  = 0.65
GRIPPER_CLOSE_THRESH = 0.35

_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── model ─────────────────────────────────────────────────────────────────────

def load_policy(ckpt_path: str, device: torch.device):
    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    policy   = ckpt.get("policy", "act")

    policy_mod = _load_policy_module(policy)
    model = policy_mod.build_model(cfg_dict, ckpt["stats"], device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"loaded {policy} checkpoint  epoch={ckpt['epoch']}  val_l1={ckpt['val_l1']:.4f}")
    return model, cfg_dict


# ── camera ────────────────────────────────────────────────────────────────────

class LiveCamera:
    """Wraps D405Camera to always expose the most recent frame."""

    def __init__(self):
        from camera import D405Camera
        self._cam = D405Camera()

    def start(self):
        self._cam.start()
        self._cam.start_recording()

    def latest_bgr(self) -> np.ndarray | None:
        with self._cam._frames_lock:
            if not self._cam._frames:
                return None
            return self._cam._frames[-1][1].copy()

    def stop(self):
        self._cam.stop_recording()
        self._cam.stop()


def frame_to_tensor(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return _IMG_TRANSFORM(t).to(device)


# ── gripper ───────────────────────────────────────────────────────────────────

class GripperHandler:
    def __init__(self, robot: FR5Controller):
        self._robot = robot
        self._state = None   # "open" | "closed" | None

    def update(self, gripper_norm: float):
        if gripper_norm >= GRIPPER_OPEN_THRESH and self._state != "open":
            self._send("open", GRIPPER_OPEN_PCT)
        elif gripper_norm <= GRIPPER_CLOSE_THRESH and self._state != "closed":
            self._send("closed", GRIPPER_CLOSE_PCT)

    def _send(self, label: str, pct: int):
        self._robot.stop_servo_mode()
        time.sleep(0.2)

        with self._robot._rpc_lock:
            self._robot._robot.ResetAllError()
        time.sleep(0.05)

        err = self._robot.send_gripper(
            GRIPPER_INDEX, pct,
            GRIPPER_VEL_PCT, GRIPPER_FORCE_PCT,
            GRIPPER_MAXTIME_MS, 1, GRIPPER_TYPE,
        )
        if err == 0:
            self._state = label
            print(f"[GRIPPER] {label.upper()}")
        else:
            print(f"[GRIPPER] MoveGripper error {err}")

        time.sleep(0.1)
        with self._robot._rpc_lock:
            self._robot._robot.RobotEnable(1)
        self._robot.start_servo_mode()


# ── main loop ─────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    model, cfg_dict = load_policy(args.checkpoint, device)
    use_image = cfg_dict["dataset"]["use_image"] and not args.no_image

    # Language-conditioned policies (dit_flow, pi0, pi05, pi0_fast) need the task
    # instruction at inference, matching what they were trained on. ACT / Diffusion
    # ignore it. Passed as a batch of one string to model.predict().
    task = [args.task] if args.task else None
    if task:
        print(f"task: {args.task!r}")

    camera  = None
    gripper = None

    try:
        if use_image:
            camera = LiveCamera()
            camera.start()
            print("[CAM] camera ready — waiting for first frame ...")
            for _ in range(30):
                if camera.latest_bgr() is not None:
                    break
                time.sleep(0.1)
            else:
                print("[CAM] no frames received — continuing without image")
                camera.stop()
                camera = None

        with FR5Controller() as robot:
            robot.start_servo_mode()
            gripper = GripperHandler(robot)

            # Activate gripper
            robot.activate_gripper(GRIPPER_INDEX)
            time.sleep(0.5)

            print(f"\nrunning policy for {args.steps} steps at {POLICY_HZ} Hz  "
                  f"({'with' if camera else 'without'} image)\n"
                  "  Ctrl+C to stop early\n")

            model.reset()
            step       = 0
            t_last_log = time.monotonic()
            last_img   = None   # holds last valid frame to avoid None on drops

            while step < args.steps:
                t0 = time.monotonic()

                # ── observation ──────────────────────────────────────────────
                joints = robot.get_joint_positions()                # list[float] × 6
                obs    = torch.tensor(joints, dtype=torch.float32).unsqueeze(0).to(device)

                img = last_img
                if camera is not None:
                    bgr = camera.latest_bgr()
                    if bgr is not None:
                        img = frame_to_tensor(bgr, device).unsqueeze(0)
                        last_img = img

                # ── policy ───────────────────────────────────────────────────
                action = model.predict(obs, img, task=task)         # (action_dim,) or (1, action_dim)
                if action.dim() == 2:
                    action = action[0]
                action = action.cpu().numpy()

                joints_cmd  = action[:6].tolist()
                gripper_cmd = float(action[6])

                # ── execute ───────────────────────────────────────────────────
                gripper.update(gripper_cmd)   # no-op unless threshold crossed
                robot.servo_j(joints_cmd)

                step += 1

                # heartbeat every 2 s
                now = time.monotonic()
                if now - t_last_log >= 2.0:
                    print(f"  step {step}/{args.steps}  "
                          f"joints={[f'{j:.1f}' for j in joints_cmd]}  "
                          f"gripper={gripper_cmd:.2f}")
                    t_last_log = now

                # ── rate limit ────────────────────────────────────────────────
                elapsed = time.monotonic() - t0
                if elapsed < POLICY_PERIOD:
                    time.sleep(POLICY_PERIOD - elapsed)

    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        if camera is not None:
            camera.stop()

    print("done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to best.pt or epoch_XXXX.pt")
    parser.add_argument("--steps",      type=int, default=150,
                        help="number of policy steps to run (default 150 = 5s at 30 Hz)")
    parser.add_argument("--no-image",   action="store_true",
                        help="ignore camera even if checkpoint was trained with images")
    parser.add_argument("--task", default="pick up the block and place it in the bin",
                        help="language instruction for language-conditioned policies "
                             "(dit_flow / pi0 / pi05 / pi0_fast); ignored by ACT / Diffusion")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
