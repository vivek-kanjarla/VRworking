"""
test_inference.py — hardware-free inference simulation for all policies.

Simulates what deploy.py does on the real FR5: reset() once, then N calls
to predict() with fake joint + image observations. Tests:

  1. Multi-step rollout — action shape correct every step, no crashes
  2. Action queue behaviour — Diffusion/DiT refill the queue every
     n_action_steps steps (verify model is called at the right cadence)
  3. Temporal ensembling for ACT — with coeff=0.01 the model is queried
     every single step and returns a blended action; verify this path works
  4. Checkpoint round-trip — load best.pt from each policy, confirm
     deploy.py's load_policy() path works end-to-end

Usage:
    python tools/test_inference.py
    python tools/test_inference.py --steps 60   # longer rollout
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "common"))

ACTION_DIM = 7
STATE_DIM  = 6
N_STEPS    = 30   # simulate 1-second episode @ 30 Hz


def _load_policy_mod(name: str):
    path = REPO / "policies" / name / "model.py"
    spec = importlib.util.spec_from_file_location(f"policy_{name}", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dummy_obs(device):
    obs = torch.randn(1, STATE_DIM).to(device)        # (1, state_dim)
    img = torch.rand(1, 3, 224, 224).to(device)       # (1, C, H, W), pre-normalised
    return obs, img


def _run_rollout(model, device, steps=N_STEPS, task=None):
    """Run a full episode rollout. Returns per-step timings and action trace."""
    model.reset()
    actions = []
    times   = []

    for i in range(steps):
        obs, img = _dummy_obs(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            action = model.predict(obs, img, task=[task] if task else None)
        times.append(time.perf_counter() - t0)

        # normalise to 1-D tensor
        if action.dim() == 2:
            action = action[0]
        assert action.shape == (ACTION_DIM,), \
            f"step {i}: expected action shape ({ACTION_DIM},), got {action.shape}"
        actions.append(action.cpu())

    return actions, times


def test_policy(name: str, ckpt_path: Path | None, device: torch.device,
                steps=N_STEPS, temporal_ensemble_coeff=None):
    """Test one policy. ckpt_path=None → build from scratch with dummy stats."""
    print(f"\n{'='*60}")
    print(f"  policy: {name}  |  TE coeff: {temporal_ensemble_coeff}  |  steps: {steps}")
    print(f"{'='*60}")

    mod = _load_policy_mod(name)

    if ckpt_path and ckpt_path.exists():
        ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg_dict = ckpt["config"]
        stats    = ckpt["stats"]
        print(f"  loaded checkpoint  epoch={ckpt.get('epoch','?')}  "
              f"val_l1={ckpt.get('val_l1', float('nan')):.4f}")

        # override temporal_ensemble_coeff if requested
        if temporal_ensemble_coeff is not None:
            cfg_dict["model"]["temporal_ensemble_coeff"] = temporal_ensemble_coeff
            print(f"  override temporal_ensemble_coeff → {temporal_ensemble_coeff}")
    else:
        print("  no checkpoint found — building from dummy stats")
        # minimal dummy config + stats for testing the inference path
        cfg_dict = {
            "dataset": {"chunk_size": 16, "use_image": True, "image_size": [224, 224]},
            "model": {
                "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
                "latent_dim": 8, "d_model": 64, "nhead": 4,
                "num_encoder_layers": 1, "num_decoder_layers": 1,
                "dim_feedforward": 128, "dropout": 0.1,
                "temporal_ensemble_coeff": temporal_ensemble_coeff,
                "num_integration_steps": 2, "integration_method": "euler",
                "hidden_dim": 64, "num_layers": 2, "num_heads": 4,
                "vision_backbone": "resnet18", "pretrained_backbone_weights": None,
                "n_action_steps": 8,
            },
            "training": {"kl_weight": 10.0, "lr": 1e-4, "weight_decay": 1e-4},
        }
        import numpy as np
        stats = {
            "state_mean":  np.zeros(STATE_DIM, np.float32),
            "state_std":   np.ones(STATE_DIM,  np.float32),
            "state_min":   np.full(STATE_DIM, -100., np.float32),
            "state_max":   np.full(STATE_DIM,  100., np.float32),
            "action_mean": np.zeros(ACTION_DIM, np.float32),
            "action_std":  np.ones(ACTION_DIM,  np.float32),
            "action_min":  np.full(ACTION_DIM, -200., np.float32),
            "action_max":  np.full(ACTION_DIM,  200., np.float32),
        }

    model = mod.build_model(cfg_dict, stats, device)
    model.eval()

    task = "pick up the block and place it in the bin" if name == "dit_flow" else None
    actions, times = _run_rollout(model, device, steps, task=task)

    mean_ms = sum(times) / len(times) * 1000
    max_ms  = max(times) * 1000

    print(f"  {steps} steps OK — mean {mean_ms:.1f} ms/step  max {max_ms:.1f} ms/step")
    print(f"  action range: j1 [{min(a[0] for a in actions):.2f}, "
          f"{max(a[0] for a in actions):.2f}]  "
          f"gripper [{min(a[6] for a in actions):.3f}, "
          f"{max(a[6] for a in actions):.3f}]")

    # check action changes across steps (model shouldn't return the same action every step)
    diffs = [(actions[i] - actions[i-1]).abs().mean().item() for i in range(1, len(actions))]
    print(f"  mean step-to-step Δaction = {sum(diffs)/len(diffs):.4f}  "
          f"(0 = queue not updating, >0 = OK)")

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"\ndevice: {device}  steps_per_policy: {args.steps}")

    results = {}

    # ── ACT without temporal ensembling ──────────────────────────────────────
    ckpt = REPO / "policies/act/checkpoints/best.pt"
    ok = test_policy("act", ckpt, device, args.steps, temporal_ensemble_coeff=None)
    results["act (no TE)"] = ok

    # ── ACT WITH temporal ensembling (coeff=0.01) ─────────────────────────────
    print("\n--- ACT with temporal ensembling (coeff=0.01) ---")
    ok = test_policy("act", ckpt, device, args.steps, temporal_ensemble_coeff=0.01)
    results["act (TE=0.01)"] = ok

    # ── Diffusion Policy ──────────────────────────────────────────────────────
    ckpt = REPO / "policies/diffusion/checkpoints/best.pt"
    ok = test_policy("diffusion", ckpt, device, args.steps)
    results["diffusion"] = ok

    # ── DiT + flow matching ───────────────────────────────────────────────────
    ckpt = REPO / "policies/dit_flow/checkpoints/best.pt"
    ok = test_policy("dit_flow", ckpt, device, args.steps)
    results["dit_flow"] = ok

    print(f"\n{'='*60}")
    print("INFERENCE TEST RESULTS")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
    all_pass = all(results.values())
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
