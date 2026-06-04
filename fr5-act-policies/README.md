# fairino-fr5-policies

Training and deploying imitation learning policies on the Fairino FR5 cobot. Data comes from teleoperation via an SO-101 leader arm, and the goal is to get the FR5 to autonomously replicate tasks like pick-and-place at 30 Hz.

This repo is set up as a centralised policy hub — one shared training/data pipeline in `common/`, and each policy type gets its own folder under `policies/`. So you can swap between ACT, Diffusion Policy, DiT etc. without touching any shared code.

---

## what's in here

| policy | folder | what it is |
|---|---|---|
| ACT | `policies/act/` | Action Chunking Transformer — CVAE + transformer, predicts 100-step chunks |
| Diffusion Policy | `policies/diffusion/` | DDPM with a 1-D U-Net denoiser, 10-step DDIM at inference |
| DiT + Flow Matching | `policies/dit_flow/` | Diffusion Transformer with flow matching objective, CLIP vision + language |

all three are wrapped around lerobot 0.5.1 and trained on the same FR5 episodes.

---

## setup

lerobot pins torch/torchvision below the newest releases so you need a venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

run everything from the repo root.

---

## getting data ready

raw recordings live in `episodes/episode_*/` — a ~100 Hz CSV log, a ~30 Hz wrist cam video, and a meta.json per episode. the converter resamples the CSV down to the camera rate (one row per frame) and outputs a LeRobot v3.0 dataset:

```bash
python common/convert_episodes.py --episodes episodes --out lerobot_dataset --extract-frames
```

`--extract-frames` pre-extracts JPEGs so training I/O is fast instead of seeking the video every step.

then update `dataset.root` in whatever config you're using to point at the output folder.

---

## training

```bash
# ACT
python common/train.py --policy act --config policies/act/config.local.yaml

# Diffusion Policy
python common/train.py --policy diffusion --config policies/diffusion/config.local.yaml

# DiT + flow matching
python common/train.py --policy dit_flow --config policies/dit_flow/config.local.yaml
```

checkpoints go to `policies/<policy>/checkpoints/`. `best.pt` updates whenever val L1 improves. each checkpoint stores which policy it came from, so deploy just reads that and loads the right model automatically.

there's also a quick smoke test that runs convert → dataset → train step → checkpoint → predict in a few seconds:

```bash
python common/smoke_test.py act
python common/smoke_test.py diffusion
python common/smoke_test.py dit_flow
```

---

## deploying on the robot

```bash
python common/deploy.py --checkpoint policies/act/checkpoints/best.pt
```

runs at 30 Hz for 150 steps by default. use `--steps N` to change that. if the model was trained with images but the camera isn't connected, `--no-image` falls back to state-only.

---

## adding a new policy

drop a folder under `policies/<name>/` with:
- `model.py` — implements `build_model(cfg, stats, device)` returning a model with `forward(obs, actions, pad, image, task) -> (loss, l1, kl)`, `reset()`, and `predict(obs, image, task)`
- `config.yaml`, `config.local.yaml`, `config.smoke.yaml`

that's it. `common/train.py --policy <name>` picks it up automatically.

---

## repo layout

```
common/
  convert_episodes.py    raw episodes → LeRobot dataset
  dataset.py             parquet + video → PyTorch DataLoader
  train.py               shared training loop (--policy <name>)
  deploy.py              load any checkpoint, run on the FR5
  smoke_test.py          quick end-to-end sanity check
policies/
  act/                   ACT wrapper + configs
  diffusion/             Diffusion Policy wrapper + configs
  dit_flow/              DiT + flow matching wrapper + configs
eda/
  eda.ipynb              explore the dataset before training
```

---

## hardware

- **Robot**: Fairino FR5 (6-DOF cobot)
- **Leader arm**: SO-101 for teleoperation
- **Camera**: Intel RealSense D405 (wrist-mounted), 640×480 → resized to 224×224
- **Control rate**: 30 Hz
- **Action space**: 6 joint angles (deg) + gripper (0–1 normalised)
- **State space**: 6 actual joint positions (deg)
