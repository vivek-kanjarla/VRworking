# ACT on the Fairino FR5 — Full Working Progress

End-to-end log of building a VR-teleoperated FR5 cobot dataset, training an ACT
(Action Chunking Transformer) policy, deploying it on the real arm, and the
debugging that explains the current results. Reflects everything done to date.

---

## 1. Goal & high-level pipeline

```
Meta Quest 3 VR teleop ──> record demos ──> LeRobot dataset ──> train ACT ──> deploy on FR5
```

Task: **"pick up the object and place it in the bin"** (single-arm, 6-DOF + gripper).

Two repos:
- **`quest3-fairino-teleop/`** (orig `Quest3-Fairino`) — teleoperation + data recording.
- **`fr5-act-policies/`** (orig `fairino-fr5-policies`) — ACT training + deployment.
Both consolidated into **`vivek-kanjarla/VRworking`** (`main`).

---

## 2. Hardware & environment

| Component | Detail |
|---|---|
| Robot | Fairino FR5, Ethernet `192.168.58.2` (CNDE), controlled via `ServoJ` @ 125 Hz |
| VR | Meta Quest 3 (WebXR / Meta Browser, `https://<wifi-ip>:8012`) |
| Wrist cam | Intel RealSense **D405** (eye-in-hand, on the arm), 640×480 @ 30 fps |
| Scene cam | Intel RealSense **D435i** (eye-to-hand, fixed) + aligned depth |
| GPU | NVIDIA **RTX 5070 Ti** 16 GB (Blackwell, sm_120) |

**Env notes (gotchas solved):**
- `lerobot==0.5.1` needs **Python 3.12**; system only had 3.10. Installed **`uv`** (from PyPI, no sudo) → `uv python install 3.12` → venv. Torch **2.10.0+cu128** (Blackwell works).
- **Fairino SDK** has no 3.12 `.so`, but `Robot.py` is pure-Python → dropped into the venv as a `fairino` package (works in 3.12).
- **pyrealsense2** installed into the venv (3.12 wheel) for deploy.

---

## 3. Teleoperation & control tuning (quest3-fairino-teleop)

Data flow: `VR pose → One-Euro filter → SE(3) retarget → DLS IK (Pinocchio) → ServoJ`.

Tuning done live on the arm (all committed):
- **Axis calibration:** `VR_FRAME_ROTATION` — flipped **forward/back** and **left/right** to match the real arm (det handled; SE(3) conjugation keeps orientation valid).
- **Gripper:** **squeeze = toggle** open/close (was hold-to-close), **independent of the trigger** (evaluated before the homing gate).
- **Wrist J4/J5:** hand-orientation coupling kept; **J4 sign flipped**, both gains halved (±0.5) to cut incidental twist.
- **J6 roll:** moved to the **LEFT controller X(CW)/Y(CCW)** buttons (plumbed through the vuer transport, shared array 12→16), trigger-independent, clamped to the per-cycle rate (fixes a runaway that hit the validator's 0.8°/cycle cap), **0.20°/cycle = 25 °/s**.
- **One-Euro filter retune:** min cutoff 1.0→**2.5 Hz** (lag 159→64 ms), pos beta 10→**6** (smoother, less jerk).
- **Rate limits:** switched to **hardware-safe** `MAX_DELTA_PER_JOINT` (≈ half the sim caps) so tracking-glitch corrections ramp gently instead of lurching.

**Known issue (hardware):** the **D405 wrist cable flexes** on the moving arm → intermittent USB drops/timeouts. The D405 itself is fine (streams perfectly when static + alongside the D435i). Needs strain relief.

---

## 4. Recording pipeline (dual-camera)

Each episode → **`episodes/episode_NNN/`**:
- `data.csv` — ~125 Hz rows: `fr5_actual_j1..6` (state), `fr5_cmd_j1..6` + `gripper_cmd` (action), `fr5_eef_*`, `fr5_vel_*`, `vr_*`, plus `index/episode_index/frame_index/task_index/next.done`.
- `meta.json` — instruction, duration, success (null until labeled), camera intrinsics, quality metrics (grasps/regrasps/pauses/smoothness).
- `wrist_cam.mp4`, `scene_cam.mp4` (640×480 @ 30), `wrist_cam_ts.npy`, `scene_cam_ts.npy`, `scene_cam_depth.npz` (aligned uint16).

Controls / robustness (all implemented):
- **A** = start/stop episode, **B** = mark success.
- **Camera auto-reconnect** — a USB drop/timeout self-heals instead of silently zeroing frames.
- **0-frame ⚠️ warning** on save if a camera dropped.
- **Log-while-not-homed** — captures the final gripper open/close after the trigger is released.
- **Auto-home before each episode** (latest) — pressing A ramps the arm to a fixed home pose, opens the gripper, *then* starts logging → **every demo's frame 0 is identical** (see §8 fix).

**Action[6] = binary gripper command** (1 = squeeze/close, 0 = open). `gripper_norm` (observed jaw position) is empty on this gripper config — only the command is recorded (fine for ACT; it's the action).

**Dataset state:** **61 clean episodes** (`episode_000..060`), all with wrist + scene + depth. Earlier wrist-cam-less captures (from D405 USB drops) were deleted and the set renumbered contiguously.

---

## 5. ACT pipeline (fr5-act-policies)

- **`common/convert_episodes.py`** → LeRobot dataset; resamples the 125 Hz CSV to the 30 Hz camera timeline; extracts JPEGs for both cameras. Patched for our format: `GRIPPER_COL = "gripper_cmd"`, reads `instruction` from meta, **dual-camera** export.
- **`common/dataset.py`** — multi-camera: loads `observation.images.{cam}` per sample.
- **`policies/act/model.py`** — `cameras` list (one VISUAL feature per camera), **configurable `vision_backbone`** (resnet18/34/50, ImageNet weights). Wraps lerobot 0.5.1 `ACTPolicy` (CVAE; shared ResNet backbone across cameras).
- **`common/train.py`** — passes cameras, feeds an image dict, **bf16 autocast**, configurable `num_workers`, `best.pt` saved on val improvement (= early stopping).

ACT feature mapping:
| ACT key | Dim | Source |
|---|---|---|
| `observation.state` | 6 | FR5 actual joints |
| `action` | 7 | FR5 cmd joints + binary gripper |
| `observation.images.wrist_cam / scene_cam` | video | D405 / D435i |
| `observation.eef_pose` | 6 | extra |

---

## 6. Training runs

| Run | Cameras | Backbone | Decoder / d_model / ff | Params | Best val L1 | Notes |
|---|---|---|---|---|---|---|
| 1 | wrist | resnet18 | 1 / 256 / 1024 | 18.7M | **0.109** | 41 eps; ~70 s/epoch |
| 2 | wrist+scene | **resnet34** | **7 / 512 / 3200** (paper-scale) | **94.0M** | **0.111** (epoch 16) | 61 eps; bf16, batch 96, GPU ~67%, ~220 s/epoch; val plateaued by ~epoch 16, stopped at 34 |

Anti-overfit: dropout 0.2, random-crop+jitter aug, weight decay 1e-4, val-based `best.pt`. Train/val gap (e.g. 0.046/0.116) confirmed we are **data-limited, not capacity-limited** — the bigger model didn't lower val.

---

## 7. Deployment (`fr5-act-policies/common/deploy_fr5.py`)

Hardware-safe deploy (NEW, custom for our stack):
- Loads checkpoint, reads **FR5 joints + both cameras**, ACT → action @ 30 Hz.
- **Velocity rate-limit** (per-joint deg/s × dt) + **joint-limit clamp** + **NaN guard** so a bad prediction can't jerk the arm.
- **Binary gripper** (≥0.5 close / else open, correct convention).
- **`--home`** (gentle ramp to mean-start pose first), **`--dry-run`** (no motion, prints commands), **temporal ensembling** (m=0.01) for smoothness.
- Dual-camera inference (opens the cameras from the checkpoint's `cameras` list).
- Clean shutdown (servo stop + disconnect).

---

## 8. Live results & the core finding

**Single-cam model (run 1):** executes the full pick→grasp→carry→release **motion** reliably, but only grasps when the object is placed **exactly at its one learned grasp spot** — positional, no generalization.

**Paper-scale dual-cam (run 2):**
- `best.pt` (epoch 16, best val) — **over-reached, never committed the grasp**.
- **`epoch_0030.pt` (last epoch)** — **committed the grasp** (sustained gripper close), clean full pick-place **motion**, **3/3 repeatable**. ⇒ **val L1 ≠ task success**; the last-epoch checkpoint beats best-val for the sparse gripper signal.
- **BUT 0/3 actual physical picks** — it grasped at a fixed spot and **missed the object** (same positional failure).

### Why the 2-camera setup didn't improve picking (diagnosed)
| Check | Result |
|---|---|
| Scene↔wrist↔action time alignment | **Fine** (≤1 frame; offset ~0 ms, drift ~13 ms) — not a plumbing bug |
| Object-position variation in demos | **Present** (~340 mm spread in grasp x/y) — the signal to learn *was* there |
| **Input-sensitivity test** (fix state, vary only images) | **Predicted reach uncorrelated with object position: corr ≈ 0.0**, pred std (1–4°) ≪ true std (7.5°) → **the model ignores BOTH cameras** (mean-collapse) |
| Start-pose spread / shortcut | Start poses varied (J2 std 17°, J4 std 21°) and **partially telegraphed the grasp** (corr start-J1↔grasp-J1 = **0.45**) → **proprioceptive shortcut** |

**Root cause:** with only ~54 training episodes and start poses that partially predicted the grasp, the policy took the **easy proprioceptive shortcut** — conditioning on joint state and **ignoring the cameras**. At deploy we always start from a fixed home, so the constant proprioception → a **constant mean reach** → grabs the same empty spot. A second camera can't help a model that ignores cameras.

---

## 9. The fix (in progress)

1. **Auto-home before each episode** (✅ implemented) → constant start state carries **zero** object info → the policy is **forced** to use the cameras.
2. **Vary the object position widely** every episode; **grasp cleanly/decisively** (sharp gripper signal).
3. **~100–150 total demos** (have 61; add ~60–90).
4. Retrain (dual-cam + ResNet34 + paper-scale is ready) and **re-run the input-sensitivity test** — predicted reach should then correlate with object position.
5. *(optional, training-side)* proprioception dropout to further discourage the shortcut.

---

## 10. Artifacts & commands

**Checkpoints** (gitignored — local + Desktop copies):
- `policies/act/checkpoints_fr5/best.pt` — single-cam (val 0.109).
- `policies/act/checkpoints_fr5_paperscale/{best.pt, epoch_0030.pt}` — dual-cam paper-scale. **Deploy `epoch_0030.pt`.**

**Commands:**
```bash
# convert (dual-cam)
.venv/bin/python common/convert_episodes.py --episodes ~/Quest3-Fairino/episodes --out lerobot_dataset --extract-frames
# train
.venv/bin/python common/train.py --policy act --config policies/act/config.fr5.yaml
# deploy: ALWAYS dry-run first, then live with --home
.venv/bin/python common/deploy_fr5.py --checkpoint policies/act/checkpoints_fr5_paperscale/epoch_0030.pt --home --dry-run
.venv/bin/python common/deploy_fr5.py --checkpoint policies/act/checkpoints_fr5_paperscale/epoch_0030.pt --home --steps 900
```

**Key takeaways for future-me:**
- Deploy the **last-epoch** checkpoint, not best-val (gripper is a sparse signal).
- The blocker is **data**, specifically **demo diversity with a fixed start pose** — not model size or the number of cameras.
- The whole pipeline (recording, dual-cam, paper-scale, deploy, safety) is built and validated; only the **data-collection protocol** changed (fixed home + varied object).
