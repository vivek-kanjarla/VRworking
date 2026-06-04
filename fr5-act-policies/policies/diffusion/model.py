"""
policies/diffusion/model.py — Diffusion Policy (DDPM/DDIM) for the FR5.

Wraps lerobot's DiffusionPolicy (lerobot==0.5.1).

Architecture overview
─────────────────────
Observation encoder
  • ResNet18 (ImageNet pretrained) image encoder  — shared with ACT
  • Linear projection                             — joint state (6 DOF)
  → global conditioning vector  c ∈ ℝᵈ (state_dim + img_features) × n_obs_steps

Action denoiser — 1-D Conditional U-Net
  • Processes the noisy action chunk as a 1-D temporal signal
  • Conditioning via FiLM layers from c
  • Predicts noise ε (epsilon-prediction) or the clean action (x-prediction)

Training objective — DDPM score matching
  • Add noise at random timestep t: a_t = √ᾱ_t · a + √(1-ᾱ_t) · ε
  • Predict ε (or a) with the U-Net
  • Loss: ‖ε_θ(a_t, t, c) − ε‖²

Inference — DDIM denoising (fast, configurable steps)
  • num_inference_steps DDIM steps from noise → clean action chunk
  • n_action_steps executed per call (remaining discarded)

Normalization (done in wrapper; lerobot policy sees IDENTITY)
  • State / action: mean-std  (same as ACT)
  • Images: ImageNet normalization already applied by dataset.py ✓
    (ResNet18 expects ImageNet stats — no renormalization needed here)
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig as _LRConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode


STATE_KEY = "observation.state"
IMAGE_KEY = "observation.images.wrist_cam"
ACTION_KEY = "action"


@dataclass
class DPConfig:
    state_dim:  int  = 6
    action_dim: int  = 7
    chunk_size: int  = 16    # horizon — U-Net predicts this many action steps
    use_image:  bool = True
    n_action_steps: int = 8  # actions actually executed per chunk (≤ chunk_size)

    # U-Net architecture
    down_dims:                tuple = field(default_factory=lambda: (256, 512, 1024))
    kernel_size:              int   = 5
    n_groups:                 int   = 8
    diffusion_step_embed_dim: int   = 128
    use_film_scale_modulation: bool = True

    # DDPM noise schedule (training)
    num_train_timesteps: int = 100
    beta_schedule:       str = "squaredcos_cap_v2"
    prediction_type:     str = "epsilon"   # "epsilon" | "sample"

    # DDIM inference (fast)
    num_inference_steps: int = 10   # DDIM steps at inference (<<100)
    noise_scheduler_type: str = "DDPM"  # train with DDPM, infer with DDIM

    # Vision backbone
    vision_backbone:             str        = "resnet18"
    pretrained_backbone_weights: str | None = None   # DP uses GroupNorm → incompatible with pretrained BN


def _lerobot_config(cfg: DPConfig) -> _LRConfig:
    input_features = {
        STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(cfg.state_dim,)),
    }
    norm_map = {
        "STATE":  NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    if cfg.use_image:
        input_features[IMAGE_KEY] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
        norm_map["VISUAL"] = NormalizationMode.IDENTITY

    return _LRConfig(
        n_obs_steps=1,
        horizon=cfg.chunk_size,
        n_action_steps=cfg.n_action_steps,
        input_features=input_features,
        output_features={ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(cfg.action_dim,))},
        normalization_mapping=norm_map,
        down_dims=tuple(cfg.down_dims),
        kernel_size=cfg.kernel_size,
        n_groups=cfg.n_groups,
        diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
        use_film_scale_modulation=cfg.use_film_scale_modulation,
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
        num_inference_steps=cfg.num_inference_steps,
        noise_scheduler_type=cfg.noise_scheduler_type,
        vision_backbone=cfg.vision_backbone,
        pretrained_backbone_weights=cfg.pretrained_backbone_weights,
        crop_shape=None,
        do_mask_loss_for_padding=False,
    )


class DiffusionPol(nn.Module):
    def __init__(self, cfg: DPConfig, stats: dict):
        super().__init__()
        self.cfg = cfg
        self.policy = DiffusionPolicy(_lerobot_config(cfg))

        # mean-std normalization buffers
        self.register_buffer("state_mean",  torch.as_tensor(stats["state_mean"]).float())
        self.register_buffer("state_std",   torch.as_tensor(stats["state_std"]).float())
        self.register_buffer("action_mean", torch.as_tensor(stats["action_mean"]).float())
        self.register_buffer("action_std",  torch.as_tensor(stats["action_std"]).float())

    # ── normalization ──────────────────────────────────────────────────────────

    def _norm_state(self, s):    return (s - self.state_mean) / self.state_std
    def _norm_action(self, a):   return (a - self.action_mean) / self.action_std
    def _unnorm_action(self, a): return a * self.action_std + self.action_mean

    # ── batch builder ──────────────────────────────────────────────────────────

    def _make_batch(self, obs_state, actions=None, action_is_pad=None,
                    obs_image=None, for_training=True):
        """
        for_training=True  (→ policy.forward / compute_loss)
          State: (B, 1, state_dim)  — compute_loss asserts shape[1] == n_obs_steps
          Image: (B, C, H, W)       — forward auto-unsqueezes for n_obs_steps==1

        for_training=False  (→ policy.select_action via queue)
          State: (B, state_dim)     — queue stacks to (B, 1, state_dim)
          Image: (B, C, H, W)       — same; queue adds the n_obs_steps dim
        """
        state = self._norm_state(obs_state)
        if for_training:
            state = state.unsqueeze(1)      # (B, 1, state_dim)
        batch = {STATE_KEY: state}

        if self.cfg.use_image and obs_image is not None:
            # Images arrive ImageNet-normalised from dataset.py — ResNet18 is happy as-is.
            batch[IMAGE_KEY] = obs_image    # (B, C, H, W)

        if actions is not None:
            batch[ACTION_KEY]      = self._norm_action(actions)
            batch["action_is_pad"] = action_is_pad

        return batch

    # ── public interface ───────────────────────────────────────────────────────

    def forward(self, obs_state, actions, action_is_pad, obs_image=None, task=None):
        """Returns (loss, loss_item, 0.0) — DDPM MSE loss, no KL term."""
        batch = self._make_batch(obs_state, actions, action_is_pad, obs_image,
                                 for_training=True)
        loss, _ = self.policy.forward(batch)
        return loss, loss.item(), 0.0

    def reset(self):
        """Call once at the start of each episode before calling predict()."""
        self.policy.reset()

    @torch.no_grad()
    def predict(self, obs_state, obs_image=None, task=None):
        """DDIM denoising → one action from the chunk, in original joint-space units."""
        action_norm = self.policy.select_action(
            self._make_batch(obs_state, obs_image=obs_image, for_training=False)
        )
        return self._unnorm_action(action_norm)


# ── policy entry point ─────────────────────────────────────────────────────────

def build_model(cfg: dict, stats: dict, device) -> DiffusionPol:
    m, d = cfg["model"], cfg["dataset"]
    model_cfg = DPConfig(
        state_dim=m["state_dim"],
        action_dim=m["action_dim"],
        chunk_size=d["chunk_size"],
        use_image=d["use_image"],
        n_action_steps=m.get("n_action_steps", d["chunk_size"] // 2),
        down_dims=tuple(m.get("down_dims", [256, 512, 1024])),
        kernel_size=m.get("kernel_size", 5),
        n_groups=m.get("n_groups", 8),
        diffusion_step_embed_dim=m.get("diffusion_step_embed_dim", 128),
        use_film_scale_modulation=m.get("use_film_scale_modulation", True),
        num_train_timesteps=m.get("num_train_timesteps", 100),
        beta_schedule=m.get("beta_schedule", "squaredcos_cap_v2"),
        prediction_type=m.get("prediction_type", "epsilon"),
        num_inference_steps=m.get("num_inference_steps", 10),
        noise_scheduler_type=m.get("noise_scheduler_type", "DDPM"),
        vision_backbone=m.get("vision_backbone", "resnet18"),
        pretrained_backbone_weights=m.get("pretrained_backbone_weights",
                                          "ResNet18_Weights.IMAGENET1K_V1"),
    )
    return DiffusionPol(model_cfg, stats).to(device)
