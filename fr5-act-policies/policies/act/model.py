"""
Wraps lerobot's ACTPolicy so train.py / deploy.py stay simple.

Tested against lerobot == 0.5.1.

Notes on the 0.5.x API (differs a lot from 0.1.x):
  * policies moved to `lerobot.policies.act`
  * config takes `input_features` / `output_features` (PolicyFeature) and a
    `normalization_mapping`, not the old input_shapes/normalization dicts
  * `n_obs_steps == 1`, so state is fed as (B, state_dim) and images as
    (B, C, H, W) — no extra time axis
  * normalization moved OUT of the policy into separate processor pipelines.
    We don't use those pipelines (they're built around lerobot's own training
    loop); instead we normalise state/action ourselves with the dataset stats
    and tell lerobot to treat every feature as IDENTITY. Images arrive already
    ImageNet-normalised from dataset.py.
  * `policy.forward(batch)` returns (loss, {"l1_loss", "kld_loss"}) where loss
    already includes kl_weight, and the dict values are plain floats.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from lerobot.policies.act.configuration_act import ACTConfig as _LRConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode


STATE_KEY = "observation.state"
ENV_KEY = "observation.environment_state"
IMAGE_KEY = "observation.images.wrist_cam"
ACTION_KEY = "action"

# ImageNet-pretrained weights per supported ResNet backbone (lerobot ACT works
# with torchvision ResNet-family backbones; ViT/DINO would need code changes).
_BACKBONE_WEIGHTS = {
    "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
    "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
    "resnet50": "ResNet50_Weights.IMAGENET1K_V1",
}


@dataclass
class ACTConfig:
    state_dim:          int   = 6
    action_dim:         int   = 7
    latent_dim:         int   = 32
    d_model:            int   = 512
    nhead:              int   = 8
    num_encoder_layers: int   = 4
    num_decoder_layers: int   = 7
    dim_feedforward:    int   = 3200
    dropout:            float = 0.1
    chunk_size:              int          = 100
    use_image:               bool         = True
    kl_weight:               float        = 10.0
    temporal_ensemble_coeff: float | None = 0.01
    cameras:                 list         = field(default_factory=lambda: ["wrist_cam"])
    vision_backbone:         str          = "resnet18"


def _lerobot_config(cfg: ACTConfig) -> _LRConfig:
    input_features = {
        STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(cfg.state_dim,)),
    }
    norm_map = {
        FeatureType.STATE:  NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.IDENTITY,
    }

    if cfg.use_image:
        for cam in cfg.cameras:
            input_features[f"observation.images.{cam}"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224))
        norm_map[FeatureType.VISUAL] = NormalizationMode.IDENTITY
    else:
        # lerobot's ACT requires at least one image or an env-state input; with no
        # camera we feed the joint state in as the environment-state feature too.
        input_features[ENV_KEY] = PolicyFeature(type=FeatureType.ENV, shape=(cfg.state_dim,))
        norm_map[FeatureType.ENV] = NormalizationMode.IDENTITY

    output_features = {ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(cfg.action_dim,))}

    # temporal ensembling requires querying the policy every step
    n_action_steps = 1 if cfg.temporal_ensemble_coeff is not None else cfg.chunk_size

    return _LRConfig(
        chunk_size=cfg.chunk_size,
        n_action_steps=n_action_steps,
        n_obs_steps=1,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping=norm_map,
        dim_model=cfg.d_model,
        n_heads=cfg.nhead,
        dim_feedforward=cfg.dim_feedforward,
        n_encoder_layers=cfg.num_encoder_layers,
        n_decoder_layers=cfg.num_decoder_layers,
        n_vae_encoder_layers=cfg.num_encoder_layers,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
        kl_weight=cfg.kl_weight,
        use_vae=True,
        temporal_ensemble_coeff=cfg.temporal_ensemble_coeff,
        vision_backbone=cfg.vision_backbone,
        pretrained_backbone_weights=(_BACKBONE_WEIGHTS.get(cfg.vision_backbone)
                                     if cfg.use_image else None),
    )


class ACT(nn.Module):
    def __init__(self, cfg: ACTConfig, stats: dict):
        super().__init__()
        self.cfg = cfg
        self.policy = ACTPolicy(_lerobot_config(cfg))

        # mean/std normalisation buffers (saved in state_dict, restored on load)
        self.register_buffer("state_mean",  torch.as_tensor(stats["state_mean"]).float())
        self.register_buffer("state_std",   torch.as_tensor(stats["state_std"]).float())
        self.register_buffer("action_mean", torch.as_tensor(stats["action_mean"]).float())
        self.register_buffer("action_std",  torch.as_tensor(stats["action_std"]).float())

    # ── normalisation helpers ──────────────────────────────────────────────
    def _norm_state(self, s):    return (s - self.state_mean) / self.state_std
    def _norm_action(self, a):   return (a - self.action_mean) / self.action_std
    def _unnorm_action(self, a): return a * self.action_std + self.action_mean

    def _make_batch(self, obs_state, actions=None, action_is_pad=None, obs_images=None):
        state = self._norm_state(obs_state)              # (B, state_dim)
        batch = {STATE_KEY: state}
        if self.cfg.use_image:
            if not obs_images:
                raise ValueError("model configured with use_image=True but no images given")
            for cam, img in obs_images.items():          # {cam: (B,C,H,W)} ImageNet-normed
                batch[f"observation.images.{cam}"] = img
        else:
            batch[ENV_KEY] = state
        if actions is not None:
            batch[ACTION_KEY] = self._norm_action(actions)   # (B, chunk, action_dim)
            batch["action_is_pad"] = action_is_pad
        return batch

    def forward(self, obs_state, actions, action_is_pad, obs_images=None, task=None):
        """Returns (loss, l1_item, kl_item). `loss` is differentiable and already
        includes kl_weight; l1_item / kl_item are floats for logging.

        lerobot's ACTPolicy.forward always computes the KL term when use_vae=True,
        but the VAE encoder only produces the latent params in training mode — so
        calling it in eval mode (our validation pass) feeds None into the KL math
        and crashes. We force train mode around the loss call so both train and
        validation produce a comparable l1 + kld; train.py wraps validation in
        no_grad, so no gradients leak."""
        batch = self._make_batch(obs_state, actions, action_is_pad, obs_images)
        was_training = self.policy.training
        self.policy.train()
        try:
            loss, loss_dict = self.policy.forward(batch)
        finally:
            self.policy.train(was_training)
        l1 = loss_dict["l1_loss"] if "l1_loss" in loss_dict else loss.item()
        kl = loss_dict.get("kld_loss", 0.0)
        return loss, l1, kl

    def reset(self):
        """Call once at the start of each episode before running predict()."""
        self.policy.reset()

    @torch.no_grad()
    def predict(self, obs_state, obs_images=None, task=None):
        action_norm = self.policy.select_action(
            self._make_batch(obs_state, obs_images=obs_images)
        )
        return self._unnorm_action(action_norm)


def build_model(cfg: dict, stats: dict, device) -> "ACT":
    """Policy entry point used by common/train.py and common/deploy.py.
    Builds an ACT model from the full config dict and dataset stats."""
    m, d, t = cfg["model"], cfg["dataset"], cfg["training"]
    model_cfg = ACTConfig(
        state_dim=m["state_dim"],
        action_dim=m["action_dim"],
        latent_dim=m["latent_dim"],
        d_model=m["d_model"],
        nhead=m["nhead"],
        num_encoder_layers=m["num_encoder_layers"],
        num_decoder_layers=m["num_decoder_layers"],
        dim_feedforward=m["dim_feedforward"],
        dropout=m["dropout"],
        chunk_size=d["chunk_size"],
        use_image=d["use_image"],
        kl_weight=t["kl_weight"],
        temporal_ensemble_coeff=m.get("temporal_ensemble_coeff"),
        cameras=d.get("cameras", ["wrist_cam"]),
        vision_backbone=m.get("vision_backbone", "resnet18"),
    )
    return ACT(model_cfg, stats).to(device)
