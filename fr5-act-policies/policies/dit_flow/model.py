"""
policies/dit_flow/model.py — Diffusion Transformer + flow-matching policy for the FR5.

Wraps lerobot's MultiTaskDiTPolicy (lerobot==0.5.1) with objective="flow_matching".

Architecture overview
─────────────────────
Observation encoder (CLIP-based)
  • CLIP ViT-B/16 image encoder  — wrist-cam frames
  • CLIP text encoder             — language task string
  • Linear projection             — joint state (6 DOF)
  → flat conditioning vector per timestep

Action denoiser — Diffusion Transformer (DiT)
  • Transformer with rotary PE (RoPE)
  • Conditioning injected via cross-attention / AdaLN
  • Predicts the velocity field  v(x_t, t, cond)

Training objective — conditional flow matching
  Path:   x_t  = t·a + (1 − (1−σ)·t)·ε,   ε ∼ 𝒩(0, I)
  Target: v    = a − (1−σ)·ε
  Loss:   ‖v_θ(x_t, t, cond) − v‖²

Inference — Euler ODE integration, t: 0 → 1
  x_{t+Δt} = x_t + Δt · v_θ(x_t, t, cond)
  (num_integration_steps steps, configurable)

Normalization (all done in this wrapper; lerobot policy sees IDENTITY norms)
  • State  / action : min-max  → [-1, 1]
  • Images : undo ImageNet (from dataset.py) → CLIP normalization
  • Language       : CLIPTokenizerFast via AutoTokenizer
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from lerobot.policies.multi_task_dit.configuration_multi_task_dit import (
    MultiTaskDiTConfig as _LRConfig,
)
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode


# ── batch key constants ────────────────────────────────────────────────────────
STATE_KEY      = "observation.state"
IMAGE_KEY      = "observation.images.wrist_cam"
ACTION_KEY     = "action"
LANG_TOKENS    = "observation.language.tokens"
LANG_ATTN_MASK = "observation.language.attention_mask"

# ImageNet statistics applied by dataset.py — we undo these before CLIP
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# CLIP ViT normalization expected by HuggingFace's CLIPModel pixel_values
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


@dataclass
class DiTFlowConfig:
    state_dim:  int  = 6
    action_dim: int  = 7
    chunk_size: int  = 32         # horizon == n_action_steps
    use_image:  bool = True

    # flow-matching / diffusion
    objective:             str   = "flow_matching"   # "flow_matching" | "diffusion"
    num_integration_steps: int   = 10                # Euler/RK4 ODE steps at inference
    integration_method:    str   = "euler"           # "euler" | "rk4"

    # DiT transformer
    hidden_dim:     int   = 512
    num_layers:     int   = 6
    num_heads:      int   = 8
    dropout:        float = 0.1

    # CLIP vision + language encoders
    vision_encoder_name:  str = "openai/clip-vit-base-patch16"
    text_encoder_name:    str = "openai/clip-vit-base-patch16"
    tokenizer_max_length: int = 77


def _lerobot_config(cfg: DiTFlowConfig) -> _LRConfig:
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

    # n_action_steps == horizon → full chunk executed per policy query (ACT-style)
    return _LRConfig(
        n_obs_steps=1,
        horizon=cfg.chunk_size,
        n_action_steps=cfg.chunk_size,
        input_features=input_features,
        output_features={ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(cfg.action_dim,))},
        normalization_mapping=norm_map,
        objective=cfg.objective,
        num_integration_steps=cfg.num_integration_steps,
        integration_method=cfg.integration_method,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        vision_encoder_name=cfg.vision_encoder_name if cfg.use_image else None,
        text_encoder_name=cfg.text_encoder_name,
        tokenizer_max_length=cfg.tokenizer_max_length,
        image_resize_shape=None,   # dataset already outputs 224×224
        image_crop_shape=None,     # no additional crop needed
        image_crop_is_random=False,
        use_rope=True,
        do_mask_loss_for_padding=False,
    )


class DiTFlow(nn.Module):
    def __init__(self, cfg: DiTFlowConfig, stats: dict):
        super().__init__()
        self.cfg = cfg
        self.policy = MultiTaskDiTPolicy(_lerobot_config(cfg))

        # language tokenizer (same CLIP checkpoint as the text encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder_name)

        # min-max normalization buffers — saved into the checkpoint
        eps = 1e-6
        self.register_buffer("state_min",   torch.as_tensor(stats["state_min"]).float())
        self.register_buffer("state_max",   torch.as_tensor(stats["state_max"]).float() + eps)
        self.register_buffer("action_min",  torch.as_tensor(stats["action_min"]).float())
        self.register_buffer("action_max",  torch.as_tensor(stats["action_max"]).float() + eps)

        # image renormalization constants (stored as buffers for device movement)
        self.register_buffer("_imagenet_mean", _IMAGENET_MEAN.clone())
        self.register_buffer("_imagenet_std",  _IMAGENET_STD.clone())
        self.register_buffer("_clip_mean",     _CLIP_MEAN.clone())
        self.register_buffer("_clip_std",      _CLIP_STD.clone())

    # ── normalization helpers ──────────────────────────────────────────────────

    def _norm_state(self, s):
        return 2.0 * (s - self.state_min) / (self.state_max - self.state_min) - 1.0

    def _norm_action(self, a):
        return 2.0 * (a - self.action_min) / (self.action_max - self.action_min) - 1.0

    def _unnorm_action(self, a):
        return (a + 1.0) / 2.0 * (self.action_max - self.action_min) + self.action_min

    def _renorm_image(self, img):
        """Undo ImageNet normalization (applied by dataset.py) → apply CLIP normalization."""
        raw = img * self._imagenet_std + self._imagenet_mean   # → [0, 1]
        return (raw - self._clip_mean) / self._clip_std

    def _tokenize(self, task, device):
        """Tokenize a list of language strings → input_ids / attention_mask tensors."""
        if isinstance(task, str):
            task = [task]
        enc = self.tokenizer(
            task,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.tokenizer_max_length,
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # ── batch builder ──────────────────────────────────────────────────────────

    def _make_batch(self, obs_state, actions=None, action_is_pad=None,
                    obs_image=None, task=None, for_training=True):
        """Build a batch dict for lerobot's MultiTaskDiTPolicy.

        Two distinct modes:

        for_training=True  (→ policy.forward)
          State: (B, 1, state_dim) — encode() reads batch[OBS_STATE].shape[:2]
          Image: (B, C, H, W)     — _prepare_batch adds camera dim, encode adds n_obs_steps

        for_training=False  (→ policy.select_action via queue system)
          State: (B, state_dim)   — queue stacks to (B, 1, state_dim) via dim=1
          Image: (B, C, H, W)     — _prepare_batch adds camera dim → (B, 1, C, H, W),
                                     queue stacks to (B, 1, 1, C, H, W) — 6D for encode

        Language tokens are always (B, max_length); encode handles the n_obs_steps
        expansion internally. They are NOT queued (not in policy._queues).
        """
        dev = obs_state.device
        B   = obs_state.shape[0]

        state = self._norm_state(obs_state)
        if for_training:
            state = state.unsqueeze(1)    # → (B, 1, state_dim) for encode()
        batch = {STATE_KEY: state}

        if self.cfg.use_image and obs_image is not None:
            # Always (B, C, H, W); _prepare_batch and/or the queue adds n_obs_steps
            batch[IMAGE_KEY] = self._renorm_image(obs_image)

        # Language tokens — always include so conditioning_dim stays constant.
        # Default to empty strings; CLIP text encoder handles them gracefully.
        if task is None:
            task = [""] * B
        ids, mask = self._tokenize(task, dev)
        batch[LANG_TOKENS]    = ids
        batch[LANG_ATTN_MASK] = mask

        if actions is not None:
            batch[ACTION_KEY]      = self._norm_action(actions)   # (B, chunk, action_dim)
            batch["action_is_pad"] = action_is_pad

        return batch

    # ── public interface (matches common/train.py + common/deploy.py) ──────────

    def forward(self, obs_state, actions, action_is_pad, obs_image=None, task=None):
        """Returns (loss, loss_item, 0.0) — flow-matching MSE, no KL term."""
        batch = self._make_batch(obs_state, actions, action_is_pad, obs_image, task,
                                 for_training=True)
        loss, _ = self.policy.forward(batch)
        return loss, loss.item(), 0.0

    def reset(self):
        """Call once before each episode during deployment."""
        self.policy.reset()

    @torch.no_grad()
    def predict(self, obs_state, obs_image=None, task=None):
        """One ODE integration step → (action_dim,) tensor in original units.

        Uses select_action which internally manages an action queue: the first call
        generates a full chunk (chunk_size actions via Euler ODE), subsequent calls
        pop actions one by one until the queue is exhausted and a new chunk is generated.
        Call reset() at the start of each episode to clear the queue.
        """
        action_norm = self.policy.select_action(
            self._make_batch(obs_state, obs_image=obs_image, task=task, for_training=False)
        )
        return self._unnorm_action(action_norm)


# ── policy entry point for common/train.py and common/deploy.py ───────────────

def build_model(cfg: dict, stats: dict, device) -> DiTFlow:
    m, d = cfg["model"], cfg["dataset"]
    model_cfg = DiTFlowConfig(
        state_dim=m["state_dim"],
        action_dim=m["action_dim"],
        chunk_size=d["chunk_size"],
        use_image=d["use_image"],
        objective=m.get("objective", "flow_matching"),
        num_integration_steps=m.get("num_integration_steps", 10),
        integration_method=m.get("integration_method", "euler"),
        hidden_dim=m.get("hidden_dim", 512),
        num_layers=m.get("num_layers", 6),
        num_heads=m.get("num_heads", 8),
        dropout=m.get("dropout", 0.1),
        vision_encoder_name=m.get("vision_encoder_name", "openai/clip-vit-base-patch16"),
        text_encoder_name=m.get("text_encoder_name", "openai/clip-vit-base-patch16"),
        tokenizer_max_length=m.get("tokenizer_max_length", 77),
    )
    return DiTFlow(model_cfg, stats).to(device)
