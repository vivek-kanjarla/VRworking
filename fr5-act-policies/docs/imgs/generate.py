"""
generate.py — produce all concept-illustration PNGs for the docs.

Run from the repo root:
    python docs/imgs/generate.py

Outputs: docs/imgs/*.png
"""

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

OUT = Path(__file__).parent

plt.rcParams.update({
    "font.family": "monospace",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. ACT — action chunking timeline
# ─────────────────────────────────────────────────────────────────────────────
def plot_action_chunking():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(-1, 22)
    ax.set_ylim(-0.6, 2.2)
    ax.axis("off")
    ax.set_title("ACT — Action Chunking (chunk_size=8 shown, repo uses 100)", pad=10)

    chunk_size = 8
    colors = ["#4C8BE2", "#E27B4C", "#4CE27B"]
    for ci, start in enumerate([0, 8, 16]):
        c = colors[ci % len(colors)]
        for t in range(chunk_size):
            rect = mpatches.FancyBboxPatch(
                (start + t + 0.05, 0.9), 0.85, 0.6,
                boxstyle="round,pad=0.05", fc=c, alpha=0.7, ec="none"
            )
            ax.add_patch(rect)
        ax.annotate(
            f"query {ci+1}\n→ predict steps {start}…{start+chunk_size-1}",
            xy=(start + chunk_size / 2, 1.55), ha="center", fontsize=9, color=c
        )
        ax.annotate("", xy=(start, 0.85), xytext=(start, 0.55),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
        ax.text(start, 0.4, f"obs t={start}", ha="center", fontsize=8, color=c)

    # timeline
    for t in range(25):
        ax.plot([t, t], [0.0, 0.25], color="grey", lw=0.8)
    ax.annotate("", xy=(24, 0.12), xytext=(-0.5, 0.12),
                arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
    ax.text(24.1, 0.08, "time (steps)", fontsize=9, color="grey")

    fig.tight_layout()
    fig.savefig(OUT / "act_action_chunking.png", bbox_inches="tight")
    plt.close(fig)
    print("  act_action_chunking.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ACT — temporal ensembling weight decay
# ─────────────────────────────────────────────────────────────────────────────
def plot_temporal_ensembling():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ages = np.arange(0, 16)
    for ax, m, title in zip(
        axes,
        [0.01, 0.3],
        ["m=0.01 (very smooth — almost uniform)", "m=0.3 (recency-weighted)"]
    ):
        w = np.exp(-m * ages)
        w /= w.sum()
        bars = ax.bar(ages, w, color="#4C8BE2", alpha=0.8, width=0.7)
        ax.set_xlabel("prediction age (steps old)")
        ax.set_ylabel("blend weight")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(ages)

    fig.suptitle("Temporal Ensembling — w_age = exp(−m·age) / Σ exp(−m·age)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "act_temporal_ensembling.png", bbox_inches="tight")
    plt.close(fig)
    print("  act_temporal_ensembling.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ACT — CVAE architecture sketch
# ─────────────────────────────────────────────────────────────────────────────
def plot_cvae_diagram():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title("ACT — CVAE latent z captures demo variability", pad=8)

    def box(x, y, w, h, label, color):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle="round,pad=0.06",
                                       fc=color, ec="none", alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=9, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.4))

    # Training path (top)
    ax.text(0.05, 0.93, "TRAIN", transform=ax.transAxes, fontsize=9,
            color="grey", style="italic")
    box(0.0, 3.2, 1.8, 0.7, "obs (joints+image)", "#AED6F1")
    box(2.2, 3.2, 1.8, 0.7, "action chunk\n(ground truth)", "#A9DFBF")
    box(4.8, 3.2, 1.5, 0.7, "CVAE\nEncoder", "#F9E79F")
    box(7.0, 3.2, 1.2, 0.7, "μ, σ → z", "#F0B27A")
    box(4.8, 2.0, 1.5, 0.7, "ACT\nDecoder", "#D2B4DE")
    box(7.0, 2.0, 1.8, 0.7, "predicted chunk\n+ KL loss", "#FADBD8")

    arrow(1.8, 3.55, 2.2, 3.55)          # obs → gt
    arrow(2.2+1.8, 3.55, 4.8, 3.55)      # gt → encoder
    arrow(4.8+1.5, 3.55, 7.0, 3.55)      # encoder → z
    arrow(7.6, 3.2, 7.6, 2.7)            # z down
    arrow(7.6, 2.35, 4.8+1.5, 2.35)      # z → decoder
    arrow(0.9, 3.2, 0.9, 2.35)           # obs → decoder (skip)
    arrow(0.9, 2.35, 4.8, 2.35)

    # Inference path (bottom)
    ax.text(0.05, 0.38, "INFER", transform=ax.transAxes, fontsize=9,
            color="grey", style="italic")
    box(0.0, 0.7, 1.8, 0.7, "obs (joints+image)", "#AED6F1")
    box(3.2, 0.7, 1.2, 0.7, "z ~ N(0,I)", "#F0B27A")
    box(4.8, 0.7, 1.5, 0.7, "ACT\nDecoder", "#D2B4DE")
    box(7.0, 0.7, 1.8, 0.7, "predicted chunk\n(execute!)", "#A9DFBF")
    arrow(1.8, 1.05, 3.2, 1.05)
    arrow(3.2+1.2, 1.05, 4.8, 1.05)
    arrow(4.8+1.5, 1.05, 7.0, 1.05)

    ax.set_xlim(-0.3, 9.2)
    ax.set_ylim(0.2, 4.3)

    fig.tight_layout()
    fig.savefig(OUT / "act_cvae_diagram.png", bbox_inches="tight")
    plt.close(fig)
    print("  act_cvae_diagram.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Diffusion — cosine noise schedule (ᾱ_t)
# ─────────────────────────────────────────────────────────────────────────────
def plot_cosine_schedule():
    T = 1000
    t = np.arange(T + 1)
    # cosine schedule: ᾱ_t = cos((t/T + 0.008)/(1+0.008) · π/2)²
    s = 0.008
    alpha_bar = np.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar /= alpha_bar[0]   # normalise to 1 at t=0

    # compare with linear
    beta_lin = np.linspace(1e-4, 0.02, T)
    alpha_lin = np.cumprod(1 - beta_lin)
    alpha_lin = np.concatenate([[1.0], alpha_lin])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, alpha_bar, label="cosine schedule ᾱ_t (used)", color="#4C8BE2", lw=2)
    ax.plot(t, alpha_lin, label="linear schedule ᾱ_t", color="#E27B4C", lw=2, ls="--")
    ax.set_xlabel("diffusion timestep t  (0 = clean, T = pure noise)")
    ax.set_ylabel("ᾱ_t  (signal fraction remaining)")
    ax.set_title("Noise Schedule: how much signal survives at step t")
    ax.legend()
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.text(T * 0.5, 0.52, "50% signal", color="grey", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "diffusion_cosine_schedule.png", bbox_inches="tight")
    plt.close(fig)
    print("  diffusion_cosine_schedule.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Diffusion — forward process: action noise levels
# ─────────────────────────────────────────────────────────────────────────────
def plot_forward_diffusion():
    np.random.seed(42)
    T = 100   # DDIM uses 10 but we show all T for illustration
    s = 0.008
    t_vals = np.arange(T + 1)
    alpha_bar = np.cos((t_vals / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar /= alpha_bar[0]

    # 7-D action (joints + gripper)
    x0 = np.array([30., -45., 60., 0., 90., 0., 0.8])   # clean action
    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "grip"]
    selected_t = [0, 10, 30, 60, 100]

    fig, axes = plt.subplots(1, len(selected_t), figsize=(13, 3.5), sharey=True)
    for ax, t in zip(axes, selected_t):
        noise = np.random.randn(len(x0))
        ab = alpha_bar[t]
        x_t = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * noise * np.abs(x0).max()
        colors = ["#4C8BE2" if i < 6 else "#E27B4C" for i in range(len(x0))]
        ax.bar(range(len(x0)), x_t, color=colors, alpha=0.8)
        ax.set_title(f"t = {t}\nᾱ={ab:.2f}", fontsize=9)
        ax.set_xticks(range(len(x0)))
        ax.set_xticklabels(joint_names, fontsize=7, rotation=45)

    axes[0].set_ylabel("action value")
    fig.suptitle("Forward Diffusion: clean action (t=0) → pure noise (t=T)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "diffusion_forward_process.png", bbox_inches="tight")
    plt.close(fig)
    print("  diffusion_forward_process.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Diffusion — U-Net architecture (1-D over time axis)
# ─────────────────────────────────────────────────────────────────────────────
def plot_unet_diagram():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis("off")
    ax.set_title("1-D U-Net Denoiser (action axis = sequence length)", pad=8)

    def rbox(x, y, w, h, label, color, fs=8.5):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle="round,pad=0.08",
                                       fc=color, ec="none", alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)

    def arr(x1, y1, x2, y2, color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.3))

    # Encoder path (down-sampling)
    levels = [
        (0.2, 2.5, 1.4, 0.8, "Conv\nLen=16", "#AED6F1"),
        (0.2, 1.4, 1.4, 0.8, "Conv↓\nLen=8", "#85C1E9"),
        (0.2, 0.3, 1.4, 0.8, "Conv↓\nLen=4", "#5DADE2"),
    ]
    for x, y, w, h, lbl, c in levels:
        rbox(x, y, w, h, lbl, c)

    # Bottleneck
    rbox(3.5, 0.3, 1.5, 0.8, "Bottleneck\nLen=4", "#F9E79F")

    # Decoder path (up-sampling) — skip connections
    dec = [
        (5.5, 0.3, 1.4, 0.8, "Conv↑+skip\nLen=8", "#A9DFBF"),
        (5.5, 1.4, 1.4, 0.8, "Conv↑+skip\nLen=16", "#7DCEA0"),
        (5.5, 2.5, 1.4, 0.8, "Conv\nLen=16", "#52BE80"),
    ]
    for x, y, w, h, lbl, c in dec:
        rbox(x, y, w, h, lbl, c)

    # Output
    rbox(8.0, 1.7, 1.4, 0.8, "predicted\nnoise ε̂", "#FADBD8")

    # Arrows: encoder down
    arr(0.9, 2.5, 0.9, 2.2)
    arr(0.9, 1.4, 0.9, 1.2)
    arr(0.9, 0.3, 3.5, 0.65)   # enc level3 → bottleneck

    # Bottleneck → decoder
    arr(5.0, 0.65, 5.5, 0.65)

    # Decoder up
    arr(6.2, 1.1, 6.2, 1.4)
    arr(6.2, 2.2, 6.2, 2.5)
    arr(6.9, 2.9, 8.0, 2.1)    # last dec → output (diagonal)

    # Skip connections (dashed)
    for enc_y, dec_y in [(2.9, 2.9), (1.8, 1.8)]:
        ax.annotate("", xy=(5.5, dec_y), xytext=(1.6, enc_y),
                    arrowprops=dict(arrowstyle="->", color="#E27B4C", lw=1.2, ls="dashed"))
    ax.text(3.5, 2.95, "skip", color="#E27B4C", fontsize=8)

    # FiLM conditioning label
    rbox(3.3, 1.7, 1.9, 0.75, "FiLM cond.\n(obs + t)", "#F0B27A")
    for y_conn in [0.65, 1.78, 2.9]:
        arr(4.25, 2.1, 4.25, y_conn, color="#F0B27A")

    ax.set_xlim(-0.3, 9.8)
    ax.set_ylim(-0.1, 4.0)

    fig.tight_layout()
    fig.savefig(OUT / "diffusion_unet.png", bbox_inches="tight")
    plt.close(fig)
    print("  diffusion_unet.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Flow Matching — straight vs curved paths
# ─────────────────────────────────────────────────────────────────────────────
def plot_flow_matching():
    np.random.seed(7)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── left: flow matching (straight) ──
    ax = axes[0]
    ax.set_title("Flow Matching (OT-CFM)\nlinear path: x_t = t·x₁ + (1-t)·x₀", fontsize=10)
    for _ in range(8):
        x0 = np.random.randn(2) * 1.2
        x1 = np.random.randn(2) * 0.4 + np.array([2.0, 1.0])
        ts = np.linspace(0, 1, 30)
        path = np.outer(1 - ts, x0) + np.outer(ts, x1)
        ax.plot(path[:, 0], path[:, 1], color="#4C8BE2", alpha=0.6, lw=1.5)
        ax.scatter(*x0, color="#E27B4C", s=40, zorder=5)
        ax.scatter(*x1, color="#4CE27B", s=40, zorder=5)
    ax.scatter([], [], color="#E27B4C", s=60, label="noise x₀ ~ N(0,I)")
    ax.scatter([], [], color="#4CE27B", s=60, label="action x₁")
    ax.legend(fontsize=8)
    ax.set_xlabel("action dim 1")
    ax.set_ylabel("action dim 2")
    ax.set_aspect("equal")

    # ── right: DDPM (curved) ──
    ax = axes[1]
    ax.set_title("DDPM (diffusion)\nmany noisy denoising steps", fontsize=10)
    for _ in range(8):
        x_final = np.random.randn(2) * 0.4 + np.array([2.0, 1.0])
        steps = 20
        x = np.random.randn(2) * 1.2
        path = [x.copy()]
        for s in range(steps):
            direction = (x_final - x) / steps
            x = x + direction + np.random.randn(2) * 0.15 * (1 - s / steps)
            path.append(x.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], color="#E27B4C", alpha=0.5, lw=1.5)
        ax.scatter(*path[0], color="#E27B4C", s=40, zorder=5)
        ax.scatter(*path[-1], color="#4CE27B", s=40, zorder=5)
    ax.scatter([], [], color="#E27B4C", s=60, label="noise x_T ~ N(0,I)")
    ax.scatter([], [], color="#4CE27B", s=60, label="action x₀")
    ax.legend(fontsize=8)
    ax.set_xlabel("action dim 1")
    ax.set_aspect("equal")

    fig.suptitle("Trajectory comparison: Flow Matching vs DDPM", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "flow_matching_vs_ddpm.png", bbox_inches="tight")
    plt.close(fig)
    print("  flow_matching_vs_ddpm.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. DiT — AdaLN-Zero block diagram
# ─────────────────────────────────────────────────────────────────────────────
def plot_adaln_block():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    ax.set_title("DiT — AdaLN-Zero Block (per transformer layer)", pad=10)

    def box(x, y, w, h, label, color, fs=9):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle="round,pad=0.08",
                                       fc=color, ec="none", alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.4))

    # Input
    box(2.0, 7.8, 2.5, 0.6, "x (tokens)", "#AED6F1")
    # Layer norm + scale/shift from c
    box(2.0, 6.8, 2.5, 0.6, "LayerNorm", "#F9E79F")
    box(5.2, 6.8, 2.5, 0.6, "γ, β, α  from c", "#F0B27A")
    box(2.0, 5.8, 2.5, 0.6, "γ·LN(x) + β", "#F9E79F")
    # Self-attention
    box(2.0, 4.7, 2.5, 0.7, "Self-Attention", "#D2B4DE")
    # Gate + residual
    box(2.0, 3.7, 2.5, 0.6, "α · Attn_out", "#A9DFBF")
    box(0.5, 3.7, 1.0, 0.6, "x", "#AED6F1")
    box(2.0, 2.8, 2.5, 0.6, "x  +  gated output", "#AED6F1")
    # FFN (same pattern)
    box(2.0, 1.8, 2.5, 0.6, "LayerNorm + FFN\n(same gating)", "#D2B4DE", fs=8)
    # Output
    box(2.0, 0.8, 2.5, 0.6, "x  (updated)", "#AED6F1")

    arr(3.25, 7.8, 3.25, 7.4)
    arr(3.25, 6.8, 3.25, 6.4)
    arr(5.2, 7.1, 4.5, 7.1)   # c → γ,β
    arr(5.2, 6.5, 4.5, 6.15)  # c → scale,shift
    arr(3.25, 5.8, 3.25, 5.4)
    arr(3.25, 4.7, 3.25, 4.3)
    arr(1.0, 4.0, 2.0, 4.0)   # skip
    arr(3.25, 3.7, 3.25, 3.4)
    arr(3.25, 2.8, 3.25, 2.4)
    arr(3.25, 1.8, 3.25, 1.4)

    # conditioning label
    ax.text(5.5, 8.0, "c = timestep t\n   + obs tokens\n   + language\n   (all concat → MLP → γ,β,α)",
            fontsize=8, color="#8B4513",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FEF9E7", ec="#F0B27A"))

    ax.set_xlim(-0.2, 9.0)
    ax.set_ylim(0.3, 9.0)

    fig.tight_layout()
    fig.savefig(OUT / "dit_adaln_block.png", bbox_inches="tight")
    plt.close(fig)
    print("  dit_adaln_block.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. pi0-FAST — DCT energy concentration
# ─────────────────────────────────────────────────────────────────────────────
def plot_dct_energy():
    np.random.seed(0)
    T = 50   # chunk_size for pi0
    t = np.linspace(0, 1, T)

    # simulated smooth joint trajectory
    traj = (
        20 * np.sin(2 * np.pi * t)
        + 8  * np.sin(4 * np.pi * t + 0.3)
        + 3  * np.sin(8 * np.pi * t + 1.1)
        + np.random.randn(T) * 1.5
    )

    # DCT-II (manual via numpy)
    N = len(traj)
    n = np.arange(N)
    coeffs = np.array([
        np.sum(traj * np.cos(np.pi * k * (2*n + 1) / (2*N)))
        * (np.sqrt(1/N) if k == 0 else np.sqrt(2/N))
        for k in range(N)
    ])
    energy = coeffs ** 2
    energy_frac = np.cumsum(energy) / energy.sum()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(t, traj, color="#4C8BE2", lw=2)
    axes[0].set_title("Joint trajectory (T=50 steps)")
    axes[0].set_xlabel("normalised time")
    axes[0].set_ylabel("angle (deg)")

    axes[1].bar(range(T), energy, color="#E27B4C", alpha=0.8)
    axes[1].axvline(8, color="red", ls="--", lw=1.5, label="first 8 coeffs")
    axes[1].set_title("DCT coefficient energy")
    axes[1].set_xlabel("DCT index k")
    axes[1].set_ylabel("energy (coeff²)")
    axes[1].legend(fontsize=8)

    axes[2].plot(range(T), energy_frac * 100, color="#4CE27B", lw=2)
    axes[2].axhline(95, color="red", ls="--", lw=1.2, label="95% threshold")
    axes[2].axvline(8, color="red", ls="--", lw=1.2)
    axes[2].set_title("Cumulative energy captured")
    axes[2].set_xlabel("keep first K coefficients")
    axes[2].set_ylabel("% energy captured")
    axes[2].legend(fontsize=8)

    fig.suptitle("π0-FAST: DCT compresses a 50-step trajectory into ~8 tokens", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "pi0_fast_dct_energy.png", bbox_inches="tight")
    plt.close(fig)
    print("  pi0_fast_dct_energy.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Inference timeline — query cadence comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_inference_cadence():
    policies = {
        "ACT (no TE)":        (100, 100),   # chunk, exec_per_query
        "ACT (TE on)":        (100, 1),
        "Diffusion Policy":   (16,  8),
        "DiT + Flow":         (32,  32),
        "π0 / π0.5":          (50,  50),
        "π0-FAST":            (50,  50),
    }
    total_steps = 120

    fig, axes = plt.subplots(len(policies), 1, figsize=(13, 7), sharex=True)
    colors_exec = "#4C8BE2"
    colors_query = "#E27B4C"

    for ax, (name, (chunk, exec_n)) in zip(axes, policies.items()):
        t = 0
        while t < total_steps:
            # query marker
            ax.axvline(t, color=colors_query, lw=1.8, alpha=0.9)
            # execution band
            end = min(t + exec_n, total_steps)
            ax.barh(0, end - t, left=t, height=0.6, color=colors_exec, alpha=0.5)
            t += exec_n

        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8, labelpad=120)
        ax.set_xlim(0, total_steps)
        ax.set_ylim(-0.5, 0.5)

    axes[-1].set_xlabel("policy step (1 step = 33 ms at 30 Hz)")

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=colors_query, lw=2, label="model query (expensive)"),
        mpatches.Patch(color=colors_exec, alpha=0.5, label="executing cached actions"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9, bbox_to_anchor=(0.98, 0.98))
    fig.suptitle("Model query cadence across 120 steps (~4 seconds @ 30 Hz)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "inference_cadence.png", bbox_inches="tight")
    plt.close(fig)
    print("  inference_cadence.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11. pi0 — architecture overview (VLM + action expert)
# ─────────────────────────────────────────────────────────────────────────────
def plot_pi0_architecture():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    ax.set_title("π0 Architecture — PaliGemma VLM prefix + Action Expert", pad=10)

    def box(x, y, w, h, label, color, fs=9):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle="round,pad=0.08",
                                       fc=color, ec="none", alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)

    def arr(x1, y1, x2, y2, label="", color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.4))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.12, label, fontsize=7, ha="center", color=color)

    # Inputs
    box(0.0, 2.8, 1.6, 0.7, "wrist image\n(224×224)", "#AED6F1")
    box(0.0, 1.8, 1.6, 0.7, "joint states\n(6-D)", "#AED6F1")
    box(0.0, 0.8, 1.6, 0.7, "language\n\"pick up block\"", "#D5F5E3")

    # PaliGemma encoder
    box(2.2, 1.8, 2.2, 1.0, "PaliGemma 2B\n(ViT-G + Gemma 2B)\nKV-cached prefix", "#F9E79F")
    box(2.2, 0.8, 2.2, 0.7, "SigLIP\nViT tokenizer", "#F9E79F")

    arr(1.6, 3.15, 2.2, 2.65)
    arr(1.6, 2.15, 2.2, 2.15)
    arr(1.6, 1.15, 2.2, 1.15)

    # Joint tokens
    box(2.2, 0.0, 2.2, 0.6, "state proj\n(6 → 512)", "#D2B4DE")
    arr(1.6, 2.0, 2.2, 0.3, color="#D2B4DE")

    # Action Expert
    box(5.2, 1.2, 2.2, 1.4, "Action Expert\n(3-layer transformer)\nconditioned on VLM prefix\n+ noisy action x_t", "#FADBD8")
    arr(4.4, 2.35, 5.2, 1.9)
    arr(4.4, 1.15, 5.2, 1.5)
    arr(4.4, 0.3,  5.2, 1.35)

    # ODE
    box(8.0, 1.4, 1.8, 0.8, "Euler ODE\n10 steps\nflow matching", "#A9DFBF")
    arr(7.4, 1.9, 8.0, 1.8)
    arr(9.8, 1.8, 10.0, 1.8)

    # Output
    box(10.0, 1.4, 1.8, 0.8, "action chunk\n(50×7)", "#52BE80")

    ax.set_xlim(-0.3, 12.2)
    ax.set_ylim(-0.2, 4.2)

    fig.tight_layout()
    fig.savefig(OUT / "pi0_architecture.png", bbox_inches="tight")
    plt.close(fig)
    print("  pi0_architecture.png")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Writing PNGs to {OUT}/")
    plot_action_chunking()
    plot_temporal_ensembling()
    plot_cvae_diagram()
    plot_cosine_schedule()
    plot_forward_diffusion()
    plot_unet_diagram()
    plot_flow_matching()
    plot_adaln_block()
    plot_dct_energy()
    plot_inference_cadence()
    plot_pi0_architecture()
    print("done — all PNGs written.")


if __name__ == "__main__":
    main()
