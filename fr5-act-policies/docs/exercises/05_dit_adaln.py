# EXERCISE — implement the functions marked TODO, then run:
#     python docs/exercises/05_dit_adaln.py
# It self-checks. Reference solution: docs/exercises/solutions/05_dit_adaln.py
# (Try hard before peeking. These build the core math of the policies in docs/.)

"""
Exercise 05: a DiT block with AdaLN-Zero conditioning.

Used by: dit_flow (and the same idea inside pi0's action expert).
Key concept: instead of feeding the conditioning vector in as another token,
AdaLN *computes the LayerNorm scale/shift (and a residual gate) from the
conditioning*, per block. "Zero" = the gate starts at 0 so the block is the
identity at initialization (stabilizes training).

Invariants checked:
  - shapes
  - AdaLN-Zero identity-at-init: with gate weights zero-initialized, the block
    returns its input unchanged on the first forward pass.
"""
import torch
import torch.nn as nn


def modulate(x, shift, scale):
    # x: (B, T, D); shift/scale: (B, D) -> broadcast over T
    # TODO: implement
    raise NotImplementedError


class DiTBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )
        # produces 6 vectors from conditioning: shift/scale/gate for attn and mlp
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        # AdaLN-Zero: zero-init the final layer so gates start at 0
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond):
        # cond: (B, D) conditioning (timestep + observation embedding)
        # TODO: implement
        raise NotImplementedError


def _check():
    torch.manual_seed(0)
    B, T, D = 2, 32, 64
    block = DiTBlock(D, n_heads=8)
    x = torch.randn(B, T, D)
    cond = torch.randn(B, D)
    y = block(x, cond)
    assert y.shape == (B, T, D)
    # AdaLN-Zero: gates are 0 at init -> block is identity on first pass
    assert torch.allclose(y, x, atol=1e-6), "AdaLN-Zero should be identity at init"
    print("05 dit_adaln: PASS")


if __name__ == "__main__":
    _check()
