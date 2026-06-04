# EXERCISE — implement the functions marked TODO, then run:
#     python docs/exercises/01_attention.py
# It self-checks. Reference solution: docs/exercises/solutions/01_attention.py
# (Try hard before peeking. These build the core math of the policies in docs/.)

"""
Exercise 01: Scaled dot-product + multi-head self-attention.

Used by: every transformer policy (ACT, DiT/dit_flow, pi0).
Invariant checked: matches torch.nn.functional.scaled_dot_product_attention.
"""
import torch
import torch.nn.functional as F


def softmax_lastdim(scores: torch.Tensor) -> torch.Tensor:
    # numerically-stable softmax over the last dim
    # TODO: implement
    raise NotImplementedError


def scaled_dot_product_attention(Q, K, V):
    """
    Q: (B, H, T, d)   queries
    K: (B, H, T, d)   keys
    V: (B, H, T, d)   values
    returns: (B, H, T, d)
    """
    # TODO: implement
    raise NotImplementedError


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.out = torch.nn.Linear(d_model, d_model)

    def forward(self, x):                  # x: (B, T, d_model)
        # TODO: implement
        raise NotImplementedError


def _check():
    torch.manual_seed(0)
    B, H, T, d = 2, 4, 5, 8
    Q, K, V = torch.randn(B, H, T, d), torch.randn(B, H, T, d), torch.randn(B, H, T, d)
    mine = scaled_dot_product_attention(Q, K, V)
    ref = F.scaled_dot_product_attention(Q, K, V)
    assert torch.allclose(mine, ref, atol=1e-5), (mine - ref).abs().max()

    mha = MultiHeadSelfAttention(16, 4)
    y = mha(torch.randn(2, 7, 16))
    assert y.shape == (2, 7, 16)
    print("01 attention: PASS")


if __name__ == "__main__":
    _check()
