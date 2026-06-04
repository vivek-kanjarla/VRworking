# EXERCISE — implement the functions marked TODO, then run:
#     python docs/exercises/06_fast_tokenizer.py
# It self-checks. Reference solution: docs/exercises/solutions/06_fast_tokenizer.py
# (Try hard before peeking. These build the core math of the policies in docs/.)

"""
Exercise 06: FAST-style action tokenization (DCT + quantize + tiny BPE).

Used by: pi0_fast. NEW from-scratch territory.
This is a miniature of how π0-FAST turns a continuous action chunk into discrete
tokens: DCT over time -> quantize -> (BPE compress). Here we do DCT + quantize +
a tiny BPE round-trip and check reconstruction.

Invariants checked:
  - idct(dct(x)) == x  (orthonormal DCT-II round-trip)
  - DCT concentrates energy of a smooth ramp in low-frequency coeffs
  - BPE encode/decode round-trips a token stream
"""
import torch


def dct_ii(x, norm_ortho=True):
    """DCT-II along the last axis, computed directly from the definition.
    x: (..., N) -> (..., N).  This is the JPEG transform: decompose into cosines.
    """
    # TODO: implement
    raise NotImplementedError


def idct_ii(X, norm_ortho=True):
    """Inverse of dct_ii (i.e. DCT-III). Reconstructs x from coefficients."""
    # TODO: implement
    raise NotImplementedError


def quantize(X, step):
    """Round coefficients to a grid of resolution `step` (the lossy part)."""
    # TODO: implement
    raise NotImplementedError


def bpe_encode(tokens, merges):
    """Greedy byte-pair encoding: repeatedly replace the most-frequent adjacent
    pair with a new symbol. `merges` is the number of merge ops to perform.
    Returns (encoded_list, merge_table) where merge_table maps new_id -> (a, b).
    """
    # TODO: implement
    raise NotImplementedError


def bpe_decode(seq, table):
    """Expand merged symbols back to the original token stream."""
    # TODO: implement
    raise NotImplementedError


def _check():
    torch.manual_seed(0)

    # DCT round-trip
    x = torch.randn(4, 7, 16)               # (B, A, T): DCT over time axis (last)
    X = dct_ii(x)
    assert torch.allclose(idct_ii(X), x, atol=1e-4), (idct_ii(X) - x).abs().max()

    # smooth ramp -> energy in low freqs
    ramp = torch.arange(1., 5.).view(1, 4)
    C = dct_ii(ramp, norm_ortho=False)[0]
    lo = C[:2].pow(2).sum(); hi = C[2:].pow(2).sum()
    assert lo > 50 * hi, (lo.item(), hi.item())

    # BPE round-trip on a repetitive stream (like quantized smooth coeffs)
    toks = [0, 0, 1, 0, 0, 1, 0, 0, 1]
    enc, table = bpe_encode(toks, merges=3)
    assert len(enc) < len(toks)
    assert bpe_decode(enc, table) == toks
    print("06 fast_tokenizer: PASS")


if __name__ == "__main__":
    _check()
