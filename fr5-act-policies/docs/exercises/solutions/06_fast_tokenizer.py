"""
SOLUTION — Exercise 06: FAST-style action tokenization (DCT + quantize + tiny BPE).

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
    N = x.shape[-1]
    n = torch.arange(N, dtype=x.dtype)
    k = torch.arange(N, dtype=x.dtype).view(N, 1)
    basis = torch.cos(torch.pi / N * (n + 0.5) * k)     # (N, N): row k = freq-k cosine
    X = x @ basis.T                                      # (..., N)
    if norm_ortho:
        scale = torch.full((N,), (2.0 / N) ** 0.5, dtype=x.dtype)
        scale[0] = (1.0 / N) ** 0.5
        X = X * scale
    return X


def idct_ii(X, norm_ortho=True):
    """Inverse of dct_ii (i.e. DCT-III). Reconstructs x from coefficients."""
    N = X.shape[-1]
    n = torch.arange(N, dtype=X.dtype)
    k = torch.arange(N, dtype=X.dtype).view(N, 1)
    basis = torch.cos(torch.pi / N * (n + 0.5) * k)     # same basis
    if norm_ortho:
        scale = torch.full((N,), (2.0 / N) ** 0.5, dtype=X.dtype)
        scale[0] = (1.0 / N) ** 0.5
        Xs = X * scale
    else:
        Xs = X * (2.0 / N); Xs[..., 0] = X[..., 0] / N
    # x_n = sum_k Xs_k cos(pi/N (n+0.5) k)
    return Xs @ basis


def quantize(X, step):
    """Round coefficients to a grid of resolution `step` (the lossy part)."""
    return torch.round(X / step) * step


def bpe_encode(tokens, merges):
    """Greedy byte-pair encoding: repeatedly replace the most-frequent adjacent
    pair with a new symbol. `merges` is the number of merge ops to perform.
    Returns (encoded_list, merge_table) where merge_table maps new_id -> (a, b).
    """
    seq = list(tokens)
    table = {}
    next_id = max(seq) + 1 if seq else 0
    for _ in range(merges):
        # count adjacent pairs
        counts = {}
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
        if not counts:
            break
        best = max(counts, key=counts.get)
        if counts[best] < 2:
            break                                   # no repetition left to exploit
        table[next_id] = best
        # replace all occurrences of `best`
        out, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best:
                out.append(next_id); i += 2
            else:
                out.append(seq[i]); i += 1
        seq, next_id = out, next_id + 1
    return seq, table


def bpe_decode(seq, table):
    """Expand merged symbols back to the original token stream."""
    changed = True
    seq = list(seq)
    while changed:
        changed = False
        out = []
        for s in seq:
            if s in table:
                out.extend(table[s]); changed = True
            else:
                out.append(s)
        seq = out
    return seq


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
