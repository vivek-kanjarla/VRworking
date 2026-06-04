"""
SOLUTION — Exercise 03: Diffusion Policy core (forward noising + DDPM/DDIM reverse).

Used by: policies/diffusion. NEW from-scratch territory for you.
Invariants checked:
  - cosine alpha_bar is monotonically decreasing from ~1 to ~0
  - q_sample then closed-form epsilon recovery is exact
  - DDIM sampling of a trivially-learnable field recovers the data direction
"""
import torch


def cosine_alpha_bar(T: int) -> torch.Tensor:
    """ᾱ_k schedule (squaredcos_cap_v2 style). Returns (T,) decreasing 1 -> ~0.
    ᾱ_k = f(k)/f(0), f(k) = cos( ((k/T + s)/(1+s)) * pi/2 )^2 , s = 0.008.
    """
    s = 0.008
    k = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((k / T + s) / (1 + s)) * torch.pi / 2) ** 2
    alpha_bar = f / f[0]
    return alpha_bar[1:].float()          # (T,)


def q_sample(a0, k, eps, alpha_bar):
    """Forward diffusion: a_k = sqrt(ᾱ_k) a0 + sqrt(1-ᾱ_k) eps.
    a0, eps: (B, T, A);  k: (B,) long indices into alpha_bar
    """
    ab = alpha_bar[k].view(-1, 1, 1)      # (B,1,1)
    return ab.sqrt() * a0 + (1 - ab).sqrt() * eps


def recover_eps(a_k, a0, k, alpha_bar):
    """Invert q_sample to get the noise that was added (used to sanity-check)."""
    ab = alpha_bar[k].view(-1, 1, 1)
    return (a_k - ab.sqrt() * a0) / (1 - ab).sqrt()


@torch.no_grad()
def ddim_sample(eps_model, shape, alpha_bar, n_steps):
    """Deterministic DDIM sampling. eps_model(a_k, k) -> predicted noise.
    Iterate over a subset of timesteps from high noise to low.
    """
    B = shape[0]
    a = torch.randn(shape)                                   # start: pure noise
    steps = torch.linspace(len(alpha_bar) - 1, 0, n_steps).round().long()
    for i, k in enumerate(steps):
        ab_k = alpha_bar[k]
        kb = torch.full((B,), k, dtype=torch.long)
        eps = eps_model(a, kb)
        a0_pred = (a - (1 - ab_k).sqrt() * eps) / ab_k.sqrt()   # predict clean
        if i == len(steps) - 1:
            a = a0_pred
        else:
            ab_next = alpha_bar[steps[i + 1]]
            a = ab_next.sqrt() * a0_pred + (1 - ab_next).sqrt() * eps  # re-noise to next level
    return a


def _check():
    torch.manual_seed(0)
    T = 100
    ab = cosine_alpha_bar(T)
    assert ab[0] > 0.99 and ab[-1] < 0.05
    assert (ab[1:] <= ab[:-1] + 1e-6).all(), "alpha_bar must be non-increasing"

    # forward/inverse consistency
    a0 = torch.randn(8, 16, 7)
    eps = torch.randn_like(a0)
    k = torch.randint(0, T, (8,))
    a_k = q_sample(a0, k, eps, ab)
    assert torch.allclose(recover_eps(a_k, a0, k, ab), eps, atol=1e-4)

    # DDIM toward a fixed target: a model that always says "the noise points away
    # from target mu" should denoise to mu. Build the oracle eps for target=mu.
    mu = torch.full((1, 16, 7), 0.7)
    def oracle(a, kb):
        ab_k = ab[kb].view(-1, 1, 1)
        return (a - ab_k.sqrt() * mu) / (1 - ab_k).sqrt()   # exact noise if a0==mu
    out = ddim_sample(oracle, (4, 16, 7), ab, n_steps=10)
    assert torch.allclose(out, mu.expand(4, 16, 7), atol=1e-3), out.mean()
    print("03 diffusion_policy: PASS")


if __name__ == "__main__":
    _check()
