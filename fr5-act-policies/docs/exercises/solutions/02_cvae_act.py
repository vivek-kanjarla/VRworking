"""
SOLUTION — Exercise 02: CVAE pieces for ACT (reparameterization + KL + chunk loss).

Used by: ACT (you already have a CVAE notebook in Robotic-ML/ACT/autoencoders.ipynb;
this ties it to action chunks and the exact L1 + KL objective the repo uses).
Invariants checked:
  - KL matches torch.distributions closed form
  - reparameterize has the right mean/std statistically
  - loss is a finite scalar with correct gradient flow
"""
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


def reparameterize(mu, logvar):
    """Sample z = mu + sigma * eps, eps ~ N(0,I).  (the 'reparameterization trick'
    — keeps the sampling differentiable w.r.t. mu/logvar.)
    mu, logvar: (B, latent_dim)  ->  z: (B, latent_dim)
    """
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + std * eps


def kl_to_standard_normal(mu, logvar):
    """KL( N(mu, sigma^2) || N(0, I) ), summed over latent dim, mean over batch.
    Closed form: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)).
    """
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()


def act_loss(pred_actions, true_actions, is_pad, mu, logvar, kl_weight=10.0):
    """ACT objective: L1 reconstruction (masking padded steps) + kl_weight * KL.
    pred/true: (B, chunk, action_dim);  is_pad: (B, chunk) bool
    """
    l1 = (F.l1_loss(pred_actions, true_actions, reduction="none")
          * ~is_pad.unsqueeze(-1)).mean()
    kl = kl_to_standard_normal(mu, logvar)
    return l1 + kl_weight * kl, l1, kl


def _check():
    torch.manual_seed(0)
    B, L = 20000, 8
    # constant mu/logvar across the batch so batch-statistics reflect only the
    # sampling noise (if mu varied per-row, z.std would also pick up var(mu)).
    mu = torch.ones(B, L) * 1.0
    logvar = torch.zeros(B, L)                # sigma = exp(0.5*0) = 1

    z = reparameterize(mu, logvar)
    assert torch.allclose(z.mean(0), torch.ones(L), atol=0.05)   # -> mu
    assert torch.allclose(z.std(0),  torch.ones(L), atol=0.05)   # -> sigma = 1

    # KL vs torch closed form
    mu2, logvar2 = torch.randn(3, 5), torch.randn(3, 5)
    mine = kl_to_standard_normal(mu2, logvar2)
    p = Normal(mu2, (0.5 * logvar2).exp())
    q = Normal(torch.zeros_like(mu2), torch.ones_like(mu2))
    ref = kl_divergence(p, q).sum(-1).mean()
    assert torch.allclose(mine, ref, atol=1e-5), (mine, ref)

    # loss is finite + differentiable
    pred = torch.randn(2, 10, 7, requires_grad=True)
    true = torch.randn(2, 10, 7)
    pad = torch.zeros(2, 10, dtype=torch.bool); pad[:, 8:] = True
    loss, l1, kl = act_loss(pred, true, pad, torch.zeros(2, 4), torch.zeros(2, 4))
    loss.backward()
    assert torch.isfinite(loss) and pred.grad is not None
    print("02 cvae_act: PASS")


if __name__ == "__main__":
    _check()
