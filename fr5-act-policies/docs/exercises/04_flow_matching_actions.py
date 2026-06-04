# EXERCISE — implement the functions marked TODO, then run:
#     python docs/exercises/04_flow_matching_actions.py
# It self-checks. Reference solution: docs/exercises/solutions/04_flow_matching_actions.py
# (Try hard before peeking. These build the core math of the policies in docs/.)

"""
Exercise 04: Conditional flow matching on action chunks.

Used by: dit_flow, pi0. You already have a deep flow-matching series in
~/Desktop/Flow matching/ — this is the robot-action-chunk flavour, tied to this repo.
Invariants checked:
  - the path endpoints are correct (t=0 -> noise, t=1 -> data)
  - velocity target identity: x_t + (1-t)*v == data  (for sigma_min=0)
  - Euler integration of the *exact* velocity field recovers the data
"""
import torch


def fm_path(a_data, eps, t, sigma_min=0.0):
    """Straight-line probability path.
    x_t = t*a_data + (1 - (1-sigma_min)*t) * eps
    a_data, eps: (B, T, A);  t: (B,) in [0,1]
    """
    # TODO: implement
    raise NotImplementedError


def fm_target_velocity(a_data, eps, sigma_min=0.0):
    """Constant velocity that carries eps -> a_data in a straight line.
    v = a_data - (1 - sigma_min) * eps
    """
    # TODO: implement
    raise NotImplementedError


@torch.no_grad()
def euler_integrate(velocity_fn, x0, n_steps):
    """Integrate dx/dt = v(x, t) from t=0 to t=1 with forward Euler.
    velocity_fn(x, t_scalar) -> velocity tensor like x.
    """
    # TODO: implement
    raise NotImplementedError


def _check():
    torch.manual_seed(0)
    a = torch.randn(8, 16, 7)
    eps = torch.randn_like(a)

    # endpoints
    x0 = fm_path(a, eps, torch.zeros(8))
    x1 = fm_path(a, eps, torch.ones(8))
    assert torch.allclose(x0, eps, atol=1e-5)
    assert torch.allclose(x1, a, atol=1e-5)

    # velocity identity: x_t + (1-t)*v == data, for any t (sigma_min=0)
    t = torch.rand(8)
    xt = fm_path(a, eps, t)
    v = fm_target_velocity(a, eps)
    lhs = xt + (1 - t).view(-1, 1, 1) * v
    assert torch.allclose(lhs, a, atol=1e-4), (lhs - a).abs().max()

    # Euler with the EXACT (constant) field recovers data from noise in 1 step,
    # because the path is a straight line.
    mu = torch.full((4, 16, 7), 0.5)
    target_eps = torch.randn(4, 16, 7)
    const_v = fm_target_velocity(mu, target_eps)        # constant in x and t
    out = euler_integrate(lambda x, t: const_v, target_eps, n_steps=1)
    assert torch.allclose(out, mu, atol=1e-4)
    # and many steps also fine
    out2 = euler_integrate(lambda x, t: const_v, target_eps, n_steps=10)
    assert torch.allclose(out2, mu, atol=1e-4)
    print("04 flow_matching_actions: PASS")


if __name__ == "__main__":
    _check()
