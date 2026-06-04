"""
SOLUTION — Exercise 07: ACT temporal ensembling.

Used by: ACT at inference (you already have a TemporalEnsembler in
Robotic-ML/ACT/act_implemetation.ipynb with m=0.01 — this is the same idea,
written as a checkable function so you can re-derive the weighting).

Idea: at each timestep the policy predicts a whole chunk. For the action to
execute NOW, several past chunks also made a prediction. Blend them with weight
w_i = exp(-m * age_i), normalized. Lower m -> trust older predictions more.

Invariants checked:
  - weights are nonneg and sum to 1
  - if every chunk predicted the same value, the blend returns that value
  - lower m puts MORE relative weight on older predictions
"""
import torch


def ensemble_weights(num_preds, m):
    """Weights for `num_preds` overlapping predictions of the current timestep.
    age 0 = the prediction made this step (newest), age num_preds-1 = oldest.
    w_i = exp(-m * age_i), normalized to sum to 1.
    """
    ages = torch.arange(num_preds, dtype=torch.float32)
    w = torch.exp(-m * ages)
    return w / w.sum()


def blend(predictions, m):
    """predictions: (num_preds, action_dim), index 0 = newest.
    returns the ensembled action (action_dim,)."""
    w = ensemble_weights(len(predictions), m)
    return (w.unsqueeze(-1) * predictions).sum(0)


def _check():
    torch.manual_seed(0)

    w = ensemble_weights(5, m=0.01)
    assert torch.allclose(w.sum(), torch.tensor(1.0))
    assert (w >= 0).all()

    # identical predictions -> blend returns the same value
    same = torch.full((5, 7), 0.3)
    assert torch.allclose(blend(same, m=0.01), torch.full((7,), 0.3), atol=1e-6)

    # lower m -> flatter weights -> MORE relative weight on the oldest prediction
    w_small = ensemble_weights(10, m=0.01)
    w_large = ensemble_weights(10, m=0.5)
    assert w_small[-1] > w_large[-1], (w_small[-1].item(), w_large[-1].item())

    # newest always has the largest weight
    assert torch.argmax(w_small).item() == 0
    print("07 temporal_ensembling: PASS")


if __name__ == "__main__":
    _check()
