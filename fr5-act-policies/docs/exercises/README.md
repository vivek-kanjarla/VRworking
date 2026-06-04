# from-scratch coding exercises

Implement the core math behind each policy yourself, in **PyTorch**, with nothing from
lerobot. Each exercise is a single file with `# TODO` stubs and a built-in self-check.

## how to use

```bash
# work on a stub, then run it — it self-checks:
python docs/exercises/01_attention.py

# see your overall progress:
python docs/exercises/check_all.py            # PASS / TODO / FAIL per file

# peek only after trying — tested reference solutions live here:
python docs/exercises/check_all.py --solutions
docs/exercises/solutions/01_attention.py
```

Each check passes by matching a **known invariant** (a torch built-in, a closed-form
identity, or a mathematical property) — not by comparing to the solution file, so you
can't cheat by importing it.

## the exercises

| # | file | concept | doc | which policies |
|---|------|---------|-----|----------------|
| 01 | `01_attention.py` | scaled dot-product + multi-head self-attention | [act.md](../act.md), [dit_flow.md](../dit_flow.md) | ACT, DiT, pi0 |
| 02 | `02_cvae_act.py` | reparameterization trick, KL, ACT's L1+KL loss | [act.md](../act.md) | ACT |
| 03 | `03_diffusion_policy.py` | cosine ᾱ schedule, forward noising, DDIM sampling | [diffusion_policy.md](../diffusion_policy.md) | Diffusion Policy |
| 04 | `04_flow_matching_actions.py` | CFM path, velocity target, Euler ODE | [dit_flow.md](../dit_flow.md), [pi0.md](../pi0.md) | DiT, pi0 |
| 05 | `05_dit_adaln.py` | a DiT block with AdaLN-Zero conditioning | [dit_flow.md](../dit_flow.md) | DiT, pi0 action expert |
| 06 | `06_fast_tokenizer.py` | DCT + quantize + tiny BPE (action tokenization) | [pi0_fast.md](../pi0_fast.md) | pi0_fast |
| 07 | `07_temporal_ensembling.py` | exponential blending of overlapping chunks | [act.md](../act.md) | ACT (inference) |

Suggested order: **01 → 02 → 07** (ACT family), then **03** (diffusion), then
**04 → 05** (flow + DiT), then **06** (tokenization).

## relation to your own notebooks

You've already built several of these from scratch — these exercises are deliberately
small and self-checking so you can re-derive the exact pieces and connect them to *this*
repo's policies. Cross-references to your existing work:

| exercise | your existing from-scratch work |
|---|---|
| 02 (CVAE), 07 (temporal ensembling) | `~/Desktop/Robotic-ML/ACT/act_implemetation.ipynb` (ACTEncoder/Decoder/Model, `TemporalEnsembler` with `m=0.01`) and `autoencoders.ipynb` (AE/VAE/CVAE) |
| 04 (flow matching) | `~/Desktop/Flow matching/` (RealNVP → CNF → flow matching → rectified flow, + your NOTES_*.md and COMPLETE_EXPLAINER.md) |
| 01, 05 (attention, DiT) | new — not in your notebooks yet |
| 03 (DDPM/DDIM) | new — your flow-matching work covers the ODE side; this is the SDE/score side |
| 06 (DCT + BPE) | new |

(Those notebooks live in other folders on this machine, outside this repo.)

## notes
- everything runs on CPU in seconds — no GPU, no lerobot, no dataset needed.
- shapes follow the FR5 convention: action chunks are `(B, T, 7)` (6 joints + gripper),
  state is 6-dim, matching the rest of the repo.
- the solutions are the *minimal* correct version; your own implementation may look
  different and still pass (the checks test behavior, not code).
