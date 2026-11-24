# grokking-progress-measures

Grokking on modular addition `(a + b) mod p` with a 1-layer transformer, following Nanda et al. 2023 (*Progress measures for grokking via mechanistic interpretability*). The repo covers:

- a reproduction;
- weight-decay × data-fraction sweeps;
- **progress measures** that move long before test accuracy does:
  - Fourier sparsity of the token embedding, i.e. the share of norm in the top-5 frequencies;
  - total weight norm.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kishorhange111/grokking-progress-measures/blob/main/notebooks/colab_sweep.ipynb)

## Pilot (CPU, 25k steps, single seed)

The pilot used p = 59, 30% of pairs for training, weight decay 1.0, AdamW lr 1e-3 and full-batch updates (`results/single_p59_frac0.3_wd1.0_seed0.*`).

| step | train acc | test acc | top-5 Fourier share of W_E | weight norm² |
|---|---|---|---|---|
| 0 | 0.011 | 0.018 | 0.20 | 468 |
| 2,500 | **1.000** | 0.025 | 0.32 | 1,855 |
| 10,000 | 1.000 | 0.041 | 0.40 | 2,206 |
| 17,500 | 1.000 | 0.096 | 0.49 | 1,976 |
| 25,000 | 1.000 | **0.243** | **0.61** | 1,848 |

- **Memorisation is immediate.** Training accuracy reaches 99% at step 200.
- **The Fourier progress measure leads test accuracy.** Between steps 2.5k and 25k, Fourier sparsity rises almost monotonically (0.32 → 0.61) while test accuracy is still near chance (2.5% → 24%). The embedding is reorganising around a few frequencies (7, 12, 20, 21, 23) well before the generalising circuit shows up in accuracy. This is the "hidden progress" Nanda et al. describe.
- **The run had not generalised by 25k steps**, and the weight norm is still near its peak rather than in the clean-up phase. The long run and the multi-seed sweep in the Colab notebook test whether this run eventually groks and how the delay depends on weight decay and data fraction.

This is a single seed on CPU, so treat it as a pilot, not a result.

## Run

All experiments run on **Google Colab** (badge above). The notebook runs:

- the sweep: 3 train fractions × 2 weight decays × 3 seeds, four processes sharing one GPU, resumable via `runs.jsonl` on Drive;
- a 100k-step single run.

Locally (CPU), for development:

```
pip install -e ".[dev]"
pytest -q
python -m grok.cli single --p 59 --frac 0.3 --wd 1.0 --steps 2000 --device cpu
python -m grok.cli sweep --fracs 0.3 0.5 --wds 0.3 1.0 --seeds 0 --device auto --workers 2
```
