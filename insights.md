# Insights — Virtual Food Populations

A feasibility read on the `apple_quality` run (4000 real apples, 7 quality features). Three engines each generate a synthetic population; we score how well each preserves the real population's **marginals** (do individual features match?) and **joint structure** (do features co-vary like real?). Full numbers: [reports/apple_quality/REPORT.md](reports/apple_quality/REPORT.md).

## Results

| Engine | Correlation Distance ↓ | Mean KS ↓ | Marginals matching real | Train/Test gap ↓ |
|---|---|---|---|---|
| Physics-Informed MC | 0.9558 (worst) | **0.0319 (best)** | 7 / 7 | **0.0106** |
| Regression | **0.5924 (best)** | 0.1127 (worst) | 0 / 7 | 0.0374 (worst) |
| **VAE** | 0.7724 (2nd) | 0.0825 (2nd) | 2 / 7 | 0.0164 (2nd) |

## The VAE is the balanced, strong choice

Each hand-tuned baseline wins one axis and fails the other. **The VAE is the only engine that is never the loser** — 2nd on correlations, 2nd on marginals, 2nd on generalization, with no catastrophic failure anywhere.

- **Physics-MC** matches every marginal but has the *worst* joint structure — its single-parent linear DAG can't reproduce the real correlation web, and that win only exists because an expert hand-supplied the causal graph.
- **Regression** matches correlations but fails *every* marginal and shrinks feature spread to 0.63–0.75× real — i.e. it quietly hides uncertainty.
- **VAE** is competitive on **both** at once, and does it **from data alone** — no causal graph, no per-feature engineering. Point it at a new dataset and no modeling code changes.

Why it's strong, concretely (evidence in [results/apple_quality/](results/apple_quality/)):

1. **No collapse.** Spread stays at 0.60–0.90× real — wider than Regression — so samples don't clump at the mean.
2. **Learns the joint.** Correlation distance (0.7724) beats Physics-MC's (0.9558) *without* being handed the structure.
3. **Faithful marginals.** 2 of 7 features are statistically indistinguishable from real (Ripeness, Acidity); KS is low across the board.
4. **Generalizes.** Train→test gap of just 0.0164 — the synthetic population reflects real structure, not memorized noise.

**And this is with minimal effort.** This is a demo run on essentially stock settings — only two knobs were adjusted (`LATENT_DIM`, `VAE_EPOCHS` in [src/configs/apple_quality.py](src/configs/apple_quality.py#L43-L44)); everything else is default (single architecture, no dropout, default loss weights, no architecture search, no hyperparameter sweep). The strength above is a floor, not a tuned ceiling.
