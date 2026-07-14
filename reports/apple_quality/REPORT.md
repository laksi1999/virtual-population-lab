# Virtual Population Lab — Report

_Auto-generated on 2026-07-14 01:29 from `apple_quality`. Re-run `make run CONFIG=apple_quality` to refresh._

## Objective

Compare modeling engines on how well each generates a synthetic population that preserves the real population's feature distributions, correlations, and uncertainty.

## Data

- Source file: `data/cleaned-scaled-apple-quality.csv`
- Source dataset: https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality
- Features (7): Size, Weight, Sweetness, Crunchiness, Juiciness, Ripeness, Acidity
- Rows after cleaning: 4000 — split into 2800 train / 1200 test (`TEST_SIZE=0.3`)

## Methodology

**Physics-Informed Monte Carlo** (`src.generators.physics_mc_generator.generate`)

Generates synthetic rows via forward Monte Carlo sampling over a
caller-supplied causal graph:

- Each variable in `root_variables` is drawn from its own real marginal
  distribution (assumed Gaussian).
- Each (parent, child) edge in `causal_graph` draws `child` from a
  linear relationship fit on the real data, plus residual noise sampled
  at that fit's real residual std.

Every conditional here is a Gaussian we can sample directly, so plain
ancestral Monte Carlo sampling (this function) is exact — there's no
intractable distribution to approximate, so no need for MCMC.

The causal structure and roots are dataset knowledge supplied by the
caller (see src/configs/apple_quality.py) — this function has no
dataset-specific assumptions baked in, so it works unchanged for a
different set of features/relationships.

**Regression** (`src.generators.regression_generator.generate`)

For each feature, fits one RandomForestRegressor (once, on train_df)
predicting it from the other features. Synthetic rows are built by
taking a base apple's other features from base_df — held out from
training, so predictions aren't leaking from rows the model has seen —
and adding noise bootstrapped from that model's real training residuals,
so outputs vary instead of collapsing to a deterministic point estimate.

**Variational Autoencoder** (`src.generators.vae_generator.generate`)

Trains on `x` (expected to already be reasonably scaled) and samples
`n_samples` new rows from the prior. Pass a fitted `scaler` only if `x`
needs to be inverse-transformed back to source units afterwards —
leave it None when the source data is already standardized, since
there's no original unscaled space to return to.

Loss is the standard VAE ELBO (reconstruction MSE plus the analytic KL
divergence between the approximate posterior N(mu, sigma^2) and the
standard normal prior, weighted by `beta`), plus a covariance-matching
term weighted by `cov_weight`: the squared difference between the
reconstructed batch's covariance matrix and the real batch's. The ELBO
alone has no direct incentive to get cross-feature correlations right —
it only rewards accurate per-point reconstruction — so this term gives
the model direct pressure toward the metric this project actually scores
correlation preservation on. `beta` is linearly annealed from 0 up to
its target value over the first `kl_warmup_frac` of training — without
this, the KL term can dominate before reconstruction has learned
anything, collapsing the model onto near-zero variance (i.e. generated
samples clustering tightly around the mean).

With `use_minibatch=True`, each epoch shuffles the data and takes one
gradient step per `batch_size` chunk instead of one step over the whole
dataset — the added stochastic noise can help escape a plateaued loss
that full-batch descent gets stuck in. Set `use_minibatch=False` for
very small datasets, where a mini-batch may be too small to estimate a
stable gradient at all.

Training stops early once the loss hasn't improved for `patience`
epochs, checked only after the KL warmup finishes — beta is still
rising during warmup, so the loss naturally shifts then regardless of
real convergence, and checking during that window would trigger a
spurious early stop.

`hidden_dim` and `dropout` control model capacity — the defaults (128,
0.0) fit apple_quality's 2800 rows comfortably, but on a small dataset
(dozens to low hundreds of rows) that many parameters can memorize
training noise instead of generalizing. Shrink `hidden_dim` and/or add
dropout (e.g. 0.1-0.3) if the generalization check shows a large
train/test gap.

`free_bits` floors the KL cost per latent dimension: once a dimension's
KL drops below this many nats, it stops contributing gradient to the KL
term (though reconstruction can still make it informative). Without
this (free_bits=0.0), KL annealing alone can still let every dimension
collapse fully to the prior on some datasets — recon_loss plateauing
at ~1.0 on standardized data (i.e. no better than always predicting the
mean) with KL near zero is the signature to watch for. Try 0.5-2.0 if
that happens.

## Results

| Engine | Correlation Distance (Euclidean) | Mean KS Statistic |
|---|---|---|
| Physics-Informed Monte Carlo | 0.9558 | **0.0319** |
| Regression | **0.5924** | 0.1127 |
| Variational Autoencoder | 0.7724 | 0.0825 |

(Lower is better for both metrics; bold = best.)

## Findings

- Best **Correlation Distance (Euclidean)**: Regression (0.5924)
- Best **Mean KS Statistic**: Physics-Informed Monte Carlo (0.0319)

Per-feature marginal fit (two-sample KS test, real vs. generated; lower ks_stat / higher p_value = closer):

- **Physics-Informed Monte Carlo**: 0/7 features statistically distinguishable from real (p < 0.05)
- **Regression**: 7/7 features statistically distinguishable from real (p < 0.05)
- **Variational Autoencoder**: 5/7 features statistically distinguishable from real (p < 0.05)

## Per-Feature KS Statistic

Lower = closer to real; bold = best per feature.

| Feature | Physics-Informed Monte Carlo | Regression | Variational Autoencoder |
|---|---|---|---|
| Size | **0.0267** | 0.1027 | 0.0723 |
| Weight | **0.0267** | 0.1275 | 0.0903 |
| Sweetness | **0.0352** | 0.0950 | 0.0615 |
| Crunchiness | **0.0330** | 0.1213 | 0.1415 |
| Juiciness | **0.0397** | 0.1315 | 0.1082 |
| Ripeness | **0.0258** | 0.0938 | 0.0530 |
| Acidity | **0.0365** | 0.1170 | 0.0503 |

## Feature Spread Comparison

**Correlation Distance (Euclidean)** is scale-invariant — Pearson correlation ignores absolute spread, so a method can score well on it while its population is uniformly shrunk or stretched relative to real. This shows each engine's per-feature standard deviation as a fraction of real's (1.00 = matches real exactly; well below 1.00 = generated population is narrower than real).

| Feature | Real Std | Physics-Informed Monte Carlo | Regression | Variational Autoencoder |
|---|---|---|---|---|
| Size | 1.933 | 1.886 (0.98x) | 1.330 (0.69x) | 1.630 (0.84x) |
| Weight | 1.609 | 1.567 (0.97x) | 1.047 (0.65x) | 1.248 (0.78x) |
| Sweetness | 1.923 | 2.029 (1.06x) | 1.435 (0.75x) | 1.725 (0.90x) |
| Crunchiness | 1.420 | 1.390 (0.98x) | 0.891 (0.63x) | 0.846 (0.60x) |
| Juiciness | 1.891 | 1.960 (1.04x) | 1.270 (0.67x) | 1.567 (0.83x) |
| Ripeness | 1.872 | 1.871 (1.00x) | 1.346 (0.72x) | 1.604 (0.86x) |
| Acidity | 2.128 | 2.153 (1.01x) | 1.438 (0.68x) | 1.895 (0.89x) |

## Generalization Check

Correlation distance for each engine's synthetic data against the train split it was fit on vs. the held-out test split it's actually scored on. A small gap is evidence results generalize rather than matching training-set noise a real (unseen) population wouldn't share.

| Engine | Correlation Dist. (vs. Train) | Correlation Dist. (vs. Test) | Gap |
|---|---|---|---|
| Physics-Informed Monte Carlo | 0.9452 | 0.9558 | 0.0106 |
| Regression | 0.5550 | 0.5924 | 0.0374 |
| Variational Autoencoder | 0.7561 | 0.7724 | 0.0164 |

## Figures

### Joint variability (correlation)

**Correlation matrices**

![Real correlation matrix](../../results/apple_quality/figures/correlation/correlation_real.png)
![Physics-Informed Monte Carlo correlation matrix](../../results/apple_quality/figures/correlation/correlation_physics_mc.png)
![Regression correlation matrix](../../results/apple_quality/figures/correlation/correlation_regression.png)
![Variational Autoencoder correlation matrix](../../results/apple_quality/figures/correlation/correlation_vae.png)

**PCA projection — real vs. each method, plus all combined**

![PCA projection of real vs generated populations](../../results/apple_quality/figures/correlation/pca_real_vs_generated.png)

**PCA projection — each population shown individually, no overlay**

![PCA projection of each population individually](../../results/apple_quality/figures/correlation/pca_individual.png)

### Marginal distributions (KS)

**KS statistic by feature and method**

![KS statistic heatmap](../../results/apple_quality/figures/marginals/ks_heatmap.png)

**Empirical CDFs — real vs. generated, per feature**

![ECDF overlay per feature](../../results/apple_quality/figures/marginals/ecdf_overlay.png)

**P-value by feature and method**

![P-value heatmap](../../results/apple_quality/figures/marginals/pvalue_heatmap.png)

**Volcano plot — effect size vs. significance**

![Volcano plot](../../results/apple_quality/figures/marginals/volcano_plot.png)

---

Raw data: `results/apple_quality/summary_metrics.csv`, `results/apple_quality/ks_marginals.csv`, `results/apple_quality/generalization_gap.csv`, `results/apple_quality/synthetic_data/`.
