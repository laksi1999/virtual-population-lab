# Supplementary analysis scripts

Reproduce the analyses in `SUPPLEMENTARY.md` / `results/_summary/supp_*.md`. Run each
from the repository root:

| Script | Reproduces | What it shows |
|---|---|---|
| `python -m src.experiments.oracle_independent` | §A / S6.1 | On independent-Gaussian data, Physics-MC sits on the oracle floor and Physics-VAE ties it — Banana/Mango being co-best on fidelity is the ceiling the data allows, not a shortfall. |
| `python -m src.experiments.structure_floor` | §B / S6.1–S6.2 | Banana correlation-distance noise floor (real train-vs-test ≈ 0.28); per-group OLS slope/correlation heterogeneity for Date, Mango, Safou. |
| `python -m src.experiments.region_conditional` | §B / S6.2 | Region-conditional physics and single-region subsetting both underperform the pooled Physics-VAE (a bias–variance outcome at small per-group n). |
| `python -m src.experiments.banana_latent_sweep` | §B / S6.1 | No latent dimension recovers a Banana correlation "win" — the metric is pinned at the noise floor. |
| `python -m src.experiments.conformal_prototype` | §C / S5.4 | Standalone prototype of the conformal recalibration that ships in `src/evaluation/loro.py` (columns `coverage_*_conf`): fixes in-region coverage, halves the off-support gap. |

These are analysis/ablation scripts, separate from the main pipeline
(`make run` / `make loro` / `python -m src.multiseed`). They are CPU-feasible; the
ones that train ensembles (`region_conditional`, `conformal_prototype`,
`banana_latent_sweep`) take a few minutes.
