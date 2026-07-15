# Insights — Virtual Food Populations

A feasibility read on the `apple_quality` run (4000 real apples, 7 quality features). Each engine generates a synthetic **virtual food population**; we score how well each preserves the real population's **marginals** (do individual features match?) and **joint structure** (do features co-vary like real?). Full numbers: [reports/apple_quality/REPORT.md](reports/apple_quality/REPORT.md).

## Results

| Engine | Correlation Distance ↓ | Mean KS ↓ | Marginals matching real | Mean spread vs real | Train/Test gap ↓ |
|---|---|---|---|---|---|
| Physics-Informed MC | 0.9558 (worst) | **0.0319 (best)** | 7 / 7 | 1.005 | **0.0106** |
| Regression | 0.5924 | 0.1127 (worst) | 0 / 7 | 0.683 (worst) | 0.0374 (worst) |
| VAE | 0.7725 | 0.0825 | 2 / 7 | 0.813 | 0.0164 |
| **Physics-Informed VAE (Hybrid)** | **0.4606 (best)** | 0.0471 (2nd) | 4 / 7 | 0.928 | 0.0209 |

## The story the evidence tells

No single engine is best at everything — but the four together tell a clear, layered story:

- **Physics-informed Monte Carlo** remains the **state-of-the-art for faithfully reproducing a *known* food population** — best marginals (7/7 indistinguishable from real) and near-perfect variability (1.005). Its cost: it needs a hand-supplied causal graph, and it has the *worst* joint structure, because a fixed linear-Gaussian chain can't recover the full correlation web.
- **Regression** preserves parts of the correlation structure but clearly **underestimates population variability** (spread collapses to 0.68×) and fails every marginal.
- **The VAE** is the **best purely data-driven generative approach** — the best balance of realistic marginals and multivariate structure, learned with *no* handcrafted assumptions, plus capabilities the statistical methods lack: a continuous latent representation, novel-individual generation, conditional sampling, and a path to digital-twin integration.
- **The hybrid physics-informed VAE combines both strengths and comes out on top.** By injecting the causal graph into the VAE as a physics-consistency loss, it takes the **best correlation distance of all four (0.4606)** while pulling marginals up to 2nd place (0.0471, just behind physics-MC) — all still learned end-to-end.

## Why the hybrid is the standout

Adding the mechanistic constraint fixed the plain VAE's two weaknesses without giving up its data-driven nature:

1. **Best joint structure of any method.** Correlation distance drops from 0.77 (VAE) to **0.46** — better even than Regression, the previous best.
2. **Variance shrinkage largely repaired.** Mean spread rises from 0.81× (VAE) to **0.93×**. The clearest case: **Crunchiness**, the plain VAE's worst feature, goes from 0.60×→**0.95×** spread and from KS 0.141 (clearly distinguishable) to **0.032 (now statistically indistinguishable from real)** — the physics term defends spread instead of collapsing it, exactly as intended.
3. **More faithful marginals.** 4 of 7 features are now statistically indistinguishable from real (Sweetness, Crunchiness, Juiciness, Acidity), up from 2/7.
4. **Still generalizes.** Train→test gap of 0.0209 (test actually slightly *better* than train) — no overfitting.

**And this is with minimal effort** — essentially stock settings, only two knobs touched ([src/configs/apple_quality.py](src/configs/apple_quality.py#L43-L44)), one physics term at default weight 1.0. Headroom, not a tuned ceiling.

## The message

Physics-informed Monte Carlo is today's best method for **reproducing a known food population**, but it doesn't scale or extend. Deep generative models are the **future direction** for scalable, extensible virtual food populations — and the results show the way forward is **hybrid**: the VAE is the strongest purely data-driven solution today, while the physics-informed VAE, which integrates mechanistic constraints with a deep generative model, is the best of both worlds and the natural next generation.

## Caveats / next steps

- Physics-MC still edges the hybrid on *pure* marginal KS (0.0319 vs 0.0471) — worth stating plainly; the hybrid's win is on the harder joint-structure axis plus balance.
- Only `apple_quality` has run. `make run CONFIG=mango_ripeness` and `make run CONFIG=banana_quality` would make this a cross-food result — mango (~100 rows) is the real test of whether the hybrid holds up on small data.
- The physics term (`VAE_PHYSICS_WEIGHT` in [src/configs/base.py](src/configs/base.py)) is untuned at 1.0 — a short sweep could push it further.
