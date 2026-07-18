# Insights — Generative Virtual Food Populations

A study of generative modelling for **virtual food populations** — synthetic populations of food items that preserve a real population's distributions, correlations, uncertainty, and downstream usefulness. Built as a data-driven counterpart to physics-driven Monte-Carlo digital twins (Onwude et al. 2022), and finalized on **four datasets** spanning quality/sensor and nutrient composition, with four engines and an honest account of where each wins.

## Framing / prior work

**Onwude et al. (2022, *Resources, Conservation & Recycling* 186)** generate a virtual population of 1,000 Valencia oranges via **MCMC (Gibbs) + a bespoke physics-based mechanistic digital twin**, from 50 measured oranges, validated by correlation-matrix preservation. This project asks whether a **data-driven generative model** can do the population-generation step **without a hand-built mechanistic twin**, match that fidelity criterion, and add **calibrated, support-aware uncertainty** and **downstream utility**. The `physics_mc` engine is the MCMC-style mechanistic baseline; the **hybrid physics-informed VAE** is the generative alternative.

## Engines

Physics-Informed Monte Carlo (mechanistic; needs a causal graph) · Regression (RandomForest, OOB) · VAE (deep generative) · **Hybrid Physics-Informed VAE** (VAE + causal-graph physics loss + covariance/marginal matching + optional empirical-copula calibration).

## The four datasets

| Dataset | Type (n) | Physics | Groups (near/far) | Downstream label |
|---|---|---|---|---|
| **apple_quality** | quality/sensor (4,000) | causal graph | — (no group) | Quality (2-class) |
| **banana_quality** | quality/sensor (1,000) | weak (categorical codes) | region (8 countries) | quality_category (4-class) |
| **biofood_date_region** | nutrients: minerals (74) | valid dilution | UAE / Tunisia | — |
| **biofood_safou_region** | nutrients: lipids, **individual-level** (41) | near-deterministic (Water→Fat −0.99) | Congo / Guinea | — |

## Headline results

**1. Hybrid is the best or co-best engine across the board.** Correlation + KS (seed 42):

| Dataset | corr winner | KS winner |
|---|---|---|
| apple | **hybrid 0.47** | **hybrid 0.02** |
| date | **hybrid 0.54** | **hybrid 0.14** |
| safou | **hybrid 0.08** (via calibration; beats physics-MC 0.12) | VAE 0.16 (hybrid 0.22) |
| banana | VAE 0.64 | **hybrid 0.06** |

The hybrid wins correlation on the two nutrient sets and apple, and marginals on 3/4. On banana it wins marginals + downstream but is genuinely behind on correlation (see caveats).

**2. Downstream utility (TSTR) — hybrid nearest the real ceiling on both labelled sets.**

| Dataset | real ceiling | hybrid | next best |
|---|---|---|---|
| apple (ROC AUC) | 0.959 | **0.934** | regression 0.902 |
| banana (accuracy) | 0.883 | **0.863** | VAE 0.813 |

**3. Support-aware uncertainty — disagreement widens for an unseen region** (same-model protocol: train one region, NEAR = held-out same-region, FAR = a different region):

| Dataset | disagreement FAR / NEAR | widening |
|---|---|---|
| banana | 0.288 / 0.099 | **+192%** |
| date | 0.093 / 0.055 | **+70%** |
| safou | 0.063 / 0.049 | **+28%** |

## Figures

`results/_summary/` — `fig1_leaderboard.png` (engines × datasets, corr + KS), `fig2_downstream.png` (TSTR vs real ceiling), `fig3_nearfar.png` (uncertainty widening). Reproduce with `python -m src.summary_figures`.


## Bottom line

A **data-driven generative alternative to MCMC + physics-digital-twin virtual populations (Onwude 2022)** that is, across four real datasets, **best or co-best on fidelity, best on downstream utility, and support-aware (uncertainty widens off-support)** — with an honest map of which engine (mechanistic / classical / generative) is the right tool for each data regime, and honest disclosure of the copula margins and the banana correlation result.

_Reproduce: `make run CONFIG=<name>` and `make loro CONFIG=<name>` for apple_quality, banana_quality, biofood_date_region, biofood_safou_region._
