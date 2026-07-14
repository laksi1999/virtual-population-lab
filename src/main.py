import inspect
import logging
import os
import re
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config_loader import DEFAULT_CONFIG, load_config
from src.data_loading import load_data
from src.evaluation.evaluate import display_name, generalization_gap, metric_display_name, run_evaluation
from src.generators import physics_mc_generator, regression_generator, vae_generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("vp-lab")

CONFIG_NAME = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
config = load_config(CONFIG_NAME)

# Namespaced by config so results/reports from different datasets never
# clobber each other.
RESULTS_DIR = f"results/{CONFIG_NAME}"
REPORT_PATH = f"reports/{CONFIG_NAME}/REPORT.md"


def log_config():
    log.info("Active config: %s", CONFIG_NAME)
    for key in dir(config):
        if key.isupper():
            log.info("  %s = %r", key, getattr(config, key))


BOLD_GREEN = "\033[1;32m"
RESET = "\033[0m"
ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _visible_len(s):
    return len(ANSI_RE.sub("", s))


def print_summary(summary):
    """Prints `summary` as a bordered ASCII table, with the best (lowest) value
    per metric column marked with '*' and, in an actual terminal, colored."""
    metrics = [c for c in summary.columns if c != "method"]
    color = sys.stdout.isatty()

    headers = ["Engine"] + [metric_display_name(m) for m in metrics]
    rows = []
    for _, row in summary.iterrows():
        cells = [display_name(row["method"])]
        for metric in metrics:
            is_best = row[metric] == summary[metric].min()
            text = f"{'*' if is_best else ' '}{row[metric]:.4f}"
            if is_best and color:
                text = f"{BOLD_GREEN}{text}{RESET}"
            cells.append(text)
        rows.append(cells)

    widths = [
        max(_visible_len(h), *(_visible_len(r[i]) for r in rows))
        for i, h in enumerate(headers)
    ]

    def format_row(cells):
        padded = []
        for i, (c, w) in enumerate(zip(cells, widths)):
            fill = " " * (w - _visible_len(c))
            padded.append(c + fill if i == 0 else fill + c)  # text left, numbers right
        return "| " + " | ".join(padded) + " |"

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    print(sep)
    print(format_row(headers))
    print(sep)
    for r in rows:
        print(format_row(r))
    print(sep)
    print("* = best (lowest) for that metric")


def write_report(df, train_df, test_df, summary, ks_table, gap_df, std_table, engines):
    """
    Rewrites reports/REPORT.md from this run's actual data and outputs.
    `engines`: dict of {key: (module, generated_df)} — methodology text comes
    straight from each module's generate() docstring instead of being
    duplicated here, so it can't drift from the code.
    """
    metrics = [c for c in summary.columns if c != "method"]
    lines = [
        "# Virtual Population Lab — Report",
        "",
        f"_Auto-generated on {datetime.now():%Y-%m-%d %H:%M} from `{CONFIG_NAME}`. "
        f"Re-run `make run CONFIG={CONFIG_NAME}` to refresh._",
        "",
        "## Objective",
        "",
        "Compare modeling engines on how well each generates a synthetic population "
        "that preserves the real population's feature distributions, correlations, "
        "and uncertainty.",
        "",
        "## Data",
        "",
        f"- Source file: `{config.DATA_PATH}`",
    ]

    if config.DATA_SOURCE_URL:
        lines.append(f"- Source dataset: {config.DATA_SOURCE_URL}")

    lines += [
        f"- Features ({len(config.FEATURES)}): {', '.join(config.FEATURES)}",
        f"- Rows after cleaning: {len(df)} — split into {len(train_df)} train / {len(test_df)} test "
        f"(`TEST_SIZE={config.TEST_SIZE}`)",
        "",
        "## Methodology",
        "",
    ]

    for key, (module, _) in engines.items():
        doc = inspect.getdoc(module.generate) or ""
        lines += [f"**{display_name(key)}** (`{module.__name__}.generate`)", "", doc, ""]

    lines += ["## Results", ""]
    lines.append("| Engine | " + " | ".join(metric_display_name(m) for m in metrics) + " |")
    lines.append("|" + "---|" * (len(metrics) + 1))
    for _, row in summary.iterrows():
        cells = [display_name(row["method"])]
        for metric in metrics:
            text = f"{row[metric]:.4f}"
            cells.append(f"**{text}**" if row[metric] == summary[metric].min() else text)
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "(Lower is better for both metrics; bold = best.)", "", "## Findings", ""]

    for metric in metrics:
        best_row = summary.loc[summary[metric].idxmin()]
        lines.append(f"- Best **{metric_display_name(metric)}**: {display_name(best_row['method'])} ({best_row[metric]:.4f})")

    lines += [
        "",
        "Per-feature marginal fit (two-sample KS test, real vs. generated; "
        "lower ks_stat / higher p_value = closer):",
        "",
    ]
    for key in engines:
        sub = ks_table[ks_table.method == key]
        significant = int((sub.p_value < 0.05).sum())
        lines.append(f"- **{display_name(key)}**: {significant}/{len(sub)} features "
                      "statistically distinguishable from real (p < 0.05)")

    ks_pivot = ks_table.pivot(index="feature", columns="method", values="ks_stat")
    ks_pivot = ks_pivot.reindex(index=config.FEATURES, columns=list(engines))
    lines += [
        "",
        "## Per-Feature KS Statistic",
        "",
        "Lower = closer to real; bold = best per feature.",
        "",
    ]
    lines.append("| Feature | " + " | ".join(display_name(k) for k in engines) + " |")
    lines.append("|" + "---|" * (len(engines) + 1))
    for feature in config.FEATURES:
        row = ks_pivot.loc[feature]
        best_key = row.idxmin()
        cells = [f"**{row[k]:.4f}**" if k == best_key else f"{row[k]:.4f}" for k in engines]
        lines.append(f"| {feature} | " + " | ".join(cells) + " |")

    std_pivot = std_table.pivot(index="feature", columns="method", values="generated_std")
    std_pivot = std_pivot.reindex(index=config.FEATURES, columns=list(engines))
    ratio_pivot = std_table.pivot(index="feature", columns="method", values="ratio_to_real")
    ratio_pivot = ratio_pivot.reindex(index=config.FEATURES, columns=list(engines))
    real_std = std_table.drop_duplicates("feature").set_index("feature")["real_std"]

    lines += [
        "",
        "## Feature Spread Comparison",
        "",
        f"**{metric_display_name('correlation_euclidean_dist')}** is scale-invariant — Pearson correlation "
        "ignores absolute spread, so a method can score well on it while its population is uniformly "
        "shrunk or stretched relative to real. This shows each engine's per-feature standard deviation as "
        "a fraction of real's (1.00 = matches real exactly; well below 1.00 = generated population is "
        "narrower than real).",
        "",
        "| Feature | Real Std | " + " | ".join(display_name(k) for k in engines) + " |",
        "|---|---|" + "---|" * len(engines),
    ]
    for feature in config.FEATURES:
        cells = [
            f"{std_pivot.loc[feature, k]:.3f} ({ratio_pivot.loc[feature, k]:.2f}x)"
            for k in engines
        ]
        lines.append(f"| {feature} | {real_std[feature]:.3f} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Generalization Check",
        "",
        "Correlation distance for each engine's synthetic data against the train split it was fit on "
        "vs. the held-out test split it's actually scored on. A small gap is evidence results generalize "
        "rather than matching training-set noise a real (unseen) population wouldn't share.",
        "",
        "| Engine | Correlation Dist. (vs. Train) | Correlation Dist. (vs. Test) | Gap |",
        "|---|---|---|---|",
    ]
    for _, row in gap_df.iterrows():
        lines.append(
            f"| {display_name(row['method'])} | {row['train_correlation_dist']:.4f} | "
            f"{row['test_correlation_dist']:.4f} | {row['gap']:.4f} |"
        )

    # REPORT.md lives at reports/<config>/REPORT.md, two directories deep,
    # so figures under results/<config>/... need ../../ to get back to root.
    rel_results = f"../../{RESULTS_DIR}"

    lines += ["", "## Figures", "", "### Joint variability (correlation)", "",
              "**Correlation matrices**", "",
              f"![Real correlation matrix]({rel_results}/figures/correlation/correlation_real.png)"]
    for key in engines:
        lines.append(f"![{display_name(key)} correlation matrix]"
                      f"({rel_results}/figures/correlation/correlation_{key}.png)")
    lines += [
        "",
        "**PCA projection — real vs. each method, plus all combined**",
        "",
        "![PCA projection of real vs generated populations]"
        f"({rel_results}/figures/correlation/pca_real_vs_generated.png)",
        "",
        "**PCA projection — each population shown individually, no overlay**",
        "",
        f"![PCA projection of each population individually]({rel_results}/figures/correlation/pca_individual.png)",
        "",
        "### Marginal distributions (KS)",
        "",
        "**KS statistic by feature and method**",
        "",
        f"![KS statistic heatmap]({rel_results}/figures/marginals/ks_heatmap.png)",
        "",
        "**Empirical CDFs — real vs. generated, per feature**",
        "",
        f"![ECDF overlay per feature]({rel_results}/figures/marginals/ecdf_overlay.png)",
        "",
        "**P-value by feature and method**",
        "",
        f"![P-value heatmap]({rel_results}/figures/marginals/pvalue_heatmap.png)",
        "",
        "**Volcano plot — effect size vs. significance**",
        "",
        f"![Volcano plot]({rel_results}/figures/marginals/volcano_plot.png)",
        "",
        "---",
        "",
        f"Raw data: `{RESULTS_DIR}/summary_metrics.csv`, `{RESULTS_DIR}/ks_marginals.csv`, "
        f"`{RESULTS_DIR}/generalization_gap.csv`, `{RESULTS_DIR}/synthetic_data/`.",
    ]

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    log_config()

    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    log.info("Loading data from %s", config.DATA_PATH)
    df = load_data(config)
    log.info("Loaded %d clean rows", len(df))

    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    log.info("Split into %d train / %d test rows", len(train_df), len(test_df))

    log.info("Running physics-informed Monte Carlo generator...")
    physics_mc_generated = physics_mc_generator.generate(
        train_df, config.FEATURES, config.CAUSAL_GRAPH, config.ROOT_VARIABLES,
        n_samples=config.N_SAMPLES,
    )
    log.info("Physics-informed MC: generated %d rows", len(physics_mc_generated))

    log.info("Running regression generator...")
    regression_generated = regression_generator.generate(
        train_df, test_df, config.FEATURES,
        n_samples=config.N_SAMPLES,
    )
    log.info("Regression: generated %d rows", len(regression_generated))

    if config.IS_PRE_SCALED:
        vae_input = train_df[config.FEATURES].values
        vae_scaler = None
    else:
        vae_scaler = StandardScaler().fit(train_df[config.FEATURES])
        vae_input = vae_scaler.transform(train_df[config.FEATURES])

    log.info("Training VAE generator...")
    vae_generated = vae_generator.generate(
        vae_input,
        config.FEATURES,
        latent_dim=config.LATENT_DIM,
        epochs=config.VAE_EPOCHS,
        beta=config.VAE_BETA,
        n_samples=config.N_SAMPLES,
        scaler=vae_scaler,
        use_minibatch=config.VAE_USE_MINIBATCH,
        batch_size=config.VAE_BATCH_SIZE,
        cov_weight=config.VAE_COV_WEIGHT,
        patience=config.VAE_PATIENCE,
        hidden_dim=config.VAE_HIDDEN_DIM,
        dropout=config.VAE_DROPOUT,
        free_bits=config.VAE_FREE_BITS,
    )
    log.info("VAE: generated %d rows", len(vae_generated))

    engines = {
        "physics_mc": (physics_mc_generator, physics_mc_generated),
        "regression": (regression_generator, regression_generated),
        "vae": (vae_generator, vae_generated),
    }
    generated = {key: gen_df for key, (_, gen_df) in engines.items()}

    os.makedirs(f"{RESULTS_DIR}/synthetic_data", exist_ok=True)
    for name, gen_df in generated.items():
        gen_df.to_csv(f"{RESULTS_DIR}/synthetic_data/{name}.csv", index=False)
    log.info("Saved synthetic data to %s/synthetic_data/", RESULTS_DIR)

    log.info("Running evaluation...")
    summary, ks_table, std_table = run_evaluation(
        test_df, generated, config.FEATURES,
        figures_dir=f"{RESULTS_DIR}/figures",
        results_dir=RESULTS_DIR,
    )
    log.info("Saved figures and metrics to %s/", RESULTS_DIR)

    print_summary(summary)

    log.info(
        "Checking generalization — correlation distance against the train split each engine "
        "was fit on vs. the held-out test split it's actually scored on (small gap = generalizes; "
        "train much better than test = overfitting signal)..."
    )
    gap_df = generalization_gap(train_df, test_df, generated, config.FEATURES)
    gap_df.to_csv(f"{RESULTS_DIR}/generalization_gap.csv", index=False)
    for _, row in gap_df.iterrows():
        log.info(
            "  %-30s corr_dist: train=%.4f  test=%.4f  gap=%.4f",
            display_name(row["method"]), row["train_correlation_dist"],
            row["test_correlation_dist"], row["gap"],
        )

    log.info(
        "Checking feature spread — each engine's per-feature std dev as a ratio of real's, "
        "averaged across all features (1.00 = matches real's spread exactly; well below 1.00 = "
        "generated population is narrower than real; this is independent of %s, "
        "which is scale-invariant and can't see this on its own)...",
        metric_display_name("correlation_euclidean_dist"),
    )
    mean_std_ratio = std_table.groupby("method")["ratio_to_real"].mean()
    for name in engines:
        log.info("  %-30s mean std ratio=%.3f", display_name(name), mean_std_ratio[name])

    write_report(df, train_df, test_df, summary, ks_table, gap_df, std_table, engines)
    log.info("Wrote %s", REPORT_PATH)


if __name__ == "__main__":
    main()
