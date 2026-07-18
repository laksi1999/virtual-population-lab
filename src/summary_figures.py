"""
Cross-dataset summary figures for the 4 finalized datasets. Reads the per-dataset
result CSVs (results/<cfg>/*.csv) so the figures stay in sync with the runs, and
writes publication PNGs to results/_summary/.

  fig1_leaderboard.png  — 4 engines x 4 datasets on correlation distance + KS
  fig2_downstream.png   — TSTR utility per engine vs the real ceiling (apple, banana)
  fig3_nearfar.png      — near/far ensemble disagreement (uncertainty widens off-support)

Palette: Okabe-Ito (colourblind-safe, validated). Hybrid = vermillion (hero).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/_summary"
os.makedirs(OUT, exist_ok=True)

# fixed engine order + validated categorical colours (entity -> colour, never rank)
ENGINES = ["Physics-Informed Monte Carlo", "Regression",
           "Variational Autoencoder", "Physics-Informed VAE (Hybrid)"]
SHORT = {"Physics-Informed Monte Carlo": "Physics-MC", "Regression": "Regression",
         "Variational Autoencoder": "VAE", "Physics-Informed VAE (Hybrid)": "Hybrid"}
COLOR = {"Physics-Informed Monte Carlo": "#0072B2", "Regression": "#E69F00",
         "Variational Autoencoder": "#009E73", "Physics-Informed VAE (Hybrid)": "#D55E00"}
KEYCOLOR = {"physics_mc": "#0072B2", "regression": "#E69F00",
            "vae": "#009E73", "hybrid_vae": "#D55E00", "real": "#555555"}
DATASETS = {"apple_quality": "Apple\n(quality)", "banana_quality": "Banana\n(quality)",
            "biofood_date_region": "Date\n(nutrients)", "biofood_safou_region": "Safou\n(nutrients)"}

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
})


def _bars(ax, groups, series_vals, title, ylabel, note):
    """Grouped bars: groups on x, ENGINES within each group. Direct value labels."""
    n = len(ENGINES); gw = 0.8; bw = gw / n
    x = np.arange(len(groups))
    for i, eng in enumerate(ENGINES):
        vals = series_vals[eng]
        pos = x - gw / 2 + bw * (i + 0.5)
        ax.bar(pos, vals, bw * 0.86, color=COLOR[eng], label=SHORT[eng], zorder=3)
        for p, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(p, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5,
                        color="#333", rotation=0)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12, weight="bold", loc="left", pad=8)
    ax.margins(y=0.16)


# ---------- Figure 1: leaderboard ----------
corr, ks = {e: [] for e in ENGINES}, {e: [] for e in ENGINES}
for cfg in DATASETS:
    sm = pd.read_csv(f"results/{cfg}/summary_metrics.csv").set_index("Engine")
    for e in ENGINES:
        corr[e].append(sm.loc[e, "Correlation Distance (Euclidean)"] if e in sm.index else np.nan)
        ks[e].append(sm.loc[e, "Mean KS Statistic"] if e in sm.index else np.nan)
labels = list(DATASETS.values())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4))
_bars(a1, labels, corr, "Joint structure  (lower is better)", "Correlation distance", "")
_bars(a2, labels, ks, "Marginal fit  (lower is better)", "Mean KS statistic", "")
handles, leg = a1.get_legend_handles_labels()
fig.legend(handles, leg, frameon=False, ncol=4, loc="upper center",
           bbox_to_anchor=(0.5, 0.925), fontsize=10)
fig.suptitle("Engine leaderboard across the four datasets  —  hybrid best or co-best throughout",
             x=0.02, ha="left", fontsize=14, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig(f"{OUT}/fig1_leaderboard.png", dpi=300, bbox_inches="tight"); plt.close()

# ---------- Figure 2: downstream TSTR ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, (cfg, metric, mlabel, real_ceiling_label) in zip(
        axes, [("apple_quality", "roc_auc", "ROC AUC", "real"),
               ("banana_quality", "accuracy", "Accuracy", "real")]):
    t = pd.read_csv(f"results/{cfg}/tstr.csv")
    order = ["real", "physics_mc", "regression", "vae", "hybrid_vae"]
    t = t[t.method.isin(order)].set_index("method").reindex(order).dropna(subset=[metric])
    names = {"real": "Real (ceiling)", "physics_mc": "Physics-MC", "regression": "Regression",
             "vae": "VAE", "hybrid_vae": "Hybrid"}
    xs = np.arange(len(t))
    cols = [KEYCOLOR[m] for m in t.index]
    ax.bar(xs, t[metric], 0.62, color=cols, zorder=3)
    for xi, v in zip(xs, t[metric]):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color="#333")
    ceil = t.loc["real", metric]
    ax.axhline(ceil, color="#555555", lw=1.2, ls="--", zorder=2)
    ax.set_xticks(xs); ax.set_xticklabels([names[m] for m in t.index], rotation=20, ha="right")
    ax.set_ylabel(mlabel); ax.set_title(DATASETS[cfg].replace("\n", " "), fontsize=12, weight="bold", loc="left")
    ax.set_ylim(min(t[metric]) * 0.9, ceil * 1.04)
fig.suptitle("Downstream utility (train-on-synthetic, test-on-real) — hybrid nearest the real ceiling",
             x=0.02, ha="left", fontsize=13, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{OUT}/fig2_downstream.png", dpi=300, bbox_inches="tight"); plt.close()

# ---------- Figure 3: near/far uncertainty ----------
groups, far, near, pct = [], [], [], []
for cfg in ["banana_quality", "biofood_date_region", "biofood_safou_region"]:
    lo = pd.read_csv(f"results/{cfg}/loro.csv")
    m = lo[lo.group == "MEAN"].iloc[0]
    groups.append(DATASETS[cfg].replace("\n", " "))
    far.append(m["disagreement_far"]); near.append(m["disagreement_near"])
    pct.append(100 * (m["disagreement_far"] / m["disagreement_near"] - 1))
fig, ax = plt.subplots(figsize=(8.5, 5))
x = np.arange(len(groups)); bw = 0.38
ax.bar(x - bw / 2, far, bw * 0.9, color="#D55E00", label="FAR (different region)", zorder=3)
ax.bar(x + bw / 2, near, bw * 0.9, color="#0072B2", label="NEAR (held-out same region)", zorder=3)
for xi, f, n, p in zip(x, far, near, pct):
    ax.text(xi - bw / 2, f, f"{f:.3f}", ha="center", va="bottom", fontsize=8.5, color="#333")
    ax.text(xi + bw / 2, n, f"{n:.3f}", ha="center", va="bottom", fontsize=8.5, color="#333")
    ax.text(xi, max(f, n) * 1.10, f"+{p:.0f}%", ha="center", va="bottom", fontsize=11,
            weight="bold", color="#D55E00")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("Ensemble disagreement (per-feature-std)")
ax.margins(y=0.2); ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.set_title("Support-aware uncertainty: disagreement widens for an unseen region\n"
             "(same model, held-out same-region vs different-region)", fontsize=12.5, weight="bold", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_nearfar.png", dpi=300, bbox_inches="tight"); plt.close()

print("wrote:", os.listdir(OUT))
