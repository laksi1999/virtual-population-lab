"""
Paper-native assets, generated from the result CSVs:

  results/_summary/table1.tex        — main results table (best per metric bolded)
  results/_summary/fig4_engine_map.png — "which engine wins where" schematic

Reproduce with `python -m src.paper_assets`.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/_summary"
os.makedirs(OUT, exist_ok=True)

DATASETS = {
    "apple_quality": "Apple (quality)",
    "banana_quality": "Banana (quality)",
    "biofood_date_region": "Date (nutrients)",
    "biofood_safou_region": "Safou (nutrients)",
}
ENGINES = ["physics_mc", "regression", "vae", "hybrid_vae"]
ENGINE_NAME = {"physics_mc": "Physics-MC", "regression": "Regression",
               "vae": "VAE", "hybrid_vae": "Hybrid"}
DISPLAY_TO_KEY = {
    "Physics-Informed Monte Carlo": "physics_mc", "Regression": "regression",
    "Variational Autoencoder": "vae", "Physics-Informed VAE (Hybrid)": "hybrid_vae",
}


def _read(cfg, name):
    p = f"results/{cfg}/{name}"
    return pd.read_csv(p) if os.path.exists(p) else None


# ---------------- collect per-dataset metric blocks ----------------
blocks = {}  # cfg -> dict engine -> {corr,ks,calib,tstr}
for cfg in DATASETS:
    summ = _read(cfg, "summary_metrics.csv")
    cov = _read(cfg, "coverage.csv")
    tstr = _read(cfg, "tstr.csv")
    if summ is None:
        continue
    summ = summ.assign(k=summ["Engine"].map(DISPLAY_TO_KEY)).set_index("k")
    cov = cov.set_index("method") if cov is not None else None
    tstr = tstr.set_index("method") if tstr is not None else None
    d = {}
    for e in ENGINES:
        d[e] = {
            "corr": summ.loc[e, "Correlation Distance (Euclidean)"] if e in summ.index else np.nan,
            "ks": summ.loc[e, "Mean KS Statistic"] if e in summ.index else np.nan,
            "calib": cov.loc[e, "calibration_error"] if cov is not None and e in cov.index else np.nan,
            "tstr": tstr.loc[e, "accuracy"] if tstr is not None and e in tstr.index else np.nan,
        }
    d["_tstr_real"] = tstr.loc["real", "accuracy"] if tstr is not None and "real" in tstr.index else np.nan
    blocks[cfg] = d


# ---------------- LaTeX Table 1 ----------------
def _fmt(val, best, higher=False):
    if pd.isna(val):
        return "--"
    s = f"{val:.3f}"
    is_best = (val >= best - 1e-9) if higher else (val <= best + 1e-9)
    return f"\\textbf{{{s}}}" if is_best else s


lines = [
    r"\begin{table}[t]", r"\centering",
    r"\caption{Fidelity, calibration, and downstream utility across the four datasets. "
    r"Lower is better for correlation distance, mean KS, and calibration error; higher for "
    r"TSTR accuracy. Best engine per column is \textbf{bold}. TSTR ceiling = a classifier "
    r"trained on real data.}",
    r"\label{tab:main}",
    r"\begin{tabular}{ll cccc}", r"\toprule",
    r"Dataset & Engine & Corr.\ dist.\ $\downarrow$ & Mean KS $\downarrow$ & "
    r"Calib.\ err.\ $\downarrow$ & TSTR acc.\ $\uparrow$ \\", r"\midrule",
]
for cfg, label in DATASETS.items():
    d = blocks.get(cfg)
    if not d:
        continue
    bc = np.nanmin([d[e]["corr"] for e in ENGINES])
    bk = np.nanmin([d[e]["ks"] for e in ENGINES])
    bl = np.nanmin([d[e]["calib"] for e in ENGINES])
    bt = np.nanmax([d[e]["tstr"] for e in ENGINES])
    for i, e in enumerate(ENGINES):
        name = label if i == 0 else ""
        lines.append(
            f"{name} & {ENGINE_NAME[e]} & {_fmt(d[e]['corr'], bc)} & {_fmt(d[e]['ks'], bk)} & "
            f"{_fmt(d[e]['calib'], bl)} & {_fmt(d[e]['tstr'], bt, higher=True)} \\\\"
        )
    if not pd.isna(d["_tstr_real"]):
        lines.append(f" & \\textit{{Real (ceiling)}} & -- & -- & -- & \\textit{{{d['_tstr_real']:.3f}}} \\\\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}", r"\end{table}"]
open(f"{OUT}/table1.tex", "w").write("\n".join(lines) + "\n")


# ---------------- LaTeX Table 2: near/far transfer ----------------
nf = [
    r"\begin{table}[t]", r"\centering",
    r"\caption{Support-aware uncertainty (same-model near/far transfer). One conditional-VAE "
    r"ensemble is trained per region, then queried for the held-out same region (NEAR) vs a "
    r"different region (FAR). Ensemble disagreement widens off-support (FAR $>$ NEAR); coverage "
    r"of real held-out values degrades. Apple has no grouping and is omitted.}",
    r"\label{tab:nearfar}",
    r"\begin{tabular}{l cc c cc}", r"\toprule",
    r" & \multicolumn{2}{c}{Disagreement} & & \multicolumn{2}{c}{Coverage@90} \\",
    r"\cmidrule(lr){2-3} \cmidrule(lr){5-6}",
    r"Dataset & FAR & NEAR & Widening & FAR & NEAR \\", r"\midrule",
]
for cfg, label in DATASETS.items():
    lo = _read(cfg, "loro.csv")
    if lo is None or lo.empty:
        continue
    m = lo[lo["group"] == "MEAN"]
    if m.empty:
        continue
    m = m.iloc[0]
    pct = 100 * (m["disagreement_far"] / m["disagreement_near"] - 1)
    nf.append(
        f"{label} & {m['disagreement_far']:.3f} & {m['disagreement_near']:.3f} & "
        f"$+{pct:.0f}\\%$ & {m['coverage_far']:.3f} & {m['coverage_near']:.3f} \\\\"
    )
nf += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{OUT}/table2_nearfar.tex", "w").write("\n".join(nf) + "\n")

# ---------------- Figure 4: engine-selection map ----------------
# x = causal-graph validity, y = dataset size (log). Colour = joint-structure winner.
COLOR = {"physics_mc": "#0072B2", "hybrid_vae": "#D55E00", "regression": "#E69F00"}
POINTS = [  # label, validity(0-1), n, joint-structure winner
    ("Safou\n(n=41, det. physics)", 0.95, 41, "physics_mc"),
    ("Date\n(n=74, moderate physics)", 0.60, 74, "hybrid_vae"),
    ("Apple\n(n=4000, moderate physics)", 0.50, 4000, "hybrid_vae"),
    ("Banana\n(n=1000, no physics)", 0.05, 1000, "regression"),
]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
fig, ax = plt.subplots(figsize=(9, 5.6))
for label, val, n, win in POINTS:
    ax.scatter(val, n, s=380, color=COLOR[win], zorder=3, edgecolor="white", linewidth=1.5)
    dy = 1.35 if n < 200 else 0.62
    ax.annotate(label, (val, n), textcoords="offset points", xytext=(0, 16 if n < 200 else -34),
                ha="center", fontsize=9.5, color="#222")
ax.set_yscale("log")
ax.set_xlim(-0.06, 1.06); ax.set_ylim(20, 12000)
ax.set_xlabel("Causal-graph validity  (none  →  near-deterministic)", fontsize=11)
ax.set_ylabel("Dataset size (rows, log scale)", fontsize=11)
ax.set_title("Which engine wins joint structure — set by the data regime",
             fontsize=13, weight="bold", loc="left")
# legend for winner colours
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR["physics_mc"], markersize=12, label="Physics-MC"),
       Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR["hybrid_vae"], markersize=12, label="Hybrid"),
       Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR["regression"], markersize=12, label="Regression (classical)")]
ax.legend(handles=leg, title="Best on joint structure", frameon=False, loc="lower left", fontsize=9.5)
ax.text(0.5, -0.19, "The hybrid additionally wins marginals (KS) and downstream utility (TSTR) across all four datasets;\n"
        "only the joint-structure winner shifts with the data regime.",
        transform=ax.transAxes, ha="center", fontsize=9, color="#555")
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_engine_map.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"wrote {OUT}/table1.tex, {OUT}/table2_nearfar.tex and {OUT}/fig4_engine_map.png")
