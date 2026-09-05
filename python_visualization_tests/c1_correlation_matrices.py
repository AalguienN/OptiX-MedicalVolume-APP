# ============================================================
# C1 — Correlation matrices per family, split by epsilon.
#
#  PART 1:  one figure per epsilon (0.01..0.50), each with a
#          full Spearman correlation heatmap for every strategy
#          family present at that epsilon.
#  PART 2:  a compact "signature" heatmap per epsilon: rows =
#          families, cols = metrics, value = correlation of that
#          metric with render_mean_ms. Lets you eyeball which
#          family loads on which driver (sparsity? skipping?).
#
# Spearman is used because render times span 2 orders of magnitude
# (0.24 ms .. 37 ms) and monotonic != linear.
# ============================================================
import math

import viz_common as vc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = vc.load()
epsilons = sorted(df["epsilon"].unique())
metrics = ["render_mean_ms", "fps_mean", "gpu_mem_gb", "vol_samples",
           "sparsity_pct", "skip_ratio_bounds"]
sig_metrics = ["vol_samples", "sparsity_pct", "skip_ratio_bounds", "fps_mean", "gpu_mem_gb"]

# ---------------- PART 1: full matrices per family ----------------
for eps in epsilons:
    sub = df[df["epsilon"] == eps]
    modes = sorted(sub["mode"].unique())
    ncol, nrow = 2, math.ceil(len(modes) / 2)
    fig, axes = plt.subplots(nrow, ncol, figsize=(14.5, 4.0 * nrow), squeeze=False)
    for ax, mode in zip(axes.ravel(), modes):
        g = sub[sub["mode"] == mode][metrics].corr(method="spearman")
        im = ax.imshow(g.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, fontsize=6, rotation=45, ha="right")
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics, fontsize=7)
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                ax.text(j, i, f"{g.values[i, j]:.2f}", ha="center", va="center",
                        fontsize=5.5, color="black")
        ax.set_title(f"{mode}  (n={len(g)})", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.ravel()[len(modes):]:
        ax.axis("off")
    fig.suptitle(f"Spearman correlation within each family — epsilon = {eps}", fontsize=13)
    fig.tight_layout()
    vc.save(fig, f"c1_eps{str(eps)}_family_correlation.png")

# ---------------- PART 2: signature (metric-vs-render rho) ----------------
fig, axes = plt.subplots(1, len(epsilons), figsize=(6.0 * len(epsilons), 4.5), squeeze=False)
all_modes = sorted(df["mode"].unique())
for ax, eps in zip(axes[0], epsilons):
    sub = df[df["epsilon"] == eps]
    modes = sorted(sub["mode"].unique())
    M = pd.DataFrame(index=modes, columns=sig_metrics, dtype=float)
    for mode in modes:
        corr = sub[sub["mode"] == mode][["render_mean_ms"] + sig_metrics].corr(method="spearman")
        M.loc[mode] = corr.loc["render_mean_ms", sig_metrics].values
    M = M.reindex(all_modes)  # keep row order identical across panels
    im = ax.imshow(M.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_yticks(range(len(all_modes)))
    ax.set_yticklabels(all_modes, fontsize=7)
    ax.set_xticks(range(len(sig_metrics)))
    ax.set_xticklabels(sig_metrics, fontsize=6.5, rotation=45, ha="right")
    for i in range(len(all_modes)):
        for j in range(len(sig_metrics)):
            v = M.values[i, j]
            ax.text(j, i, "NaN" if np.isnan(v) else f"{v:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="black" if not np.isnan(v) else "gray")
    ax.set_title(f"epsilon = {eps}", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Correlation of each metric with render_mean_ms per family (Spearman) — "
             "NaNs = family not tested at that epsilon", fontsize=12)
fig.tight_layout()
vc.save(fig, "c1_epsilon_signature.png")

# machine-readable digest
rows = []
for eps in epsilons:
    sub = df[df["epsilon"] == eps]
    for mode in sorted(sub["mode"].unique()):
        corr = sub[sub["mode"] == mode][["render_mean_ms"] + sig_metrics].corr(method="spearman")
        r = {"epsilon": eps, "mode": mode}
        r.update({f"rho_{c}": round(corr.loc["render_mean_ms", c], 3) for c in sig_metrics})
        rows.append(r)
digest = pd.DataFrame(rows).set_index(["epsilon", "mode"])
print("\n=== rho(metric, render_mean_ms) per (epsilon, family) ===")
print(digest.to_string())