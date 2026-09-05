# ============================================================
# A9 — Correlation report: Pearson & Spearman matrices for the
#   numeric features. Spearman catches monotonic (not just
#   linear) relations, useful with the wide dynamic ranges.
# ============================================================
import viz_common as vc
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

df = vc.load()
cols = ["render_mean_ms", "fps_mean", "gpu_mem_gb", "vol_samples", "dist_reads",
        "leaps", "skip_ratio_bounds", "sparsity_pct", "uploaded_mb", "num_slices"]
X = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

pear = X.corr(method="pearson")
spea = X.corr(method="spearman")

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
for ax, cm, title in [(axes[0], pear, "Pearson (linear)"), (axes[1], spea, "Spearman (monotonic)")]:
    im = ax.imshow(cm.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
vc.save(fig, "a9_correlation_matrices.png")

top = (spea["render_mean_ms"].drop("render_mean_ms").abs().sort_values(ascending=False))
print("=== Top predictors of render_mean_ms (Spearman |rho|) ===")
print(top.round(3).head(5).to_string())