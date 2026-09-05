# ============================================================
# B9 — Pairplot: the numerical signature of every family.
#   Diagonal = KDE, off-diagonal = 2-D KDE/scatter density.
#   Restrict to eps=0.01 runs so every family is comparable.
# ============================================================
import viz_common as vc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = vc.load()
df = df[df["epsilon"] == 0.01]
cols = ["sparsity_pct", "vol_samples", "skip_ratio_bounds", "fps_mean", "render_mean_ms"]
for c in cols:
    if df[c].min() <= 0:
        df[c] = df[c] + 1e-6
df[cols] = df[cols].apply(lambda s: np.log(s))

g = sns.pairplot(df[cols + ["mode"]], hue="mode", corner=True, diag_kind="kde",
                 plot_kws=dict(alpha=0.45, s=16, edgecolor="none"),
                 diag_kws=dict(fill=True, alpha=0.4))
g.figure.suptitle("Pairplot of log-metrics (epsilon=0.01)", y=1.02)
g.figure.tight_layout()
g.figure.savefig(vc.PLOT_DIR + "/b9_pairplot.png", dpi=150, bbox_inches="tight")
print("  saved " + vc.PLOT_DIR + "/b9_pairplot.png")