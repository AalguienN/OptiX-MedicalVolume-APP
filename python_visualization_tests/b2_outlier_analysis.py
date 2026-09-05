# ============================================================
# B2 — Outlier analysis: how many points fall outside 1.5*IQR
#   per family, and how much the mean is dragged by them vs
#   the trimmed median. Plus a boxplot of raw vs winsorized.
# ============================================================
import viz_common as vc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = vc.load()
y = "render_mean_ms"

rows = []
for mode, g in df.groupby("mode"):
    out = vc.iqr_outliers(g[y])
    rows.append({
        "mode": mode,
        "n_outliers": int(out.sum()),
        "outlier_pct": 100 * out.mean(),
        "mean": g[y].mean(),
        "trimmed_mean": g.loc[~out, y].mean(),
        "median": g[y].median(),
    })
report = pd.DataFrame(rows).sort_values("outlier_pct", ascending=False)
print("=== Outlier report (render_mean_ms) ===")
print(report.round(3).to_string(index=False))
print("\n=== Outlier rows: which datasets trigger them ===")
for mode, g in df.groupby("mode"):
    out = g[vc.iqr_outliers(g[y])]
    if len(out):
        print(f"{mode:18s} {len(out):2d} outliers, slowest dataset: "
              f"{out.loc[out[y].idxmax(), 'dataset']} ({out[y].max():.2f} ms)")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.boxplot(data=df, x="mode", y=y, order=sorted(df["mode"].unique()), ax=axes[0], fliersize=2)
axes[0].set_title("Raw render time (outliers as visible dots)")
w = df.copy()
w[y] = vc.clip_col(df[y])
sns.boxplot(data=w, x="mode", y=y, order=sorted(df["mode"].unique()), ax=axes[1], fliersize=2)
axes[1].set_title("Winsorized render time (clipped at 1%/99% percentile)")
for ax in axes:
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
vc.save(fig, "b2_outlier_analysis.png")