# ============================================================
# A8 — Navajo of acceleration: leaps and dist_reads are the
#   counters that adaptive/bricked strategies actually skip.
#   Do higher skip counts translate into bigger speedup?
# ============================================================
import viz_common as vc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df = vc.load()
base = df[df["mode"] == "manual"].groupby("dataset")["render_mean_ms"].median()
df["speedup"] = df.apply(lambda r: base[r["dataset"]] / r["render_mean_ms"], axis=1)

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
sns.scatterplot(data=df, x="leaps", y="speedup", hue="mode", alpha=0.55, s=22, ax=axes[0])
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_title("Speedup vs bounds leaps (log-log)")

sns.scatterplot(data=df, x="skip_ratio_bounds", y="speedup", hue="mode", alpha=0.55, s=22, ax=axes[1])
axes[1].set_yscale("log")
axes[1].set_title("Speedup vs skip_ratio_bounds")

sns.scatterplot(data=df, x="dist_reads", y="speedup", hue="mode", alpha=0.55, s=22, ax=axes[2])
axes[2].set_xscale("log"); axes[2].set_yscale("log")
axes[2].set_title("Speedup vs distance reads (log-log)")

for ax in axes:
    ax.legend([], [], frameon=False)
fig.tight_layout()
vc.save(fig, "a8_skip_counters_vs_speedup.png")

for col in ["leaps", "skip_ratio_bounds", "dist_reads"]:
    rho, p = stats.spearmanr(df[col], df["speedup"])
    print(f"speedup ~ {col}: rho={rho:+.2f} (p={p:.1e})")