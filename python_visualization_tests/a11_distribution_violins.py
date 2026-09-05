# ============================================================
# A11 — Full distributions: violin/KDE of the per-dataset
#   render time (log) and speedup, per strategy family.
# ============================================================
import viz_common as vc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = vc.load()
base = df[df["mode"] == "manual"].groupby("dataset")["render_mean_ms"].median()
df["speedup"] = df.apply(lambda r: base[r["dataset"]] / r["render_mean_ms"], axis=1)

fig, axes = plt.subplots(1, 2, figsize=(17, 6))
order = sorted(df["mode"].unique())
sns.violinplot(data=df, x="mode", y="render_mean_ms", order=order, ax=axes[0],
               inner="quartile", density_norm="width", cut=0)
axes[0].set_yscale("log")
axes[0].set_title("Render time distribution per family (log)")

sns.violinplot(data=df, x="mode", y="speedup", order=order, ax=axes[1],
               inner="quartile", density_norm="width", cut=0)
axes[1].set_yscale("log")
axes[1].axhline(1, ls="--", color="crimson")
axes[1].set_title("Speedup vs manual distribution (log, >1 = faster)")
for ax in axes:
    ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
vc.save(fig, "a11_distribution_violins.png")