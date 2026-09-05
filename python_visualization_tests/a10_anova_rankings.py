# ============================================================
# A10 — Is the family effect significant? One-way ANOVA on
#   render time across the 7 families + a ranking map of which
#   strategy wins on each of the 47 datasets (1 = fastest).
# ============================================================
import viz_common as vc
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = vc.load()
groups = [g["render_mean_ms"].values for _, g in df.groupby("mode")]
f, p = stats.f_oneway(*groups)
print(f"One-way ANOVA render_mean_ms ~ mode: F={f:.2f}, p={p:.1e}\n")

rank = df.pivot_table(index="mode", columns="dataset", values="render_mean_ms")
rank_mean = rank.rank(axis=0, method="average").mean(axis=1).sort_values()
print("=== Average rank per family across datasets (1 = fastest) ===")
print(rank_mean.round(2).to_string())

best = rank.idxmin(axis=0)
print("\n=== Dataset-level winner (fastest render per dataset) ===")
print(best.value_counts().to_string())

fig, ax = plt.subplots(figsize=(11, 4.5))
rank_mean.plot.barh(color="steelblue", ax=ax)
ax.set_title("Average competition rank per strategy family (lower = faster)")
ax.set_xlabel("mean rank across the 47 datasets")
fig.tight_layout()
vc.save(fig, "a10_anova_rankings.png")