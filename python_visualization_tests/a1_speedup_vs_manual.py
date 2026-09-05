# ============================================================
# A1 — Speedup vs the 'manual' baseline, per dataset (paired).
#   speedup > 1 means the strategy beat the full-sampling
#   manual render on that same dataset. Shows median gain,
#   spread, and % of datasets where each strategy wins.
# ============================================================
import viz_common as vc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = vc.load()
pivot = df.pivot_table(index="dataset", columns="mode", values="render_mean_ms")
speedup = pivot.div(pivot["manual"], axis=0).drop(columns="manual")
m = speedup.reset_index().melt(id_vars="dataset", var_name="mode", value_name="speedup")
m = m[m["speedup"].notna()]

wins = (m.groupby("mode")["speedup"]
         .apply(lambda s: (s > 1).mean() * 100).round(1).sort_values())
print("=== % of datasets where the strategy is faster than manual ===")
print(wins.to_string())
print(f"\nMedian speedup: {m.groupby('mode')['speedup'].median().round(3).to_string()}")

fig, ax = plt.subplots(figsize=(13, 7))
sns.boxplot(data=m, x="mode", y="speedup", ax=ax, fliersize=2)
ax.axhline(1, ls="--", color="crimson", lw=1.5)
ax.set_yscale("log")
ax.set_title("Speedup vs manual baseline per dataset (log scale, >1 = faster)")
ax.set_ylabel("speedup = manual_render / strategy_render")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
vc.save(fig, "a1_speedup_vs_manual.png")