# ============================================================
# B11 — Speedup also vs the OPTIX baseline (not just manual):
#   the interesting benchmark for a real pipeline is "do we
#   beat NVIDIA OptiX?" Boxplot of optix/strategy ratio and
#   % of datasets won vs each baseline.
# ============================================================
import viz_common as vc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = vc.load()
base = df[df["mode"] == "optix"].groupby("dataset")["render_mean_ms"].median()
df["speedup_vs_optix"] = df.apply(lambda r: base[r["dataset"]] / r["render_mean_ms"], axis=1)

m = df[df["mode"] != "optix"]
fig, ax = plt.subplots(figsize=(13, 6))
order = sorted(m["mode"].unique())
sns.boxplot(data=m, x="mode", y="speedup_vs_optix", order=order, ax=ax, fliersize=2)
ax.axhline(1, ls="--", color="crimson")
ax.set_yscale("log")
ax.set_title("Speedup vs OptiX baseline (1 = equal to OptiX, >1 = faster)")
ax.set_ylabel("optix_render / strategy_render (log)")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
vc.save(fig, "b11_speedup_vs_optix.png")

print("=== % of datasets beating OptiX per family ===")
print((m.groupby("mode")["speedup_vs_optix"].apply(lambda s: (s > 1).mean() * 100)
       .round(1).sort_values().to_string()))